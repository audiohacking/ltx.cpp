// ltx-generate.cpp – LTX-Video text-to-video / image-to-video inference
//
// Usage (text-to-video):
//   ltx-generate
//     --dit   models/ltxv-2b-Q8_0.gguf
//     --vae   models/ltxv-vae-BF16.gguf
//     --t5    models/t5-xxl-Q8_0.gguf
//     --prompt "A cat sitting on a bench, sunlit park"
//     --steps 40 --cfg 3.0 --shift 3.0
//     --frames 25 --height 480 --width 704
//     --out output/video
//
// Usage (image-to-video – animate a reference image):
//   ltx-generate ... --start-frame photo.ppm
//
// Usage (first+last frame – keyframe interpolation):
//   ltx-generate ... --start-frame begin.ppm --end-frame end.ppm
//
// Outputs: output/video_0000.ppm ... output/video_NNNN.ppm

#include "ltx_common.hpp"
#include "t5_encoder.hpp"
#include "video_vae.hpp"
#include "ltx_dit.hpp"
#include "scheduler.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>   // mkdir
#if defined(_WIN32)
#  include <direct.h>   // _mkdir
#endif

#if defined(__APPLE__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#elif defined(__linux__)
#  include <sys/sysinfo.h>
#endif

// Up to 90% of system RAM for DiT scratch (min 1 GB). Avoids pulling sys/param.h in the header on macOS.
size_t dit_scratch_size_bytes() {
    size_t total_bytes = 0;
#if defined(__APPLE__)
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, nullptr, 0) == 0 && memsize > 0)
        total_bytes = (size_t)memsize;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0 && si.totalram > 0)
        total_bytes = (size_t)si.totalram * (size_t)si.mem_unit;
#endif
    if (total_bytes == 0)
        return (size_t)8 * 1024 * 1024 * 1024; // fallback 8 GB
    size_t ninety = (size_t)((double)total_bytes * 0.9);
    size_t min_scratch = (size_t)1 * 1024 * 1024 * 1024; // at least 1 GB
    return ninety > min_scratch ? ninety : min_scratch;
}

// ── Image loading (single TU to avoid duplicate stb symbols) ─────────────────
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_ONLY_PNM
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

VideoBuffer load_image(const std::string & path) {
    int W = 0, H = 0, channels = 0;
    uint8_t * data = stbi_load(path.c_str(), &W, &H, &channels, 3);
    if (!data) {
        LTX_ERR("failed to load image '%s': %s", path.c_str(), stbi_failure_reason());
        return VideoBuffer(0, 0, 0);
    }
    VideoBuffer buf(1, H, W);
    memcpy(buf.frame(0), data, (size_t)W * H * 3);
    stbi_image_free(data);
    LTX_LOG("loaded image: %s (%dx%d, original channels=%d)", path.c_str(), W, H, channels);
    return buf;
}

std::vector<uint8_t> resize_bilinear(
        const uint8_t * src, int W_src, int H_src, int W_dst, int H_dst)
{
    std::vector<uint8_t> out(W_dst * H_dst * 3);
    float sx = (float)W_src / W_dst, sy = (float)H_src / H_dst;
    for (int yd = 0; yd < H_dst; ++yd)
    for (int xd = 0; xd < W_dst; ++xd) {
        float xf = (xd + 0.5f) * sx - 0.5f, yf = (yd + 0.5f) * sy - 0.5f;
        int x0 = std::max(0, (int)xf), x1 = std::min(W_src - 1, x0 + 1);
        int y0 = std::max(0, (int)yf), y1 = std::min(H_src - 1, y0 + 1);
        float qx = xf - x0, qy = yf - y0;
        for (int c = 0; c < 3; ++c) {
            float v00 = src[(y0 * W_src + x0) * 3 + c], v10 = src[(y0 * W_src + x1) * 3 + c];
            float v01 = src[(y1 * W_src + x0) * 3 + c], v11 = src[(y1 * W_src + x1) * 3 + c];
            float v = (1 - qy) * ((1 - qx) * v00 + qx * v10) + qy * ((1 - qx) * v01 + qx * v11);
            out[(yd * W_dst + xd) * 3 + c] = (uint8_t)(v + 0.5f);
        }
    }
    return out;
}

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    std::string dit_path;
    std::string vae_path;
    std::string t5_path;
    std::string prompt         = "A beautiful scenic landscape with flowing water.";
    std::string negative_prompt = "";
    std::string out_prefix     = "output/frame";
    // Image-to-video conditioning.
    std::string start_frame_path;  // path to start/reference frame (PPM)
    std::string end_frame_path;    // path to end frame (PPM), for keyframe interpolation
    float       frame_strength    = 1.0f; // conditioning strength [0,1]; 1=fully pinned
    int    frames              = 25;   // LTX rule: 8*n+1 (e.g. 25, 33, 97)
    int    height              = 480;  // must be divisible by 32
    int    width               = 704;  // must be divisible by 32
    int    steps               = 20;   // reference: LTXVScheduler 20 steps
    float  cfg_scale           = 4.0f; // reference: CFG 4 first stage
    float  shift               = 3.0f;
    uint64_t seed              = 42;
    int    threads             = 4;
    bool   verbose             = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  --dit    <path>   DiT model GGUF file\n"
        "  --vae    <path>   VAE model GGUF file\n"
        "  --t5     <path>   T5 text encoder GGUF file\n"
        "\n"
        "Generation:\n"
        "  --prompt  <text>  Positive prompt (default: landscape)\n"
        "  --neg     <text>  Negative prompt (default: empty)\n"
        "  --frames  <N>     Number of video frames, 8*n+1 (default: 25)\n"
        "  --height  <H>     Video height, divisible by 32 (default: 480)\n"
        "  --width   <W>     Video width, divisible by 32 (default: 704)\n"
        "  --steps   <N>     Denoising steps (default: 20)\n"
        "  --cfg     <f>     Classifier-free guidance scale (default: 4.0)\n"
        "  --shift   <f>     Flow-shift parameter (default: 3.0)\n"
        "  --seed    <N>     RNG seed (default: 42)\n"
        "  --out     <pfx>   Output frame prefix (default: output/frame)\n"
        "\n"
        "Image-to-video (I2V) conditioning:\n"
        "  --start-frame <path>  PNG/JPG/BMP/TGA/PPM image to use as the first frame / reference\n"
        "  --end-frame   <path>  PNG/JPG/BMP/TGA/PPM image to use as the last frame (keyframe interp)\n"
        "  --frame-strength <f> Conditioning strength [0..1] (default: 1.0)\n"
        "                        1.0 = fully pin the frame, 0.0 = no conditioning\n"
        "\n"
        "Performance:\n"
        "  --threads <N>     CPU threads (default: 4)\n"
        "  -v                Verbose logging\n",
        prog);
}

static Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto nextarg = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };
        std::string arg = argv[i];
        if      (arg == "--dit")     a.dit_path        = nextarg();
        else if (arg == "--vae")     a.vae_path        = nextarg();
        else if (arg == "--t5")      a.t5_path         = nextarg();
        else if (arg == "--prompt" || arg == "-p") a.prompt = nextarg();
        else if (arg == "--neg"    || arg == "-n") a.negative_prompt = nextarg();
        else if (arg == "--frames")  a.frames          = std::atoi(nextarg());
        else if (arg == "--height")  a.height          = std::atoi(nextarg());
        else if (arg == "--width")   a.width           = std::atoi(nextarg());
        else if (arg == "--steps")   a.steps           = std::atoi(nextarg());
        else if (arg == "--cfg")     a.cfg_scale       = std::atof(nextarg());
        else if (arg == "--shift")   a.shift           = std::atof(nextarg());
        else if (arg == "--seed")    a.seed            = std::stoull(nextarg());
        else if (arg == "--out")     a.out_prefix      = nextarg();
        else if (arg == "--threads") a.threads         = std::atoi(nextarg());
        else if (arg == "--start-frame")    a.start_frame_path  = nextarg();
        else if (arg == "--end-frame")      a.end_frame_path    = nextarg();
        else if (arg == "--frame-strength") a.frame_strength    = std::atof(nextarg());
        else if (arg == "-v")        a.verbose         = true;
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    return a;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Round up to nearest multiple of k.
[[maybe_unused]] static int round_up_mult(int v, int k) { return ((v + k - 1) / k) * k; }

// Compute latent dimensions from pixel dimensions.
static void latent_dims(const Args & a, VaeConfig & vc,
                        int & T_lat, int & H_lat, int & W_lat) {
    T_lat = (a.frames - 1) / vc.temporal_scale + 1;
    H_lat = a.height / vc.spatial_scale;
    W_lat = a.width  / vc.spatial_scale;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    Args args = parse_args(argc, argv);

    // Validate required paths.
    if (args.dit_path.empty() || args.vae_path.empty() || args.t5_path.empty()) {
        fprintf(stderr, "Error: --dit, --vae, and --t5 are all required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // LTX reference: frames = 8*n+1, width/height divisible by 32.
    if ((args.frames - 1) % 8 != 0) {
        LTX_ERR("frames must be 8*n+1 (e.g. 25, 33, 97), got %d", args.frames);
        return 1;
    }
    if (args.width % 32 != 0 || args.height % 32 != 0) {
        LTX_ERR("width and height must be divisible by 32, got %d x %d",
                args.width, args.height);
        return 1;
    }

    LTX_LOG("ltx-generate v0.1.0");
    LTX_LOG("prompt:   %s", args.prompt.c_str());
    LTX_LOG("frames=%d  height=%d  width=%d  steps=%d  cfg=%.1f  shift=%.1f  seed=%llu",
            args.frames, args.height, args.width, args.steps,
            (double)args.cfg_scale, (double)args.shift, (unsigned long long)args.seed);

    // ── Load models ───────────────────────────────────────────────────────────

    LTX_LOG("loading T5 text encoder: %s", args.t5_path.c_str());
    LtxGgufModel t5_model;
    if (!t5_model.open(args.t5_path)) return 1;
    T5Encoder t5;
    if (!t5.load(t5_model)) return 1;

    LTX_LOG("loading VAE decoder: %s", args.vae_path.c_str());
    LtxGgufModel vae_model;
    bool is_safetensors = (args.vae_path.find(".safetensors") != std::string::npos);
    bool vae_ok = is_safetensors
        ? vae_model.open_safetensors(args.vae_path)
        : vae_model.open(args.vae_path);
    if (!vae_ok) return 1;
    VaeDecoder vae;
    if (!vae.load(vae_model)) return 1;

    // Load VAE encoder (shares the same GGUF as the decoder).
    VaeEncoder vae_enc;
    vae_enc.load(vae_model);

    LTX_LOG("loading DiT: %s", args.dit_path.c_str());
    LtxGgufModel dit_model;
    if (!dit_model.open(args.dit_path)) return 1;
    LtxDiT dit;
    if (!dit.load(dit_model)) return 1;

    // ── Text encoding ─────────────────────────────────────────────────────────

    LTX_LOG("encoding prompt ...");
    std::vector<float> text_emb = t5.encode_text(args.prompt);
    int seq_len = (int)(text_emb.size() / t5.cfg.d_model);

    std::vector<float> uncond_emb;
    bool do_cfg = (args.cfg_scale > 1.0f);
    if (do_cfg) {
        LTX_LOG("encoding negative prompt ...");
        uncond_emb = t5.encode_text(args.negative_prompt);
    }

    // ── Latent dimensions ────────────────────────────────────────────────────

    int T_lat, H_lat, W_lat;
    latent_dims(args, vae.cfg, T_lat, H_lat, W_lat);

    // Ensure spatial dims are multiples of patch size.
    int pt = dit.cfg.patch_t, ph = dit.cfg.patch_h, pw = dit.cfg.patch_w;
    if (T_lat % pt != 0 || H_lat % ph != 0 || W_lat % pw != 0) {
        LTX_ERR("Latent dimensions (%d,%d,%d) not divisible by patch size (%d,%d,%d)",
                T_lat, H_lat, W_lat, pt, ph, pw);
        return 1;
    }

    int C = dit.cfg.latent_channels;
    int n_tok = (T_lat / pt) * (H_lat / ph) * (W_lat / pw);
    int Pd = dit.cfg.patch_dim();

    LTX_LOG("latent: T=%d H=%d W=%d C=%d  → %d tokens (patch_dim=%d)",
            T_lat, H_lat, W_lat, C, n_tok, Pd);

    // ── Encode reference frames (I2V conditioning) ────────────────────────────

    size_t frame_lat_size = (size_t)H_lat * W_lat * C;

    // start_lat / end_lat: encoded reference frame latents (empty = not set).
    std::vector<float> start_lat, end_lat;
    bool has_start = !args.start_frame_path.empty();
    bool has_end   = !args.end_frame_path.empty();

    if (has_start) {
        LTX_LOG("loading start frame: %s", args.start_frame_path.c_str());
        VideoBuffer img = load_image(args.start_frame_path);
        if (img.frames == 0) return 1;
        start_lat = vae_enc.encode_frame(img.frame(0),
            img.height, img.width, H_lat, W_lat);
        LTX_LOG("start frame encoded to latent [%d x %d x %d]", H_lat, W_lat, C);
    }

    if (has_end) {
        LTX_LOG("loading end frame: %s", args.end_frame_path.c_str());
        VideoBuffer img = load_image(args.end_frame_path);
        if (img.frames == 0) return 1;
        end_lat = vae_enc.encode_frame(img.frame(0),
            img.height, img.width, H_lat, W_lat);
        LTX_LOG("end frame encoded to latent [%d x %d x %d]", H_lat, W_lat, C);
    }

    if (has_start)
        LTX_LOG("mode: image-to-video (I2V) with start frame, strength=%.2f",
                (double)args.frame_strength);
    if (has_end)
        LTX_LOG("mode: keyframe interpolation with end frame");

    // ── Initialize random latents ─────────────────────────────────────────────

    LtxRng rng(args.seed);
    size_t lat_size = (size_t)T_lat * H_lat * W_lat * C;
    std::vector<float> latents(lat_size);
    rng.fill(latents.data(), lat_size);

    // ── Persistent DiT scratch (gpt-2 style: one buffer, reused each forward) ─
    size_t dit_scratch_size = dit_scratch_size_bytes();
    void * dit_scratch_buf = std::malloc(dit_scratch_size);
    if (!dit_scratch_buf) {
        LTX_ERR("failed to allocate DiT scratch (%.1f GB)", (double)dit_scratch_size / (1024 * 1024 * 1024));
        return 1;
    }
    LTX_LOG("DiT scratch: %.1f GB", (double)dit_scratch_size / (1024 * 1024 * 1024));

    // ── Denoising loop ────────────────────────────────────────────────────────

    RFScheduler sched(args.steps, args.shift, do_cfg);
    std::vector<float> ts = sched.timesteps();

    LTX_LOG("starting denoising (%d steps) ...", args.steps);

    for (int step = 0; step < args.steps; ++step) {
        float t_cur  = ts[step];
        float t_next = ts[step + 1];

        if (args.verbose) {
            LTX_LOG("  step %d/%d  t=%.4f → %.4f", step + 1, args.steps, (double)t_cur, (double)t_next);
        } else {
            fprintf(stderr, "\r[ltx] step %d/%d  t=%.3f", step + 1, args.steps, (double)t_cur);
            fflush(stderr);
        }

        // Patchify latent.
        std::vector<float> patches = patchify(
            latents.data(), T_lat, H_lat, W_lat, C, pt, ph, pw);

        // Conditional velocity.
        std::vector<float> v_cond = dit.forward(
            patches.data(), n_tok, text_emb.data(), seq_len, t_cur,
            dit_scratch_buf, dit_scratch_size);

        if (v_cond.empty()) { std::free(dit_scratch_buf); return 1; }

        // Unpatchify velocity.
        std::vector<float> vel_cond = unpatchify(
            v_cond.data(), T_lat, H_lat, W_lat, C, pt, ph, pw);

        std::vector<float> velocity(lat_size);

        if (do_cfg) {
            // Unconditional velocity.
            std::vector<float> v_uncond = dit.forward(
                patches.data(), n_tok, uncond_emb.data(), seq_len, t_cur,
                dit_scratch_buf, dit_scratch_size);
            if (v_uncond.empty()) { std::free(dit_scratch_buf); return 1; }
            std::vector<float> vel_uncond = unpatchify(
                v_uncond.data(), T_lat, H_lat, W_lat, C, pt, ph, pw);
            RFScheduler::apply_cfg(
                velocity.data(), vel_cond.data(), vel_uncond.data(),
                args.cfg_scale, lat_size);
        } else {
            velocity = vel_cond;
        }

        // Euler step.
        RFScheduler::euler_step(latents.data(), velocity.data(),
                                t_cur, t_next, lat_size);

        // ── Frame conditioning: pin start / end latent frames ──────────────
        // After each Euler step we re-impose the reference frame(s) to prevent
        // the denoising process from drifting away from the conditioning.
        // Blend weight increases linearly from 0 (at t=1, pure noise) to
        // frame_strength (at t=0, clean image), so early steps allow global
        // structure to form freely while later steps are progressively more
        // pinned to the reference frame.
        //
        //   blend = frame_strength * (1 - t_next)   ∈ [0, frame_strength]
        //   lat_frame = lat_frame * (1 - blend) + ref_lat * blend
        //
        // This approach requires no modifications to the DiT architecture.
        if ((has_start || has_end) && args.frame_strength > 0.0f) {
            // Blend increases as t_next approaches 0 (clean image).
            float blend = args.frame_strength * (1.0f - t_next);
            blend = std::max(0.0f, std::min(1.0f, blend));

            if (has_start && blend > 0.0f) {
                float * lat_t0 = latents.data(); // first temporal frame
                for (size_t i = 0; i < frame_lat_size; ++i)
                    lat_t0[i] = lat_t0[i] * (1.0f - blend) + start_lat[i] * blend;
            }
            if (has_end && blend > 0.0f) {
                float * lat_tn = latents.data() + (T_lat - 1) * frame_lat_size;
                for (size_t i = 0; i < frame_lat_size; ++i)
                    lat_tn[i] = lat_tn[i] * (1.0f - blend) + end_lat[i] * blend;
            }
        }
    }
    fprintf(stderr, "\n");
    std::free(dit_scratch_buf);
    dit_scratch_buf = nullptr;

    LTX_LOG("denoising complete, decoding with VAE ...");

    // ── Hard-pin reference frames at t=0 (post-denoising) ─────────────────────
    // After denoising completes, fully replace the first/last latent with the
    // encoded reference frame.  This ensures the output frame exactly matches
    // the reference image in appearance regardless of frame_strength.
    if (has_start && args.frame_strength >= 1.0f) {
        float * lat_t0 = latents.data();
        memcpy(lat_t0, start_lat.data(), frame_lat_size * sizeof(float));
        LTX_LOG("start frame latent hard-pinned at t=0");
    }
    if (has_end && args.frame_strength >= 1.0f) {
        float * lat_tn = latents.data() + (T_lat - 1) * frame_lat_size;
        memcpy(lat_tn, end_lat.data(), frame_lat_size * sizeof(float));
        LTX_LOG("end frame latent hard-pinned at t=0");
    }

    // ── VAE decode ────────────────────────────────────────────────────────────

    std::vector<float> pixels = vae.decode(latents.data(), T_lat, H_lat, W_lat);

    int T_vid = (T_lat - 1) * vae.cfg.temporal_scale + 1;
    int H_vid = H_lat * vae.cfg.spatial_scale;
    int W_vid = W_lat * vae.cfg.spatial_scale;

    // ── Save frames ───────────────────────────────────────────────────────────

    // Create output directory if prefix has a directory component.
    {
        std::string pfx = args.out_prefix;
        size_t slash = pfx.rfind('/');
        if (slash != std::string::npos) {
            std::string dir = pfx.substr(0, slash);
#if defined(_WIN32)
            // On Windows use _mkdir (no -p equivalent, best effort).
            (void)_mkdir(dir.c_str());
#else
            // Best-effort recursive directory creation.
            for (size_t i = 1; i <= dir.size(); ++i) {
                if (i == dir.size() || dir[i] == '/') {
                    std::string sub = dir.substr(0, i);
                    mkdir(sub.c_str(), 0755);
                }
            }
#endif
        }
    }

    VideoBuffer vbuf(T_vid, H_vid, W_vid);
    for (int f = 0; f < T_vid; ++f) {
        const float * src = pixels.data() + f * H_vid * W_vid * 3;
        uint8_t * dst = vbuf.frame(f);
        for (int i = 0; i < H_vid * W_vid * 3; ++i)
            dst[i] = VideoBuffer::clamp_u8(src[i]);
    }

    write_video_frames(vbuf, args.out_prefix);

    LTX_LOG("done. %d frames written to '%s_XXXX.ppm'", T_vid, args.out_prefix.c_str());
    if (has_start || has_end) {
        LTX_LOG("I2V conditioning applied: start=%s  end=%s  strength=%.2f",
                has_start ? args.start_frame_path.c_str() : "(none)",
                has_end   ? args.end_frame_path.c_str()   : "(none)",
                (double)args.frame_strength);
    }
    LTX_LOG("tip: convert PPM frames to MP4 with:");
    LTX_LOG("  ffmpeg -framerate 24 -i '%s_%%04d.ppm' -c:v libx264 output.mp4",
            args.out_prefix.c_str());

    return 0;
}

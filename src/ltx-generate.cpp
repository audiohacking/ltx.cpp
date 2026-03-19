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
#include "ltx_lora.hpp"
#include "audio_vae.hpp"
#include "scheduler.hpp"
#include "ltx_perf.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <sys/stat.h>   // mkdir
#if defined(_WIN32)
#  include <direct.h>   // _mkdir
#endif

// ── WAV output (AV pipeline) ──────────────────────────────────────────────────
// Writes 16-bit mono PCM WAV. sample_rate typically 16000.
static bool write_wav(const std::string & path,
        const float * samples, size_t num_samples, int sample_rate)
{
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        LTX_ERR("cannot create WAV file: %s", path.c_str());
        return false;
    }
    size_t data_bytes = num_samples * 2;  // 16-bit
    unsigned char header[44] = {
        'R','I','F','F',
        0,0,0,0,  // file size - 8
        'W','A','V','E',
        'f','m','t',' ',
        16,0,0,0,  // fmt chunk size
        1,0,       // PCM
        1,0,       // mono
        0,0,0,0,  // sample rate (fill below)
        0,0,0,0,  // byte rate
        2,0,       // block align
        16,0,     // bits per sample
        'd','a','t','a',
        0,0,0,0   // data size
    };
    uint32_t file_size = (uint32_t)(36 + data_bytes);
    uint32_t sr = (uint32_t)sample_rate;
    uint32_t byte_rate = sr * 2;
    header[4] = (unsigned char)(file_size);       header[5] = (unsigned char)(file_size>>8);
    header[6] = (unsigned char)(file_size>>16); header[7] = (unsigned char)(file_size>>24);
    header[24] = (unsigned char)(sr);           header[25] = (unsigned char)(sr>>8);
    header[26] = (unsigned char)(sr>>16);       header[27] = (unsigned char)(sr>>24);
    header[28] = (unsigned char)(byte_rate);    header[29] = (unsigned char)(byte_rate>>8);
    header[30] = (unsigned char)(byte_rate>>16); header[31] = (unsigned char)(byte_rate>>24);
    uint32_t ds = (uint32_t)data_bytes;
    header[40] = (unsigned char)(ds);            header[41] = (unsigned char)(ds>>8);
    header[42] = (unsigned char)(ds>>16);        header[43] = (unsigned char)(ds>>24);
    if (fwrite(header, 1, 44, f) != 44) { fclose(f); return false; }
    for (size_t i = 0; i < num_samples; ++i) {
        float s = samples[i];
        s = std::max(-1.0f, std::min(1.0f, s));
        int16_t v = (int16_t)(s * 32767.0f);
        unsigned char b[2] = { (unsigned char)(v & 0xff), (unsigned char)(v >> 8) };
        if (fwrite(b, 1, 2, f) != 2) { fclose(f); return false; }
    }
    fclose(f);
    LTX_LOG("WAV written: %s (%zu samples, %d Hz)", path.c_str(), num_samples, sample_rate);
    return true;
}

// Build a crude waveform from audio latent for AV pipeline (no full audio VAE decoder yet).
// Latent [T, 8, 16] -> fake mel (T*4, 64) -> overlap-add with sinusoids -> float samples.
// sample_rate=16000, hop_length=160, mel_bins=64 (LTX reference).
static std::vector<float> latent_to_waveform(
        const float * lat, int T_lat, int sample_rate, int hop_length, int mel_bins)
{
    const int T_mel = T_lat * 4;  // LATENT_DOWNSAMPLE_FACTOR
    const size_t num_samples = (size_t)T_mel * hop_length;
    std::vector<float> out(num_samples, 0.0f);
    // Mel center frequencies (Hz) for 64 bins, 0..8000 Hz approx
    std::vector<float> mel_centers((size_t)mel_bins);
    for (int b = 0; b < mel_bins; ++b) {
        float m = (float)b / (mel_bins - 1) * 2595.0f * std::log10(1.0f + 8000.0f / 700.0f);
        mel_centers[(size_t)b] = 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f);
    }
    for (int t = 0; t < T_mel; ++t) {
        int t_lat = t / 4;
        if (t_lat >= T_lat) t_lat = T_lat - 1;
        size_t sample_start = (size_t)t * hop_length;
        for (int b = 0; b < mel_bins; ++b) {
            // Map 64 mel bins from latent (8 ch, 16 freq): use ch = b/16, f = b%16 (first 4 ch)
            int c = b / 16, f = b % 16;
            if (c >= 8) c = 7;
            float mag = lat[((size_t)t_lat * 8 + c) * 16 + f];
            mag = std::max(0.0f, std::min(1.0f, 0.5f + 0.5f * mag));  // scale latent to positive
            float phase = 0.0f;  // fixed phase for simplicity
            float f_hz = mel_centers[(size_t)b];
            for (int i = 0; i < hop_length && sample_start + i < num_samples; ++i) {
                float x = (float)((int)sample_start + i) / (float)sample_rate;
                out[sample_start + i] += mag * std::cos(2.0f * 3.14159265f * f_hz * x + phase);
            }
        }
    }
    // Normalize
    float max_val = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        float a = std::fabs(out[i]);
        if (a > max_val) max_val = a;
    }
    if (max_val > 1e-6f) {
        float scale = 0.95f / max_val;
        for (size_t i = 0; i < num_samples; ++i) out[i] *= scale;
    }
    return out;
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
    // Audio-video: enable AV path, audio VAE path, WAV output
    bool        av              = false;
    std::string audio_vae_path;    // required when --av
    std::string out_wav;           // WAV output path when --av (default: out_prefix + .wav)
    // Image-to-video conditioning.
    std::string start_frame_path;  // path to start/reference frame (PPM)
    std::string end_frame_path;    // path to end frame (PPM), for keyframe interpolation
    float       frame_strength    = 1.0f; // conditioning strength [0,1]; 1=fully pinned
    int    frames              = 25;   // LTX rule: 8*n+1 (e.g. 25, 33, 97)
    int    height              = 480;  // must be divisible by 32
    int    width               = 704;  // must be divisible by 32
    int    frames_per_second   = 24;   // frame rate (for conditioning + output mux hint)
    int    steps               = 20;   // reference: LTXVScheduler 20 steps
    float  cfg_scale           = 4.0f; // reference: CFG 4 first stage
    float  shift               = 0.0f; // legacy fixed shift (0 = use adaptive formula)
    float  max_shift           = 2.05f;  // LTXVScheduler adaptive shift max
    float  base_shift          = 0.95f;  // LTXVScheduler adaptive shift base
    float  terminal            = 0.1f;   // stop sigma (LTXVScheduler default)
    bool   stretch             = true;   // stretch sigmas to [terminal, 1.0]
    // LoRA (distilled model)
    std::string lora_path;               // path to .safetensors LoRA
    float  lora_scale          = 1.0f;   // LoRA strength multiplier
    // Two-stage pipeline
    bool   two_stage           = false;  // run stage-1 at half-res, stage-2 with LoRA
    uint64_t seed              = 42;
    int    threads             = 4;
    bool   verbose             = false;
    bool   perf               = false;   // --perf: background CPU/RAM stats every 10 s
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
        "  --fps     <N>     Output frame rate for conditioning (default: 24)\n"
        "  --steps   <N>     Denoising steps (default: 20)\n"
        "  --cfg     <f>     Classifier-free guidance scale (default: 4.0)\n"
        "  --shift   <f>     Fixed flow-shift (overrides adaptive; default: adaptive)\n"
        "  --max-shift <f>   Adaptive shift max (default: 2.05)\n"
        "  --base-shift <f>  Adaptive shift base (default: 0.95)\n"
        "  --terminal <f>    Stop sigma; skip near-clean tail (default: 0.1)\n"
        "  --no-stretch      Disable sigma stretch (stretch on by default)\n"
        "  --lora    <path>  Distilled LoRA safetensors (optional)\n"
        "  --lora-scale <f> LoRA strength (default: 1.0)\n"
        "  --two-stage       Run two-stage pipeline: stage-1 half-res + stage-2 full-res with LoRA\n"
        "  --seed    <N>     RNG seed (default: 42)\n"
        "  --out     <pfx>   Output frame prefix (default: output/frame)\n"
        "\n"
        "Audio-video (AV) pipeline:\n"
        "  --av              Enable audio+video (concat video+audio latent, DiT, split, decode both)\n"
        "  --audio-vae <path>  Audio VAE safetensors (optional; when omitted, audio from latent fallback)\n"
        "  --out-wav  <path>   Output WAV path (default: <out prefix>.wav when --av)\n"
        "\n"
        "Image-to-video (I2V) conditioning:\n"
        "  --start-frame <path>  PNG/JPG/BMP/TGA/PPM image to use as the first frame / reference\n"
        "  --end-frame   <path>  PNG/JPG/BMP/TGA/PPM image to use as the last frame (keyframe interp)\n"
        "  --frame-strength <f> Conditioning strength [0..1] (default: 1.0)\n"
        "                        1.0 = fully pin the frame, 0.0 = no conditioning\n"
        "\n"
        "Performance:\n"
        "  --threads <N>     CPU threads (default: 4)\n"
        "  -v                Verbose logging\n"
        "  --perf            Print CPU/RAM stats to stderr every 10 s\n",
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
        else if (arg == "--fps")        a.frames_per_second = std::atoi(nextarg());
        else if (arg == "--steps")      a.steps           = std::atoi(nextarg());
        else if (arg == "--cfg")        a.cfg_scale       = std::atof(nextarg());
        else if (arg == "--shift")      a.shift           = std::atof(nextarg());
        else if (arg == "--max-shift")  a.max_shift       = std::atof(nextarg());
        else if (arg == "--base-shift") a.base_shift      = std::atof(nextarg());
        else if (arg == "--terminal")   a.terminal        = std::atof(nextarg());
        else if (arg == "--no-stretch")  a.stretch         = false;
        else if (arg == "--lora")        a.lora_path       = nextarg();
        else if (arg == "--lora-scale")  a.lora_scale      = std::atof(nextarg());
        else if (arg == "--two-stage")   a.two_stage       = true;
        else if (arg == "--seed")        a.seed            = std::stoull(nextarg());
        else if (arg == "--out")     a.out_prefix      = nextarg();
        else if (arg == "--threads") a.threads         = std::atoi(nextarg());
        else if (arg == "--start-frame")    a.start_frame_path  = nextarg();
        else if (arg == "--end-frame")      a.end_frame_path    = nextarg();
        else if (arg == "--frame-strength") a.frame_strength    = std::atof(nextarg());
        else if (arg == "--av")             a.av                = true;
        else if (arg == "--audio-vae")      a.audio_vae_path    = nextarg();
        else if (arg == "--out-wav")        a.out_wav           = nextarg();
        else if (arg == "-v")        a.verbose         = true;
        else if (arg == "--perf")    a.perf            = true;
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    return a;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Round up to nearest multiple of k.
[[maybe_unused]] static int round_up_mult(int v, int k) { return ((v + k - 1) / k) * k; }

// Bilinear upscale latent [T, H_src, W_src, C] → [T, H_dst, W_dst, C].
// Used for stage-1→stage-2 transition in the two-stage pipeline.
static std::vector<float> latent_upsample_bilinear(
        const float * src, int T, int H_src, int W_src, int C,
        int H_dst, int W_dst)
{
    std::vector<float> dst((size_t)T * H_dst * W_dst * C, 0.0f);
    float sx = (float)W_src / W_dst, sy = (float)H_src / H_dst;
    for (int t = 0; t < T; ++t)
    for (int yd = 0; yd < H_dst; ++yd)
    for (int xd = 0; xd < W_dst; ++xd) {
        float xf = (xd + 0.5f) * sx - 0.5f, yf = (yd + 0.5f) * sy - 0.5f;
        int x0 = std::max(0, (int)xf), x1 = std::min(W_src - 1, x0 + 1);
        int y0 = std::max(0, (int)yf), y1 = std::min(H_src - 1, y0 + 1);
        float qx = xf - x0, qy = yf - y0;
        const float * s = src + (size_t)t * H_src * W_src * C;
        float * d = dst.data() + ((size_t)t * H_dst * W_dst + yd * W_dst + xd) * C;
        for (int c = 0; c < C; ++c) {
            float v00 = s[(y0 * W_src + x0) * C + c], v10 = s[(y0 * W_src + x1) * C + c];
            float v01 = s[(y1 * W_src + x0) * C + c], v11 = s[(y1 * W_src + x1) * C + c];
            d[c] = (1 - qy) * ((1 - qx) * v00 + qx * v10) + qy * ((1 - qx) * v01 + qx * v11);
        }
    }
    return dst;
}

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
    // --audio-vae is optional with --av: when omitted, audio is synthesized from the denoised latent (fallback).
    if (args.av && args.out_wav.empty()) {
        args.out_wav = args.out_prefix;
        size_t slash = args.out_wav.rfind('/');
        size_t dot   = args.out_wav.find('.', slash == std::string::npos ? 0 : slash);
        if (dot != std::string::npos)
            args.out_wav = args.out_wav.substr(0, dot);
        args.out_wav += ".wav";
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

    // Optional audio VAE decoder.
    AudioVaeDecoder audio_vae;
    bool audio_vae_loaded = false;
    if (args.av && !args.audio_vae_path.empty()) {
        LTX_LOG("loading audio VAE: %s", args.audio_vae_path.c_str());
        audio_vae_loaded = audio_vae.load(args.audio_vae_path);
        if (!audio_vae_loaded)
            LTX_ERR("audio VAE load failed — falling back to latent synthesis");
    }

    // Optional LoRA (distilled model).
    LtxLoRA lora;
    const LtxLoRA * lora_ptr = nullptr;
    std::vector<ggml_backend_buffer_t> lora_weight_buffers;
    if (!args.lora_path.empty()) {
        LTX_LOG("loading LoRA: %s  scale=%.2f", args.lora_path.c_str(), (double)args.lora_scale);
        if (!lora.load(args.lora_path, args.lora_scale)) {
            LTX_ERR("LoRA load failed — continuing without LoRA");
        } else {
            lora_ptr = &lora;
        }
    }
    // --two-stage requires a LoRA.
    if (args.two_stage && !lora_ptr) {
        LTX_ERR("--two-stage requires --lora <path>");
        return 1;
    }

    // ── Backend: load all plugins, then build Metal+CPU scheduler ───────────
    ggml_backend_load_all();

    int n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
    LTX_LOG("CPU threads: %d", n_threads);

    ggml_backend_t gpu_backend = ggml_backend_init_best();
    if (gpu_backend && ggml_backend_is_cpu(gpu_backend)) {
        // init_best returned CPU — we already have one
        ggml_backend_free(gpu_backend);
        gpu_backend = nullptr;
    }

    std::vector<ggml_backend_buffer_t> dit_weight_buffers;
    ggml_backend_t migrate_target = gpu_backend ? gpu_backend : cpu_backend;
    {
        LTX_LOG("backend: %s", ggml_backend_name(migrate_target));
        int n_bufs = ltx_backend_migrate_ctx(dit_model.ggml_ctx, migrate_target, dit_weight_buffers);
        if (n_bufs > 0) {
            size_t total_mb = 0;
            for (ggml_backend_buffer_t b : dit_weight_buffers) total_mb += ggml_backend_buffer_get_size(b) / (1024 * 1024);
            LTX_LOG("DiT weights on %s (%d buffers, %zu MB)", ggml_backend_name(migrate_target), n_bufs, total_mb);
        } else {
            LTX_LOG("DiT weight migration failed — running on CPU");
            if (gpu_backend) { ggml_backend_free(gpu_backend); gpu_backend = nullptr; }
        }
    }

    // Scheduler: [GPU, CPU] (or CPU-only). Automatically routes ops to the best available backend.
    ggml_backend_t sched_backends[2];
    int n_sched = 0;
    if (gpu_backend) sched_backends[n_sched++] = gpu_backend;
    sched_backends[n_sched++] = cpu_backend;
    ggml_backend_sched_t sched = ggml_backend_sched_new(sched_backends, nullptr, n_sched, 8192, false, true);
    LTX_LOG("scheduler: %d backend(s)", n_sched);

    // ── LoRA GPU upload ───────────────────────────────────────────────────────
    // Migrate all LoRA A/B matrices to the same backend as the DiT weights so
    // the LoRA delta matmuls run on Metal/GPU instead of CPU.
    if (lora_ptr) {
        ggml_backend_t lora_backend = gpu_backend ? gpu_backend : cpu_backend;
        LTX_LOG("uploading LoRA weights to %s ...", ggml_backend_name(lora_backend));
        if (!lora.gpu_upload(lora_backend, lora_weight_buffers)) {
            LTX_ERR("LoRA GPU upload failed — LoRA delta matmuls will run on CPU (slow)");
        }
    }

    // ── Perf monitor (optional --perf flag) ──────────────────────────────────
    size_t gpu_weight_mb = 0;
    for (ggml_backend_buffer_t b : dit_weight_buffers) gpu_weight_mb += ggml_backend_buffer_get_size(b) / (1024 * 1024);
    std::string backend_name = gpu_backend ? std::string(ggml_backend_name(gpu_backend)) : "CPU";
    std::unique_ptr<LtxPerfMonitor> perf_mon;
    if (args.perf)
        perf_mon = std::make_unique<LtxPerfMonitor>(10, backend_name, gpu_weight_mb);

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

    // Audio latent (AV pipeline): same T as video, C_audio=8, mel_bins=16 → n_audio_tok = T_lat, Pd_audio=128
    const int C_audio = 8, F_audio = 16;
    int n_audio_tok = args.av ? T_lat : 0;
    int n_tok_total = n_tok + n_audio_tok;

    LTX_LOG("latent: T=%d H=%d W=%d C=%d  → %d tokens (patch_dim=%d)",
            T_lat, H_lat, W_lat, C, n_tok, Pd);
    if (args.av)
        LTX_LOG("AV: audio latent T=%d C=%d F=%d  → %d audio tokens, total tokens %d",
                T_lat, C_audio, F_audio, n_audio_tok, n_tok_total);

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

    size_t audio_lat_size = args.av ? (size_t)T_lat * C_audio * F_audio : 0;
    std::vector<float> audio_latents;
    if (args.av) {
        audio_latents.resize(audio_lat_size);
        rng.fill(audio_latents.data(), audio_lat_size);
    }

    // ── Two-stage pipeline: stage-1 at half-resolution ───────────────────────
    // ComfyUI reference: stage-1 at 640×360 (H/2, W/2), 20 steps, CFG=4, dev model (no LoRA).
    // Then upsample latent and run stage-2 with LoRA, 4 steps, CFG=1.

    using wall_clock = std::chrono::steady_clock;
    using wall_sec   = std::chrono::duration<double>;

    if (args.two_stage) {
        // Stage-1 latent dims: half H/W, same T.
        int H_lat_1 = (H_lat / 2 / ph) * ph;  // align to patch size
        int W_lat_1 = (W_lat / 2 / pw) * pw;
        if (H_lat_1 < ph) H_lat_1 = ph;
        if (W_lat_1 < pw) W_lat_1 = pw;
        int n_tok_1 = (T_lat / pt) * (H_lat_1 / ph) * (W_lat_1 / pw);
        size_t lat_size_1 = (size_t)T_lat * H_lat_1 * W_lat_1 * C;

        LTX_LOG("Two-stage: stage-1 T=%d H=%d W=%d (%d tokens), 20 steps, cfg=%.1f",
                T_lat, H_lat_1, W_lat_1, n_tok_1, (double)args.cfg_scale);

        // Fresh latents at half-res (same seed for reproducibility).
        std::vector<float> latents_1(lat_size_1);
        { LtxRng rng1(args.seed); rng1.fill(latents_1.data(), lat_size_1); }

        // Stage-1 scheduler: same LTXVScheduler params, 20 steps, full CFG.
        const int s1_steps = 20;
        RFScheduler sched_1(s1_steps, 0.0f, args.cfg_scale > 1.0f,
                            args.max_shift, args.base_shift, args.terminal, args.stretch);
        sched_1.set_shift_from_tokens(n_tok_1);
        std::vector<float> ts_1 = sched_1.timesteps();
        LTX_LOG("stage-1 scheduler: shift=%.3f", (double)sched_1.shift);

        auto t_s1_start = wall_clock::now();
        for (int step = 0; step < s1_steps; ++step) {
            float t_cur = ts_1[step], t_next = ts_1[step + 1];
            auto t_s = wall_clock::now();
            fprintf(stderr, "\r[ltx] stage-1 step %2d/%d  t=%.3f  ", step + 1, s1_steps, (double)t_cur);
            fflush(stderr);

            std::vector<float> patches_1 = patchify(
                latents_1.data(), T_lat, H_lat_1, W_lat_1, C, pt, ph, pw);
            float fps = (float)args.frames_per_second;

            // Stage-1: dev model, no LoRA.
            std::vector<float> vc_1 = dit.forward(
                patches_1.data(), n_tok_1, text_emb.data(), seq_len, t_cur, fps, sched, nullptr);
            if (vc_1.empty()) { LTX_ERR("stage-1 DiT forward failed"); return 1; }

            std::vector<float> vel_c_1 = unpatchify(vc_1.data(), T_lat, H_lat_1, W_lat_1, C, pt, ph, pw);
            std::vector<float> vel_1(lat_size_1);
            if (args.cfg_scale > 1.0f) {
                std::vector<float> vu_1 = dit.forward(
                    patches_1.data(), n_tok_1, uncond_emb.data(), seq_len, t_cur, fps, sched, nullptr);
                if (vu_1.empty()) { LTX_ERR("stage-1 uncond DiT forward failed"); return 1; }
                std::vector<float> vel_u_1 = unpatchify(vu_1.data(), T_lat, H_lat_1, W_lat_1, C, pt, ph, pw);
                RFScheduler::apply_cfg(vel_1.data(), vel_c_1.data(), vel_u_1.data(),
                                       args.cfg_scale, lat_size_1);
            } else {
                vel_1 = vel_c_1;
            }
            RFScheduler::euler_step(latents_1.data(), vel_1.data(), t_cur, t_next, lat_size_1);

            double step_s = wall_sec(wall_clock::now() - t_s).count();
            fprintf(stderr, "  %.1fs/step\n", step_s);
        }
        fprintf(stderr, "\n");
        double s1_elapsed = wall_sec(wall_clock::now() - t_s1_start).count();
        LTX_LOG("stage-1 done in %.0fs", s1_elapsed);

        // Upsample stage-1 result to full resolution.
        LTX_LOG("upsampling latent %dx%d → %dx%d ...", H_lat_1, W_lat_1, H_lat, W_lat);
        latents = latent_upsample_bilinear(latents_1.data(), T_lat, H_lat_1, W_lat_1, C, H_lat, W_lat);

        // Stage-2: override scheduler to explicit distilled sigmas, disable CFG.
        // ComfyUI uses sigmas = [0.5, 0.125, 0.0625, 0.0].
        LTX_LOG("stage-2: full-res T=%d H=%d W=%d (%d tokens), %d steps, lora, cfg=1.0",
                T_lat, H_lat, W_lat, n_tok, args.steps);
        do_cfg = false;  // distilled model: CFG=1
    }

    // ── Denoising loop (stage-2 or single-stage) ──────────────────────────────

    // Stage-2 explicit sigmas when --two-stage (ComfyUI: [0.5, 0.125, 0.0625, 0.0]).
    const std::vector<float> stage2_sigmas = {0.5f, 0.125f, 0.0625f, 0.0f};
    int loop_steps = args.two_stage ? (int)(stage2_sigmas.size() - 1) : args.steps;

    RFScheduler rf_sched(args.steps,
                         args.shift > 0.0f ? args.shift : 0.0f,  // 0 = use adaptive
                         do_cfg,
                         args.shift > 0.0f ? 0.0f : args.max_shift,
                         args.base_shift,
                         args.terminal,
                         args.stretch);
    rf_sched.set_shift_from_tokens(n_tok_total > 0 ? n_tok_total : n_tok);
    std::vector<float> ts = args.two_stage ? stage2_sigmas : rf_sched.timesteps();

    LTX_LOG("scheduler: shift=%.3f terminal=%.2f stretch=%s steps=%d",
            (double)rf_sched.shift, (double)args.terminal,
            args.stretch ? "on" : "off", loop_steps);
    LTX_LOG("starting denoising (%d steps) ...", loop_steps);
    auto t_denoise_start = wall_clock::now();

    for (int step = 0; step < loop_steps; ++step) {
        float t_cur  = ts[step];
        float t_next = ts[step + 1];
        auto t_step_start = wall_clock::now();

        if (args.verbose) {
            LTX_LOG("  step %d/%d  t=%.4f → %.4f", step + 1, loop_steps, (double)t_cur, (double)t_next);
        } else {
            fprintf(stderr, "\r[ltx] step %2d/%d  t=%.3f  ", step + 1, loop_steps, (double)t_cur);
            fflush(stderr);
        }

        // Patchify: video [n_tok, Pd]; if AV, audio [n_audio_tok, Pd]; combined = [video; audio].
        std::vector<float> patches = patchify(
            latents.data(), T_lat, H_lat, W_lat, C, pt, ph, pw);

        std::vector<float> combined_patches;
        const float * dit_input = patches.data();
        int dit_n_tok = n_tok;
        if (args.av) {
            combined_patches.resize((size_t)n_tok_total * Pd);
            memcpy(combined_patches.data(), patches.data(), (size_t)n_tok * Pd * sizeof(float));
            std::vector<float> a_patches = patchify_audio(
                audio_latents.data(), T_lat, C_audio, F_audio);
            memcpy(combined_patches.data() + (size_t)n_tok * Pd,
                   a_patches.data(), (size_t)n_audio_tok * Pd * sizeof(float));
            dit_input = combined_patches.data();
            dit_n_tok = n_tok_total;
        }

        // Conditional velocity (DiT on video-only or combined AV).
        float fps = (float)args.frames_per_second;
        std::vector<float> v_cond = dit.forward(
            dit_input, dit_n_tok, text_emb.data(), seq_len, t_cur, fps, sched, lora_ptr);

        if (v_cond.empty()) { ggml_backend_sched_free(sched); for (auto b : dit_weight_buffers) ggml_backend_buffer_free(b); if (gpu_backend) ggml_backend_free(gpu_backend); ggml_backend_free(cpu_backend); return 1; }

        // Split AV output: first n_tok tokens → video velocity, rest → audio velocity.
        std::vector<float> vel_cond = unpatchify(
            v_cond.data(), T_lat, H_lat, W_lat, C, pt, ph, pw);

        std::vector<float> velocity(lat_size);
        std::vector<float> v_uncond;  // unconditional DiT output (when do_cfg), for video + audio split
        if (do_cfg) {
            v_uncond = dit.forward(
                dit_input, dit_n_tok, uncond_emb.data(), seq_len, t_cur, fps, sched, lora_ptr);
            if (v_uncond.empty()) { ggml_backend_sched_free(sched); for (auto b : dit_weight_buffers) ggml_backend_buffer_free(b); if (gpu_backend) ggml_backend_free(gpu_backend); ggml_backend_free(cpu_backend); return 1; }
            std::vector<float> vel_uncond = unpatchify(
                v_uncond.data(), T_lat, H_lat, W_lat, C, pt, ph, pw);
            RFScheduler::apply_cfg(
                velocity.data(), vel_cond.data(), vel_uncond.data(),
                args.cfg_scale, lat_size);
        } else {
            velocity = vel_cond;
        }

        // Euler step on video latent.
        RFScheduler::euler_step(latents.data(), velocity.data(),
                                t_cur, t_next, lat_size);

        // Euler step on audio latent (AV).
        if (args.av) {
            std::vector<float> audio_vel_cond = unpatchify_audio(
                v_cond.data() + (size_t)n_tok * Pd, T_lat, C_audio, F_audio);
            std::vector<float> audio_velocity(audio_lat_size);
            if (do_cfg) {
                std::vector<float> audio_vel_uncond = unpatchify_audio(
                    v_uncond.data() + (size_t)n_tok * Pd, T_lat, C_audio, F_audio);
                RFScheduler::apply_cfg(
                    audio_velocity.data(), audio_vel_cond.data(), audio_vel_uncond.data(),
                    args.cfg_scale, audio_lat_size);
            } else {
                audio_velocity = audio_vel_cond;
            }
            RFScheduler::euler_step(audio_latents.data(), audio_velocity.data(),
                                    t_cur, t_next, audio_lat_size);
        }

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

        double step_s = wall_sec(wall_clock::now() - t_step_start).count();
        double elapsed = wall_sec(wall_clock::now() - t_denoise_start).count();
        fprintf(stderr, "  %.1fs/step  elapsed=%.0fs\n", step_s, elapsed);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    ggml_backend_sched_free(sched);
    for (auto b : dit_weight_buffers) ggml_backend_buffer_free(b);
    for (auto b : lora_weight_buffers) ggml_backend_buffer_free(b);
    dit_weight_buffers.clear();
    lora_weight_buffers.clear();
    if (gpu_backend) { ggml_backend_free(gpu_backend); gpu_backend = nullptr; }
    ggml_backend_free(cpu_backend); cpu_backend = nullptr;

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

    // ── Audio output (AV pipeline) ─────────────────────────────────────────────
    if (args.av && !audio_latents.empty()) {
        const int sample_rate = 16000;
        std::vector<float> waveform;

        if (audio_vae_loaded) {
            // audio_latents layout: [T_lat, C=8, F=16] — transpose to [C, T, F] for decoder.
            std::vector<float> lat_ctf((size_t)C_audio * T_lat * F_audio);
            for (int t = 0; t < T_lat; ++t)
                for (int c = 0; c < C_audio; ++c)
                    for (int f = 0; f < F_audio; ++f)
                        lat_ctf[((size_t)c * T_lat + t) * F_audio + f] =
                            audio_latents[((size_t)t * C_audio + c) * F_audio + f];

            LTX_LOG("decoding audio latent with AudioVAE ...");
            std::vector<float> mel = audio_vae.decode(lat_ctf.data(), T_lat);
            if (!mel.empty()) {
                int T_mel = T_lat * 4;
                waveform = audio_vae.mel_to_waveform(mel.data(), T_mel);
            }
        }

        if (waveform.empty()) {
            // Fallback: crude sinusoid synthesis from raw latent.
            const int hop_length = 160, mel_bins = 64;
            waveform = latent_to_waveform(audio_latents.data(), T_lat, sample_rate, hop_length, mel_bins);
        }

        if (!waveform.empty() && write_wav(args.out_wav, waveform.data(), waveform.size(), sample_rate))
            LTX_LOG("audio written: %s", args.out_wav.c_str());
        else
            LTX_ERR("failed to write WAV: %s", args.out_wav.c_str());
    }

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

#pragma once

// audio_vae.hpp – Audio VAE decoder for LTX-Video 2.3
//
// Loads decoder weights from ltx-2.3-22b-dev_audio_vae.safetensors.
//
// Architecture (from inspected tensor shapes):
//   Input:  [C=8, T_lat, F=16]  float32 unpatchified latent
//   conv_in: Conv2D(8→512, 3×3, pad=1)
//   mid.block_1 / block_2: ResBlock(512→512)
//   up.2: 3×ResBlock(512→512) + Upsample2× + Conv(512,512,3×3)
//   up.1: ResBlock(512→256) + 2×ResBlock(256→256) + Upsample2× + Conv(256,256,3×3)
//   up.0: ResBlock(256→128) + 2×ResBlock(128→128)  [no upsample]
//   conv_out: Conv2D(128→2, 3×3, pad=1)
//   Output: [2, T_mel=T_lat*4, F_mel=64]   (channel 0 = real, channel 1 = imag)
//
// ResBlock(C_in, C_out):
//   h = SiLU(input)
//   h = conv1(h)       [C_in → C_out, 3×3]
//   h = SiLU(h)
//   h = conv2(h)       [C_out → C_out, 3×3]
//   if C_in != C_out: input = nin_shortcut(input) [C_in → C_out, 1×1]
//   output = h + input
//
// Waveform reconstruction:
//   mel_basis    [64, 257]: maps linear STFT bins → mel bins
//   inverse_basis [514, 1, 512]: ISTFT synthesis windows
//   From [2, T_mel, 64]:
//     1. Invert mel:  complex_linear[t, f] = sum_m mel_T[f,m] * (re[t,m] + j*im[t,m])
//     2. ISTFT:       y[t*hop : t*hop+512] += inv_cos[f]*re_lin[t,f] + inv_sin[f]*im_lin[t,f]
//     3. Overlap-add normalisation

#include "ltx_common.hpp"
#include "safetensors_loader.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// ── CPU 2D convolution helpers ────────────────────────────────────────────────

// input  [C_in, H, W]
// weight [C_out, C_in, kH, kW]   (PyTorch layout)
// output [C_out, H_out, W_out]
// padding: 'pad' pixels on each side (same-size output when pad=(kH-1)/2)
static void conv2d_f32(
        const float * input,  int C_in,  int H,     int W,
        const float * weight, int C_out, int kH,    int kW,
        const float * bias,
        float * output, int pad_h, int pad_w)
{
    int H_out = H + 2 * pad_h - kH + 1;
    int W_out = W + 2 * pad_w - kW + 1;
    for (int co = 0; co < C_out; ++co) {
        float b = bias ? bias[co] : 0.0f;
        for (int oh = 0; oh < H_out; ++oh)
        for (int ow = 0; ow < W_out; ++ow) {
            float sum = b;
            for (int ci = 0; ci < C_in; ++ci)
            for (int kh = 0; kh < kH; ++kh)
            for (int kw = 0; kw < kW; ++kw) {
                int ih = oh + kh - pad_h;
                int iw = ow + kw - pad_w;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                float x = input[(ci * H + ih) * W + iw];
                float w = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                sum += x * w;
            }
            output[(co * H_out + oh) * W_out + ow] = sum;
        }
    }
}

// Nearest-neighbour 2× upsample: [C, H, W] → [C, H*2, W*2]
static void upsample2x_nearest(
        const float * in, int C, int H, int W,
        float * out)  // [C, H*2, W*2]
{
    for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        float v = in[(c * H + h) * W + w];
        out[(c * (H*2) + h*2    ) * (W*2) + w*2    ] = v;
        out[(c * (H*2) + h*2    ) * (W*2) + w*2 + 1] = v;
        out[(c * (H*2) + h*2 + 1) * (W*2) + w*2    ] = v;
        out[(c * (H*2) + h*2 + 1) * (W*2) + w*2 + 1] = v;
    }
}

static inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

static void silu_inplace(float * x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = silu_f(x[i]);
}

// ── AudioVaeDecoder ───────────────────────────────────────────────────────────

struct ConvWeights {
    std::vector<float> weight, bias;
    int C_out = 0, C_in = 0, kH = 0, kW = 0;
    bool loaded() const { return !weight.empty(); }
};

struct ResBlockW {
    ConvWeights conv1, conv2, shortcut;  // shortcut only when C_in != C_out
};

struct AudioVaeDecoder {
    // Decoder weights
    ConvWeights conv_in, conv_out;
    ResBlockW mid_block1, mid_block2;
    ResBlockW up2_blocks[3];   ConvWeights up2_up;
    ResBlockW up1_blocks[3];   ConvWeights up1_up;
    ResBlockW up0_blocks[3];

    // Vocoder STFT/mel tensors (for ISTFT reconstruction)
    std::vector<float> mel_basis;      // [64, 257]: mel filterbank
    std::vector<float> inv_basis;      // [514, 512]: ISTFT synthesis windows (flattened)
    int n_mels = 64, n_fft = 512, n_stft = 257;
    int hop_length = 160;              // samples per mel frame

    bool loaded_ = false;

    bool load(const std::string & path) {
        SafetensorsLoader st;
        if (!st.load(path)) { LTX_ERR("AudioVAE: failed to open %s", path.c_str()); return false; }

        auto load_conv = [&](ConvWeights & cw, const std::string & pfx) {
            auto shape = st.tensor_shape(pfx + ".conv.weight");
            if (shape.empty()) { shape = st.tensor_shape(pfx + ".weight"); }
            if (shape.empty()) return;
            bool has_sub = !st.tensor_shape(pfx + ".conv.weight").empty();
            std::string wsuf = has_sub ? ".conv.weight" : ".weight";
            std::string bsuf = has_sub ? ".conv.bias"   : ".bias";
            auto s = st.tensor_shape(pfx + wsuf);
            if (s.size() < 2) return;
            cw.C_out = (int)s[0]; cw.C_in = (int)s[1];
            cw.kH = s.size() > 2 ? (int)s[2] : 1;
            cw.kW = s.size() > 3 ? (int)s[3] : 1;
            cw.weight = st.tensor_f32(pfx + wsuf);
            cw.bias   = st.tensor_f32(pfx + bsuf);
        };

        auto load_resblock = [&](ResBlockW & rb, const std::string & pfx) {
            load_conv(rb.conv1,    pfx + ".conv1");
            load_conv(rb.conv2,    pfx + ".conv2");
            load_conv(rb.shortcut, pfx + ".nin_shortcut");
        };

        const std::string dp = "audio_vae.decoder.";

        load_conv(conv_in,  dp + "conv_in");
        load_conv(conv_out, dp + "conv_out");

        load_resblock(mid_block1, dp + "mid.block_1");
        load_resblock(mid_block2, dp + "mid.block_2");

        for (int b = 0; b < 3; ++b) {
            load_resblock(up2_blocks[b], dp + "up.2.block." + std::to_string(b));
            load_resblock(up1_blocks[b], dp + "up.1.block." + std::to_string(b));
            load_resblock(up0_blocks[b], dp + "up.0.block." + std::to_string(b));
        }
        load_conv(up2_up, dp + "up.2.upsample.conv");
        load_conv(up1_up, dp + "up.1.upsample.conv");

        // Load STFT/mel tensors from vocoder section.
        mel_basis = st.tensor_f32("vocoder.mel_stft.mel_basis");
        auto inv  = st.tensor_f32("vocoder.mel_stft.stft_fn.inverse_basis");
        if (!inv.empty()) {
            // Shape [514, 1, 512] → flatten to [514, 512]
            inv_basis.resize(514 * 512);
            for (int i = 0; i < 514; ++i)
                memcpy(inv_basis.data() + i * 512, inv.data() + i * 512, 512 * sizeof(float));
        }

        loaded_ = conv_in.loaded() && conv_out.loaded();
        if (loaded_)
            LTX_LOG("AudioVAE decoder loaded: conv_in [%d→%d], mel_basis=%s, inv_basis=%s",
                    conv_in.C_in, conv_in.C_out,
                    mel_basis.empty() ? "missing" : "ok",
                    inv_basis.empty() ? "missing" : "ok");
        else
            LTX_ERR("AudioVAE: failed to load decoder weights from %s", path.c_str());
        return loaded_;
    }

    // Apply one ResBlock in-place.
    // input/output: [C, H, W] — returns new buffer
    std::vector<float> apply_resblock(const ResBlockW & rb,
                                       const float * x, int C, int H, int W) const
    {
        int C_out = rb.conv1.C_out;
        std::vector<float> h(x, x + (size_t)C * H * W);
        silu_inplace(h.data(), h.size());

        std::vector<float> h2((size_t)C_out * H * W);
        conv2d_f32(h.data(), C, H, W, rb.conv1.weight.data(), C_out, rb.conv1.kH, rb.conv1.kW,
                   rb.conv1.bias.data(), h2.data(), rb.conv1.kH / 2, rb.conv1.kW / 2);

        silu_inplace(h2.data(), h2.size());

        std::vector<float> h3((size_t)C_out * H * W);
        conv2d_f32(h2.data(), C_out, H, W, rb.conv2.weight.data(), C_out, rb.conv2.kH, rb.conv2.kW,
                   rb.conv2.bias.data(), h3.data(), rb.conv2.kH / 2, rb.conv2.kW / 2);

        // Shortcut
        if (rb.shortcut.loaded()) {
            std::vector<float> sc((size_t)C_out * H * W);
            conv2d_f32(x, C, H, W, rb.shortcut.weight.data(), C_out, 1, 1,
                       rb.shortcut.bias.data(), sc.data(), 0, 0);
            for (size_t i = 0; i < h3.size(); ++i) h3[i] += sc[i];
        } else {
            // Identity shortcut (C_in == C_out)
            for (size_t i = 0; i < h3.size(); ++i) h3[i] += x[i];
        }
        return h3;
    }

    // Decode audio latent → [2, T_mel, 64] complex mel spectrogram.
    // lat: [C=8, T_lat, F=16] row-major float
    std::vector<float> decode(const float * lat, int T_lat,
                              int C_lat = 8, int F_lat = 16) const
    {
        if (!loaded_) return {};

        int H = T_lat, W = F_lat;  // treat (T, F) as spatial (H, W)

        // conv_in
        std::vector<float> x((size_t)conv_in.C_out * H * W);
        conv2d_f32(lat, C_lat, H, W,
                   conv_in.weight.data(), conv_in.C_out, conv_in.kH, conv_in.kW,
                   conv_in.bias.data(), x.data(), conv_in.kH / 2, conv_in.kW / 2);

        int C = conv_in.C_out;  // 512

        // Mid blocks
        x = apply_resblock(mid_block1, x.data(), C, H, W);
        x = apply_resblock(mid_block2, x.data(), C, H, W);

        // up.2: 3 ResBlocks + upsample
        for (int b = 0; b < 3; ++b)
            x = apply_resblock(up2_blocks[b], x.data(), C, H, W);
        if (up2_up.loaded()) {
            // Nearest-neighbour 2× upsample then conv
            std::vector<float> up((size_t)C * (H*2) * (W*2));
            upsample2x_nearest(x.data(), C, H, W, up.data());
            H *= 2; W *= 2;
            x.resize((size_t)C * H * W);
            conv2d_f32(up.data(), C, H, W,
                       up2_up.weight.data(), up2_up.C_out, up2_up.kH, up2_up.kW,
                       up2_up.bias.data(), x.data(), up2_up.kH / 2, up2_up.kW / 2);
            C = up2_up.C_out;  // still 512
        }

        // up.1: 3 ResBlocks + upsample
        for (int b = 0; b < 3; ++b) {
            int C_next = up1_blocks[b].conv1.C_out;
            x = apply_resblock(up1_blocks[b], x.data(), C, H, W);
            C = C_next;
        }
        if (up1_up.loaded()) {
            std::vector<float> up((size_t)C * (H*2) * (W*2));
            upsample2x_nearest(x.data(), C, H, W, up.data());
            H *= 2; W *= 2;
            x.resize((size_t)C * H * W);
            conv2d_f32(up.data(), C, H, W,
                       up1_up.weight.data(), up1_up.C_out, up1_up.kH, up1_up.kW,
                       up1_up.bias.data(), x.data(), up1_up.kH / 2, up1_up.kW / 2);
            C = up1_up.C_out;  // 256
        }

        // up.0: 3 ResBlocks, no upsample
        for (int b = 0; b < 3; ++b) {
            int C_next = up0_blocks[b].conv1.C_out;
            x = apply_resblock(up0_blocks[b], x.data(), C, H, W);
            C = C_next;
        }
        // C = 128, H = T_lat*4, W = F_lat*4 = 64

        // conv_out → [2, T_mel, 64]
        std::vector<float> out((size_t)conv_out.C_out * H * W);
        conv2d_f32(x.data(), C, H, W,
                   conv_out.weight.data(), conv_out.C_out, conv_out.kH, conv_out.kW,
                   conv_out.bias.data(), out.data(), conv_out.kH / 2, conv_out.kW / 2);

        return out;  // [2, T_mel, F_mel=64]
    }

    // Convert decoder output [2, T_mel, 64] to waveform samples.
    // Uses saved mel_basis and inverse_basis for proper ISTFT reconstruction.
    // Returns float PCM samples (normalised to [-1, 1]).
    std::vector<float> mel_to_waveform(const float * mel_out,
                                        int T_mel, int F_mel = 64) const
    {
        // 1. Invert mel filterbank: mel[T_mel, 64] → linear[T_mel, 257]
        //    Use mel_basis.T as pseudo-inverse (suitable when mel bins don't overlap much).
        //    mel_basis: [64, 257]; each row sums to ~1 over its support.
        //    Inverse: linear[t,f] = sum_m mel_basis[m,f] * mel_in[t,m]  (transpose multiply).
        //    We process both channels (real + imaginary).
        if (inv_basis.empty() || mel_basis.empty()) {
            // Fallback: simple sine synthesis from magnitude
            return mel_fallback(mel_out, T_mel, F_mel);
        }

        std::vector<float> re_lin(T_mel * n_stft, 0.0f);  // [T_mel, 257]
        std::vector<float> im_lin(T_mel * n_stft, 0.0f);

        const float * re_mel = mel_out;                    // [T_mel, F_mel]
        const float * im_mel = mel_out + T_mel * F_mel;   // [T_mel, F_mel]

        for (int t = 0; t < T_mel; ++t)
        for (int m = 0; m < F_mel; ++m) {
            float re_m = re_mel[t * F_mel + m];
            float im_m = im_mel[t * F_mel + m];
            for (int f = 0; f < n_stft; ++f) {
                float mb = mel_basis[m * n_stft + f];
                re_lin[t * n_stft + f] += mb * re_m;
                im_lin[t * n_stft + f] += mb * im_m;
            }
        }

        // 2. ISTFT overlap-add.
        //    inv_basis: [514, 512] — rows 0..256 = cosine windows, rows 257..513 = sine windows.
        //    For frame t at sample offset t*hop_length:
        //      y[t*hop : t*hop+n_fft] += sum_f (cos_win[f] * re_lin[t,f] - sin_win[f] * im_lin[t,f])
        //    Then normalise by overlap-add of a flat window (hop_length / n_fft).

        int n_samples = (T_mel - 1) * hop_length + n_fft;
        std::vector<float> y(n_samples, 0.0f);
        std::vector<float> norm(n_samples, 0.0f);

        const float * cos_win = inv_basis.data();           // rows 0..n_stft-1, each n_fft long
        const float * sin_win = inv_basis.data() + n_stft * n_fft;  // rows n_stft..514-1

        for (int t = 0; t < T_mel; ++t) {
            int off = t * hop_length;
            for (int f = 0; f < n_stft; ++f) {
                float re = re_lin[t * n_stft + f];
                float im = im_lin[t * n_stft + f];
                const float * cw = cos_win + f * n_fft;
                const float * sw = sin_win + f * n_fft;
                for (int n = 0; n < n_fft && off + n < n_samples; ++n)
                    y[off + n] += re * cw[n] - im * sw[n];
            }
            // Accumulate normalisation window (sum of squared cosine windows).
            for (int f = 0; f < n_stft; ++f) {
                const float * cw = cos_win + f * n_fft;
                for (int n = 0; n < n_fft && off + n < n_samples; ++n)
                    norm[off + n] += cw[n] * cw[n];
            }
        }

        // Normalise.
        for (int i = 0; i < n_samples; ++i)
            if (norm[i] > 1e-8f) y[i] /= norm[i];

        // Peak normalise to [-0.95, 0.95].
        float peak = 0.0f;
        for (float v : y) peak = std::max(peak, std::fabs(v));
        if (peak > 1e-6f) { float s = 0.95f / peak; for (float & v : y) v *= s; }

        return y;
    }

private:
    // Fallback: magnitude → mel freq → overlap-add sinusoids (no ISTFT).
    std::vector<float> mel_fallback(const float * mel_out,
                                     int T_mel, int F_mel) const
    {
        const int sample_rate = 16000;
        size_t n_samples = (size_t)T_mel * hop_length;
        std::vector<float> out(n_samples, 0.0f);

        for (int t = 0; t < T_mel; ++t) {
            int off = t * hop_length;
            for (int m = 0; m < F_mel; ++m) {
                // Magnitude from both channels.
                float re = mel_out[t * F_mel + m];
                float im = mel_out[(T_mel + t) * F_mel + m];
                float mag = 0.5f + 0.5f * std::sqrt(re * re + im * im);
                // Mel centre frequency.
                float freq_m = 2595.0f * std::log10f(1.0f + 8000.0f * m / ((F_mel - 1) * 700.0f));
                float freq_hz = 700.0f * (std::powf(10.0f, freq_m / 2595.0f) - 1.0f);
                for (int i = 0; i < hop_length && off + i < (int)n_samples; ++i) {
                    float t_s = (float)(off + i) / (float)sample_rate;
                    out[off + i] += mag * std::cosf(2.0f * 3.14159265f * freq_hz * t_s);
                }
            }
        }
        float peak = 0.0f;
        for (float v : out) peak = std::max(peak, std::fabs(v));
        if (peak > 1e-6f) { float s = 0.95f / peak; for (float & v : out) v *= s; }
        return out;
    }
};

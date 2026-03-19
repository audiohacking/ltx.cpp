#pragma once

// scheduler.hpp – Flow-matching / Euler scheduler for LTX-Video
//
// LTX-Video uses Rectified Flow (RF) training, so the forward process is:
//   x_t = (1 - t) * x_0 + t * noise    t in [0, 1]
//
// The model predicts the velocity: v = dx/dt = noise - x_0
// Euler ODE solver: x_{t-dt} = x_t - dt * v_theta(x_t, t)
//
// LTXVScheduler (ComfyUI reference):
//   shift   = base_shift + (max_shift - base_shift) * (n_tok / 4096)  (token-adaptive)
//   stretch = rescale the [0,1] sigma range to [terminal, 1] so the schedule
//             spends no time on near-clean signal (t < terminal)
//   terminal= minimum sigma; sigmas stop at terminal rather than 0

#include <algorithm>
#include <cmath>
#include <vector>

struct RFScheduler {
    int   steps;
    float shift;       // flow-shift (overridden by adaptive formula when max_shift set)
    float max_shift;   // LTXVScheduler: 2.05 (set >0 to use adaptive shift)
    float base_shift;  // LTXVScheduler: 0.95
    float terminal;    // stop sigma; 0.1 in ComfyUI flow (0.0 = full range)
    bool  stretch;     // rescale sigmas into [terminal, 1.0]
    bool  cfg;

    explicit RFScheduler(int steps = 50, float shift = 3.0f, bool cfg = true,
                         float max_shift = 0.0f, float base_shift = 0.95f,
                         float terminal = 0.0f, bool stretch = false)
        : steps(steps), shift(shift), max_shift(max_shift), base_shift(base_shift),
          terminal(terminal), stretch(stretch), cfg(cfg) {}

    // Compute adaptive shift from token count (LTXVScheduler formula).
    // Call this after knowing n_tok; updates shift in-place.
    void set_shift_from_tokens(int n_tok) {
        if (max_shift > 0.0f) {
            shift = base_shift + (max_shift - base_shift) * ((float)n_tok / 4096.0f);
        }
    }

    // Build timestep schedule in [1, terminal] with optional stretch.
    // Returns length steps+1: ts[0]=1.0 (full noise), ts[steps]=terminal (or 0).
    std::vector<float> timesteps() const {
        std::vector<float> ts(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            float alpha = (float)(steps - i) / (float)steps; // 1 → 0
            float t = alpha * shift / (1.0f + (shift - 1.0f) * alpha);
            ts[i] = t;
        }

        if (stretch && terminal > 0.0f && terminal < 1.0f) {
            // Linearly rescale all sigmas from [0,1] into [terminal, 1].
            // sigma_new = terminal + sigma_old * (1 - terminal)
            float lo = terminal, hi = 1.0f;
            for (float & t : ts)
                t = lo + t * (hi - lo);
        } else if (terminal > 0.0f) {
            // Clamp: replace the final 0.0 with terminal.
            ts[steps] = terminal;
        }

        return ts;
    }

    // Single Euler step (in-place).
    static void euler_step(
            float * x_t,
            const float * v,
            float t_cur, float t_next,
            size_t n)
    {
        float dt = t_next - t_cur; // negative
        for (size_t i = 0; i < n; ++i)
            x_t[i] += dt * v[i];
    }

    // Classifier-free guidance.
    static void apply_cfg(
            float * out,
            const float * v_cond, const float * v_uncond,
            float guidance_scale,
            size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            out[i] = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
    }

    // Build a schedule from explicit sigma values (distilled stage 2).
    // sigmas is already the full ts vector including the final 0 or terminal value.
    static std::vector<float> from_sigmas(const std::vector<float> & sigmas) {
        return sigmas;
    }
};

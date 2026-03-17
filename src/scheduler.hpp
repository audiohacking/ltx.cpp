#pragma once

// scheduler.hpp – Flow-matching / Euler scheduler for LTX-Video
//
// LTX-Video uses Rectified Flow (RF) training, so the forward process is:
//   x_t = (1 - t) * x_0 + t * noise    t in [0, 1]
//
// The model predicts the velocity: v = dx/dt = noise - x_0
// Euler ODE solver: x_{t-dt} = x_t - dt * v_theta(x_t, t)
//
// For distilled (few-step) models the shift parameter `shift` rescales
// the timestep schedule: t' = t / (t + (1-t)/shift)

#include <algorithm>
#include <cmath>
#include <vector>

struct RFScheduler {
    int   steps;    // number of denoising steps
    float shift;    // flow-shift parameter (default 3.0 for LTX-Video)
    bool  cfg;      // use classifier-free guidance?

    explicit RFScheduler(int steps = 50, float shift = 3.0f, bool cfg = true)
        : steps(steps), shift(shift), cfg(cfg) {}

    // Build linearly-spaced timestep schedule in [1, 0].
    // Returns a vector of length steps+1 with t[0]=1.0 (full noise), t[N]=0.0 (clean).
    std::vector<float> timesteps() const {
        std::vector<float> ts(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            float alpha = (float)(steps - i) / (float)steps; // 1 → 0
            // Apply flow shift rescaling.
            float t = alpha * shift / (1.0f + (shift - 1.0f) * alpha);
            ts[i] = t;
        }
        return ts;
    }

    // Single Euler step: predict velocity and advance latent.
    // x_t:     noisy latent (in-place modified)
    // v:       predicted velocity from the model
    // t_cur:   current timestep
    // t_next:  next (smaller) timestep
    // n:       number of elements in x_t and v
    static void euler_step(
            float * x_t,
            const float * v,
            float t_cur, float t_next,
            size_t n)
    {
        float dt = t_next - t_cur; // negative (going from t→0)
        for (size_t i = 0; i < n; ++i)
            x_t[i] += dt * v[i];
    }

    // Classifier-free guidance: combine conditional and unconditional predictions.
    // v_cond: [n]  velocity from conditioned model
    // v_uncond: [n] velocity from unconditioned model (empty prompt)
    // out: [n]
    static void apply_cfg(
            float * out,
            const float * v_cond, const float * v_uncond,
            float guidance_scale,
            size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            out[i] = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
    }
};

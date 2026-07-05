/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    enum class DepthPriorType : int {
        Auto = 0,
        Disparity = 1,
        Depth = 2,
    };

    // Final/diagnostic slots at the front of the partials buffer.
    namespace depth_loss_slots {
        constexpr int kValid = 0;
        constexpr int kModel = 1; // 0 = disparity-space fit, 1 = depth-space fit
        constexpr int kScale = 2;
        constexpr int kShift = 3;
        constexpr int kFloor = 4;
        constexpr int kInvNorm = 5;
        constexpr int kSumAlpha = 6;
        constexpr int kCount = 7;
        constexpr int kCorrDisparity = 8;
        constexpr int kCorrDepth = 9;
        constexpr int kMeanExpectedDepth = 10;
        constexpr int kSigmaP = 11;
        constexpr int kMeanTarget = 12;
        constexpr int kVarTarget = 13;
        constexpr int kScaleDepthModel = 14;
        constexpr int kShiftDepthModel = 15;
        constexpr int kSlotCount = 16;
    } // namespace depth_loss_slots

    // Softening floor for depth inversion, as a fraction of the weighted mean
    // expected depth. Bounds the leverage of near-camera floaters in both the
    // alignment fit and the gradients.
    constexpr float kDepthLossFloorFraction = 0.05f;
    constexpr float kDepthLossMinAlpha = 1.0e-3f;
    // Ridge term on the prior variance in the affine fits. Near the 8-bit
    // quantization noise floor the slope shrinks toward zero.
    constexpr float kDepthLossTargetVarRidge = 1.5e-5f;
    // Below this prior variance the prior is considered flat and is rejected for
    // anchored supervision. A constant target has no geometry signal.
    constexpr float kDepthLossFlatPriorVar = 4.0f * kDepthLossTargetVarRidge;
    // Unanchored (render-fitted) alignment is rejected below this correlation:
    // fitting the prior onto an uncorrelated render only injects noise.
    constexpr float kDepthLossMinSelfFitCorr = 0.20f;

    struct DepthAnchorCandidate {
        bool valid = false;
        float scale = 0.0f;
        float shift = 0.0f;
        float corr = 0.0f;
        int samples = 0;
    };

    // Per-camera alignment of the depth prior against sparse anchor points
    // (COLMAP / init point cloud), fitted once at startup. Keeps the target
    // depth absolute and multi-view consistent instead of chasing the render.
    struct DepthAnchor {
        bool valid = false;
        int model = 0; // 0 = disparity-space fit, 1 = depth-space fit
        float scale = 0.0f;
        float shift = 0.0f;
        float floor = 0.0f;
        float corr = 0.0f;
        int samples = 0;
        DepthAnchorCandidate disparity;
        DepthAnchorCandidate depth;
    };

    // Projects sparse points into the camera, samples the prior, and fits both
    // affine models (prior -> inverse depth, prior -> depth) with one trimmed
    // refit. Synchronizes the stream; startup use only.
    // aabb_lo/aabb_hi: world-space bounds gating the anchor points (robust
    // scene bbox); pass -inf/+inf to disable.
    [[nodiscard]] DepthAnchor fit_depth_anchor(
        const float* points_xyz, // [N,3] CUDA
        size_t num_points,
        const float* w2c, // [16] CUDA row-major world-to-camera
        float fx,
        float fy,
        float cx,
        float cy,
        const float* prior, // [H,W] CUDA
        int width,
        int height,
        float near_plane,
        const float aabb_lo[3],
        const float aabb_hi[3],
        cudaStream_t stream = nullptr);

    [[nodiscard]] size_t depth_loss_partial_count(size_t num_pixels);

    // Scale-and-shift-invariant depth supervision on alpha-normalized expected
    // depth in inverse-depth space: per-image weighted least-squares affine
    // alignment of the monocular prior (disparity- or depth-convention, chosen
    // by |correlation| in Auto mode), sigma-normalized L1 plus a
    // gradient-alignment term. Emits gradients w.r.t. both the accumulated
    // depth map and the alpha map.
    // error_map_out (optional, may be null): per-pixel |residual|/sigma clamped
    // to [0,1] — a densification signal for regions whose depth disagrees with
    // the prior. Zeroed when the loss is inactive.
    // anchor (optional, may be null): fixed per-camera alignment; when valid it
    // replaces the per-iteration self-fit against the render.
    // prior_quantization_step: quantization step of the prior in target units
    // (1/255 for 8-bit priors, 1/65535 for 16-bit, 0 for float). Residuals and
    // gradient-alignment differences inside the quantizer's half-step corridor
    // carry no loss and no gradient; without this, sign gradients drag smooth
    // surfaces onto the prior's staircase (terracing on coarse 8-bit priors).
    void launch_depth_loss(
        const float* rendered_depth_accum,
        const float* rendered_alpha_accum,
        const float* target_depth,
        float* grad_depth,
        float* grad_alpha,
        float* error_map_out,
        float* loss_out,
        float* partial_sums,
        int width,
        int height,
        float weight,
        float gradient_term_weight,
        DepthPriorType prior,
        float prior_quantization_step = 0.0f,
        const DepthAnchor* anchor = nullptr,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels

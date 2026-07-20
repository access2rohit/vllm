/*
 * OP-301: torch extension binding for the fused MoE-finalize + MNNVL oneshot
 * Lamport AllReduce + residual + RMSNorm kernel.
 *
 * Built with torch.utils.cpp_extension.load() (see
 * vllm/model_executor/layers/fused_moe/moe_finalize_ar_rms.py). Needs the
 * FlashInfer header tree on the include path:
 *   -I <site-packages>/flashinfer/data/include
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "moe_finalize_ar_rms_kernel.cuh"

namespace {

using moe_finalize_ar_rms::MoeFinalizeARRMSKernelParams;
using moe_finalize_ar_rms::moeFinalizeARRMSDispatch;

void moe_finalize_ar_rms_fused(
    // MoE finalize prologue inputs
    at::Tensor gemm2_output,             // [num_permuted, hidden] bf16
    at::Tensor expert_weights,           // [num_tokens, top_k] bf16 or fp32
    at::Tensor expanded_idx_to_permuted_idx,  // [num_tokens, top_k] int32
    at::Tensor shared_expert_output,     // [num_tokens, hidden] bf16
    double routed_scaling_factor,
    // AR + RMSNorm state
    at::Tensor residual_in,              // [num_tokens, hidden] bf16
    at::Tensor gamma,                    // [hidden] bf16
    double epsilon,
    // MNNVL workspace (from MNNVLAllReduceFusionWorkspace)
    int64_t multicast_buffer_ptr,        // workspace.mc_ptr
    int64_t buffer_ptrs_dev,             // workspace.uc_ptrs_dev
    at::Tensor buffer_flags,             // workspace.buffer_flags (uint32[>=9])
    int64_t nranks, int64_t rank,
    bool launch_with_pdl,
    // outputs
    at::Tensor norm_out,                 // [num_tokens, hidden] bf16
    at::Tensor residual_out) {           // [num_tokens, hidden] bf16
  TORCH_CHECK(gemm2_output.is_cuda(), "gemm2_output must be CUDA");
  TORCH_CHECK(gemm2_output.scalar_type() == at::kBFloat16,
              "moe_finalize_ar_rms currently supports bf16 activations only");
  TORCH_CHECK(gemm2_output.is_contiguous() && shared_expert_output.is_contiguous() &&
                  residual_in.is_contiguous() && gamma.is_contiguous() &&
                  norm_out.is_contiguous() && residual_out.is_contiguous(),
              "all activation tensors must be contiguous");
  TORCH_CHECK(expanded_idx_to_permuted_idx.scalar_type() == at::kInt,
              "expanded_idx_to_permuted_idx must be int32");
  TORCH_CHECK(expanded_idx_to_permuted_idx.is_contiguous() && expert_weights.is_contiguous(),
              "index/scale tensors must be contiguous");

  int64_t const num_tokens = shared_expert_output.size(0);
  int64_t const hidden = shared_expert_output.size(1);
  int64_t const top_k = expert_weights.size(-1);

  TORCH_CHECK(gemm2_output.size(-1) == hidden, "gemm2_output hidden dim mismatch");
  TORCH_CHECK(expanded_idx_to_permuted_idx.numel() == num_tokens * top_k,
              "expanded_idx_to_permuted_idx must have num_tokens*top_k elements");
  TORCH_CHECK(expert_weights.numel() == num_tokens * top_k,
              "expert_weights must have num_tokens*top_k elements");
  TORCH_CHECK(residual_in.size(0) == num_tokens && residual_in.size(1) == hidden,
              "residual_in shape mismatch");
  TORCH_CHECK(norm_out.size(0) == num_tokens && norm_out.size(1) == hidden,
              "norm_out shape mismatch");
  TORCH_CHECK(residual_out.size(0) == num_tokens && residual_out.size(1) == hidden,
              "residual_out shape mismatch");
  TORCH_CHECK(gamma.numel() == hidden, "gamma shape mismatch");
  TORCH_CHECK(hidden % (sizeof(float4) / sizeof(__nv_bfloat16)) == 0,
              "hidden dim must be divisible by 8 for bf16");
  TORCH_CHECK(nranks == 2 || nranks == 4 || nranks == 8, "nranks must be in {2, 4, 8}");
  TORCH_CHECK(rank >= 0 && rank < nranks, "invalid rank");
  TORCH_CHECK(buffer_flags.scalar_type() == at::kUInt32 && buffer_flags.numel() >= 9,
              "buffer_flags must be uint32 with >= 9 elements");

  at::cuda::CUDAGuard device_guard(gemm2_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using T = __nv_bfloat16;

  auto launch = [&](auto scale_tag) -> cudaError_t {
    using ScaleT = decltype(scale_tag);
    MoeFinalizeARRMSKernelParams<T, ScaleT> params{};
    params.gemm2Output = reinterpret_cast<T const*>(gemm2_output.data_ptr());
    params.expertWeights = reinterpret_cast<ScaleT const*>(expert_weights.data_ptr());
    params.expandedIdxToPermutedIdx =
        reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx.data_ptr());
    params.sharedExpertOutput = reinterpret_cast<T const*>(shared_expert_output.data_ptr());
    params.routedScalingFactor = static_cast<float>(routed_scaling_factor);
    params.topK = static_cast<int>(top_k);
    params.outputPtr = reinterpret_cast<T*>(norm_out.data_ptr());
    params.prenormedPtr = reinterpret_cast<T*>(residual_out.data_ptr());
    params.residualInPtr = reinterpret_cast<T const*>(residual_in.data_ptr());
    params.gammaPtr = reinterpret_cast<T const*>(gamma.data_ptr());
    params.inputPtrs = reinterpret_cast<T**>(buffer_ptrs_dev);
    params.mcastPtr = reinterpret_cast<T*>(multicast_buffer_ptr);
    params.nRanks = static_cast<int>(nranks);
    params.rank = static_cast<int>(rank);
    params.numTokens = static_cast<int>(num_tokens);
    params.tokenDim = static_cast<int>(hidden);
    params.epsilon = static_cast<float>(epsilon);
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr());
    return moeFinalizeARRMSDispatch<T, ScaleT>(params, launch_with_pdl, stream);
  };

  cudaError_t status;
  switch (expert_weights.scalar_type()) {
    case at::kBFloat16:
      status = launch(__nv_bfloat16{});
      break;
    case at::kFloat:
      status = launch(float{});
      break;
    default:
      TORCH_CHECK(false, "expert_weights must be bf16 or fp32");
  }
  TORCH_CHECK(status == cudaSuccess,
              "moe_finalize_ar_rms_fused launch failed: ", cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_finalize_ar_rms_fused", &moe_finalize_ar_rms_fused,
        "Fused MoE-finalize + MNNVL oneshot Lamport AllReduce + residual + RMSNorm (bf16)");
}

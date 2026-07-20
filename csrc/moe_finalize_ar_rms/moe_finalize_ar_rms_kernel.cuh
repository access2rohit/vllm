/*
 * OP-301: MoE-finalize + MNNVL oneshot Lamport AllReduce + residual + RMSNorm
 * fused kernel.
 *
 * This is a prologue variant of FlashInfer's MNNVL oneshot allreduce kernel
 * (flashinfer/comm/trtllm_mnnvl_allreduce.cuh :: oneshotAllreduceFusionKernel).
 * The Lamport core (mcast broadcast store, per-rank volatile polling,
 * deterministic fp32 reduction, residual add, RMSNorm epilogue, buffer flag
 * rotation) is copied verbatim from that kernel. ONLY the input-load stage is
 * changed: instead of loading a pre-finalized [num_tokens, hidden] shard, each
 * thread performs the MoE finalize inline:
 *
 *   for k in top_k:
 *     permuted_idx = expanded_idx_to_permuted_idx[token * top_k + k]
 *     if permuted_idx < 0: skip                      (padding / EP-dropped)
 *     accum_fp32 += gemm2_output[permuted_idx, :] * expert_weights[token, k]
 *   accum_fp32 *= routed_scaling_factor
 *   accum_fp32 += shared_expert_output[token, :]
 *
 * The top-k combine accumulates in fp32 (matching the baseline TRT-LLM-gen
 * finalizeKernel fp32 accumulator) and the 8-rank Lamport reduction accumulates
 * in fp32 (matching reduceOneshotDeterministic) -> lossless vs the baseline
 * finalize + add_mul + mnnvl AR+RMSNorm chain, output bf16.
 *
 * Grid geometry, PDL behaviour and the Lamport flag/rotation protocol are
 * unchanged, so this kernel is a drop-in for the oneshot regime (num_tokens
 * <= oneshot threshold; the caller gates dispatch to BS <= 16 shapes).
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Reuse the MNNVL Lamport machinery from FlashInfer (header-only utils:
// LamportFlags, PackedVec, loadPacked/loadPackedVolatile, isLamportDirty,
// sanitizeLamportPayload, waitOneshotRemoteRanks, reduceOneshotDeterministic,
// blockReduceSum, adjustGridConfig, to/fromFloat).
#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"

namespace moe_finalize_ar_rms {

namespace fi = flashinfer::trtllm_mnnvl_allreduce;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

template <typename T, typename ScaleT>
struct MoeFinalizeARRMSKernelParams {
  // --- MoE finalize prologue inputs ---
  // [num_permuted, tokenDim]; rows indexed by expandedIdxToPermutedIdx.
  T const* gemm2Output;
  // [numTokens, topK] expert scales (routing weights).
  ScaleT const* expertWeights;
  // [numTokens * topK]; -1 marks a skipped slot.
  int32_t const* expandedIdxToPermutedIdx;
  // [numTokens, tokenDim]
  T const* sharedExpertOutput;
  float routedScalingFactor;
  int topK;

  // --- Unchanged MNNVL oneshot AR + RMSNorm state (see AllReduceKernelParams)
  T* outputPtr;      // norm_out            [numTokens, tokenDim]
  T* prenormedPtr;   // residual_out        [numTokens, tokenDim]
  T const* residualInPtr;
  T const* gammaPtr;
  T** inputPtrs;     // per-rank unicast Lamport buffer base ptrs (device array)
  T* mcastPtr;       // multicast Lamport buffer base ptr
  int nRanks;
  int rank;
  int numTokens;
  int tokenDim;
  float epsilon;
  uint32_t* bufferFlags;
};

// ---------------------------------------------------------------------------
// Kernel: verbatim oneshotAllreduceFusionKernel<WorldSize, T, /*RMSNorm=*/true,
// QuantType::kNone> with the input-load stage replaced by the MoE finalize
// prologue. Structure and ordering of every Lamport step is preserved.
// ---------------------------------------------------------------------------

template <uint8_t WorldSize, typename T, typename ScaleT, typename PackedType = float4>
__global__ void __launch_bounds__(1024)
    moeFinalizeARRMSFusionKernel(MoeFinalizeARRMSKernelParams<T, ScaleT> params) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr uint32_t kELT_SIZE = sizeof(T);
  int const tokenDim = params.tokenDim;
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int packedIdx = cluster.thread_rank();
  int token = blockIdx.x;
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
  // We only use 1 stage for the oneshot allreduce
  constexpr bool kUseLamportCGA = true;
  fi::utils::LamportFlags<PackedType, kUseLamportCGA> flag(params.bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));

  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);
    return;
  }

  // ==================== MoE finalize prologue (CHANGED vs baseline) =========
  // Replaces: val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
  fi::utils::PackedVec<PackedType, T> val;
  {
    float accum[kELTS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      accum[i] = 0.f;
    }
    int const topK = params.topK;
    int const rowOffset = packedIdx * kELTS_PER_THREAD;
#pragma unroll 1
    for (int k = 0; k < topK; k++) {
      int const expandedIdx = token * topK + k;
      int32_t const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
      if (permutedIdx < 0) {
        continue;
      }
      float const scale = fi::utils::toFloat<ScaleT>(params.expertWeights[expandedIdx]);
      fi::utils::PackedVec<PackedType, T> row;
      row.packed = fi::utils::loadPacked<PackedType>(
          &params.gemm2Output[static_cast<int64_t>(permutedIdx) * tokenDim + rowOffset]);
#pragma unroll
      for (int i = 0; i < kELTS_PER_THREAD; i++) {
        accum[i] += fi::utils::toFloat<T>(row.elements[i]) * scale;
      }
    }
    // routed scaling + shared expert add, still in fp32.
    fi::utils::PackedVec<PackedType, T> sharedRow;
    sharedRow.packed =
        fi::utils::loadPacked<PackedType>(&params.sharedExpertOutput[threadOffset]);
    float const rsf = params.routedScalingFactor;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      val.elements[i] =
          fi::utils::fromFloat<T>(accum[i] * rsf + fi::utils::toFloat<T>(sharedRow.elements[i]));
    }
  }
  fi::utils::sanitizeLamportPayload<PackedType, T>(val);

  // ==================== Broadcast tokens to each rank (UNCHANGED) ===========
  reinterpret_cast<PackedType*>(
      &stagePtrMcast[token * tokenDim * WorldSize + params.rank * tokenDim])[packedIdx] =
      val.packed;

  flag.ctaArrive();
  // ============ Lamport sync + clear previous iteration's buffer ============
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

  // ======================= Reduction (UNCHANGED) ============================
  // Fully deterministic: every rank uses the exact same reduction order.
  fi::utils::PackedVec<PackedType, T> packedAccum;
  static_assert(WorldSize <= 8, "moe_finalize_ar_rms supports WorldSize <= 8");
  packedAccum = val;
#define RUN_ONESHOT_LOCAL_RANK(LOCAL_RANK)                                                       \
  case LOCAL_RANK:                                                                               \
    if constexpr (WorldSize > LOCAL_RANK) {                                                     \
      fi::utils::PackedVec<PackedType, T> remoteValues[WorldSize];                              \
      fi::utils::waitOneshotRemoteRanks<WorldSize, LOCAL_RANK, T, PackedType,                   \
                                        kELTS_PER_THREAD>(remoteValues, stagePtrLocal, token,   \
                                                          tokenDim, packedIdx);                 \
      packedAccum =                                                                             \
          fi::utils::reduceOneshotDeterministic<WorldSize, LOCAL_RANK, T, PackedType,           \
                                                kELTS_PER_THREAD>(remoteValues, val);           \
    }                                                                                           \
    break

  switch (params.rank) {
    RUN_ONESHOT_LOCAL_RANK(0);
    RUN_ONESHOT_LOCAL_RANK(1);
    RUN_ONESHOT_LOCAL_RANK(2);
    RUN_ONESHOT_LOCAL_RANK(3);
    RUN_ONESHOT_LOCAL_RANK(4);
    RUN_ONESHOT_LOCAL_RANK(5);
    RUN_ONESHOT_LOCAL_RANK(6);
    RUN_ONESHOT_LOCAL_RANK(7);
  }
#undef RUN_ONESHOT_LOCAL_RANK

  cudaTriggerProgrammaticLaunchCompletion();
  // =============================== Residual (UNCHANGED) =====================
  {
    fi::utils::PackedVec<PackedType, T> residualIn;
    residualIn.packed = *reinterpret_cast<PackedType const*>(&params.residualInPtr[threadOffset]);
    packedAccum += residualIn;
    if (params.prenormedPtr != nullptr) {
      *reinterpret_cast<PackedType*>(&params.prenormedPtr[threadOffset]) = packedAccum.packed;
    }
    // =============================== RMSNorm (UNCHANGED) ====================
    fi::utils::PackedVec<PackedType, T> gamma;
    gamma.packed =
        *reinterpret_cast<PackedType const*>(&params.gammaPtr[packedIdx * kELTS_PER_THREAD]);

    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      threadSum += fi::utils::toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    float blockSum = fi::utils::blockReduceSum<float, true>(threadSum);

    __shared__ float sharedVal[8];  // Temporary variable to share the sum within block
    float fullSum = blockSum;
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
      // Need to reduce over the entire cluster
      int const blockRank = cluster.block_rank();
      if (threadIdx.x < numBlocks) {
        cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
      }
      cluster.barrier_wait(cluster.barrier_arrive());
      for (int i = 0; i < numBlocks; ++i) {
        fullSum += sharedVal[i];
      }
    }
    float rcpRms = rsqrtf(fullSum / tokenDim + params.epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fi::utils::fromFloat<T>(
          fi::utils::toFloat<T>(packedAccum.elements[i]) * rcpRms *
          fi::utils::toFloat<T>(gamma.elements[i]));
    }
  }
  if (params.outputPtr != nullptr) {
    reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = packedAccum.packed;
  }
  flag.waitAndUpdate(
      {static_cast<uint32_t>(params.numTokens * tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

// ---------------------------------------------------------------------------
// Host dispatch (mirrors oneshotAllreduceFusionDispatch, RMSNorm-only path)
// ---------------------------------------------------------------------------

template <typename T, typename ScaleT>
cudaError_t moeFinalizeARRMSDispatch(MoeFinalizeARRMSKernelParams<T, ScaleT> const& params,
                                     bool launchWithPdl, cudaStream_t stream) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);

  auto [blockSize, clusterSize, loadsPerThread] =
      fi::utils::adjustGridConfig(numTokens, tokenDim, eltsPerThread, /*useCluster=*/true);
  dim3 grid(numTokens, clusterSize, 1);

  FLASHINFER_CHECK(blockSize <= 1024 && loadsPerThread == 1,
                   "moe_finalize_ar_rms: hidden dimension %d exceeds the maximum supported "
                   "hidden dimension (%d)",
                   tokenDim, 1024 * 8 * eltsPerThread);

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = launchWithPdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = clusterSize;
  attrs[1].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{
      .gridDim = grid,
      .blockDim = static_cast<unsigned int>(blockSize),
      .dynamicSmemBytes = 0,
      .stream = stream,
      .attrs = attrs,
      .numAttrs = 2,
  };

#define LAUNCH_MOE_FIN_AR_KERNEL(WORLD_SIZE)                                           \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                             \
      &config, &moeFinalizeARRMSFusionKernel<WORLD_SIZE, T, ScaleT, float4>, params));

  switch (params.nRanks) {
    case 2:
      LAUNCH_MOE_FIN_AR_KERNEL(2);
      break;
    case 4:
      LAUNCH_MOE_FIN_AR_KERNEL(4);
      break;
    case 8:
      LAUNCH_MOE_FIN_AR_KERNEL(8);
      break;
    default:
      FLASHINFER_ERROR("moe_finalize_ar_rms: unsupported world_size " +
                       std::to_string(params.nRanks) + ". Supported sizes: {2, 4, 8}");
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_MOE_FIN_AR_KERNEL
  return cudaSuccess;
}

}  // namespace moe_finalize_ar_rms

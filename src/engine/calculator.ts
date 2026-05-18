import type {
  CalculationInput,
  CalculationResult,
  MemoryBreakdownItem,
  ModelVariant,
} from "./types";
import { getBytesPerParam, getKvCacheBytes } from "./quantization";
import { getEffectiveVram, getGpuFactor, getGpuTdp } from "./gpu";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const GB = 1_000_000_000;

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

// ---------------------------------------------------------------------------
// Attention structure – determines KV head count
// ---------------------------------------------------------------------------

function getNumKvHeads(model: ModelVariant): number {
  if (model.num_key_value_heads && model.num_key_value_heads > 0) {
    return model.num_key_value_heads;
  }

  const arch = (model.attention_structure || "mha").toLowerCase();
  if (arch === "mqa") return 1;

  const hiddenDim = model.hidden_dim_size || 4096;
  // Standard head_dim = 128 for most modern models
  const headDim = 128;
  const numQueryHeads = model.num_attention_heads && model.num_attention_heads > 0
    ? model.num_attention_heads
    : Math.round(hiddenDim / headDim);

  if (arch === "gqa") {
    // For GQA, KV heads is typically num_query_heads / num_groups
    // Common grouping: 4 or 8
    // We infer from hidden_dim: typical GQA models have explicit kv_heads
    // Fall back to heuristic
    return Math.max(1, Math.ceil(numQueryHeads / 4));
  }

  // MLA (DeepSeek-style) uses a compressed KV representation
  if (arch === "mla") {
    // MLA reduces KV cache by using a low-rank joint compression
    // The effective KV dimension is much smaller
    // DeepSeek-V2 uses q_lora_rank=1536, kv_lora_rank=512
    // For a 3B model with MLA, the effective per-head KV dim is much smaller
    return Math.max(1, Math.round(hiddenDim / 128 / 8));
  }

  // MHA: same as query heads
  return numQueryHeads;
}

// ---------------------------------------------------------------------------
// Multi-GPU interconnect efficiency
// ---------------------------------------------------------------------------

function getInterconnectEfficiency(
  interconnect: string | null,
  numGpus: number
): number {
  if (numGpus <= 1) return 1.0;

  const map: Record<string, number> = {
    nvlink_gen4: 0.95,
    nvlink_gen3: 0.92,
    nvlink_gen2: 0.88,
    nvlink_bridge: 0.82,
    infiniband_hdr: 0.88,
    infiniband_edr: 0.84,
    pcie5: 0.82,
    pcie4: 0.78,
    pcie3: 0.72,
    ethernet400g: 0.8,
    ethernet200g: 0.75,
    ethernet100g: 0.7,
    ethernet25g: 0.55,
    ethernet10g: 0.4,
    ethernet1g: 0.2,
  };

  return interconnect ? (map[interconnect] ?? 0.75) : 0.8;
}

function getMultiGpuOverhead(baseVram: number, numGpus: number): number {
  if (numGpus <= 1) return 0;
  return baseVram * 0.05 * Math.sqrt(numGpus - 1);
}

// ---------------------------------------------------------------------------
// Core memory calculations
// ---------------------------------------------------------------------------

function calcModelWeights(model: ModelVariant, bytesPerParam: number): number {
  const params = model.num_of_params * 1e9;
  return round2((params * bytesPerParam) / GB);
}

function calcExpertWeights(model: ModelVariant, bytesPerParam: number): number {
  const experts = model.num_of_expert_params;
  if (!experts || experts <= 0) return 0;
  return round2((experts * 1e9 * bytesPerParam) / GB);
}

function calcSharedWeights(
  model: ModelVariant,
  bytesPerParam: number
): number {
  const params = model.num_of_params * 1e9;
  const experts = (model.num_of_expert_params || 0) * 1e9;
  const shared = Math.max(0, params - experts);
  return round2((shared * bytesPerParam) / GB);
}

function calcKvCache(
  model: ModelVariant,
  seqLen: number,
  batchSize: number,
  bytesPerKv: number
): number {
  const numLayers = model.num_of_layers;
  const numKvHeads = getNumKvHeads(model);
  const headDim = 128; // standard head dimension for KV projection

  const size =
    (2 * numLayers * numKvHeads * headDim * seqLen * batchSize * bytesPerKv) /
    GB;

  return round2(size);
}

function calcActivations(
  model: ModelVariant,
  seqLen: number,
  batchSize: number,
  bytesPerElem: number
): number {
  const hiddenDim = model.hidden_dim_size;
  const numLayers = model.num_of_layers;

  // K factor calibrated against WASM:
  //   For 7B GQA: K≈2.0 gives 0.54GB at seq=1024 (matches WASM exactly)
  //   For MoE:    K≈2.0 gives 2.15GB at seq=4096
  //   Formula: batch * seq * hidden * layers * 2 * K / GB
  const arch = model.architecture?.toLowerCase() || "dense";
  const attention = (model.attention_structure || "mha").toLowerCase();

  let K = 2.0;

  if (arch === "moe") K *= 0.7;
  if (attention === "mqa") K *= 0.9;
  if (attention === "gqa") K *= 0.95;
  if (attention === "mla") K *= 0.75;

  const size =
    (batchSize * seqLen * hiddenDim * numLayers * bytesPerElem * K) / GB;

  return round2(size);
}

function calcOptimizerStates(
  numParams: number,
  bytesPerState: number
): number {
  // AdamW stores 2 states per parameter
  return round2((numParams * 1e9 * 2 * bytesPerState) / GB);
}

function calcGradients(
  numParams: number,
  bytesPerGrad: number
): number {
  return round2((numParams * 1e9 * bytesPerGrad) / GB);
}

function calcLoRAOverhead(
  model: ModelVariant,
  rank: number,
  bytesPerParam: number,
  bytesPerState: number
): number {
  // Approximate LoRA adapter parameter count
  // LoRA adds parameters for attention Q,K,V,O projections + optionally MLP
  const hiddenDim = model.hidden_dim_size;
  const numLayers = model.num_of_layers;

  // Per layer: Q(rank*hidden + hidden*rank) + V(rank*hidden + hidden*rank)
  // = 4 * rank * hidden per layer (for Q and V)
  const loraParamsPerLayer = 4 * rank * hiddenDim;
  const totalLoraParams = numLayers * loraParamsPerLayer;

  // LoRA params + optimizer states (2x) + gradients
  const paramMem = (totalLoraParams * bytesPerParam) / GB;
  const optMem = (totalLoraParams * 2 * bytesPerState) / GB;
  const gradMem = (totalLoraParams * bytesPerParam) / GB;

  return round2(paramMem + optMem + gradMem);
}

function calcFrameworkOverhead(_weightsGb: number, extraGb: number): number {
  // Wasm-calibrated: ~1GB base + small percentage of non-weight memory
  return round2(1.0 + (extraGb * 0.005));
}

// ---------------------------------------------------------------------------
// Performance estimation (TPS)
// ---------------------------------------------------------------------------

function estimateTps(
  model: ModelVariant,
  bytesPerParam: number,
  seqLen: number,
  batchSize: number,
  gpuKey: string,
  numGpus: number,
  interconnect: string | null,
  isTraining: boolean
): {
  latencyTps: number;
  throughputTps: number;
  perUserTps: number;
  concurrentUsers: number;
  msPerToken: number;
  tftt: number;
} {
  const factor = getGpuFactor(gpuKey);
  const eff = getInterconnectEfficiency(interconnect, numGpus);

  // Baseline: RTX 3060 (factor=1.0, ~360 GB/s bandwidth) on 7B FP16
  // Real-world llama.cpp: ~20-30 tok/s for 7B Q4, ~15-20 for 7B FP16
  const baseParams = 7;
  const modelParams = model.num_of_params;
  const paramScale = baseParams / modelParams;
  const precisionScale = 2.0 / bytesPerParam;

  let baseTps = 20 * paramScale * precisionScale;

  // Adjust for attention structure
  const attention = (model.attention_structure || "mha").toLowerCase();
  if (attention === "gqa") baseTps *= 1.1;
  if (attention === "mqa") baseTps *= 1.15;
  if (attention === "mla") baseTps *= 1.05; // MLA has compute overhead

  // MoE: fewer active params per token → faster inference
  if (model.architecture === "moe") {
    const active = model.num_of_active_experts || 2;
    const total = model.num_of_experts || 8;
    // Speedup roughly: total_params / active_params
    const moeSpeedup = Math.min(3, total / Math.max(1, active));
    baseTps *= moeSpeedup * 0.7; // 0.7 dampening for routing overhead
  }

  // Sequence length effect: longer sequences = more compute per token
  if (seqLen > 1024) {
    baseTps *= Math.pow(1024 / seqLen, 0.15); // mild penalty
  }

  // Apply GPU factor and multi-GPU scaling
  const singleGpuTps = baseTps * factor;
  const rawTps = singleGpuTps * (1 + (numGpus - 1) * eff);

  // Training overhead: ~3x slower per token due to backward pass
  const speedMultiplier = isTraining ? 1 / 3 : 1;
  const latencyTps = round4(rawTps * speedMultiplier);

  // Throughput = latency * batch_size (but sub-linear scaling for large batches)
  const batchEfficiency = Math.pow(batchSize, 0.85);
  const throughputTps = round4(latencyTps * batchEfficiency);

  // Per-user TPS
  const concurrentUsers = 1; // used for the TPS calc
  const perUserTps = round4(latencyTps / concurrentUsers);

  // Time per token (ms)
  const msPerToken = latencyTps > 0 ? round4(1000 / latencyTps) : 0;

  // Time to First Token — prompt is processed in one parallel forward pass
  // TTFT ~= time for ~15 "effective tokens" of compute (heuristic)
  const tftt = msPerToken > 0 ? Math.round(msPerToken * 15) : 0;

  return { latencyTps, throughputTps, perUserTps, concurrentUsers, msPerToken, tftt };
}

// ---------------------------------------------------------------------------
// Main calculation entry point
// ---------------------------------------------------------------------------

export function calculateVram(input: CalculationInput): CalculationResult {
  const {
    model_variant: model,
    quantization,
    kv_cache_quantization,
    gpu_key,
    custom_vram,
    num_gpus,
    batch_size,
    calc_mode,
    finetuning_method,
    fine_tuning_quantization,
    seq_length,
    lora_rank,
    concurrent_users,
    gradient_accumulation_steps,
    enable_offloading,
    offload_kv_cache,
    num_offload_layers,
    carbon_intensity,
    interconnect_type,
    num_samples,
    tokens_per_sample,
    num_epochs,
    energy_cost_per_kwh,
  } = input;

  const bytesPerParam = getBytesPerParam(quantization);
  const bytesPerKv = getKvCacheBytes(kv_cache_quantization);
  const isTraining = calc_mode === "finetuning";
  const gpuVram = getEffectiveVram(gpu_key, custom_vram);

  // --- Memory breakdown ---
  const breakdown: MemoryBreakdownItem[] = [];

  // Model weights
  let weightsGb: number;
  let sharedWeightsGb = 0;
  let expertWeightsGb = 0;

  if (model.architecture === "moe" && model.num_of_expert_params) {
    sharedWeightsGb = calcSharedWeights(model, bytesPerParam);
    expertWeightsGb = calcExpertWeights(model, bytesPerParam);
    weightsGb = round2(sharedWeightsGb + expertWeightsGb);
  } else {
    weightsGb = calcModelWeights(model, bytesPerParam);
  }

  // KV Cache (inference only)
  const kvCacheGb = isTraining ? 0 : calcKvCache(model, seq_length, batch_size, bytesPerKv);

  // KV Cache scales with concurrent users (one per user)
  const kvCachePerUser = kvCacheGb;
  const kvCacheTotal = kvCacheGb * Math.max(1, concurrent_users);

  // Activations are shared across batched users
  const effectiveBatchSize = isTraining ? batch_size * gradient_accumulation_steps : batch_size;
  const activationsGb = calcActivations(model, seq_length, effectiveBatchSize, bytesPerParam);

  // Optimizer + Gradients (training)
  let optimizerGb = 0;
  let gradientsGb = 0;
  let loraOverheadGb = 0;
  let tempBuffersGb = 0;

  if (isTraining) {
    const trainableParams = model.num_of_params;
    const ftBytes = getBytesPerParam(fine_tuning_quantization || "fp16");

    if (finetuning_method === "lora" || finetuning_method === "qlora") {
      const baseQuant = finetuning_method === "qlora" ? "q4" : quantization;
      weightsGb = calcModelWeights(model, getBytesPerParam(baseQuant));
      loraOverheadGb = calcLoRAOverhead(model, lora_rank || 16, ftBytes, 4);
    } else {
      optimizerGb = calcOptimizerStates(trainableParams, 4);
      gradientsGb = calcGradients(trainableParams, ftBytes);
      tempBuffersGb = round2(activationsGb * 0.3);
    }
  }

  // Multi-GPU: shard model weights, optimizer states, and gradients across GPUs
  // KV cache and activations stay per-GPU (each GPU processes its own shard)
  if (num_gpus > 1) {
    weightsGb = round2(weightsGb / num_gpus);
    optimizerGb = round2(optimizerGb / num_gpus);
    gradientsGb = round2(gradientsGb / num_gpus);
    loraOverheadGb = round2(loraOverheadGb / num_gpus);
    // Activations and KV cache are already per-GPU
  }

  // Compute subtotal BEFORE overhead to feed into overhead formula
  const subtotalGb = round2(
    weightsGb + kvCacheTotal + activationsGb + optimizerGb +
    gradientsGb + loraOverheadGb + tempBuffersGb
  );
  const nonWeightGb = subtotalGb - weightsGb;

  // Framework overhead (calibrated: ~1GB base + small fraction of non-weight memory)
  let overheadGb = calcFrameworkOverhead(weightsGb, nonWeightGb);
  const multiGpuOverheadGb = num_gpus > 1 ? getMultiGpuOverhead(subtotalGb, num_gpus) : 0;
  if (num_gpus > 1) {
    overheadGb = round2(overheadGb + multiGpuOverheadGb);
  }

  // --- Total ---
  const totalUsage = round2(subtotalGb + overheadGb);

  // --- Derived ---
  const staticShared = round2(
    weightsGb + optimizerGb + gradientsGb + loraOverheadGb + overheadGb
  );
  const perUserMemory = round2(
    kvCachePerUser + (activationsGb / Math.max(1, concurrent_users))
  );

  // --- Offloading ---
  let offloadedGb = 0;
  let offloadedWeightsGb = 0;
  let offloadedKvGb = 0;
  if (enable_offloading) {
    const layersToOffload =
      num_offload_layers != null
        ? Math.min(num_offload_layers, model.num_of_layers)
        : model.num_of_layers;
    const offloadFraction = layersToOffload / model.num_of_layers;

    offloadedWeightsGb = round2(weightsGb * 0.85 * offloadFraction);
    if (offload_kv_cache) {
      offloadedKvGb = kvCacheTotal;
    }
    offloadedGb = round2(offloadedWeightsGb + offloadedKvGb);
  }

  const effectiveWeightsGb = round2(Math.max(0, weightsGb - offloadedWeightsGb));
  const effectiveKvCacheGb = round2(Math.max(0, kvCacheTotal - offloadedKvGb));
  const effectiveUsage = round2(Math.max(0, totalUsage - offloadedGb));
  const breakdownTotalGb = Math.max(0.0001, effectiveUsage);

  // --- Build breakdown (post-offload, aligned with displayed effective usage) ---
  if (model.architecture === "moe" && model.num_of_expert_params) {
    const shardedSharedGb = num_gpus > 1 ? round2(sharedWeightsGb / num_gpus) : sharedWeightsGb;
    const shardedExpertGb = num_gpus > 1 ? round2(expertWeightsGb / num_gpus) : expertWeightsGb;
    const shardedWeightTotalGb = Math.max(0.0001, round2(shardedSharedGb + shardedExpertGb));
    const sharedRatio = shardedSharedGb / shardedWeightTotalGb;
    const effectiveSharedGb = round2(effectiveWeightsGb * sharedRatio);
    const effectiveExpertGb = round2(Math.max(0, effectiveWeightsGb - effectiveSharedGb));
    breakdown.push({ label: "Shared Backbone Weights", value: round4((effectiveSharedGb / breakdownTotalGb) * 100), size_gb: effectiveSharedGb });
    breakdown.push({ label: "All Expert Weights", value: round4((effectiveExpertGb / breakdownTotalGb) * 100), size_gb: effectiveExpertGb });
  } else {
    breakdown.push({ label: "Base Model Weights", value: round4((effectiveWeightsGb / breakdownTotalGb) * 100), size_gb: effectiveWeightsGb });
  }

  if (activationsGb > 0) {
    breakdown.push({ label: "Activations", value: round4((activationsGb / breakdownTotalGb) * 100), size_gb: activationsGb });
  }

  if (isTraining && optimizerGb > 0) {
    breakdown.push({ label: "Optimizer States", value: round4((optimizerGb / breakdownTotalGb) * 100), size_gb: optimizerGb });
    breakdown.push({ label: "Gradients", value: round4((gradientsGb / breakdownTotalGb) * 100), size_gb: gradientsGb });
    if (tempBuffersGb > 0) {
      breakdown.push({ label: "Temp Buffers", value: round4((tempBuffersGb / breakdownTotalGb) * 100), size_gb: tempBuffersGb });
    }
  }

  if (effectiveKvCacheGb > 0) {
    breakdown.push({ label: "KV Cache", value: round4((effectiveKvCacheGb / breakdownTotalGb) * 100), size_gb: effectiveKvCacheGb });
  }

  if (loraOverheadGb > 0) {
    breakdown.push({ label: "LoRA Adapters, Optimizer & Gradients", value: round4((loraOverheadGb / breakdownTotalGb) * 100), size_gb: loraOverheadGb });
  }

  if (num_gpus > 1 && multiGpuOverheadGb > 0) {
    breakdown.push({ label: "Multi-GPU Overhead", value: round4((multiGpuOverheadGb / breakdownTotalGb) * 100), size_gb: multiGpuOverheadGb });
  }

  breakdown.push({ label: "Framework Overhead", value: round4((overheadGb / breakdownTotalGb) * 100), size_gb: overheadGb });
  // Per GPU percentage; WASM caps at 100
  const perGpuVram = gpuVram;
  const rawPct = round4((effectiveUsage / perGpuVram) * 100);
  const rawActualPct = round4((totalUsage / perGpuVram) * 100);
  const vramPct = Math.min(100, rawPct);
  const actualPct = Math.min(100, rawActualPct);

  // Memory status based on effective per-GPU usage (after offloading).
  // This aligns status with the displayed usage/progress.
  //   "Sufficient" < ~50%, "Okay" < ~65%, "Moderate" < ~80%, "High" < ~95%, "Insufficient"
  let memoryStatus: string;
  if (rawPct >= 95 || effectiveUsage >= perGpuVram) memoryStatus = "Insufficient";
  else if (rawPct >= 80) memoryStatus = "High";
  else if (rawPct >= 65) memoryStatus = "Moderate";
  else if (rawPct >= 50) memoryStatus = "Okay";
  else memoryStatus = "Sufficient";

  // --- Performance ---
  const perf = estimateTps(
    model,
    bytesPerParam,
    seq_length,
    batch_size,
    gpu_key,
    num_gpus,
    interconnect_type,
    isTraining
  );

  // --- Power draw ---
  const tdp = getGpuTdp(gpu_key);
  const tdpUtilization = Math.min(1, actualPct / 100);
  const powerDraw = Math.round(tdp * num_gpus * (0.3 + 0.7 * tdpUtilization));

  // --- System RAM ---
  const systemRam = round2(effectiveUsage * 1.2 + 4);

  // --- Carbon emissions ---
  let carbonPerHour: number | undefined;
  let carbonPerDay: number | undefined;
  let carbonPerMonth: number | undefined;
  let carbonPerYear: number | undefined;

  if (carbon_intensity != null && carbon_intensity > 0) {
    const powerKw = powerDraw / 1000;
    carbonPerHour = round4(powerKw * carbon_intensity / 1000); // kg CO2
    carbonPerDay = round4(carbonPerHour * 24);
    carbonPerMonth = round4(carbonPerDay * 30);
    carbonPerYear = round4(carbonPerDay * 365);
  }

  // --- Training-specific metrics & Energy cost ---
  let trainingTps: number | undefined;
  let samplesPerSec: number | undefined;
  let stepsPerSec: number | undefined;
  let totalTokens: number | undefined;
  let totalTrainingTimeHours: number | undefined;
  let energyCostPerHour: number | undefined;
  let energyCostTotal: number | undefined;

  if (isTraining && num_samples && tokens_per_sample && num_epochs) {
    trainingTps = perf.latencyTps;
    const tokensPerStep = effectiveBatchSize * seq_length;
    stepsPerSec = trainingTps / tokensPerStep;
    const secsPerStep = 1 / Math.max(stepsPerSec, 0.0001);
    samplesPerSec = round4(effectiveBatchSize / secsPerStep);

    totalTokens = num_samples * tokens_per_sample * num_epochs;
    const totalSteps = Math.ceil(
      (num_samples * num_epochs) / effectiveBatchSize
    );
    totalTrainingTimeHours = round4((totalSteps * secsPerStep) / 3600);
  }

  if (energy_cost_per_kwh != null && energy_cost_per_kwh > 0) {
    const powerKw = powerDraw / 1000;
    energyCostPerHour = round4(powerKw * energy_cost_per_kwh * num_gpus);
    if (totalTrainingTimeHours != null) {
      energyCostTotal = round4(energyCostPerHour * totalTrainingTimeHours);
    }
  }

  // Apply concurrent users to throughput for inference
  const inferenceThroughput = isTraining
    ? perf.throughputTps
    : round4(perf.latencyTps * Math.pow(concurrent_users, 0.7));
  const inferencePerUser = isTraining
    ? perf.perUserTps
    : round4(perf.latencyTps / Math.max(1, concurrent_users));

  return {
    vram_usage: effectiveUsage,
    vram_percentage: vramPct,
    actual_vram_percentage: actualPct,
    memory_status: memoryStatus,
    memory_breakdown: breakdown,
    static_shared_memory: staticShared,
    per_user_memory: perUserMemory,
    offloaded_memory: offloadedGb,
    estimated_latency_tps: perf.latencyTps,
    estimated_throughput_tps: inferenceThroughput,
    per_user_tps: inferencePerUser,
    ms_per_token: perf.msPerToken,
    tftt: perf.tftt,
    estimated_power_draw: powerDraw,
    estimated_system_ram_required: systemRam,
    training_tps: trainingTps,
    samples_per_second: samplesPerSec,
    steps_per_second: stepsPerSec,
    total_tokens: totalTokens,
    total_training_time_hours: totalTrainingTimeHours,
    carbon_emissions_per_hour: carbonPerHour,
    carbon_emissions_per_day: carbonPerDay,
    carbon_emissions_per_month: carbonPerMonth,
    carbon_emissions_per_year: carbonPerYear,
    energy_cost_per_hour_usd: energyCostPerHour,
    energy_cost_total_usd: energyCostTotal,
  };
}

export function initEngine(): void {
  // No-op — pure TypeScript needs no init
}

export function isEngineReady(): boolean {
  return true;
}

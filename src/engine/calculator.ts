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
  if (
    typeof model.num_key_value_heads === "number" &&
    Number.isFinite(model.num_key_value_heads) &&
    model.num_key_value_heads > 0
  ) {
    return model.num_key_value_heads;
  }

  const attentionStructure = (model.attention_structure ?? "").toLowerCase();
  switch (attentionStructure) {
    case "mqa":
      return 1;
    case "gqa":
      return Math.max(
        1,
        Math.floor(
          model.num_attention_heads && model.num_attention_heads > 0
            ? model.num_attention_heads / 8
            : 8,
        ),
      );
    default:
      return Math.max(
        1,
        model.num_attention_heads && model.num_attention_heads > 0
          ? model.num_attention_heads
          : 32,
      );
  }
}

function getKvElementsPerTokenPerLayer(model: ModelVariant): number {
  if (
    typeof model.kv_lora_rank === "number" &&
    Number.isFinite(model.kv_lora_rank) &&
    model.kv_lora_rank > 0 &&
    typeof model.qk_rope_head_dim === "number" &&
    Number.isFinite(model.qk_rope_head_dim) &&
    model.qk_rope_head_dim >= 0
  ) {
    return model.kv_lora_rank + model.qk_rope_head_dim;
  }

  const kvHeads = getNumKvHeads(model);
  const headDim =
    typeof model.head_dim === "number" &&
    Number.isFinite(model.head_dim) &&
    model.head_dim > 0
      ? model.head_dim
      : model.num_attention_heads && model.num_attention_heads > 0
        ? model.hidden_dim_size / model.num_attention_heads
        : 128;

  return 2 * kvHeads * headDim;
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

function calcKvCache(
  model: ModelVariant,
  sequenceLength: number,
  batchSize: number,
  bytesPerElement: number,
): number {
  const elements =
    (model.num_kv_cache_layers ?? model.num_of_layers) *
    getKvElementsPerTokenPerLayer(model) *
    sequenceLength *
    batchSize;
  return (elements * bytesPerElement) / GB;
}

function calcActivations(
  model: ModelVariant,
  seqLen: number,
  batchSize: number,
  bytesPerElement: number,
  activationCheckpointing: boolean,
): number {
  const architectureFactor =
    model.architecture.toLowerCase() === "moe" ? 0.7 : 1;
  const attention = model.attention_structure.toLowerCase();
  const attentionFactor =
    attention === "mqa"
      ? 0.9
      : attention === "gqa"
        ? 0.95
        : attention === "mla" || attention === "dsa"
          ? 0.75
          : 1;
  const checkpointingFactor = activationCheckpointing ? 0.35 : 1;
  const size =
    (batchSize *
      seqLen *
      model.hidden_dim_size *
      model.num_of_layers *
      bytesPerElement *
      2 *
      architectureFactor *
      attentionFactor *
      checkpointingFactor) /
    GB;

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
  targetModules: number,
): { parameters: number; optimizer: number; gradients: number } {
  const paramsPerLayer =
    2 * rank * model.hidden_dim_size * targetModules;
  const totalParams = paramsPerLayer * model.num_of_layers;
  return {
    parameters: (totalParams * 2) / GB,
    optimizer: (totalParams * 8) / GB,
    gradients: (totalParams * 2) / GB,
  };
}

function calcFrameworkOverhead(memoryGb: number): number {
  return round2(1 + memoryGb * 0.005);
}

function getZeroStage(config: CalculationInput["optimization_config"]): 0 | 2 | 3 {
  if (config?.zero_stage === 2 || config?.zero_stage === 3) {
    return config.zero_stage;
  }

  const preset = config?.preset;
  switch (preset) {
    case undefined:
    case "default":
      return 0;
    case "zero2":
      return 2;
    case "zero3":
      return 3;
    default:
      throw new Error(`Unsupported optimization preset: ${preset}`);
  }
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
  isTraining: boolean,
): {
  latencyTps: number;
  throughputTps: number;
  perUserTps: number;
  msPerToken: number;
  tftt: number;
} {
  const activeModelParams = model.num_of_active_params ?? model.num_of_params;
  let baseTps = 25 * Math.pow(7 / activeModelParams, 0.85);

  if (
    model.num_of_active_params &&
    model.num_of_active_params < model.num_of_params
  ) {
    baseTps *= 0.7;
  }

  baseTps *=
    bytesPerParam <= 0.5 ? 2.2 : bytesPerParam <= 1 ? 1.8 : 1.0;

  if (seqLen > 1024) {
    baseTps *= Math.pow(1024 / seqLen, 0.15);
  }

  const gpuFactor = getGpuFactor(gpuKey);
  const interconnectEfficiency = getInterconnectEfficiency(
    interconnect,
    numGpus,
  );
  const singleGpuTps = baseTps * gpuFactor;
  const rawTps =
    singleGpuTps * (1 + (numGpus - 1) * interconnectEfficiency);
  const latencyTps = round4(rawTps * (isTraining ? 1 / 3 : 1));
  const throughputTps = round4(
    latencyTps * Math.pow(Math.max(1, batchSize), 0.85),
  );
  const msPerToken = latencyTps > 0 ? round4(1000 / latencyTps) : 0;

  return {
    latencyTps,
    throughputTps,
    perUserTps: latencyTps,
    msPerToken,
    tftt: msPerToken > 0 ? Math.round(msPerToken * 15) : 0,
  };
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
    offload_target,
    num_offload_layers,
    percentage_offload,
    offload_kv_cache,
    optimization_config,
    carbon_intensity,
    interconnect_type,
    num_samples,
    tokens_per_sample,
    num_epochs,
    energy_cost_per_kwh,
  } = input;

  const gpuCount = Math.max(1, num_gpus);
  const batchSize = Math.max(1, batch_size);
  const sequenceLength = Math.max(1, seq_length);
  const userCount = Math.max(1, concurrent_users);
  const isTraining = calc_mode === "finetuning";
  const weightQuantization = isTraining
    ? finetuning_method === "qlora"
      ? "q4"
      : fine_tuning_quantization || "fp16"
    : quantization;
  const bytesPerParam = getBytesPerParam(weightQuantization);
  const bytesPerKv = getKvCacheBytes(kv_cache_quantization);

  let weightsGb = calcModelWeights(model, bytesPerParam);
  let kvCachePerUserGb = isTraining
    ? 0
    : calcKvCache(model, sequenceLength, batchSize, bytesPerKv);
  let kvCacheGb = kvCachePerUserGb * userCount;
  const activationsGb = calcActivations(
    model,
    sequenceLength,
    batchSize,
    isTraining && finetuning_method === "full" ? bytesPerParam : 2,
    optimization_config?.gradient_checkpointing ?? false,
  );

  let optimizerGb = 0;
  let gradientsGb = 0;
  let loraParametersGb = 0;
  let loraOptimizerGb = 0;
  let loraGradientsGb = 0;

  if (isTraining && finetuning_method === "full") {
    optimizerGb = calcOptimizerStates(model.num_of_params, 4);
    gradientsGb = calcGradients(model.num_of_params, bytesPerParam);
  } else if (
    isTraining &&
    (finetuning_method === "lora" || finetuning_method === "qlora")
  ) {
    const lora = calcLoRAOverhead(model, Math.max(1, lora_rank ?? 8), 4);
    loraParametersGb = lora.parameters;
    loraOptimizerGb = lora.optimizer;
    loraGradientsGb = lora.gradients;
  }

  const zeroStage = isTraining ? getZeroStage(optimization_config) : 0;
  if (gpuCount > 1) {
    if (!isTraining || zeroStage === 3) {
      weightsGb /= gpuCount;
      loraParametersGb /= gpuCount;
    }
    if (!isTraining || zeroStage >= 2) {
      optimizerGb /= gpuCount;
      gradientsGb /= gpuCount;
      loraOptimizerGb /= gpuCount;
      loraGradientsGb /= gpuCount;
    }
    if (!isTraining) {
      kvCachePerUserGb /= gpuCount;
      kvCacheGb /= gpuCount;
    }
  }

  let offloadedWeightsGb = 0;
  let offloadedKvCacheGb = 0;
  if (enable_offloading) {
    const requestedFraction =
      num_offload_layers !== null
        ? num_offload_layers / Math.max(1, model.num_of_layers)
        : percentage_offload !== null
          ? percentage_offload / 100
          : 1;
    const offloadFraction = Math.max(0, Math.min(1, requestedFraction));
    offloadedWeightsGb = weightsGb * offloadFraction;
    if (!isTraining && offload_kv_cache) {
      offloadedKvCacheGb = kvCacheGb;
    }
  }

  const residentWeightsGb = weightsGb - offloadedWeightsGb;
  const residentKvCacheGb = kvCacheGb - offloadedKvCacheGb;
  const loraOverheadGb =
    loraParametersGb + loraOptimizerGb + loraGradientsGb;
  const componentMemoryGb =
    residentWeightsGb +
    residentKvCacheGb +
    activationsGb +
    optimizerGb +
    gradientsGb +
    loraOverheadGb;
  const frameworkOverheadGb = calcFrameworkOverhead(componentMemoryGb);
  const multiGpuOverheadGb = getMultiGpuOverhead(
    componentMemoryGb,
    gpuCount,
  );
  const allocatedVramGb =
    componentMemoryGb + frameworkOverheadGb + multiGpuOverheadGb;
  const effectiveUsage = allocatedVramGb / 0.9;
  const reservedHeadroomGb = effectiveUsage - allocatedVramGb;

  const breakdownParts: Array<{ label: string; sizeGb: number }> = [
    { label: "Model Weights", sizeGb: residentWeightsGb },
    { label: "KV Cache", sizeGb: residentKvCacheGb },
    { label: "Activations", sizeGb: activationsGb },
    { label: "Optimizer States", sizeGb: optimizerGb },
    { label: "Gradients", sizeGb: gradientsGb },
    { label: "LoRA Parameters & States", sizeGb: loraOverheadGb },
    { label: "Framework Overhead", sizeGb: frameworkOverheadGb },
    { label: "Multi-GPU Overhead", sizeGb: multiGpuOverheadGb },
    { label: "Reserved Headroom", sizeGb: reservedHeadroomGb },
  ];
  const breakdown: MemoryBreakdownItem[] = breakdownParts
    .filter((part) => part.sizeGb > 0)
    .map((part) => ({
      label: part.label,
      value: round4((part.sizeGb / effectiveUsage) * 100),
      size_gb: round4(part.sizeGb),
    }));

  const perGpuVram = getEffectiveVram(gpu_key, custom_vram);
  const vramPercentage = (effectiveUsage / perGpuVram) * 100;
  const actualVramPercentage = (allocatedVramGb / perGpuVram) * 100;
  let memoryStatus: string;
  if (vramPercentage >= 95) memoryStatus = "Insufficient";
  else if (vramPercentage >= 80) memoryStatus = "High";
  else if (vramPercentage >= 65) memoryStatus = "Moderate";
  else if (vramPercentage >= 50) memoryStatus = "Okay";
  else memoryStatus = "Sufficient";

  const performance = estimateTps(
    model,
    bytesPerParam,
    sequenceLength,
    batchSize,
    gpu_key,
    gpuCount,
    interconnect_type,
    isTraining,
  );
  const inferenceThroughput = isTraining
    ? performance.throughputTps
    : round4(performance.latencyTps * Math.pow(userCount, 0.7));
  const inferencePerUser = isTraining
    ? performance.perUserTps
    : round4(performance.latencyTps / userCount);

  const tdp = getGpuTdp(gpu_key);
  const tdpUtilization = Math.min(1, actualVramPercentage / 100);
  const powerDraw = Math.round(
    tdp * gpuCount * (0.3 + 0.7 * tdpUtilization),
  );
  const aggregateOffloadedGb =
    (offloadedWeightsGb + offloadedKvCacheGb) * gpuCount;
  const cpuOffloadedGb =
    enable_offloading && offload_target !== "nvme"
      ? aggregateOffloadedGb
      : 0;
  const systemRamGb =
    allocatedVramGb * gpuCount * 0.2 + cpuOffloadedGb + 4;

  let carbonPerHour: number | undefined;
  let carbonPerDay: number | undefined;
  let carbonPerMonth: number | undefined;
  let carbonPerYear: number | undefined;
  if (carbon_intensity !== null && carbon_intensity > 0) {
    carbonPerHour = round4((powerDraw / 1000) * (carbon_intensity / 1000));
    carbonPerDay = round4(carbonPerHour * 24);
    carbonPerMonth = round4(carbonPerDay * 30);
    carbonPerYear = round4(carbonPerDay * 365);
  }

  let trainingTps: number | undefined;
  let samplesPerSecond: number | undefined;
  let stepsPerSecond: number | undefined;
  let totalTokens: number | undefined;
  let totalTrainingTimeHours: number | undefined;
  if (
    isTraining &&
    num_samples !== null &&
    tokens_per_sample !== null &&
    num_epochs !== null
  ) {
    const sampleTokens = Math.max(1, tokens_per_sample);
    trainingTps = performance.throughputTps;
    samplesPerSecond = round4(trainingTps / sampleTokens);
    const effectiveGlobalBatch =
      batchSize *
      Math.max(1, gradient_accumulation_steps) *
      gpuCount;
    stepsPerSecond = round4(samplesPerSecond / effectiveGlobalBatch);
    totalTokens = num_samples * sampleTokens * num_epochs;
    totalTrainingTimeHours = round4(totalTokens / trainingTps / 3600);
  }

  let energyCostPerHour: number | undefined;
  let energyCostTotal: number | undefined;
  if (energy_cost_per_kwh !== null && energy_cost_per_kwh > 0) {
    energyCostPerHour = round4(
      (powerDraw / 1000) * energy_cost_per_kwh,
    );
    if (totalTrainingTimeHours !== undefined) {
      energyCostTotal = round4(
        energyCostPerHour * totalTrainingTimeHours,
      );
    }
  }

  return {
    vram_usage: round4(effectiveUsage),
    vram_percentage: round4(vramPercentage),
    actual_vram_percentage: round4(actualVramPercentage),
    memory_status: memoryStatus,
    memory_breakdown: breakdown,
    static_shared_memory: round4(effectiveUsage - residentKvCacheGb),
    per_user_memory: round4(
      kvCachePerUserGb + activationsGb / userCount,
    ),
    offloaded_memory: round4(offloadedWeightsGb + offloadedKvCacheGb),
    estimated_latency_tps: performance.latencyTps,
    estimated_throughput_tps: inferenceThroughput,
    per_user_tps: inferencePerUser,
    ms_per_token: performance.msPerToken,
    tftt: performance.tftt,
    estimated_power_draw: powerDraw,
    estimated_system_ram_required: round4(systemRamGb),
    training_tps: trainingTps,
    samples_per_second: samplesPerSecond,
    steps_per_second: stepsPerSecond,
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

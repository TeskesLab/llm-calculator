import { describe, expect, it } from "vitest";
import {
  calculateVram,
  type CalculationInput,
  type CalculationResult,
  type ModelVariant,
} from "./index";
import { getModelBySlug, models } from "../data/models";

function baseInput(overrides: Partial<CalculationInput> = {}): CalculationInput {
  const model = getModelBySlug("deepseek-v4-flash");
  if (!model) throw new Error("Model deepseek-v4-flash not found");

  return {
    model_variant: {
      num_of_params: parseFloat(model.num_of_params),
      architecture: model.architecture,
      modality: model.modality || "text",
      hidden_dim_size: model.hidden_dim_size ?? 4096,
      num_of_layers: model.num_of_layers ?? 32,
      num_kv_cache_layers: model.num_kv_cache_layers || undefined,
      num_of_active_params: model.num_of_active_params || undefined,
      num_of_experts: model.num_of_experts || undefined,
      num_of_active_experts: model.num_of_active_experts || undefined,
      attention_structure: model.attention_structure,
      num_attention_heads: model.num_attention_heads || undefined,
      num_key_value_heads: model.num_key_value_heads || undefined,
      head_dim: model.head_dim || undefined,
      kv_lora_rank: model.kv_lora_rank || undefined,
      qk_rope_head_dim: model.qk_rope_head_dim || undefined,
      position_embedding: model.position_embedding,
    },
    quantization: "nvfp4",
    kv_cache_quantization: "int8",
    gpu_key: "gb10_128",
    custom_vram: null,
    num_gpus: 1,
    batch_size: 1,
    calc_mode: "inference",
    finetuning_method: null,
    fine_tuning_quantization: null,
    seq_length: 1024,
    lora_rank: null,
    concurrent_users: 1,
    gradient_accumulation_steps: 1,
    enable_offloading: false,
    offload_target: null,
    num_offload_layers: null,
    percentage_offload: null,
    offload_kv_cache: false,
    optimization_config: null,
    carbon_intensity: null,
    interconnect_type: null,
    num_samples: null,
    tokens_per_sample: null,
    num_epochs: null,
    energy_cost_per_kwh: null,
    ...overrides,
  };
}

const compactDenseModel: ModelVariant = {
  num_of_params: 1,
  architecture: "dense",
  modality: "text",
  hidden_dim_size: 1024,
  num_of_layers: 8,
  attention_structure: "gqa",
  num_attention_heads: 8,
  num_key_value_heads: 2,
  head_dim: 128,
};

function trainingInput(
  overrides: Partial<CalculationInput> = {},
): CalculationInput {
  return baseInput({
    model_variant: compactDenseModel,
    quantization: "fp16",
    num_gpus: 2,
    batch_size: 2,
    calc_mode: "finetuning",
    finetuning_method: "full",
    fine_tuning_quantization: "fp16",
    num_samples: 100,
    tokens_per_sample: 512,
    num_epochs: 2,
    ...overrides,
  });
}

function breakdownSize(
  result: CalculationResult,
  label: string,
): number {
  return result.memory_breakdown.find((item) => item.label === label)?.size_gb ?? 0;
}

describe("calculator sanity", () => {
  it("reduces per-GPU VRAM when adding GPUs for DeepSeek V4 Flash", () => {
    const one = calculateVram(baseInput({ num_gpus: 1 }));
    const two = calculateVram(baseInput({ num_gpus: 2 }));
    const four = calculateVram(baseInput({ num_gpus: 4 }));

    expect(two.vram_usage).toBeLessThan(one.vram_usage);
    expect(four.vram_usage).toBeLessThan(two.vram_usage);
  });

  it("reduces per-GPU VRAM at max context when adding GPUs", () => {
    const one = calculateVram(baseInput({ num_gpus: 1, seq_length: 1_000_000 }));
    const two = calculateVram(baseInput({ num_gpus: 2, seq_length: 1_000_000 }));
    const four = calculateVram(baseInput({ num_gpus: 4, seq_length: 1_000_000 }));

    expect(two.vram_usage).toBeLessThan(one.vram_usage);
    expect(four.vram_usage).toBeLessThan(two.vram_usage);
  });

  it("offloads total KV cache across concurrent users", () => {
    const noOffload = calculateVram(baseInput({ concurrent_users: 8, enable_offloading: true, offload_kv_cache: false }));
    const withKvOffload = calculateVram(baseInput({ concurrent_users: 8, enable_offloading: true, offload_kv_cache: true }));

    expect(withKvOffload.vram_usage).toBeLessThan(noOffload.vram_usage);
    expect(withKvOffload.offloaded_memory).toBeGreaterThan(noOffload.offloaded_memory);
  });

  it("keeps memory status aligned with effective VRAM percentage", () => {
    const result = calculateVram(baseInput({ num_gpus: 2 }));
    if (result.vram_percentage >= 95) expect(result.memory_status).toBe("Insufficient");
    if (result.vram_percentage < 95 && result.vram_percentage >= 80) expect(result.memory_status).toBe("High");
    if (result.vram_percentage < 80 && result.vram_percentage >= 65) expect(result.memory_status).toBe("Moderate");
    if (result.vram_percentage < 65 && result.vram_percentage >= 50) expect(result.memory_status).toBe("Okay");
    if (result.vram_percentage < 50) expect(result.memory_status).toBe("Sufficient");
  });

  it("uses compressed latent dimensions for MLA KV cache", () => {
    const model: ModelVariant = {
      ...compactDenseModel,
      num_of_layers: 61,
      hidden_dim_size: 7168,
      attention_structure: "mla",
      num_attention_heads: 128,
      num_key_value_heads: 128,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
    };
    const result = calculateVram(
      baseInput({
        model_variant: model,
        quantization: "fp16",
        kv_cache_quantization: "int8",
      }),
    );
    const expectedKvGb = (61 * (512 + 64) * 1024) / 1_000_000_000;

    expect(breakdownSize(result, "KV Cache")).toBeCloseTo(expectedKvGb, 4);
  });

  it("uses tokens per sample for training duration", () => {
    const result = calculateVram(trainingInput());
    const expectedTokens = 100 * 512 * 2;

    expect(result.total_tokens).toBe(expectedTokens);
    expect(result.total_training_time_hours).toBeCloseTo(
      expectedTokens / result.training_tps! / 3600,
      4,
    );
  });

  it("charges aggregate GPU power exactly once", () => {
    const result = calculateVram(
      trainingInput({ num_gpus: 4, energy_cost_per_kwh: 0.2 }),
    );
    const expectedHourlyCost =
      Math.round((result.estimated_power_draw / 1000) * 0.2 * 10_000) /
      10_000;

    expect(result.energy_cost_per_hour_usd).toBe(expectedHourlyCost);
  });

  it("applies ZeRO preset sharding to the correct training states", () => {
    const defaultResult = calculateVram(
      trainingInput({ optimization_config: { preset: "default" } }),
    );
    const zero2 = calculateVram(
      trainingInput({ optimization_config: { preset: "zero2" } }),
    );
    const zero3 = calculateVram(
      trainingInput({ optimization_config: { preset: "zero3" } }),
    );

    expect(zero2.vram_usage).toBeLessThan(defaultResult.vram_usage);
    expect(zero3.vram_usage).toBeLessThan(zero2.vram_usage);
  });

  it("uses training precision for full-finetuning weights and gradients", () => {
    const fp16 = calculateVram(
      trainingInput({ fine_tuning_quantization: "fp16" }),
    );
    const fp32 = calculateVram(
      trainingInput({ fine_tuning_quantization: "fp32" }),
    );

    expect(fp32.vram_usage).toBeGreaterThan(fp16.vram_usage);
    expect(breakdownSize(fp32, "Model Weights")).toBe(
      breakdownSize(fp16, "Model Weights") * 2,
    );
  });

  it("changes optimizer cadence without inflating activation memory", () => {
    const oneStep = calculateVram(
      trainingInput({ gradient_accumulation_steps: 1 }),
    );
    const eightSteps = calculateVram(
      trainingInput({ gradient_accumulation_steps: 8 }),
    );

    expect(eightSteps.vram_usage).toBe(oneStep.vram_usage);
    expect(eightSteps.steps_per_second).toBeLessThan(oneStep.steps_per_second!);
  });

  it("reports one complete multi-GPU memory breakdown", () => {
    const result = calculateVram(trainingInput());
    const multiGpuEntries = result.memory_breakdown.filter(
      (item) => item.label === "Multi-GPU Overhead",
    );
    const percentageTotal = result.memory_breakdown.reduce(
      (total, item) => total + item.value,
      0,
    );

    expect(multiGpuEntries).toHaveLength(1);
    expect(percentageTotal).toBeCloseTo(100, 2);
  });

  it("accounts CPU offload in system RAM while keeping NVMe off host RAM", () => {
    const noOffload = calculateVram(
      baseInput({ model_variant: compactDenseModel, quantization: "q4" }),
    );
    const zeroLayers = calculateVram(
      baseInput({
        model_variant: compactDenseModel,
        quantization: "q4",
        enable_offloading: true,
        offload_target: "cpu",
        num_offload_layers: 0,
      }),
    );
    const cpuOffload = calculateVram(
      baseInput({
        model_variant: compactDenseModel,
        quantization: "q4",
        enable_offloading: true,
        offload_target: "cpu",
        percentage_offload: 100,
      }),
    );
    const nvmeOffload = calculateVram(
      baseInput({
        model_variant: compactDenseModel,
        quantization: "q4",
        enable_offloading: true,
        offload_target: "nvme",
        percentage_offload: 100,
      }),
    );

    expect(zeroLayers.vram_usage).toBe(noOffload.vram_usage);
    expect(cpuOffload.estimated_system_ram_required).toBeGreaterThan(
      noOffload.estimated_system_ram_required,
    );
    expect(nvmeOffload.estimated_system_ram_required).toBeLessThan(
      noOffload.estimated_system_ram_required,
    );
  });

  it("uses total MoE parameters for weights and active parameters for speed", () => {
    const moeModel: ModelVariant = {
      ...compactDenseModel,
      architecture: "moe",
      num_of_params: 671,
      num_of_active_params: 37,
      num_of_experts: 256,
      num_of_active_experts: 8,
    };
    const activeResult = calculateVram(
      baseInput({ model_variant: moeModel, quantization: "q4" }),
    );
    const totalOnlyResult = calculateVram(
      baseInput({
        model_variant: { ...moeModel, num_of_active_params: undefined },
        quantization: "q4",
      }),
    );

    expect(breakdownSize(activeResult, "Model Weights")).toBe(335.5);
    expect(activeResult.estimated_latency_tps).toBeGreaterThan(
      totalOnlyResult.estimated_latency_tps,
    );
  });

  it("sizes KV cache only for cache-bearing hybrid layers", () => {
    const model = getModelBySlug("qwen3-coder-next-80b-a3b");
    if (!model) throw new Error("Model qwen3-coder-next-80b-a3b not found");

    const variant: ModelVariant = {
      num_of_params: parseFloat(model.num_of_params),
      architecture: model.architecture,
      modality: model.modality,
      hidden_dim_size: model.hidden_dim_size,
      num_of_layers: model.num_of_layers,
      num_kv_cache_layers: model.num_kv_cache_layers || undefined,
      num_of_active_params: model.num_of_active_params || undefined,
      num_of_experts: model.num_of_experts || undefined,
      num_of_active_experts: model.num_of_active_experts || undefined,
      attention_structure: model.attention_structure,
      num_attention_heads: model.num_attention_heads || undefined,
      num_key_value_heads: model.num_key_value_heads || undefined,
      head_dim: model.head_dim || undefined,
    };
    const hybrid = calculateVram(
      baseInput({ model_variant: variant, seq_length: 65536 }),
    );
    const allLayers = calculateVram(
      baseInput({
        model_variant: { ...variant, num_kv_cache_layers: undefined },
        seq_length: 65536,
      }),
    );

    expect(breakdownSize(hybrid, "KV Cache")).toBe(
      breakdownSize(allLayers, "KV Cache") / 4,
    );
  });

  it("loads corrected architecture metadata", () => {
    const mistral = getModelBySlug("mistral-small-2501");
    const deepseek = getModelBySlug("deepseek-v3");
    const kimi = getModelBySlug("kimi-k25");
    const codestral = getModelBySlug("codestral-2501");
    const commandA = getModelBySlug("command-a");

    expect(mistral).toMatchObject({
      hidden_dim_size: 5120,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      head_dim: 128,
    });
    expect(deepseek).toMatchObject({
      num_of_experts: 256,
      num_of_active_experts: 8,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
    });
    expect(kimi?.num_of_active_params).toBe(32);
    expect(codestral).toMatchObject({
      num_of_params: "22.00",
      context_length: 256000,
      attention_structure: "gqa",
      position_embedding: "rope",
      head_dim: 128,
    });
    expect(commandA).toMatchObject({
      hidden_dim_size: 12288,
      ffn_intermediate_size: 36864,
      vocab_size: 256000,
      num_of_layers: 64,
      num_attention_heads: 96,
      num_key_value_heads: 8,
      head_dim: 128,
    });
  });
  it("loads recent open-weight model releases", () => {
    const expectedModels = [
      ["glm-52", "744.00", 40, 1048576],
      ["kimi-k27-code", "1000.00", 32, 262144],
      ["minimax-m27", "229.00", 10, 204800],
      ["minimax-m3", "428.00", 23, 1048576],
      ["mistral-medium-35-128b", "128.00", null, 262144],
      ["mistral-small-4-119b-a6b", "119.00", 6.5, 262144],
      ["nvidia-nemotron-3-super-120b-a12b", "120.00", 12, 1048576],
      ["qwen3-coder-next-80b-a3b", "80.00", 3, 262144],
      ["qwen36-27b", "27.00", null, 262144],
    ] as const;

    for (const [slug, total, active, context] of expectedModels) {
      expect(getModelBySlug(slug)).toMatchObject({
        num_of_params: total,
        num_of_active_params: active,
        context_length: context,
      });
    }
  });

  it("keeps active MoE parameters at or below total parameters", () => {
    for (const model of models) {
      if (model.num_of_active_params === null) continue;
      expect(model.num_of_active_params).toBeLessThanOrEqual(
        parseFloat(model.num_of_params),
      );
    }
  });

});

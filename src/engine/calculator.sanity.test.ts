import { describe, expect, it } from "vitest";
import { calculateVram, type CalculationInput } from "./index";
import { getModelBySlug } from "../data/models";

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
      num_of_expert_params: model.num_of_expert_params || 0,
      num_of_experts: model.num_of_experts || undefined,
      num_of_active_experts: model.num_of_active_experts || undefined,
      attention_structure: model.attention_structure,
      num_attention_heads: model.num_attention_heads || undefined,
      num_key_value_heads: model.num_key_value_heads || undefined,
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
});

export interface GpuInfo {
  key: string;
  label: string;
  vendor: string;
  type: string;
  memory: number;
  factor: number;
  tdp_watts: number;
  hourly_price_usd: number | null;
  category: string;
}

export interface ModelVariant {
  name?: string;
  num_of_params: number;
  architecture: string;
  modality?: string;
  hidden_dim_size: number;
  num_of_layers: number;
  num_of_expert_params?: number;
  num_of_experts?: number;
  num_of_active_experts?: number;
  attention_structure: string;
  position_embedding?: string;
  context_length?: number;
  ffn_intermediate_size?: number;
  vocab_size?: number;
}

export interface OptimizationConfig {
  preset?: string;
  flash_attention?: boolean;
  gradient_checkpointing?: boolean;
  eight_bit_optimizer?: boolean;
  paged_optimizer?: boolean;
  fused_kernels?: boolean;
  activation_offload?: boolean;
  zero_stage?: number;
}

export interface CalculationInput {
  model_variant: ModelVariant;
  quantization: string;
  kv_cache_quantization: string;
  gpu_key: string;
  custom_vram: number | null;
  num_gpus: number;
  batch_size: number;
  calc_mode: "inference" | "finetuning";
  finetuning_method: string | null;
  fine_tuning_quantization: string | null;
  seq_length: number;
  lora_rank: number | null;
  concurrent_users: number;
  gradient_accumulation_steps: number;
  enable_offloading: boolean;
  offload_target: string | null;
  num_offload_layers: number | null;
  percentage_offload: number | null;
  offload_kv_cache: boolean;
  optimization_config: OptimizationConfig | null;
  carbon_intensity: number | null;
  interconnect_type: string | null;
  num_samples: number | null;
  tokens_per_sample: number | null;
  num_epochs: number | null;
}

export interface MemoryBreakdownItem {
  label: string;
  value: number;
  size_gb: number;
}

export interface CalculationResult {
  vram_usage: number;
  vram_percentage: number;
  actual_vram_percentage: number;
  memory_status: string;
  memory_breakdown: MemoryBreakdownItem[];
  static_shared_memory: number;
  per_user_memory: number;
  offloaded_memory: number;
  estimated_latency_tps: number;
  estimated_throughput_tps: number;
  per_user_tps: number;
  ms_per_token: number;
  tftt: number;
  estimated_power_draw: number;
  estimated_system_ram_required: number;
  training_tps?: number;
  samples_per_second?: number;
  steps_per_second?: number;
  active_overhead_multiplier?: number;
  carbon_emissions_per_hour?: number;
  carbon_emissions_per_day?: number;
  carbon_emissions_per_month?: number;
  carbon_emissions_per_year?: number;
  total_tokens?: number;
  total_training_time_hours?: number;
  energy_cost_per_hour_usd?: number;
  energy_cost_total_usd?: number;
}

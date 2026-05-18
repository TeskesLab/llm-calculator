import rawModels from "../models-data.json";

export interface ModelInfo {
  id: number;
  slug: string;
  family: string;
  provider: string;
  name: string;
  z_score: string;
  z_score_coding: string;
  rank: number;
  rank_coding: number;
  num_of_params: string;
  modality: string;
  release_date: string;
  hidden_dim_size: number;
  ffn_intermediate_size: number | null;
  vocab_size: number | null;
  num_of_layers: number;
  num_of_experts: number | null;
  num_of_active_experts: number | null;
  num_attention_heads: number | null;
  num_key_value_heads: number | null;
  architecture: string;
  attention_structure: string;
  position_embedding: string;
  context_length: number;
  num_of_expert_params: number | null;
  additional_notes?: string | null;
}

export const models = (rawModels as unknown as ModelInfo[]);

export function getModelBySlug(slug: string): ModelInfo | undefined {
  return models.find((m) => m.slug === slug);
}

export function getModelByName(name: string): ModelInfo | undefined {
  return models.find((m) => m.name === name);
}

export function getModelOptions() {
  return models.map((m) => ({
    value: m.slug,
    label: m.name,
    group: m.provider,
  }));
}

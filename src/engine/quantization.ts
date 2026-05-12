// Bytes-per-parameter map for weight quantization
export const BYTES_PER_PARAM: Record<string, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  nvfp6: 0.75,
  nvfp4: 0.5,
  int8: 1,
  int4: 0.5,
  q8: 1,
  q7: 0.875,
  q6_k: 0.8125,
  q6: 0.75,
  q5_k_s: 0.6,
  q5_k_m: 0.6875,
  q5: 0.625,
  q4_k_s: 0.5375,
  q4_k_m: 0.575,
  q4: 0.5,
  q3_k_s: 0.425,
  q3_k_m: 0.4375,
  q3_k_l: 0.4875,
  q3: 0.375,
  q2_k: 0.32,
  q2: 0.25,
  q1: 0.125,
};

// Bytes-per-value for KV cache
export const KV_CACHE_BYTES: Record<string, number> = {
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
};

export function getBytesPerParam(quant: string): number {
  return BYTES_PER_PARAM[quant] ?? 2; // default FP16
}

export function getKvCacheBytes(quant: string): number {
  return KV_CACHE_BYTES[quant] ?? 2; // default FP16
}

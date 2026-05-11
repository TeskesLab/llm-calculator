/**
 * WASM loader for vram_calculator_wasm_bg.wasm
 * Wraps the wasm-bindgen compiled VRAM calculator module
 */

let wasmInstance: WebAssembly.Instance | null = null;
const textDecoder = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
const textEncoder = new TextEncoder();
let cachedStrLen = 0;

textDecoder.decode(); // Force init

function getMemory(): Uint8Array {
  if (!wasmInstance) throw new Error("WASM not initialized");
  const mem = wasmInstance.exports.memory as WebAssembly.Memory;
  return new Uint8Array(mem.buffer);
}

function wasmFree(ptr: number, len: number): void {
  if (!wasmInstance) return;
  try {
    (wasmInstance.exports.__wbindgen_free as Function)(ptr, len, 1);
  } catch { /* ignore */ }
}

function passStringToWasm(str: string): { ptr: number; len: number } {
  if (!wasmInstance) throw new Error("WASM not initialized");
  const { __wbindgen_malloc: malloc } = wasmInstance.exports as Record<string, Function>;
  const encoded = textEncoder.encode(str);
  const ptr = (malloc as (size: number, align: number) => number)(encoded.length, 1) >>> 0;
  getMemory().subarray(ptr, ptr + encoded.length).set(encoded);
  return { ptr, len: encoded.length };
}

function getStringFromWasm(ptr: number, len: number): string {
  return textDecoder.decode(getMemory().subarray(ptr, ptr + len));
}

export interface ModelVariant {
  num_of_params?: number;
  architecture?: string;
  modality?: string;
  hidden_dim_size?: number;
  num_of_layers?: number;
  num_of_expert_params?: number | null;
  attention_structure?: string;
  position_embedding?: string;
}

export interface CalculationInput {
  model_variant?: ModelVariant;
  quantization?: string;
  kv_cache_quantization?: string;
  gpu_key?: string;
  custom_vram?: number | null;
  num_gpus?: number;
  batch_size?: number;
  calc_mode?: "inference" | "finetuning";
  finetuning_method?: string | null;
  fine_tuning_quantization?: string | null;
  seq_length?: number;
  lora_rank?: number | null;
  concurrent_users?: number;
  gradient_accumulation_steps?: number;
  enable_offloading?: boolean;
  offload_target?: string | null;
  num_offload_layers?: number | null;
  percentage_offload?: number | null;
  offload_kv_cache?: boolean;
  optimization_config?: Record<string, unknown> | null;
  carbon_intensity?: number | null;
  interconnect_type?: string | null;
  num_samples?: number | null;
  tokens_per_sample?: number | null;
  num_epochs?: number | null;
}

export interface MemoryBreakdownItem {
  label: string;
  value: number;
  size_gb?: number;
}

export interface CalculationResult {
  vram_usage?: number;
  vramUsage?: number;
  vram_percentage?: number;
  vramPercentage?: number;
  actual_vram_percentage?: number;
  actualVramPercentage?: number;
  memory_status?: string;
  memoryStatus?: string;
  memory_details?: Record<string, number>;
  memory_breakdown?: MemoryBreakdownItem[];
  memoryBreakdown?: MemoryBreakdownItem[];
  static_shared_memory?: number;
  staticSharedMemory?: number;
  per_user_memory?: number;
  perUserMemory?: number;
  estimated_latency_tps?: number;
  estimatedLatencyTps?: number;
  estimated_throughput_tps?: number;
  estimatedThroughputTps?: number;
  per_user_tps?: number;
  perUserTps?: number;
  ms_per_token?: number;
  msPerToken?: number;
  tftt?: number;
  estimated_power_draw?: number;
  estimatedPowerDraw?: number;
  estimated_system_ram_required?: number;
  estimatedSystemRamRequired?: number;
  offloaded_memory?: number;
  offloadedMemory?: number;
  carbon_emissions_per_hour?: number;
  carbonEmissionsPerHour?: number;
  carbon_emissions_per_day?: number;
  carbonEmissionsPerDay?: number;
  carbon_emissions_per_month?: number;
  carbonEmissionsPerMonth?: number;
  carbon_emissions_per_year?: number;
  carbonEmissionsPerYear?: number;
  training_tps?: number;
  trainingTps?: number;
  samples_per_second?: number;
  samplesPerSecond?: number;
  steps_per_second?: number;
  stepsPerSecond?: number;
  total_tokens?: number;
  totalTokens?: number;
  total_training_time_hours?: number;
  totalTrainingTimeHours?: number;
}

export interface GpuInfo {
  key: string;
  label: string;
  vendor: string;
  type: string;
  memory: number;
  factor: number;
  tdp_watts: number;
  hourly_price_usd: number;
  category: string;
  bandwidth?: number;
}

const wasmImports = {
  "./vram_calculator_wasm_bg.js": {
    __wbindgen_cast_0000000000000001(idx: number, tableIdx: number) {
      const ptr = idx >>> 0;
      cachedStrLen += tableIdx;
      if (cachedStrLen >= 0x7ff00000) {
        textDecoder.decode();
        cachedStrLen = tableIdx;
      }
      return textDecoder.decode(getMemory().subarray(ptr, ptr + tableIdx));
    },

    __wbindgen_init_externref_table() {
      const table = wasmInstance!.exports.__wbindgen_externrefs as WebAssembly.Table;
      const base = table.grow(4);
      table.set(0, undefined);
      table.set(base + 0, undefined);
      table.set(base + 1, null);
      table.set(base + 2, true);
      table.set(base + 3, false);
    },
  },
};

let initPromise: Promise<void> | null = null;

export async function initWasm(wasmUrl?: string): Promise<void> {
  if (wasmInstance) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const url = wasmUrl || "/vram_calculator_wasm_bg.wasm";
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load WASM: ${response.status}`);
    const wasmBytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(wasmBytes, wasmImports);
    wasmInstance = instance;

    const startFn = wasmInstance.exports.__wbindgen_start as Function | undefined;
    if (startFn) startFn();

    cachedStrLen = 0;
  })();

  return initPromise;
}

export function isWasmReady(): boolean {
  return wasmInstance !== null;
}

export function getGpuList(): GpuInfo[] {
  if (!wasmInstance) throw new Error("WASM not initialized. Call initWasm() first.");

  const get_gpu_list = wasmInstance.exports.get_gpu_list as () => [number, number, number, number];
  const [ptr, len] = get_gpu_list();

  if (len === 0) throw new Error("get_gpu_list returned empty");

  const json = getStringFromWasm(ptr, len);
  wasmFree(ptr, len);

  return JSON.parse(json) as GpuInfo[];
}

export function calculateVram(input: CalculationInput): CalculationResult {
  if (!wasmInstance) throw new Error("WASM not initialized. Call initWasm() first.");

  const calculate_vram = wasmInstance.exports.calculate_vram as (
    ptr: number,
    len: number
  ) => [number, number, number, number];

  const json = JSON.stringify(input);
  const { ptr, len } = passStringToWasm(json);

  let resultPtr = 0;
  let resultLen = 0;

  try {
    const [rptr, rlen, extref, err] = calculate_vram(ptr, len);

    if (err) {
      // Error case: extref contains error info
      throw new Error(`WASM calculation error (extref=${extref})`);
    }

    resultPtr = rptr;
    resultLen = rlen;

    if (resultLen === 0) throw new Error("Calculation returned empty result");

    const resultJson = getStringFromWasm(resultPtr, resultLen);
    return JSON.parse(resultJson) as CalculationResult;
  } finally {
    wasmFree(ptr, len);
    if (resultPtr) wasmFree(resultPtr, resultLen);
  }
}

export default initWasm;

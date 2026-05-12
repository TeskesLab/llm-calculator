import rawGpus from "../data/gpu-data.json";
import type { GpuInfo } from "./types";

export const gpuList: GpuInfo[] = rawGpus as GpuInfo[];

export function getGpuByKey(key: string): GpuInfo | undefined {
  return gpuList.find((g) => g.key === key);
}

export function getGpuList(): GpuInfo[] {
  return gpuList;
}

export function getEffectiveVram(gpuKey: string, customVram: number | null): number {
  if (
    gpuKey === "custom_discrete" ||
    gpuKey === "custom_apple_silicon"
  ) {
    return customVram ?? 24;
  }
  const gpu = getGpuByKey(gpuKey);
  return gpu?.memory ?? 24;
}

/**
 * Estimate a GPU speed factor for custom GPUs based on VRAM amount.
 * Heuristic: higher VRAM GPUs tend to be faster.
 */
export function getGpuFactor(gpuKey: string): number {
  const gpu = getGpuByKey(gpuKey);
  if (gpu && gpu.factor != null) return gpu.factor;

  // Fallback: guess factor based on VRAM
  const vram = gpu?.memory ?? 24;
  if (vram <= 8) return 0.8;
  if (vram <= 12) return 1.0;
  if (vram <= 16) return 1.5;
  if (vram <= 24) return 2.0;
  if (vram <= 32) return 2.5;
  if (vram <= 48) return 3.0;
  if (vram <= 64) return 3.5;
  if (vram <= 80) return 4.0;
  if (vram <= 96) return 4.5;
  if (vram <= 128) return 5.0;
  return 6.0;
}

export function getGpuTdp(gpuKey: string): number {
  const gpu = getGpuByKey(gpuKey);
  return gpu?.tdp_watts ?? 250;
}

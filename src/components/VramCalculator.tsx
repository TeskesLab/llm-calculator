import { useState, useCallback, useEffect, useMemo } from "react";
import {
  Paper,
  Title,
  Text,
  Select,
  NumberInput,
  Slider,
  Switch,
  Group,
  Stack,
  Grid,
  Card,
  Progress,
  Badge,
  Tabs,
  Divider,
  Box,
  SimpleGrid,
} from "@mantine/core";
import { useDebouncedValue } from "@mantine/hooks";
import {
  IconCpu,
  IconBolt,
  IconClock,
  IconUsers,
  IconChartPie,
} from "@tabler/icons-react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { calculateVram, getGpuList, type GpuInfo, type CalculationInput, type CalculationResult } from "../engine";
import { models, getModelBySlug } from "../data/models";

const QUANTIZATION_OPTIONS = [
  { value: "fp32", label: "FP32" },
  { value: "fp16", label: "FP16" },
  { value: "fp8", label: "FP8" },
  { value: "nvfp6", label: "NVFP6" },
  { value: "nvfp4", label: "NVFP4" },
  { value: "int8", label: "INT8" },
  { value: "int4", label: "INT4" },
  { value: "q8", label: "Q8" },
  { value: "q7", label: "Q7" },
  { value: "q6_k", label: "Q6_K" },
  { value: "q6", label: "Q6" },
  { value: "q5_k_s", label: "Q5_K_S" },
  { value: "q5_k_m", label: "Q5_K_M" },
  { value: "q5", label: "Q5" },
  { value: "q4_k_s", label: "Q4_K_S" },
  { value: "q4_k_m", label: "Q4_K_M" },
  { value: "q4", label: "Q4" },
  { value: "q3_k_s", label: "Q3_K_S" },
  { value: "q3_k_m", label: "Q3_K_M" },
  { value: "q3_k_l", label: "Q3_K_L" },
  { value: "q3", label: "Q3" },
  { value: "q2_k", label: "Q2_K" },
  { value: "q2", label: "Q2" },
  { value: "q1", label: "Q1" },
];

const KV_CACHE_OPTIONS = [
  { value: "fp16", label: "FP16 (default)" },
  { value: "fp8", label: "FP8" },
  { value: "int8", label: "INT8" },
  { value: "int4", label: "INT4 (experimental)" },
];

const FINE_TUNING_METHODS = [
  { value: "full", label: "Full Fine-Tuning" },
  { value: "lora", label: "LoRA" },
  { value: "qlora", label: "QLoRA" },
];

const FINE_TUNING_QUANT_OPTIONS = [
  { value: "fp32", label: "FP32" },
  { value: "fp16", label: "FP16 / BF16" },
  { value: "fp8", label: "FP8" },
];

const LORA_BASE_PRECISION = [
  { value: "fp16", label: "FP16 / BF16 (Base)" },
];

const QLORA_BASE_PRECISION = [
  { value: "q4", label: "4-bit (Base)" },
];

const INTERCONNECT_OPTIONS = [
  { group: "High-Speed", items: [
    { value: "nvlink_gen4", label: "NVLink Gen 4" },
    { value: "nvlink_gen3", label: "NVLink Gen 3" },
    { value: "nvlink_gen2", label: "NVLink Gen 2" },
    { value: "nvlink_bridge", label: "NVLink Bridge" },
    { value: "infiniband_hdr", label: "InfiniBand HDR" },
    { value: "infiniband_edr", label: "InfiniBand EDR" },
  ]},
  { group: "PCIe", items: [
    { value: "pcie5", label: "PCIe 5.0" },
    { value: "pcie4", label: "PCIe 4.0" },
    { value: "pcie3", label: "PCIe 3.0" },
  ]},
  { group: "Network", items: [
    { value: "ethernet400g", label: "Ethernet 400G" },
    { value: "ethernet200g", label: "Ethernet 200G" },
    { value: "ethernet100g", label: "Ethernet 100G" },
  ]},
];

const OPTIMIZATION_PRESETS = [
  { value: "unsloth", label: "Unsloth" },
  { value: "deepspeed_zero2", label: "DeepSpeed ZeRO-2" },
  { value: "deepspeed_zero3", label: "DeepSpeed ZeRO-3" },
  { value: "peft_minimal", label: "PEFT Minimal" },
  { value: "default", label: "Default" },
  { value: "custom", label: "Custom" },
];

const PIE_COLORS = [
  "#50E3C2", "#4A90D9", "#F5A623", "#7ED321",
  "#BD10E0", "#D0021B", "#F8E71C", "#8B572A",
  "#417505", "#9013FE",
];

const CARBON_REGIONS = [
  { value: "475", label: "Global Average (475 g/kWh)" },
  { value: "380", label: "US (380 g/kWh)" },
  { value: "250", label: "EU (250 g/kWh)" },
  { value: "560", label: "China (560 g/kWh)" },
  { value: "55", label: "France (55 g/kWh)" },
  { value: "680", label: "India (680 g/kWh)" },
  { value: "460", label: "Japan (460 g/kWh)" },
  { value: "130", label: "Brazil (130 g/kWh)" },
];

function formatBytes(gb: number): string {
  if (gb >= 1000) return `${(gb / 1000).toFixed(1)} TB`;
  return `${gb.toFixed(1)} GB`;
}

function formatSpeed(tps: number): string {
  if (tps === 0 || !isFinite(tps)) return "N/A";
  if (tps >= 1000) return `${(tps / 1000).toFixed(1)}k tok/s`;
  return `${tps.toFixed(1)} tok/s`;
}

const defaultModel = models.find(
  (m) => m.name === "DeepSeek-R1 3B"
) || models[0];

export function VramCalculator() {
  // GPU list loaded synchronously
  const [gpuList] = useState<GpuInfo[]>(() => getGpuList());

  // Calculation mode
  const [calcMode, setCalcMode] = useState<"inference" | "finetuning">("inference");

  // Model selection
  const [modelSlug, setModelSlug] = useState<string>(defaultModel?.slug || "");
  const selectedModel = useMemo(() => getModelBySlug(modelSlug), [modelSlug]);

  // Quantization
  const [quantization, setQuantization] = useState<string>("fp16");
  const [kvCacheQuant, setKvCacheQuant] = useState<string>("fp16");

  // Hardware
  const [selectedGpuKey, setSelectedGpuKey] = useState<string>("4090_24");
  const selectedGpu = useMemo(
    () => gpuList.find((g) => g.key === selectedGpuKey),
    [gpuList, selectedGpuKey]
  );
  const [customVram, setCustomVram] = useState<number>(24);
  const isCustomGpu = selectedGpuKey === "custom_discrete" || selectedGpuKey === "custom_apple_silicon";
  const [numGpus, setNumGpus] = useState<number>(1);

  // Input parameters
  const [batchSize, setBatchSize] = useState<number>(1);
  const [seqLength, setSeqLength] = useState<number>(1024);
  const [concurrentUsers, setConcurrentUsers] = useState<number>(1);

  // Offloading
  const [enableOffloading, setEnableOffloading] = useState(false);
  const [offloadTarget, setOffloadTarget] = useState<string | null>(null);
  const [numOffloadLayers, setNumOffloadLayers] = useState<number | null>(null);
  const [offloadKvCache, setOffloadKvCache] = useState(false);

  // Fine-tuning specific
  const [finetuningMethod, setFinetuningMethod] = useState<string>("full");
  const [finetuningQuant, setFinetuningQuant] = useState<string>("fp16");
  const [loraRank, setLoraRank] = useState<number>(16);
  const [gradAccumSteps, setGradAccumSteps] = useState<number>(1);
  const [optimizationPreset, setOptimizationPreset] = useState<string>("default");
  const [numSamples, setNumSamples] = useState<number | null>(1000);
  const [tokensPerSample, setTokensPerSample] = useState<number | null>(1024);
  const [numEpochs, setNumEpochs] = useState<number | null>(3);
  const [energyCostPerKwh, setEnergyCostPerKwh] = useState<number | null>(null);
  const [carbonIntensity, setCarbonIntensity] = useState<number | null>(null);

  // Interconnect
  const [interconnectType, setInterconnectType] = useState<string | null>(null);

  // Results
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [calculating, setCalculating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-calculate on input changes
  const [debouncedSeqLength] = useDebouncedValue(seqLength, 300);

  const doCalculate = useCallback(() => {
    if (!selectedModel) return;

    setCalculating(true);
    setError(null);

    try {
      const input: CalculationInput = {
        model_variant: {
          num_of_params: parseFloat(selectedModel.num_of_params),
          architecture: selectedModel.architecture,
          modality: selectedModel.modality || "text",
          hidden_dim_size: selectedModel.hidden_dim_size,
          num_of_layers: selectedModel.num_of_layers,
          num_of_expert_params: selectedModel.num_of_expert_params || 0,
          attention_structure: selectedModel.attention_structure,
          position_embedding: selectedModel.position_embedding,
        },
        quantization,
        kv_cache_quantization: calcMode === "inference" ? kvCacheQuant : "fp16",
        gpu_key: selectedGpuKey,
        custom_vram: isCustomGpu ? customVram : null,
        num_gpus: numGpus,
        batch_size: batchSize,
        calc_mode: calcMode,
        finetuning_method: calcMode === "finetuning" ? finetuningMethod : null,
        fine_tuning_quantization:
          calcMode === "finetuning" ? finetuningQuant : null,
        seq_length: debouncedSeqLength,
        lora_rank:
          calcMode === "finetuning" &&
          (finetuningMethod === "lora" || finetuningMethod === "qlora")
            ? loraRank
            : null,
        concurrent_users: calcMode === "inference" ? concurrentUsers : 1,
        gradient_accumulation_steps:
          calcMode === "finetuning" ? gradAccumSteps : 1,
        enable_offloading: enableOffloading,
        offload_target: enableOffloading ? offloadTarget : null,
        num_offload_layers:
          enableOffloading && numOffloadLayers ? numOffloadLayers : null,
        percentage_offload: null,
        offload_kv_cache: offloadKvCache,
        optimization_config:
          calcMode === "finetuning"
            ? { preset: optimizationPreset }
            : null,
        carbon_intensity: carbonIntensity,
        interconnect_type: numGpus > 1 ? interconnectType : null,
        num_samples: calcMode === "finetuning" ? numSamples : null,
        tokens_per_sample: calcMode === "finetuning" ? tokensPerSample : null,
        num_epochs: calcMode === "finetuning" ? numEpochs : null,
        energy_cost_per_kwh: energyCostPerKwh,
      };

      const res = calculateVram(input);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Calculation failed");
    } finally {
      setCalculating(false);
    }
  }, [
    selectedModel, quantization, kvCacheQuant, selectedGpuKey, isCustomGpu,
    customVram, numGpus, batchSize, debouncedSeqLength, concurrentUsers,
    enableOffloading, offloadTarget, numOffloadLayers, offloadKvCache,
    calcMode, finetuningMethod, finetuningQuant, loraRank, gradAccumSteps,
    optimizationPreset, numSamples, tokensPerSample, numEpochs,
    carbonIntensity, interconnectType, energyCostPerKwh,
  ]);

  useEffect(() => {
    if (selectedModel) doCalculate();
  }, [doCalculate]);

  // Build GPU select options grouped by vendor
  const gpuSelectData = useMemo(() => {
    const nvidiaGpus = gpuList.filter(
      (g) => g.vendor === "nvidia" && g.type !== "apu"
    );
    const nvidiaApus = gpuList.filter(
      (g) => g.vendor === "nvidia" && g.type === "apu"
    );
    const amdGpus = gpuList.filter(
      (g) => g.vendor === "amd" && g.type !== "apu"
    );
    const amdApus = gpuList.filter(
      (g) => g.vendor === "amd" && g.type === "apu"
    );
    const appleGpus = gpuList.filter((g) => g.vendor === "apple");
    const intelGpus = gpuList.filter((g) => g.vendor === "intel");

    const groups: { group: string; items: { value: string; label: string }[] }[] = [];

    if (nvidiaGpus.length > 0)
      groups.push({
        group: "NVIDIA GPUs",
        items: nvidiaGpus.map((g) => ({ value: g.key, label: g.label })),
      });
    if (nvidiaApus.length > 0)
      groups.push({
        group: "NVIDIA Superchips",
        items: nvidiaApus.map((g) => ({ value: g.key, label: g.label })),
      });
    if (amdGpus.length > 0)
      groups.push({
        group: "AMD GPUs",
        items: amdGpus.map((g) => ({ value: g.key, label: g.label })),
      });
    if (amdApus.length > 0)
      groups.push({
        group: "AMD Ryzen APUs",
        items: amdApus.map((g) => ({ value: g.key, label: g.label })),
      });
    if (appleGpus.length > 0)
      groups.push({
        group: "Apple Silicon",
        items: appleGpus.map((g) => ({ value: g.key, label: g.label })),
      });
    if (intelGpus.length > 0)
      groups.push({
        group: "Intel GPUs",
        items: intelGpus.map((g) => ({ value: g.key, label: g.label })),
      });

    // Deduplicate: remove items that appear in earlier groups
    const seen = new Set<string>();
    const deduped = groups
      .map((g) => ({
        group: g.group,
        items: g.items.filter((item) => {
          if (seen.has(item.value)) return false;
          seen.add(item.value);
          return true;
        }),
      }))
      .filter((g) => g.items.length > 0);

    return deduped;
  }, [gpuList]);

  // Model select data - group by provider
  const modelSelectData = useMemo(() => {
    const byProvider: Record<string, { value: string; label: string }[]> = {};
    for (const m of models) {
      const provider = m.provider || "Other";
      if (!byProvider[provider]) byProvider[provider] = [];
      byProvider[provider].push({
        value: m.slug,
        label: `${m.name} (${m.num_of_params}B)`,
      });
    }
    return Object.entries(byProvider).map(([group, items]) => ({
      group,
      items,
    }));
  }, []);

  // Get effective GPU VRAM
  const effectiveVram = useMemo(() => {
    if (!selectedGpu) return 24;
    return selectedGpu.memory;
  }, [selectedGpu]);

  const vramUsage = result?.vram_usage ?? 0;
  const vramPct = result?.vram_percentage ?? 0;
  const memoryStatus = result?.memory_status ?? "Ready";
  const tps = result?.estimated_latency_tps ?? 0;
  const throughputTps = result?.estimated_throughput_tps ?? 0;
  const perUserTps = result?.per_user_tps ?? 0;
  const tftt = result?.tftt ?? 0;
  const powerDraw = result?.estimated_power_draw ?? 0;
  const sysRam = result?.estimated_system_ram_required ?? 0;
  const offloadedMem = result?.offloaded_memory ?? 0;
  const breakdown = result?.memory_breakdown ?? [];
  const trainingTps = result?.training_tps ?? 0;
  const totalTrainingTime = result?.total_training_time_hours ?? 0;
  const energyCostPerHour = result?.energy_cost_per_hour_usd ?? 0;

  const getStatusColor = (status: string) => {
    const s = status.toLowerCase();
    if (s.includes("error") || s.includes("exceed") || s.includes("insufficient")) return "red";
    if (s.includes("okay") || s.includes("sufficient") || s.includes("low")) return "green";
    if (s.includes("moderate")) return "yellow";
    if (s.includes("high") || s.includes("warning")) return "orange";
    return "violet";
  };

  const formatStatus = (status: string) => {
    return status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();
  };

  return (
    <Stack gap="lg">
      {/* Main Calculator Card */}
      <Paper withBorder p="lg" radius="md">
        <Group justify="space-between" mb="md">
          <Title order={3}>
            LLM Inference: VRAM & Performance Calculator
          </Title>
        </Group>

        {/* Mode Tabs */}
        <Tabs
          value={calcMode}
          onChange={(v) => setCalcMode(v as "inference" | "finetuning")}
          mb="lg"
        >
          <Tabs.List>
            <Tabs.Tab value="inference">Inference</Tabs.Tab>
            <Tabs.Tab value="finetuning">Fine-tuning</Tabs.Tab>
          </Tabs.List>
        </Tabs>

        <Grid>
          {/* Left Column - Inputs */}
          <Grid.Col span={{ md: 6 }}>
            <Stack gap="md">
              {/* Model Selection */}
              <Box>
                <Text size="sm" fw={500} mb={4}>
                  Select Model <span style={{ color: "red" }}>*</span>
                </Text>
                <Select
                  data={modelSelectData}
                  value={modelSlug}
                  onChange={(v) => v && setModelSlug(v)}
                  searchable
                  placeholder="Search models..."
                  nothingFoundMessage="No models found"
                />
                {selectedModel && (
                  <Text size="xs" c="dimmed" mt={4}>
                    {selectedModel.num_of_params}B params |{" "}
                    {selectedModel.hidden_dim_size} hidden |{" "}
                    {selectedModel.num_of_layers} layers |{" "}
                    {selectedModel.attention_structure?.toUpperCase() || "MHA"}
                    {selectedModel.architecture === "moe" &&
                      ` | MoE (${selectedModel.num_of_experts} experts)`}
                  </Text>
                )}
              </Box>

              {/* Quantization */}
              <Box>
                <Text size="sm" fw={500} mb={4}>
                  Inference Quantization
                </Text>
                <Text size="xs" c="dimmed" mb={4}>
                  Precision for model weights during inference. Lower uses less
                  VRAM but may affect quality.
                </Text>
                <Select
                  data={QUANTIZATION_OPTIONS}
                  value={quantization}
                  onChange={(v) => v && setQuantization(v)}
                />
              </Box>

              {calcMode === "inference" && (
                <Box>
                  <Text size="sm" fw={500} mb={4}>
                    KV Cache Quantization
                  </Text>
                  <Text size="xs" c="dimmed" mb={4}>
                    KV Cache precision. Lower values reduce VRAM, especially
                    for long sequences.
                  </Text>
                  <Select
                    data={KV_CACHE_OPTIONS}
                    value={kvCacheQuant}
                    onChange={(v) => v && setKvCacheQuant(v)}
                  />
                </Box>
              )}

              {/* Fine-tuning specific */}
              {calcMode === "finetuning" && (
                <>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Fine-tuning Method
                    </Text>
                    <Select
                      data={FINE_TUNING_METHODS}
                      value={finetuningMethod}
                      onChange={(v) => v && setFinetuningMethod(v)}
                    />
                  </Box>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Base Model Precision
                    </Text>
                    <Select
                      data={
                        finetuningMethod === "qlora"
                          ? QLORA_BASE_PRECISION
                          : finetuningMethod === "lora"
                          ? LORA_BASE_PRECISION
                          : FINE_TUNING_QUANT_OPTIONS
                      }
                      value={finetuningQuant}
                      onChange={(v) => v && setFinetuningQuant(v)}
                    />
                  </Box>
                  {(finetuningMethod === "lora" || finetuningMethod === "qlora") && (
                    <Box>
                      <Text size="sm" fw={500} mb={4}>
                        LoRA Rank
                      </Text>
                      <NumberInput
                        value={loraRank}
                        onChange={(v) => setLoraRank(Number(v) || 16)}
                        min={1}
                        max={256}
                      />
                    </Box>
                  )}
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Optimization Preset
                    </Text>
                    <Select
                      data={OPTIMIZATION_PRESETS}
                      value={optimizationPreset}
                      onChange={(v) => v && setOptimizationPreset(v)}
                    />
                  </Box>
                </>
              )}

              {/* Hardware Configuration */}
              <Box>
                <Text size="sm" fw={500} mb={4}>
                  Hardware Configuration
                </Text>
                <Text size="xs" c="dimmed" mb={4}>
                  Select your GPU or set custom VRAM
                </Text>
                <Select
                  data={gpuSelectData}
                  value={selectedGpuKey}
                  onChange={(v) => v && setSelectedGpuKey(v)}
                  searchable
                  placeholder="Select GPU..."
                />
              </Box>

              {isCustomGpu && (
                <Box>
                  <Text size="sm" fw={500} mb={4}>
                    Custom VRAM (GB)
                  </Text>
                  <NumberInput
                    value={customVram}
                    onChange={(v) => setCustomVram(Number(v) || 24)}
                    min={1}
                    max={selectedGpuKey === "custom_apple_silicon" ? 512 : 142 * 1024}
                    suffix=" GB"
                  />
                </Box>
              )}

              <Box>
                <Text size="sm" fw={500} mb={4}>
                  Num GPUs
                </Text>
                <Text size="xs" c="dimmed" mb={4}>
                  Devices for parallel inference
                </Text>
                <NumberInput
                  value={numGpus}
                  onChange={(v) => setNumGpus(Number(v) || 1)}
                  min={1}
                  max={128}
                />
              </Box>

              {numGpus > 1 && (
                <Box>
                  <Text size="sm" fw={500} mb={4}>
                    Interconnect Type
                  </Text>
                  <Select
                    data={INTERCONNECT_OPTIONS}
                    value={interconnectType}
                    onChange={(v) => setInterconnectType(v)}
                    placeholder="Auto-detect"
                    clearable
                  />
                </Box>
              )}

              {/* Input Parameters */}
              <Divider label="Input Parameters" labelPosition="center" />

              <Box>
                <Group justify="space-between" mb={4}>
                  <Text size="sm" fw={500}>
                    Batch Size
                  </Text>
                  <Badge variant="light">{batchSize}</Badge>
                </Group>
                <Group gap="xs" mb={4}>
                  {[1, 8, 16, 32].map((v) => (
                    <Badge
                      key={v}
                      variant={batchSize === v ? "filled" : "outline"}
                      style={{ cursor: "pointer" }}
                      onClick={() => setBatchSize(v)}
                    >
                      {v}
                    </Badge>
                  ))}
                </Group>
                <Slider
                  value={batchSize}
                  onChange={setBatchSize}
                  min={1}
                  max={256}
                  label={(v) => v.toString()}
                />
              </Box>

              <Box>
                <Group justify="space-between" mb={4}>
                  <Text size="sm" fw={500}>
                    Sequence Length
                  </Text>
                  <Badge variant="light">
                    {seqLength >= 1024
                      ? `${(seqLength / 1024).toFixed(0)}K`
                      : seqLength}
                  </Badge>
                </Group>
                <Group gap="xs" mb={4}>
                  {[1024, 8192, 16384, 32768, 65536, 131072].map((v) => (
                    <Badge
                      key={v}
                      variant={seqLength === v ? "filled" : "outline"}
                      style={{ cursor: "pointer" }}
                      onClick={() => setSeqLength(v)}
                    >
                      {v >= 1024 ? `${(v / 1024).toFixed(0)}K` : v}
                    </Badge>
                  ))}
                </Group>
                <Slider
                  value={seqLength}
                  onChange={setSeqLength}
                  min={128}
                  max={131072}
                  step={128}
                  label={(v) =>
                    v >= 1024 ? `${(v / 1024).toFixed(0)}K` : v.toString()
                  }
                />
              </Box>

              {calcMode === "inference" && (
                <Box>
                  <Group justify="space-between" mb={4}>
                    <Text size="sm" fw={500}>
                      Concurrent Users
                    </Text>
                    <Badge variant="light">{concurrentUsers}</Badge>
                  </Group>
                  <Group gap="xs" mb={4}>
                    {[1, 4, 8, 16, 32].map((v) => (
                      <Badge
                        key={v}
                        variant={concurrentUsers === v ? "filled" : "outline"}
                        style={{ cursor: "pointer" }}
                        onClick={() => setConcurrentUsers(v)}
                      >
                        {v}
                      </Badge>
                    ))}
                  </Group>
                  <Slider
                    value={concurrentUsers}
                    onChange={setConcurrentUsers}
                    min={1}
                    max={128}
                    label={(v) => v.toString()}
                  />
                </Box>
              )}

              {/* Training-specific parameters */}
              {calcMode === "finetuning" && (
                <>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Gradient Accumulation Steps
                    </Text>
                    <NumberInput
                      value={gradAccumSteps}
                      onChange={(v) => setGradAccumSteps(Number(v) || 1)}
                      min={1}
                      max={128}
                    />
                  </Box>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Number of Samples
                    </Text>
                    <NumberInput
                      value={numSamples || ""}
                      onChange={(v) =>
                        setNumSamples(v === "" ? null : Number(v))
                      }
                      min={1}
                    />
                  </Box>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Tokens per Sample
                    </Text>
                    <NumberInput
                      value={tokensPerSample || ""}
                      onChange={(v) =>
                        setTokensPerSample(v === "" ? null : Number(v))
                      }
                      min={1}
                    />
                  </Box>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Number of Epochs
                    </Text>
                    <NumberInput
                      value={numEpochs || ""}
                      onChange={(v) =>
                        setNumEpochs(v === "" ? null : Number(v))
                      }
                      min={1}
                      max={1000}
                    />
                  </Box>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Carbon Intensity
                    </Text>
                    <Select
                      data={CARBON_REGIONS}
                      value={carbonIntensity?.toString() || null}
                      onChange={(v) =>
                        setCarbonIntensity(v ? parseFloat(v) : null)
                      }
                      placeholder="None"
                      clearable
                    />
                  </Box>
                </>
              )}

              {/* Offloading */}
              <Box>
                <Switch
                  label="Enable Offloading to CPU/RAM or NVMe"
                  checked={enableOffloading}
                  onChange={(e) => setEnableOffloading(e.currentTarget.checked)}
                />
              </Box>

              {enableOffloading && (
                <>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Offload Target
                    </Text>
                    <Select
                      data={[
                        { value: "cpu_ram", label: "CPU RAM" },
                        { value: "nvme", label: "NVMe" },
                      ]}
                      value={offloadTarget}
                      onChange={setOffloadTarget}
                    />
                  </Box>
                  <Box>
                    <Text size="sm" fw={500} mb={4}>
                      Number of Layers to Offload (optional)
                    </Text>
                    <NumberInput
                      value={numOffloadLayers || ""}
                      onChange={(v) =>
                        setNumOffloadLayers(v === "" ? null : Number(v))
                      }
                      min={0}
                      placeholder="All layers"
                    />
                  </Box>
                  <Switch
                    label="Offload KV Cache"
                    checked={offloadKvCache}
                    onChange={(e) =>
                      setOffloadKvCache(e.currentTarget.checked)
                    }
                  />
                </>
              )}

              {/* Energy Cost */}
              <Box>
                <Text size="sm" fw={500} mb={4}>
                  Electricity Price ($/kWh)
                </Text>
                <Text size="xs" c="dimmed" mb={4}>
                  Optional. Used to estimate hourly operating cost from GPU power draw.
                </Text>
                <NumberInput
                  value={energyCostPerKwh ?? ""}
                  onChange={(v) =>
                    setEnergyCostPerKwh(v === "" ? null : Number(v))
                  }
                  placeholder="e.g. 0.16"
                  min={0}
                  max={10}
                  step={0.01}
                  decimalScale={4}
                  prefix="$"
                />
              </Box>
            </Stack>
          </Grid.Col>

          {/* Right Column - Results */}
          <Grid.Col span={{ md: 6 }}>
            <Stack gap="md">
              {/* Model Details */}
              {selectedModel && (
                <Card withBorder>
                  <Group gap="xs" mb="xs">
                    <Badge variant="light" size="sm">
                      {selectedModel.provider}
                    </Badge>
                  </Group>
                  <Text fw={700} size="sm" mb="md">
                    {selectedModel.name}
                  </Text>
                  <SimpleGrid cols={2} spacing="xs">
                    <Box>
                      <Text size="xs" c="dimmed">
                        Weights
                      </Text>
                      <Text size="sm" fw={600}>
                        {quantization.toUpperCase()}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="xs" c="dimmed">
                        KV Cache
                      </Text>
                      <Text size="sm" fw={600}>
                        {calcMode === "inference"
                          ? kvCacheQuant.toUpperCase()
                          : "N/A"}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="xs" c="dimmed">
                        Attention
                      </Text>
                      <Text size="sm" fw={600}>
                        {selectedModel.attention_structure?.toUpperCase() || "MHA"}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="xs" c="dimmed">
                        Pos. Embedding
                      </Text>
                      <Text size="sm" fw={600}>
                        {selectedModel.position_embedding || "N/A"}
                      </Text>
                    </Box>
                  </SimpleGrid>
                </Card>
              )}

              {/* VRAM Usage */}
              <Card withBorder>
                <Group justify="space-between" mb="xs">
                  <Text size="sm" fw={500}>
                    VRAM Usage
                  </Text>
                  <Badge color={getStatusColor(memoryStatus)}>
                    {formatStatus(memoryStatus)}
                  </Badge>
                </Group>
                <Progress
                  value={Math.min(vramPct, 100)}
                  color={
                    vramPct > 90 ? "red" : vramPct > 70 ? "orange" : "violet"
                  }
                  size="xl"
                  mb="xs"
                  animated={calculating}
                />
                <Group justify="space-between">
                  <Text fw={700} size="xl">
                    {formatBytes(vramUsage)}
                  </Text>
                  <Text c="dimmed" size="sm">
                    of {formatBytes(effectiveVram * numGpus)} VRAM
                  </Text>
                </Group>
              </Card>

              {/* Speed Metrics */}
              <Grid>
                <Grid.Col span={6}>
                  <Card withBorder h="100%">
                    <Group gap="xs" mb={4}>
                      <IconBolt size={16} />
                      <Text size="xs" c="dimmed">
                        Generation Speed
                      </Text>
                    </Group>
                    <Text fw={700} size="lg">
                      {formatSpeed(tps)}
                    </Text>
                  </Card>
                </Grid.Col>
                <Grid.Col span={6}>
                  <Card withBorder h="100%">
                    <Group gap="xs" mb={4}>
                      <IconClock size={16} />
                      <Text size="xs" c="dimmed">
                        Time to First Token
                      </Text>
                    </Group>
                    <Text fw={700} size="lg">
                      {tftt ? `~${Math.round(tftt)}ms` : "N/A"}
                    </Text>
                  </Card>
                </Grid.Col>
              </Grid>

              {calcMode === "inference" && (
                <Grid>
                  <Grid.Col span={6}>
                    <Card withBorder h="100%">
                      <Group gap="xs" mb={4}>
                        <IconUsers size={16} />
                        <Text size="xs" c="dimmed">
                          Total Throughput
                        </Text>
                      </Group>
                      <Text fw={700} size="lg">
                        {formatSpeed(throughputTps)}
                      </Text>
                    </Card>
                  </Grid.Col>
                  <Grid.Col span={6}>
                    <Card withBorder h="100%">
                      <Group gap="xs" mb={4}>
                        <IconCpu size={16} />
                        <Text size="xs" c="dimmed">
                          Per User TPS
                        </Text>
                      </Group>
                      <Text fw={700} size="lg">
                        {formatSpeed(perUserTps)}
                      </Text>
                    </Card>
                  </Grid.Col>
                </Grid>
              )}

              {calcMode === "finetuning" && trainingTps > 0 && (
                <Grid>
                  <Grid.Col span={6}>
                    <Card withBorder h="100%">
                      <Text size="xs" c="dimmed" mb={4}>
                        Training TPS
                      </Text>
                      <Text fw={700} size="lg">
                        {formatSpeed(trainingTps)}
                      </Text>
                    </Card>
                  </Grid.Col>
                  <Grid.Col span={6}>
                    <Card withBorder h="100%">
                      <Text size="xs" c="dimmed" mb={4}>
                        Training Time
                      </Text>
                      <Text fw={700} size="lg">
                        {totalTrainingTime > 0
                          ? `${totalTrainingTime.toFixed(1)}h`
                          : "N/A"}
                      </Text>
                    </Card>
                  </Grid.Col>
                </Grid>
              )}

              {powerDraw > 0 && (
                <Card withBorder>
                  <Text size="xs" c="dimmed" mb={4}>
                    Estimated Power Draw
                  </Text>
                  <Text fw={700}>{Math.round(powerDraw)}W</Text>
                </Card>
              )}

              {sysRam > 0 && (
                <Card withBorder>
                  <Text size="xs" c="dimmed" mb={4}>
                    System RAM Required
                  </Text>
                  <Text fw={700}>{formatBytes(sysRam)}</Text>
                </Card>
              )}

               {offloadedMem > 0 && (
                 <Card withBorder>
                   <Text size="xs" c="dimmed" mb={4}>
                     Offloaded Memory
                   </Text>
                   <Text fw={700}>{formatBytes(offloadedMem)}</Text>
                 </Card>
               )}

               {energyCostPerHour > 0 && (
                 <Card withBorder>
                   <Text size="xs" c="dimmed" mb={4}>
                     Energy Cost
                   </Text>
                   <Text fw={700}>
                     ${energyCostPerHour.toFixed(3)}/h
                   </Text>
                 </Card>
               )}

              {/* Memory Breakdown */}
              {breakdown.length > 0 && (
                <Card withBorder>
                  <Group gap="xs" mb="sm">
                    <IconChartPie size={18} />
                    <Title order={5}>
                      Memory Breakdown
                    </Title>
                  </Group>
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie
                        data={breakdown.filter(
                          (item) => item.size_gb > 0
                        )}
                        cx="50%"
                        cy="50%"
                        innerRadius={55}
                        outerRadius={100}
                        paddingAngle={2}
                        dataKey="size_gb"
                        nameKey="label"
                        labelLine={{
                          stroke: "#7CB0C1",
                          strokeWidth: 1,
                        }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        label={(props: any) => {
                          const pct = vramUsage > 0
                            ? ((props.value / vramUsage) * 100).toFixed(0)
                            : "0";
                          return `${pct}%`;
                        }}
                        style={{ fontSize: 10, fontFamily: "'Share Tech Mono', monospace" }}
                      >
                        {breakdown
                          .filter((item) => item.size_gb > 0)
                          .map((_, i) => (
                            <Cell
                              key={i}
                              fill={PIE_COLORS[i % PIE_COLORS.length]}
                              strokeWidth={0}
                            />
                          ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          background: "#20323B",
                          border: "1px solid #7CB0C1",
                          borderRadius: 4,
                          color: "#F5F7FA",
                          fontFamily: "'Share Tech Mono', monospace",
                        }}
                        formatter={(_value: unknown) =>
                          `${formatBytes(_value as number)}`
                        }
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <Stack gap="xs" mt="sm">
                    {breakdown
                      .filter((item) => item.size_gb > 0)
                      .map((item, i) => (
                        <Group key={i} justify="space-between">
                          <Group gap="xs">
                            <Box
                              w={10}
                              h={10}
                              style={{
                                borderRadius: 2,
                                backgroundColor:
                                  PIE_COLORS[i % PIE_COLORS.length],
                              }}
                            />
                            <Text size="xs">{item.label}</Text>
                          </Group>
                          <Text size="xs" fw={500}>
                            {formatBytes(item.size_gb)}
                          </Text>
                        </Group>
                      ))}
                  </Stack>
                </Card>
              )}

              {/* Config Summary */}
              <Card withBorder>
                <Text size="xs" c="dimmed">
                  Mode: {calcMode === "inference" ? "Inference" : "Fine-tuning"}
                  {" | "}
                  {quantization.toUpperCase()} Weights{" | "}
                  {calcMode === "inference"
                    ? `${kvCacheQuant.toUpperCase()} KV Cache`
                    : `${finetuningMethod.toUpperCase()}`}
                  {" | "}
                  {selectedGpu?.label || "Custom GPU"}
                  {selectedModel &&
                    ` | Input: ${debouncedSeqLength >= 1024 ? `${(debouncedSeqLength / 1024).toFixed(0)}K` : debouncedSeqLength} tokens`}
                </Text>
              </Card>

              {error && (
                <Card withBorder bg="red.0">
                  <Text c="red" size="sm">
                    {error}
                  </Text>
                </Card>
              )}
            </Stack>
          </Grid.Col>
        </Grid>
      </Paper>
    </Stack>
  );
}

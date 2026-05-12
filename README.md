# LLM Calc — VRAM & Performance Calculator

Estimate GPU VRAM requirements and generation speed for running large language models locally or in the cloud.

**[llmcalc.teske.live](https://llmcalc.teske.live)**

## Features

- 174 LLM models with architecture details
- 135 GPUs (NVIDIA, AMD, Intel, Apple Silicon)
- Inference & fine-tuning modes
- Memory breakdown donut chart
- Speed & throughput estimates
- Energy cost from user-defined kWh price
- Dark theme matching [Teske's Lab](https://lab.teske.live)

## Contributing

### Adding or Updating GPUs

Edit [`src/data/gpu-data.json`](src/data/gpu-data.json). Each entry follows this format:

```json
{
  "key": "gpu_key_name",
  "label": "GPU Display Name",
  "vendor": "nvidia | amd | apple | intel",
  "type": "discrete | apu",
  "memory": 24,
  "factor": 4.0,
  "tdp_watts": 450,
  "hourly_price_usd": 0.79,
  "category": "consumer | datacenter | workstation"
}
```

- **key** — unique snake_case identifier used internally
- **label** — human-readable name shown in the dropdown
- **vendor** — manufacturer: `nvidia`, `amd`, `apple`, or `intel`
- **type** — `discrete` for dedicated GPUs, `apu` for integrated / Apple Silicon
- **memory** — VRAM in GB (or unified memory for Apple Silicon)
- **factor** — relative performance multiplier (RTX 3060 = 1.0 baseline)
- **tdp_watts** — typical power draw in watts (used for energy cost calculation)
- **hourly_price_usd** — cloud rental price (informational, not used in calculations)
- **category** — `consumer`, `datacenter`, or `workstation`

The entry will automatically appear in the GPU selector dropdown grouped by vendor.

### Adding or Updating Models

Edit [`src/models-data.json`](src/models-data.json). Field reference:

| Field | Description |
|-------|-------------|
| `name` | Display name |
| `slug` | URL-safe identifier |
| `provider` | Organization / company |
| `num_of_params` | Parameter count in billions |
| `architecture` | `dense` or `moe` |
| `hidden_dim_size` | Hidden layer dimension |
| `num_of_layers` | Number of transformer layers |
| `attention_structure` | `mha`, `gqa`, `mqa`, or `mla` |
| `position_embedding` | `rope`, `alibi`, `learned`, etc. |
| `num_of_expert_params` | Expert parameter count (MoE only) |
| `context_length` | Maximum context window |

### Modifying Calculation Logic

The VRAM and TPS formulas live in [`src/engine/calculator.ts`](src/engine/calculator.ts). Quantization byte-per-param maps are in [`src/engine/quantization.ts`](src/engine/quantization.ts).

## Development

```bash
npm install
npm run dev        # Start dev server at localhost:5173
npm run build      # Production build to dist/
npm run build:ci   # CI build (uses dist/ directly)
```

## Tech Stack

React 19, TypeScript, Vite, Mantine UI, recharts

## License

MIT — see [LICENSE](LICENSE)

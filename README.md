# Quantization Benchmarks

Command-line benchmarks for CNN and LLM quantization, adapted from the notebooks in this project.

## What is included

- `cnn_quant_benchmark.py`  
  CIFAR-10 training + quantization benchmarking for:
  - `resnet18`
  - `mobilenetv2`
  - `googlenet`
  - `vgg16_bn`

  Techniques:
  - FP32
  - Dynamic PTQ
  - Static PTQ
  - QAT

- `llm_quant_benchmark.py`  
  Inference benchmarking for Hugging Face causal LLMs using:
  - Baseline
  - BitsAndBytes INT8
  - BitsAndBytes INT4
  - optional quantized KV-cache if supported by the installed `transformers`

By default, both scripts print **only the final table**.  
Pass `--plots` to also show graphs.

## Why the code is structured this way

For CNNs, eager static PTQ and QAT in PyTorch are CPU quantization flows, which is why the script measures quantized CNN latency on CPU for a fair comparison. PyTorch’s official static quantization tutorial explicitly notes that quantization is currently supported for CPUs in that flow. citeturn468215search1turn468215search10

For `resnet18` and `mobilenetv2`, the script uses TorchVision’s **quantization-native model builders** for static PTQ and QAT because these architectures have residual / inverted-residual add paths that need quantizable implementations. TorchVision provides dedicated quantizable builders for both models.

For LLMs, the script uses Hugging Face `BitsAndBytesConfig`, which is the official loading path for 8-bit and 4-bit quantization in Transformers.

## Installation

Create a fresh virtual environment, then install:

```bash
pip install -U torch torchvision pandas matplotlib transformers accelerate bitsandbytes sentencepiece
```

### Notes

- **CNN quantization**
  - Static PTQ and QAT are benchmarked on CPU.
  - Training still uses GPU if available.

- **LLM quantization**
  - BitsAndBytes requires a CUDA-capable GPU for the standard workflow.
  - On Windows, you may want to redirect the Hugging Face cache to a larger drive with `--hf-home`.

## Usage

### CNN benchmark

Run one model:

```bash
python cnn_quant_benchmark.py --models resnet18
```

Run multiple models:

```bash
python cnn_quant_benchmark.py --models resnet18 mobilenetv2 googlenet vgg16_bn
```

Custom training setup:

```bash
python cnn_quant_benchmark.py --models resnet18 vgg16_bn --epochs 10 --qat-epochs 3 --batch-size 128
```

Show plots:

```bash
python cnn_quant_benchmark.py --models resnet18 googlenet --plots
```

Save the final table:

```bash
python cnn_quant_benchmark.py --models resnet18 googlenet --save-csv results/cnn_results.csv
```

### LLM benchmark

Run one LLM:

```bash
python llm_quant_benchmark.py --models gpt2
```

Run multiple LLMs:

```bash
python llm_quant_benchmark.py --models distilgpt2 gpt2 facebook/opt-350m
```

Custom prompt and generation length:

```bash
python llm_quant_benchmark.py --models gpt2 --prompt "Quantization improves inference because" --max-new-tokens 64
```

Show plots:

```bash
python llm_quant_benchmark.py --models gpt2 facebook/opt-350m --plots
```

Save the final table:

```bash
python llm_quant_benchmark.py --models gpt2 facebook/opt-350m --save-csv results/llm_results.csv
```

Use a custom Hugging Face cache directory:

```bash
python llm_quant_benchmark.py --models gpt2 facebook/opt-350m --hf-home D:\hf_cache
```

Try quantized KV-cache if your Transformers version supports it:

```bash
python llm_quant_benchmark.py --models gpt2 --include-kv-cache
```

## Output format

### CNN output

The CNN script prints one final table with columns:

- `model`
- `technique`
- `accuracy`
- `latency_s`
- `size_mb`

### LLM output

The LLM script prints one final table with columns:

- `model`
- `technique`
- `latency_s`
- `tokens_per_sec`
- `peak_gpu_mem_mb`

## Suggested repository structure

```text
.
├── cnn_quant_benchmark.py
├── llm_quant_benchmark.py
└── README.md
```

## Practical tips

- Start with fewer models first to validate the environment.
- On lower-VRAM GPUs, prefer smaller LLMs such as:
  - `distilgpt2`
  - `gpt2`
  - `facebook/opt-350m`
- If disk space becomes an issue for Hugging Face downloads, move the cache with `--hf-home`.

#!/usr/bin/env python3
import argparse
import gc
import os
import time
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import QuantizedCacheConfig
    HAS_QUANTIZED_CACHE = True
except Exception:
    HAS_QUANTIZED_CACHE = False


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_baseline_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )


def load_int8_model(model_name: str):
    qconfig = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qconfig,
        device_map="auto",
    )


def load_int4_model(model_name: str):
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=get_dtype(),
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qconfig,
        device_map="auto",
    )


@torch.no_grad()
def benchmark_generate(model, tokenizer, prompt: str, max_new_tokens: int, generation_kwargs=None):
    generation_kwargs = generation_kwargs or {}
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    _ = model.generate(
        **inputs,
        max_new_tokens=min(16, max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    latency = time.time() - start
    out_len = outputs.shape[1]
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_len - in_len
    tokens_per_sec = new_tokens / latency if latency > 0 else 0.0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "latency_s": round(latency, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_gpu_mem_mb": round(peak_mem_mb, 1),
    }


def plot_results(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    for tech in df["technique"].unique():
        sub = df[df["technique"] == tech]
        plt.plot(sub["model"], sub["latency_s"], marker="o", label=tech)
    plt.title("LLM Latency Comparison")
    plt.ylabel("Latency (s)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for tech in df["technique"].unique():
        sub = df[df["technique"] == tech]
        plt.plot(sub["model"], sub["peak_gpu_mem_mb"], marker="o", label=tech)
    plt.title("LLM Peak GPU Memory Comparison")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for tech in df["technique"].unique():
        sub = df[df["technique"] == tech]
        plt.plot(sub["model"], sub["tokens_per_sec"], marker="o", label=tech)
    plt.title("LLM Throughput Comparison")
    plt.ylabel("Tokens / second")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM quantization benchmark.")
    parser.add_argument("--models", nargs="+", required=True, help="One or more Hugging Face model IDs.")
    parser.add_argument("--prompt", type=str, default="Quantization improves transformer inference because")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--hf-home", type=str, default="", help="Optional Hugging Face cache root (useful on Windows).")
    parser.add_argument("--plots", action="store_true", help="Show latency/memory/throughput plots.")
    parser.add_argument("--save-csv", type=str, default="", help="Optional path to save the result table as CSV.")
    parser.add_argument("--include-kv-cache", action="store_true", help="Also benchmark quantized KV cache if supported.")
    return parser.parse_args()


def configure_hf_cache(hf_home: str):
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
        os.environ["HF_ASSETS_CACHE"] = os.path.join(hf_home, "assets")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    args = parse_args()
    configure_hf_cache(args.hf_home)

    rows: List[Dict] = []

    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Baseline
        reset_gpu_stats()
        model = load_baseline_model(model_name)
        res = benchmark_generate(model, tokenizer, args.prompt, args.max_new_tokens)
        rows.append({"model": model_name, "technique": "Baseline", **res})
        del model
        gc.collect()
        reset_gpu_stats()

        # INT8
        reset_gpu_stats()
        model = load_int8_model(model_name)
        res = benchmark_generate(model, tokenizer, args.prompt, args.max_new_tokens)
        rows.append({"model": model_name, "technique": "BitsAndBytes INT8", **res})
        del model
        gc.collect()
        reset_gpu_stats()

        # INT4
        reset_gpu_stats()
        model = load_int4_model(model_name)
        res = benchmark_generate(model, tokenizer, args.prompt, args.max_new_tokens)
        rows.append({"model": model_name, "technique": "BitsAndBytes INT4", **res})

        if args.include_kv_cache and HAS_QUANTIZED_CACHE:
            try:
                kv_cfg = QuantizedCacheConfig(backend="quanto", nbits=4, axis_key=0, axis_value=0)
                res = benchmark_generate(
                    model,
                    tokenizer,
                    args.prompt,
                    args.max_new_tokens,
                    generation_kwargs={
                        "cache_implementation": "quantized",
                        "cache_config": kv_cfg,
                        "use_cache": True,
                    },
                )
                rows.append({"model": model_name, "technique": "INT4 + Quantized KV Cache", **res})
            except Exception:
                pass

        del model
        gc.collect()
        reset_gpu_stats()

    df = pd.DataFrame(rows).sort_values(["model", "technique"]).reset_index(drop=True)

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        df.to_csv(args.save_csv, index=False)

    print(df.to_string(index=False))

    if args.plots:
        plot_results(df)


if __name__ == "__main__":
    main()

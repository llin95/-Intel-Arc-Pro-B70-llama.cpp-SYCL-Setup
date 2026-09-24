# Intel Arc Pro B70 + vLLM XPU + Qwen3.8-27B

Date: 2026-09-24

This note records one working single-GPU configuration for serving Qwen3.8-27B on an Intel Arc Pro B70 32 GB card with vLLM XPU. It is meant as a reproducible field note, not a universal tuning guide.

## Performance Snapshot

The current best local vLLM profile is `float16 + GPTQ INT4 + fp8 KV + native MTP4`. On real Copilot/agent-style traffic with long prompts, it has been faster than the earlier `bfloat16` profile and much faster than the local llama.cpp GGUF path.

| Runtime profile | Observed decode throughput | Notes |
|---|---:|---|
| vLLM v0.30, `float16`, MTP4, 180K context | `~56-57 tok/s` cumulative; `~83 tok/s` short high-throughput segment | Current preferred local profile. Uses `top_k=20`, `top_p=0.90`, `temperature=0.7`, and default `reasoning_effort=medium`. |
| vLLM v0.30, `bfloat16`, MTP4 | `~53 tok/s` cumulative; `~63 tok/s` best segment | Stable, but slower than the current `float16` native MTP4 path on this workload. |
| vLLM v0.30, `bfloat16`, MTP2 | `~47-50 tok/s` | Conservative fallback profile. |
| llama.cpp b11149, GGUF, q8 KV, MTP4 | `~22-27 tok/s` | Useful comparison point, but not competitive with vLLM for Qwen3.8 GPTQ serving. |

One recent `float16 + MTP4` live metrics snapshot:

```text
requests completed: 38
prompt tokens: 3,821,893
generation tokens: 47,793
decode throughput: ~57 tok/s
prefix cache hit rate: ~94%
MTP acceptance: ~68.6%
KV usage: ~67%
preemptions: 0
```

The public `84.65 tok/s` single-B70 result is plausible for short-context benchmark traffic. For long agent workloads, expect a lower average because prompts commonly reach `80K-120K+` tokens and MTP acceptance varies by request.

## Context And KV Budget Details

This setup advertises a long single-request context, but it is not a concurrent 180K-serving profile. The useful mental model is one large active request at a time, with prefix caching helping repeated agent turns.

| Setting or metric | Value | Meaning |
|---|---:|---|
| `--max-model-len` | `180000` | Maximum sequence length exposed by the server for one request. This includes prompt plus generated tokens. |
| vLLM reported XPU KV cache capacity | about `209,763 tokens` | Startup capacity reported by vLLM for this B70 run. |
| Capacity ratio | about `1.17x` | `209,763 / 180,000`; enough for one full-length request, not enough for two. |
| `--max-num-seqs` | `1` | Intentional. A second long request waits instead of competing for KV memory. |
| `--kv-cache-dtype` | `fp8` | Required for this long-context single-card setup; fp16 KV would consume too much VRAM. |
| `--gpu-memory-utilization` | `0.96` | Aggressive allocation to make 180K practical on one 32 GB B70. |
| `xpu-smi` VRAM after load | about `30.4 GiB / 32.7 GiB`, `93%` | Normal for vLLM: model weights, graph memory, and reserved KV cache stay allocated even when idle. |
| Typical live KV usage | `40-70%` in observed agent traffic | Depends on active prompt length; idle returns to `0%` in vLLM's KV metric, while VRAM remains reserved. |

Observed startup line:

```text
XPU KV cache size: 209,763 tokens, Maximum concurrency for 180,000 tokens per request: 1.17x
```

Practical interpretation:

```text
180K max_model_len is a per-request ceiling, not a throughput target.
For daily agent use, expect many requests around 80K-120K prompt tokens.
With max_num_seqs=1, concurrent requests serialize by design.
If num_requests_waiting rises while kv_cache_usage_perc is high, the service is capacity-bound, not broken.
```

Safer context variants:

| Profile | Suggested values | When to use |
|---|---|---|
| Peak benchmark | `MAX_MODEL_LEN=100000 GPU_UTIL=0.88` | Reproduce short-context public throughput numbers with more VRAM headroom. |
| Balanced long-context | `MAX_MODEL_LEN=161000 GPU_UTIL=0.90` | Closer to published quality recipe, less aggressive than 180K. |
| Current local long-agent profile | `MAX_MODEL_LEN=180000 GPU_UTIL=0.96` | Best local tradeoff for large Copilot-style prompts on one B70. |

Do not increase `--max-num-seqs` while keeping `MAX_MODEL_LEN=180000`. If concurrent serving matters more than maximum context, lower `MAX_MODEL_LEN` first, then test queueing and preemption counters.

## Hardware And Runtime

| Item | Value |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB VRAM |
| Host OS | Linux, `xe` driver, Level Zero |
| vLLM image | `vllm/vllm-openai-xpu:v0.30.0` |
| Model format | Hugging Face safetensors |
| Model | `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` |
| Quantization | GPTQ INT4, symmetric, group size 128 |
| KV cache | `fp8` |
| Speculative decoding | Native MTP, `num_speculative_tokens=4` |
| API mode | OpenAI-compatible server |

## Current Recommended Profile

This profile is tuned for an agent / coding assistant endpoint rather than a short benchmark run. It keeps long-context capacity, tool calling, prefix caching, and less aggressive sampling defaults.

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai-xpu:v0.30.0}"
CONTAINER="${CONTAINER:-qwen38-vllm-v030}"
HOST_PORT="${HOST_PORT:-11444}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-180000}"
GPU_UTIL="${GPU_UTIL:-0.96}"
SPEC_TOKENS="${SPEC_TOKENS:-4}"
DTYPE="${DTYPE:-float16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.90}"
TOP_K="${TOP_K:-20}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
HOST_TZ="${HOST_TZ:-$(cat /etc/timezone 2>/dev/null || echo Asia/Taipei)}"

cd "$HOME/intel-arc-pro-b70-qwen38-vllm"

GENERATION_CONFIG_DIR="${GENERATION_CONFIG_DIR:-$PWD/generation_config.qwen38-stable}"
CHAT_TEMPLATE_DIR="${CHAT_TEMPLATE_DIR:-$PWD/chat_template.qwen38-stable}"

case "$REASONING_EFFORT" in
  low|medium|xhigh) ;;
  *)
    echo "Invalid REASONING_EFFORT: $REASONING_EFFORT (expected low, medium, or xhigh)" >&2
    exit 1
    ;;
esac

mkdir -p "$GENERATION_CONFIG_DIR"
cat > "$GENERATION_CONFIG_DIR/generation_config.json" <<EOF
{
  "do_sample": true,
  "temperature": $TEMPERATURE,
  "top_p": $TOP_P,
  "top_k": $TOP_K,
  "repetition_penalty": 1.0
}
EOF

mkdir -p "$CHAT_TEMPLATE_DIR"
sed "0,/xhigh/s//$REASONING_EFFORT/" \
  "$MODEL_DIR/chat_template.jinja" > "$CHAT_TEMPLATE_DIR/chat_template.jinja"

sudo docker pull "$IMAGE"
sudo docker rm -f "$CONTAINER" 2>/dev/null || true

sudo docker run -d \
  --name "$CONTAINER" \
  --network llm-shared \
  --entrypoint bash \
  --device /dev/dri:/dev/dri \
  -v /dev/dri:/dev/dri \
  -e TZ="$HOST_TZ" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$GENERATION_CONFIG_DIR:/generation-config:ro" \
  -v "$CHAT_TEMPLATE_DIR:/chat-template:ro" \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p "$HOST_PORT:8000" \
  "$IMAGE" \
  -lc "exec vllm serve /model \
    --quantization gptq \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --kv-cache-dtype fp8 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --generation-config /generation-config \
    --served-model-name $SERVED_MODEL_NAME \
    --chat-template /chat-template/chat_template.jinja \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser $TOOL_PARSER \
    --limit-mm-per-prompt '{\"image\":4}' \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}'"
```

## Why These Defaults

| Setting | Current value | Reason |
|---|---:|---|
| `--dtype` | `float16` | Best match for the public single-B70 Qwen3.8 GPTQ INT4 recipes and currently faster than local `bfloat16` tests for native MTP4. |
| `--kv-cache-dtype` | `fp8` | Required to keep useful long context on a 32 GB B70. |
| `--max-model-len` | `180000` | Fits on this machine with `gpu_memory_utilization=0.96`; vLLM reports about 209K KV capacity, or roughly 1.17x of the configured 180K request length. |
| `--gpu-memory-utilization` | `0.96` | Aggressive but works on this host; leaves little headroom, so keep `max_num_seqs=1`. |
| `--max-num-seqs` | `1` | Avoids concurrent long-context requests overrunning the single-card KV budget. |
| `--max-num-batched-tokens` | `8192` | Good prefill scheduler budget for this B70 setup. |
| `--enable-prefix-caching` | enabled | Helps repeated agent/Copilot-style prompts; metrics often show 90%+ prefix hit rate. |
| MTP | `num_speculative_tokens=4` | Higher peak throughput than MTP2 when acceptance is good. |
| Sampling | `temperature=0.7`, `top_p=0.90`, `top_k=20` | Reduces Qwen3.8 thinking-mode drift compared with very open defaults. |
| Reasoning | default `medium` | The upstream Qwen3.8 chat template defaults to `xhigh`; this local wrapper changes only the default while still allowing request-level override. |

## float16 vs bfloat16

For this specific single-B70 path, use `float16` as the default:

```text
B70 + vLLM XPU + Qwen3.8 GPTQ INT4 + fp8 KV + native MTP4 -> float16
```

Reasons:

- The public single-B70 Qwen3.8 GPTQ INT4 recipes from the B70 vLLM repos use `--dtype float16`.
- Local `float16` testing reached a cumulative decode rate around `56-57 tok/s`, with a short high-throughput segment around `83 tok/s`.
- Earlier local `bfloat16` testing was stable but lower: roughly `53 tok/s` cumulative and about `63 tok/s` in the best segment.

Keep `bfloat16` as the rollback option:

```bash
DTYPE=bfloat16 ./run-qwen38-v030-stable.sh
```

Use `bfloat16` for DFlash2-style draft experiments. A vLLM XPU issue on B70/Qwen3.8 reported DFlash2 `--dtype float16` causing NaNs and 0% draft acceptance, while `bfloat16` worked. That issue is DFlash2-specific; native MTP4 has worked with `float16` locally.

## Expected Startup Log

A good startup should include these lines or equivalents:

```text
version 0.30.0
Resolved architecture: Qwen3_5ForConditionalGeneration
Using max model len 180000
Using fp8 data type to store kv cache
Resolved architecture: Qwen3_5MTP
speculative_config: SpeculativeConfig(method='mtp', model='/model', num_spec_tokens=4)
Using XPUwNa16LinearKernel for AutoGPTQLinearMethod
Using Triton/FLA GDN prefill kernel
Using Flash Attention backend
XPU KV cache size: 209,763 tokens, Maximum concurrency for 180,000 tokens per request: 1.17x
Default vLLM sampling parameters have been overridden by /generation-config: {'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.9}
Application startup complete
```

Warnings seen locally and considered non-fatal:

```text
Failed to import the DeepSelect extension (vllm._deepselect_C)
Casting torch.float16 to torch.bfloat16
Speculative decoding (method=mtp) is enabled but no KV cache group could be identified as the draft model's
```

## Current Local Performance Notes

These numbers are from live `/metrics` counters, not a clean isolated benchmark. They are still useful for agent-workload sanity checks.

| Profile | Observed decode throughput | Notes |
|---|---:|---|
| vLLM v0.30, `bfloat16`, MTP2 | `47-50 tok/s` | Stable baseline. |
| vLLM v0.30, `bfloat16`, MTP4 | `~53 tok/s` cumulative, `~63 tok/s` best segment | MTP acceptance around `75-81%`. |
| vLLM v0.30, `float16`, MTP4 | `~56-57 tok/s` cumulative, `~83 tok/s` short segment | Current preferred fast path. |
| llama.cpp b11149, GGUF, q8 KV, MTP4 | `~22-27 tok/s` | Useful comparison, much slower than vLLM for this model. |

A recent `float16 + MTP4` live snapshot showed:

```text
requests: 38
prompt tokens: 3,821,893
generation tokens: 47,793
TG: ~57 tok/s
prefix cache hit rate: ~94%
MTP acceptance: ~68.6%
KV usage: ~67%
preemptions: 0
```

Another short segment reached:

```text
TG: ~83 tok/s
prefix cache hit rate: ~95%
MTP acceptance: ~67%
```

This explains why public `84.65 tok/s` short-context results are plausible, while long agent workloads usually average lower.

## Monitoring Commands

```bash
curl -s http://127.0.0.1:11444/v1/models
curl -s http://127.0.0.1:11444/metrics | grep -E 'vllm:(prompt_tokens_total|generation_tokens_total|time_to_first_token_seconds_sum|e2e_request_latency_seconds_sum|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|request_prompt_tokens_count|prefix_cache_hits_total|prefix_cache_queries_total|num_requests_running|num_requests_waiting|num_preemptions_total|kv_cache_usage_perc)'
```

GPU memory:

```bash
xpu-smi --query-gpu=memory.used,memory.total,memory.free,utilization.memory,power.draw --id=0 --format=csv
```

Docker logs:

```bash
sudo docker logs -f qwen38-vllm-v030
```

## Throughput Formulas

From vLLM `/metrics` counters:

```text
TG = generation_tokens_total / (e2e_request_latency_seconds_sum - time_to_first_token_seconds_sum)
MTP acceptance = spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total
prefix cache hit rate = prefix_cache_hits_total / prefix_cache_queries_total
effective PP = prompt_tokens_total / time_to_first_token_seconds_sum
uncached PP estimate = (prompt_tokens_total - prefix_cache_hits_total) / time_to_first_token_seconds_sum
```

When `num_requests_running > 0`, treat throughput as approximate because token counters can include an in-flight request while E2E latency only includes completed requests.

## Practical Verdict

For a single Arc Pro B70 serving Qwen3.8-27B as an OpenAI-compatible agent endpoint, the current best local profile is:

```text
vLLM XPU v0.30.0
Qwen3.8-27B GPTQ INT4 G128 + BF16 MTP tensors
float16 runtime dtype
fp8 KV cache
MTP4
180K context
max_num_seqs=1
prefix caching enabled
server default reasoning_effort=medium
top_k=20, top_p=0.90, temperature=0.7
```

For pure benchmark reproduction of public peak numbers, reduce context to `100K-131K`, use `gpu_memory_utilization=0.88-0.90`, consider disabling prefix cache, and benchmark short prompts separately. For daily Copilot/agent use, the 180K profile above is the better tradeoff.

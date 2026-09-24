# Intel Arc Pro B70 + llama.cpp SYCL: Current Working Setup

Related vLLM setup: [Intel Arc Pro B70 + vLLM XPU + Qwen3.8-27B](qwen38-vllm-xpu-b70.md)

Date: 2026-09-03

GPU: Intel Arc Pro B70, Battlemage / Xe2, 32 GB VRAM

Primary workload: `llama.cpp` SYCL serving with Qwen-class GGUF models

This is my current field-tested setup for running `llama.cpp` on an Intel Arc Pro B70 with the SYCL backend. It is not a generic tuning guide. It is a record of what is currently working well on one B70 system, what I have measured, and which tempting settings turned out to be traps.

The short version:

- Use a recent `llama.cpp` SYCL build. My validated production baseline is around `b10705`; the latest checked release is `b10774`, but it does not add a must-have B70 SYCL change.
- Use the newer Intel GPU runtime stack: compute-runtime `26.31.39395.13` + IGC `2.40.13`.
- For the current Qwen3.6-35B-A3B GGUF production path, use f16 KV cache, `--parallel 1`, `-b 8192`, `-ub 4096`, and flash attention on.
- For Qwen3.8-27B GGUF, keep expectations separate: it is a dense 27B model, so decode is much slower than Qwen3.6 A3B MoE. Practical interactive context is around 32K to 64K on one B70.
- DFlash2 is worth testing for short follow-up workloads, but it is not a long-context cure.
- Do not blindly copy `q8_0/q4_1` KV recommendations from other B70 posts; on this setup, f16 KV is still the best current production default for the validated GGUF path.

## Hardware And Runtime

Tested hardware and software baseline:

| Component | Current baseline |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB VRAM |
| Driver path | Linux `xe` driver + Level Zero |
| GPU runtime | `libze-intel-gpu1` / `intel-opencl-icd` / `intel-ocloc` `26.31.39395.13` |
| IGC | `2.40.13` |
| xpu-smi | `2.1.1` works correctly on this setup |
| llama.cpp | `b10705` validated; `b10774` checked, no urgent SYCL/B70 trigger |
| Backend | SYCL FP16 build |

The runtime update to compute-runtime `26.31.39395.13` was a real improvement for this machine. With the same `llama.cpp` build and model, increasing the physical ubatch to `-ub 4096` became a strong prefill win and remained stable in long-context HTTP smoke tests.

## Environment

These are the environment variables I keep explicit for the B70 SYCL server:

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export SYCL_CACHE_PERSISTENT=0
export SYCL_DEVICE_FILTER=level_zero
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZE_AFFINITY_MASK=0
```

Notes:

- `ONEAPI_DEVICE_SELECTOR=level_zero:0` keeps the service on the B70 instead of accidentally using the iGPU.
- I leave SYCL optimization enabled. Do not set `GGML_SYCL_ENABLE_OPT=0` unless you are explicitly debugging a compiler/runtime issue.
- Older workarounds around `GGML_SYCL_ENABLE_MKL_FA=0` are no longer required for the validated `b10705` path, but I still re-check this when changing `llama.cpp` releases.

## Build

For a local build:

```bash
cmake -S . -B build-sycl-f16 \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-sycl-f16 --config Release -j --target llama-server llama-bench
```

The official Ubuntu SYCL FP16 release assets are also usable, but I still prefer validating each promoted build with local `llama-bench` and an HTTP smoke test.

## Current Production Profile: Qwen3.6-35B-A3B GGUF

This is the current known-good style of configuration for Qwen3.6-35B-A3B on one B70:

```bash
./llama-server \
  -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8089 \
  -ngl 99 \
  -fa on \
  --ctx-size 262144 \
  --parallel 1 \
  -b 8192 \
  -ub 4096
```

I intentionally omit `-ctk` and `-ctv` here. That means the KV cache stays at the default f16 precision.

Validated behavior with compute-runtime `26.31.39395.13`:

| Test | Result |
|---|---:|
| Short decode | about `87.9-88.0 tok/s` |
| 200K HTTP single request | about `1148 tok/s` prompt processing, `32-34 tok/s` decode |
| 200K HTTP concurrent x2 with `--parallel 1` | serialized as expected, about `360 s` wall for two requests |

The important 2026-09-02 finding was `-ub 4096`: compared with `-ub 1024`, it improved large-prompt prefill by roughly 17-21% in the real 200K HTTP path, with no observed errors. Raising `-b` from `8192` to `16384` did not help, so `-b 8192 -ub 4096` is the current production recommendation.

## KV Cache: Why I Still Use f16

Other B70 recipes often recommend asymmetric quantized KV, especially:

```bash
-fa on -ctk q8_0 -ctv q4_1
```

That recommendation is reasonable for some models and stacks, especially when the priority is fitting larger context. It is not my default for the current production GGUF path.

On this B70, f16 KV remains faster for the validated Qwen3.6 long-context server path. After `#26689`, quantized KV decode improved a lot, but it still did not beat f16 KV for this workload:

| Qwen3.6-35B-A3B, `d204800` | Result |
|---|---:|
| f16 KV, HTTP long-context baseline | about `33-35 tok/s` decode |
| `q8_0/q8_0` direct bench | `23.23 tok/s` decode |

For Qwen3.8-27B dense GGUF, quantized KV also did not help the long-context decode path in local testing:

| Qwen3.8-27B UD-Q4_K_XL, `d204800` | Result |
|---|---:|
| f16 KV | about `7.0-7.3 tok/s` decode |
| `q8_0/q8_0` KV | about `5.61 tok/s` decode |

So my current rule is simple: use f16 KV unless the problem is strictly VRAM capacity and I am willing to trade away decode speed.

## Qwen3.8-27B GGUF: Separate Expectations

Qwen3.8-27B is a dense 27B model. Do not compare its token generation speed directly with Qwen3.6-35B-A3B MoE.

Local B70 results for `Qwen3.8-27B UD-Q4_K_XL`:

| Test | Result |
|---|---:|
| Short `tg512` | about `23 tok/s` |
| `d32768` | about `16.4 tok/s` |
| `d65536` | about `13.2 tok/s` |
| `d131072` | about `9.5 tok/s` |
| `d204800` f16 KV | about `7.0-7.3 tok/s` |

This is why I treat 32K as the comfortable interactive context and 64K as the practical upper daily-use range. The model can advertise 262K context, and it can allocate larger contexts, but that does not make 128K or 200K pleasant for interactive use on a single B70.

Recommended Qwen3.8 GGUF baseline:

```bash
./llama-server \
  -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 8090 \
  -ngl 99 \
  -fa on \
  --ctx-size 32768 \
  --parallel 1 \
  -b 8192 \
  -ub 4096 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

For non-thinking / faster instruction mode:

```bash
--temp 0.7 \
--top-p 0.80 \
--top-k 20 \
--min-p 0.0 \
--presence-penalty 1.5 \
--repeat-penalty 1.0 \
--chat-template-kwargs '{"reasoning_effort":"low"}'
```

## DFlash2 For Qwen3.8

DFlash2 is interesting, but workload-sensitive.

Local testing showed:

| Scenario | Result |
|---|---:|
| Qwen3.8 no draft, short decode | about `23 tok/s` |
| Qwen3.8 + DFlash2 at `ctx=102400`, short follow-up | `31.77 tok/s`, acceptance `0.863` |
| Qwen3.8 + DFlash2 at `ctx=204800`, long-prompt pressure | about `1.6 tok/s` |

So my current DFlash2 rule is:

- Use it for short repeated/chat follow-ups where acceptance stays high.
- Do not use it as the default long-context ingestion profile.
- Test `--spec-draft-n-max 2` and `4`; do not assume the larger number wins.

Candidate DFlash2 profile:

```bash
./llama-server \
  -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 4 \
  --ctx-size 32768 \
  --parallel 1 \
  -b 8192 \
  -ub 4096 \
  -fa on
```

If acceptance is unstable, retry with:

```bash
--spec-draft-n-max 2
```

## What I Would Not Copy Blindly

Several public B70 recipes are useful, but the numbers are not interchangeable.

Sergio Barrientos' B70 recipe is useful for build flags, power tiers, `-ub 4096`, and benchmarking discipline. However, its `q8_0/q4_1` KV cache recommendation came from a different model/version/measurement set. On this machine, f16 KV remains the better production default for the validated GGUF path.

The fast Qwen3.8-27B numbers in Sergio's cookbook are mostly a separate vLLM XPU route:

```bash
vllm serve /model \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 100000 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

That is GPTQ-INT4 + fp8 KV + vLLM XPU + MTP. It is not comparable to raw `llama.cpp` GGUF decode.

BeeLlama.cpp is also worth watching. It adds KVarN, KV precision tails, and DFlash controls. The most interesting experimental idea is:

```bash
--cache-type-k kvarn5 --cache-type-v kvarn4 --kv-tail-tokens 1024
```

But Bee's own documentation says SYCL was not hardware-verified for the KVarN / precision-tail path in that release. I would treat it as an experiment, not a production setting, until measured on the actual B70 SYCL stack.

## Validation Checklist

Before promoting a new build or flag set, I run both a direct benchmark and a real server path.

Short-path `llama-bench`:

```bash
./llama-bench \
  -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 \
  -fa 1 \
  -p 512 \
  -n 512 \
  -d 0,2,4 \
  -b 8192 \
  -ub 4096 \
  -r 2 \
  -dev SYCL0
```

Deep-context discriminator:

```bash
./llama-bench \
  -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 \
  -fa 1 \
  -p 512 \
  -n 512 \
  -d 204800 \
  -b 8192 \
  -ub 4096 \
  -r 2 \
  -dev SYCL0
```

Then run a real HTTP smoke test at the target context, because `llama-bench` cannot catch every server scheduling, prompt-cache, or long-prompt behavior.

## Current Verdict

For this B70 system, the current best production setup remains:

```bash
llama.cpp SYCL FP16 build
compute-runtime 26.31.39395.13 + IGC 2.40.13
Qwen3.6-35B-A3B Q4_K_M
f16 KV cache
--parallel 1
-b 8192 -ub 4096
-fa on
```

For Qwen3.8-27B GGUF, the model is useful but should be treated as a different performance class:

- 32K is the comfortable interactive target.
- 64K is usable but slower.
- 128K+ is an experiment, not my default serving mode.
- DFlash2 can help short follow-up workloads, but it is not a replacement for a faster dense-model runtime.

If the goal is maximum Qwen3.8 throughput on B70, the credible next track is not another `llama.cpp` flag. It is a separate vLLM XPU GPTQ-INT4 + fp8 KV + MTP deployment, validated with its own quality checks.
[中文](gpu-validation.md) · **English**

# GPU Validation Guide: reproducing this project's conclusions with the official vLLM / SGLang benchmarks

> Every experiment in this project runs on `codesign_sim`, a cost model. Once you rent GPUs, use this guide to reproduce four core conclusions on real engines, and to know where the simulator and real systems will disagree.
>
> **Version baseline**: vLLM **v0.30.0** (PyPI 2026-09-22) and SGLang **v0.5.20** (PyPI 2026-09-18). Every command and flag below was checked against the official docs and source code of these two versions (links collected in Section 9). Flags may be renamed in other versions; check `vllm bench serve --help` / `python -m sglang.bench_serving --help` first.
>
> The commands in this guide **have not been run on a GPU** (this project's development machine has no NVIDIA GPU), so no "expected milliseconds" are given here — only what to record and how to compare with the simulator.

## 0. The four conclusions to reproduce

| # | Conclusion | Where in this project | Switch in real systems |
|---|---|---|---|
| A | The higher the prefix-cache hit rate, the lower the TTFT; near saturation, the effect of the hit rate on tail TTFT is **non-linear** (less prefill also means less queueing) | [Module 02](../modules/02_inference_engines/README.en.md) Section 4 and Lab 3 | vLLM `--enable-prefix-caching` / `--no-enable-prefix-caching`; SGLang `--disable-radix-cache` |
| B | The token budget (chunk size) is a knob between TTFT and TPOT: the larger the budget, the bigger the TPOT / ITL spikes for decodes in the same batch, worst with chunked prefill off; too small a budget starves prefill and TTFT explodes. The vLLM docs also say larger budgets help TTFT — the simulator has no chunking overhead so this is invisible in simulation, which is exactly what a real GPU should check | Module 02 Section 5 and Labs 1, 2 | vLLM `--max-num-batched-tokens`, `--no-enable-chunked-prefill`; SGLang `--chunked-prefill-size`, `--enable-mixed-chunk` |
| C | PD disaggregation first improves isolation and controllability of tail ITL; whether it raises goodput under SLOs depends on whether the P : D ratio matches the workload — on homogeneous hardware, with few GPUs and no SLOs, it does not raise peak throughput | Module 02 Section 9 | vLLM `--kv-transfer-config` (NixlConnector); SGLang `--disaggregation-mode` |
| D | In multi-turn agent workloads the hit rate is set by reuse distance: the more sessions are active at once and the more KV is written between two uses, the more likely a prefix is evicted before reuse, and TTFT rises | Module 02 Section 4.1, [Module 05](../modules/05_cache_prefill/README.en.md) | vLLM `--gpu-memory-utilization` (changes the KV pool size), `--max-active-conversations` of the multi-turn benchmark |

## 1. Setup

**Hardware.** Experiments A and B need one 80 GB GPU (H100 / A100); C needs 2 GPUs in the same machine (cross-node needs RDMA NICs).

**Model.** Written as `$MODEL` below. Both projects' docs use `meta-llama/Llama-3.1-8B-Instruct` in their examples (it requires requesting access on Hugging Face); any 8B-class model works. What matters is **the same model and the same dtype on both engines**.

**Installation** (in two separate virtual environments to avoid dependency conflicts):

```bash
# vLLM (as in the official quickstart, with the version pinned)
uv pip install vllm==0.30.0 --torch-backend=auto

# SGLang (as in the official install docs, with the version pinned)
uv pip install --prerelease=allow "sglang==0.5.20"
```

## 2. Measurement discipline (read this first)

1. **Cache contamination**: the vLLM docs warn about this explicitly — repeating `vllm bench serve` against the same server can hit the prefix cache left by the previous run, and the random dataset generates the same prompts for a fixed `--seed` (default 0). Remedies: restart the server between runs, change `--seed`, or use `vllm bench sweep serve` (which resets caches between runs). vLLM's `/reset_prefix_cache` endpoint is only registered when the server is started with `VLLM_SERVER_DEV_MODE=1`. SGLang's bench has `--flush-cache` (calls `/flush_cache`).
2. **Fixed output lengths**: add `--ignore-eos` for vLLM; SGLang's bench ignores EOS by default (turn that off with `--disable-ignore-eos`).
3. **The length-distribution trap**: `--random-range-ratio` means **different things** in the two tools. vLLM samples from `[L·(1−r), L·(1+r)]`, and the default r = 0 is a fixed length; SGLang samples from `[L·r, L]`, and the default r = 0 is **uniform from 1 to L**. For fixed lengths in SGLang, set `--random-range-ratio 1` explicitly.
4. **Warm-up**: vLLM `--num-warmups N` (default 0); SGLang `--warmup-requests N` (default 1).
5. **Keep the raw data**: vLLM `--save-result --save-detailed --result-dir DIR --label NAME`; SGLang `--output-file FILE.jsonl --output-details`. Means alone are meaningless; record at least p50 and p99.
6. **Metric definitions**: both tools define TPOT as `(end-to-end latency − TTFT) / (output tokens − 1)`, computed per request and then aggregated; ITL is the gap between consecutive streamed outputs. With speculative decoding one streamed output can carry several tokens, and ITL and TPOT diverge noticeably (the vLLM docs have a diagram). The simulator's TPOT uses the same definition.

### What to record for every run

| Category | Item | Where to get it |
|---|---|---|
| Configuration | Engine and version, GPU model and count, model, dtype, every non-default launch flag, the bench command | Write it down; vLLM can store it in the result JSON with `--metadata key=value` |
| Latency | TTFT p50 / p99, TPOT p50 / p99, ITL p99 | Bench output; for vLLM use `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99` |
| Throughput | Output-token throughput, request throughput; goodput when you have SLOs | Bench output; vLLM `--goodput ttft:<ms> tpot:<ms>` (DistServe's definition) |
| Cache | Prefix-cache hit rate (in tokens) | vLLM: the ratio of the deltas of the two `/metrics` counters `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries` before and after the run (Prometheus exposes them with a `_total` suffix); SGLang: add `--cache-report` to the bench, which prints `Cache hit rate` |
| Memory pressure | Preemption count, KV usage, queued requests | vLLM: `vllm:num_preemptions`, `vllm:kv_cache_usage_perc`, `vllm:num_requests_waiting`, and `Preemptions: N` in the periodic stats log (the `PreemptionMode.RECOMPUTE` sample log line in the official tuning docs dates from V0; v0.30.0 no longer prints it) |

## 3. Experiment A: prefix caching and TTFT

**Idea.** Build a "P shared prefixes × N requests per prefix" workload, run it against servers with prefix caching on and off, compare TTFT, and use the hit rate to match the simulator's `cached_tokens / prompt_tokens`.

**vLLM** (`prefix_repetition` dataset: each request = one shared prefix + a random suffix):

```bash
# Server: V1 enables prefix caching by default; the control group adds --no-enable-prefix-caching
vllm serve $MODEL --port 8000
vllm serve $MODEL --port 8000 --no-enable-prefix-caching

# Client: 5 prefixes × 20 requests, 4096-token prefix + 512-token suffix, 128 output tokens
curl -s localhost:8000/metrics | grep -E '^vllm:prefix_cache_(queries|hits)' > before.txt
vllm bench serve --backend vllm --model $MODEL --port 8000 \
  --dataset-name prefix_repetition --num-prompts 100 \
  --prefix-repetition-prefix-len 4096 --prefix-repetition-suffix-len 512 \
  --prefix-repetition-num-prefixes 5 --prefix-repetition-output-len 128 \
  --request-rate 4 --ignore-eos --seed 0 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
  --save-result --result-dir results/A --label vllm_apc_on
curl -s localhost:8000/metrics | grep -E '^vllm:prefix_cache_(queries|hits)' > after.txt
```

**SGLang** (`generated-shared-prefix` dataset: long system prompts + short questions):

```bash
# Server: radix cache is on by default; the control group adds --disable-radix-cache
python -m sglang.launch_server --model-path $MODEL --port 30000
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache

python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --model $MODEL \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 5 --gsp-prompts-per-group 20 \
  --gsp-system-prompt-len 4096 --gsp-question-len 512 --gsp-output-len 128 \
  --request-rate 4 --flush-cache --cache-report \
  --output-file results/A/sglang.jsonl --output-details
```

(In v0.5.20 `sglang.bench_serving` is marked as a legacy path and prints a FutureWarning; the new path is `python -m sglang.benchmark.serving` with the same flags.)

**How to read it.**
- Theoretical upper bound on the hit rate: the first request for each prefix must miss, so the bound ≈ (100 − 5) / 100 × 4096 / 4608 ≈ **84%** (in tokens). vLLM also caches only full blocks, which takes a little more off. A measured value well below this usually means requests for the same prefix arrived before the first one finished its prefill — raise the request rate and you will see the hit rate fall.
- Sweep `--request-rate` (e.g. 1, 2, 4, 8) and plot "TTFT p99 vs request rate, cache on and off". The cache-off curve bends upward earlier — that is the "non-linear" part of conclusion A.
- **Compare with the simulator**: plug the measured hit rate into `cached_tokens` in Module 02 Lab 3, and compare "the TTFT ratio between cache on and off" and "the request rate where the curve bends", **not absolute milliseconds**.

## 4. Experiment B: chunked prefill and the token budget

**Idea.** Hit one server simultaneously with a "background decode stream" (short prompts, long outputs, fixed concurrency) and a "long-prompt stream"; vary the token budget and watch the tail TPOT / ITL of the background stream trade off against the TTFT of the long prompts. Run the two streams in two terminals at the same time. Turn prefix caching off in every group so that repeated prompts do not hit the cache and muddy the conclusion.

**vLLM servers** (one at a time):

```bash
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --max-num-batched-tokens 2048
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --max-num-batched-tokens 8192
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --max-num-batched-tokens 16384
# With chunked prefill off, the budget must be >= max_model_len or startup fails
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 20480
```

**vLLM clients** (start both terminals together):

```bash
# Terminal 1: background decode stream, 256 input / 1024 output, concurrency 64
vllm bench serve --backend vllm --model $MODEL --dataset-name random \
  --random-input-len 256 --random-output-len 1024 \
  --num-prompts 640 --max-concurrency 64 --ignore-eos --seed 1 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
  --save-result --result-dir results/B --label bg_budget2048

# Terminal 2: long-prompt stream, 16000 input / 16 output, 0.5 req/s
vllm bench serve --backend vllm --model $MODEL --dataset-name random \
  --random-input-len 16000 --random-output-len 16 \
  --num-prompts 60 --request-rate 0.5 --ignore-eos --seed 2 \
  --percentile-metrics ttft,e2el --metric-percentiles 50,99 \
  --save-result --result-dir results/B --label long_budget2048
```

vLLM v0.30.0's bench also has `--probe-request-rate`: it additionally sends single-token probe requests at the given rate, bypassing `--max-concurrency`, and reports their latency separately — built precisely to quantify "how long the main workload stalls unrelated requests", and usable as a third measurement.

**SGLang servers** (one at a time; for the clients, replace the two commands above with `python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name random --random-range-ratio 1 ...`, keeping input / output lengths, concurrency and request rate the same):

```bash
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size 2048
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size 8192
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size -1   # chunking off
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size 2048 --enable-mixed-chunk
```

SGLang's `random` dataset samples from the ShareGPT token space and downloads that dataset on first use; to avoid the download, use `--dataset-name random-ids` (random token ids).

**How to read it.**
- Plot two lines: budget on the x-axis, and on the y-axis "ITL p99 of the background stream" and "TTFT p50 of the long prompts". Expected direction: the larger the budget, the higher the former and the lower the latter; chunking off gives the highest former.
- Compare with the table from Module 02 Lab 2: compare **direction** and **relative magnitude** (e.g. by what factor each metric changes when the budget goes from 2048 to 16384).
- The SGLang `--enable-mixed-chunk` group is interesting: without mixing (the default), prefill chunks and decodes are batched separately, so the background stream's ITL pattern differs from vLLM's. The simulator implements only the "mixed" variant.

## 5. Experiment C: PD disaggregation

**Be clear about what this can show.** Two GPUs allow only 1 P + 1 D, which is exactly the unfavourable case DistServe itself concedes (few GPUs, no room to tune the ratio), so the 2-GPU experiment only **observes isolation**: whether long prefills still push up tail ITL, and how much KV transfer adds to TTFT. To test goodput / throughput conclusions you need at least 4–8 GPUs, sweeping the P : D ratio (e.g. 1:3, 2:2, 3:1) and the input / output length ratio, and the conclusion should rest on "goodput versus request rate" curves rather than a single total-throughput number. The 2-GPU commands come first, followed by how to scale to more instances.

**vLLM: NixlConnector on one machine with two GPUs.** The commands below come from vLLM's NixlConnector usage guide; the proxy script lives under `tests/` in the vLLM source repository, so first `git clone --branch v0.30.0 https://github.com/vllm-project/vllm.git`.

```bash
uv pip install nixl

# Prefill instance (GPU 0)
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve $MODEL --port 8100 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# Decode instance (GPU 1)
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve $MODEL --port 8200 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# Proxy: send to P first, then to D
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200

# Baseline: the same 2 GPUs, data parallel
vllm serve $MODEL --port 8000 --data-parallel-size 2 --enforce-eager
```

The docs' example includes `--enforce-eager` (CUDA Graphs off), which slows decode; for a performance comparison **the baseline and the PD group must use the same such flags** (either both with it or both without).

```bash
# Hit the PD proxy (8192) and the baseline (8000) separately
vllm bench serve --backend vllm --model $MODEL --port 8192 --dataset-name random \
  --random-input-len 8000 --random-output-len 256 \
  --num-prompts 300 --request-rate 2 --ignore-eos --seed 3 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --goodput ttft:2000 tpot:50 \
  --save-result --result-dir results/C --label pd_1p1d
```

**SGLang: Mooncake or NIXL transfer.** The single-machine example from the SGLang docs (the Mooncake backend needs an RDMA NIC; put your device name in `--disaggregation-ib-device`):

```bash
python -m sglang.launch_server --model-path $MODEL --disaggregation-mode prefill \
  --port 30000 --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server --model-path $MODEL --disaggregation-mode decode \
  --port 30001 --base-gpu-id 1 --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000

# Baseline: data parallel on the same 2 GPUs
python -m sglang.launch_server --model-path $MODEL --port 30000 --dp-size 2

# Benchmark PD (the bench has a dedicated --pd-separated switch)
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 8000 --pd-separated \
  --dataset-name random --random-input-len 8000 --random-output-len 256 --random-range-ratio 1 \
  --num-prompts 300 --request-rate 2 --output-file results/C/sglang_pd.jsonl --output-details
```

In v0.5.20 the choices for `--disaggregation-transfer-backend` include `mooncake` (the default) and `nixl`, among others; whether `nixl` works on a single machine without RDMA depends on how NIXL / UCX were built, and this guide has not tested it.

**Scaling to more instances and sweeping the P : D ratio.** The vLLM toy proxy's `--prefiller-hosts / --prefiller-ports / --decoder-hosts / --decoder-ports` all accept multiple values; for example 2 P + 2 D on 4 GPUs:

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost localhost --prefiller-ports 8100 8101 \
  --decoder-hosts localhost localhost --decoder-ports 8200 8201
```

Give each instance its own `CUDA_VISIBLE_DEVICES`, port and `VLLM_NIXL_SIDE_CHANNEL_PORT` (the NixlConnector docs require a unique side-channel port per worker on the same host). In the SGLang router `--prefill` / `--decode` are list arguments; repeat them to attach more instances.

**How to read it.**
- For each ratio, sweep `--request-rate` (e.g. 1, 2, 4, 6) and record TTFT p99, ITL p99, output throughput and goodput (`--goodput ttft:<ms> tpot:<ms>`); also scrape `vllm:num_requests_waiting` from each instance's `/metrics` (which side is queueing, P or D), NixlConnector's transfer metrics (see the NIXL section of the vLLM metrics docs) and GPU utilization.
- Expected: the PD group's ITL p99 is steadier. Whether goodput beats a DP baseline with the same GPU count depends on whether the ratio fits the workload — with long-input, short-output traffic such as 8000 in / 256 out, P-side queues blow up first when P is short (the ratio mental math in Module 02 Section 9). Read the vLLM docs' "Disaggregated prefill DOES NOT improve throughput" in the context of its design goals (tuning TTFT and ITL separately, controlling tail ITL); do not generalize a 2-GPU 1:1 result into "PD disaggregation is useless" — DistServe's goodput gains and Splitwise's same-cost throughput gains both rely on the right ratio, scale or heterogeneous hardware.
- Work out the KV transfer volume: 8000 tokens × KV bytes per token (128 KiB for Llama-3.1-8B) ≈ 1 GiB per request, all over one GPU's link with TP1. The TTFT increment will look very different over NVLink / PCIe within a node versus RDMA across nodes.
- The simulator's `codesign_sim.pd` can make the same comparison: Lab 4 of Module 02 compares colocated × 4 with 2P+2D / 3P+1D on the same 32 GPUs, in goodput under two SLOs. Check the **direction** of your real goodput-versus-request-rate curves against it — with a strict TPOT SLO, P/D with the right ratio wins; with a loose SLO, colocation wins; with the wrong ratio, P/D loses under any SLO; the simulator's P/D simplifications are listed in Section 8.

## 6. Experiment D: multi-turn agent replay and reuse distance

**Idea.** Replay an agent workload as multi-turn sessions: each session has a common prefix shared by all sessions, its own unique prefix, and user messages and model replies appended turn by turn. Changing "the number of simultaneously active sessions" changes how many other requests land between two visits to the same session (reuse distance); watch how the token-level hit rate and TTFT respond.

**vLLM: `benchmarks/multi_turn/benchmark_serving_multi_turn.py`** (in the v0.30.0 source repository; download `pg1184.txt` as its README describes to generate synthetic conversations):

```bash
vllm serve $MODEL --port 8000                       # to see the effect of capacity, shrink the KV pool with --gpu-memory-utilization
cd vllm/benchmarks/multi_turn && pip install -r requirements.txt
# Edit generate_multi_turn.json: common_prefix_num_tokens (shared by all sessions), prefix_num_tokens (unique per session),
# num_turns, per-turn num_tokens, output num_tokens
python benchmark_serving_multi_turn.py --model $MODEL --url http://localhost:8000 \
  --input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6 \
  --stats-json-output results/D/stats_active6.json
```

Sweep `--max-active-conversations` (e.g. 6, 24, 96, 384): clients cycle through their active sessions (the default `--conversation-sampling round_robin`), so the more sessions are active, the more other requests fall between two turns of the same session. `--request-rate` (a per-client Poisson rate) can emulate the gaps from tool waits. Scrape `/metrics` before and after each run to compute the token-level hit rate.

**SGLang:** `python -m sglang.bench_serving --backend sglang-oai-chat --dataset-name agentic-trace --dataset-path <trace.json> --cache-report` replays multi-turn agent traces round by round, feeding the server's real replies into the next round; you have to bring your own trace, in the format described in the docstring of `python/sglang/benchmark/datasets/agentic_trace.py` in the v0.5.20 source (OpenHands / SWE-smith style). Without a trace, `--gsp-num-turns` of the `generated-shared-prefix` dataset also produces multi-turn questions sharing a system prefix.

**How to read it.** Plot "token-level hit rate / TTFT p50 / TTFT p99 versus the number of active sessions". Expected: once active sessions × KV per session exceed the free part of the KV pool, the hit rate starts to fall and TTFT climbs steeply (the residency formula in Module 02 Section 4.1); shrink the KV pool and the knee moves left. This group pairs with the cache-policy experiments in [Module 05](../modules/05_cache_prefill/README.en.md).

## 7. How to compare with the simulator

| Simulator (`codesign_sim`) | vLLM v0.30.0 | SGLang v0.5.20 |
|---|---|---|
| `EngineConfig.max_batch_tokens` | `--max-num-batched-tokens` | `--chunked-prefill-size` |
| `chunked_prefill=True` with `max_batch_tokens` ≥ the longest prompt (decode first, effectively no chunking; row 2 of Module 02 Lab 1) | `--no-enable-chunked-prefill` (the budget must then be ≥ `--max-model-len`) | — |
| `chunked_prefill=False` (prefill first, decodes paused in that step; row 1 of Lab 1) | No counterpart (V0 / Orca style) | `--chunked-prefill-size -1` (no mixing by default: a new prefill batch runs first) |
| `Request.cached_tokens / prompt_tokens` | `Δvllm:prefix_cache_hits / Δvllm:prefix_cache_queries` | `Cache hit rate` from `--cache-report` |
| `EngineConfig.n_gpus` (TP communication counted only if `tp` is set) | `--tensor-parallel-size` | `--tp-size` |
| `EngineConfig.kv_capacity_tokens` | Startup log `GPU KV cache size: N tokens` | `#tokens: N` of the KV pool in the startup log |
| `preemptions` with `EngineConfig(kv_reserve="incremental")` | `vllm:num_preemptions` | — |
| Closed-loop capacity in Lab 2 (64 concurrent clients) | Output throughput of `vllm bench serve --request-rate inf --max-concurrency 64` | Output throughput of `python -m sglang.bench_serving --max-concurrency 64` |
| `pd.simulate_pd(n_prefill, n_decode)` / `pd.simulate_colocated(n_instances)` (Lab 4) | Experiment C's multi-instance proxy / `--data-parallel-size` baseline | Experiment C's router / `--dp-size` baseline |
| Results `ttft_p50` / `ttft_p99` / `tpot_p50` / `tpot_p99` / `itl_max` | Bench TTFT / TPOT / ITL percentiles | Bench TTFT / TPOT / ITL percentiles |

**Three steps to compare.**
1. **Calibrate**: run two single-request groups with `--max-concurrency 1` — the TTFT of a long prompt with 1 output token ≈ one prefill; the TPOT of a short prompt with a long output ≈ a batch-1 decode step. Dividing the theoretical `2 × parameters × prompt length / peak FLOPS` by the former gives the achieved compute utilization; `weight bytes / TPOT / HBM bandwidth` gives the achieved bandwidth utilization. Compare them with the simulator's default assumptions (`mfu=0.6`, `mbu=0.8` in `step_time_s`).
2. **Compare relative quantities**: "the TTFT ratio between cache on and off", "by what factor TPOT p99 changes when the budget doubles", "at which request rate the knee appears" — not absolute milliseconds.
3. **Explain the differences**: go through Section 8 line by line; most discrepancies can be attributed to one of them.

## 8. Known deviations between the simulator and real systems

As of the current codesign_sim (behaviour changes are listed in Section 11 of the [API docs](codesign-sim.en.md)):

| Deviation | Simulator | Real systems | Effect on conclusions |
|---|---|---|---|
| KV allocation | Default `kv_reserve="oracle"`: reserves prompt + the **true output length** at admission (information a real scheduler does not have), never preempts; with `"incremental"` KV grows on demand and, when memory runs out, the most recently admitted sequence is preempted and recomputed | Blocks allocated on demand; vLLM V1 only has recompute preemption | The default mode shows no preemption spikes; use `incremental` to compare preemption |
| Scheduling | `chunked_prefill=False` means prefill first with decodes paused; with chunking on, decode first and always mixed | vLLM V1 is always decode-first + mixing (turning chunking off only stops splitting); SGLang defaults to prefill first without mixing, where chunking only caps each prefill step (`--enable-mixed-chunk` turns mixing on) | Pick the matching row: vLLM with chunking off ↔ the large-budget row; SGLang's default ↔ the prefill-first row (though SGLang still chunks) |
| Step time | Roofline: max(compute, memory) with constant efficiencies | Kernel efficiency varies with shape (tile alignment, poor efficiency at small batches); chunking adds overhead (Sarathi-Serve reports at most about 25% at chunk 512) | The simulator is optimistic at small budgets |
| Communication | Not counted by default; setting `tp` / `ep` adds a simplified bandwidth + latency model, without DeepSeek-style same-node dedup + NVLink forwarding | 2 all-reduces per layer (TP); all-to-all for MoE | The default configuration is optimistic with multiple GPUs, more so at larger TP |
| Fixed overheads | By default no scheduling, CUDA Graph / kernel launch, sampling, detokenization or HTTP costs (`step_overhead_s` can add a constant) | All present; client-measured TTFT has a floor of a few milliseconds | Largest relative error for short requests and small models; small budgets pay the per-step cost more often, so the real capacity-optimal budget is larger than the simulator's (near the knee) |
| Prefix caching | `cached_tokens` is given by the caller, exact to the token; the single-instance engine does not model eviction (the multi-instance router does) | vLLM caches only full blocks; the cache can be evicted before reuse; hashing has a cost | Real hit rate ≤ nominal sharing ratio |
| P/D disaggregation | `pd.py` is a minimal model: the decode side counts prompt + generated tokens against KV capacity and queues (or preempts and recomputes) when it runs out; no DP / EP inside the pools; transfer does not contend with compute for links; with layer-wise pipelining only about 1/n_layers of the transfer is exposed | Transfer queueing, resharding when P and D use different parallelism, link contention | P/D in the simulator is optimistic, especially under high load |
| Output length | Fixed | Decided by EOS (unless `--ignore-eos`) | Always benchmark with fixed lengths before comparing |

## 9. References (all matching the versions above)

- vLLM Benchmark CLI (`vllm bench serve`, datasets, metric definitions, cache-contamination warning): <https://docs.vllm.ai/en/v0.30.0/benchmarking/cli.html>
- vLLM bench serve flag definitions: <https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/benchmarks/serve.py>; dataset flags: <https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/benchmarks/datasets/datasets.py>
- vLLM optimization and tuning (preemption, chunked prefill, `max_num_batched_tokens`): <https://docs.vllm.ai/en/v0.30.0/configuration/optimization.html>
- vLLM's default `max_num_batched_tokens` chosen by GPU memory (`get_batch_defaults`): <https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py>
- vLLM automatic prefix caching and its design doc: <https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching.html>, <https://docs.vllm.ai/en/v0.30.0/design/prefix_caching.html>
- vLLM metrics (Prometheus `/metrics`): <https://docs.vllm.ai/en/v0.30.0/usage/metrics.html>; counter definitions: <https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/metrics/loggers.py>
- vLLM PD disaggregation and NixlConnector: <https://docs.vllm.ai/en/v0.30.0/features/disagg_prefill.html>, <https://docs.vllm.ai/en/v0.30.0/features/nixl_connector_usage.html>
- vLLM installation: <https://docs.vllm.ai/en/v0.30.0/getting_started/quickstart.html>
- SGLang Bench Serving guide: <https://docs.sglang.io/docs/developer_guide/bench_serving>; flag definitions: <https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/benchmark/serving.py>
- SGLang PD disaggregation: <https://docs.sglang.io/docs/advanced_features/pd_disaggregation>
- SGLang server flags (`--chunked-prefill-size`, `--enable-mixed-chunk`, `--disable-radix-cache`, PD flags): <https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/arg_groups/fields>
- SGLang's default `--chunked-prefill-size` by GPU memory: <https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/arg_groups/memory_hook.py>
- SGLang installation: <https://github.com/sgl-project/sglang/blob/v0.5.20/docs/docs/get-started/install.mdx>
- vLLM multi-turn benchmark (Experiment D): <https://github.com/vllm-project/vllm/tree/v0.30.0/benchmarks/multi_turn>; PD toy proxy: <https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py>
- SGLang agentic-trace dataset format: <https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/benchmark/datasets/agentic_trace.py>
- DistServe (definition of goodput): <https://arxiv.org/abs/2401.09670>; Sarathi-Serve (chunking overhead): <https://arxiv.org/abs/2403.02310>

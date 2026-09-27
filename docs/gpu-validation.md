**中文** · [English](gpu-validation.en.md)

# GPU 验证指南：用 vLLM / SGLang 官方 benchmark 复现本项目的结论

> 本项目的实验都跑在 `codesign_sim` 这个成本模型上。等你租到 GPU，用这份指南在真实引擎上复现四条核心结论，并且知道模拟器和真实系统会在哪里对不上。
>
> **版本基线**：vLLM **v0.30.0**（PyPI 2026-09-22）、SGLang **v0.5.20**（PyPI 2026-09-18）。下面每一条命令和参数名都对照过这两个版本的官方文档和源码（链接汇总在第 9 节）。换版本后参数可能改名，请先 `vllm bench serve --help` / `python -m sglang.bench_serving --help` 核对。
>
> 本指南的命令**没有在 GPU 上实际跑过**（本项目的开发机没有 NVIDIA GPU），所以这里不给任何「预期毫秒数」，只给要记录什么、怎么和模拟器对照。

## 0. 要复现的四条结论

| # | 结论 | 本项目出处 | 真实系统里的开关 |
|---|---|---|---|
| A | 前缀缓存命中率越高，TTFT 越低；负载接近饱和时，命中率对 TTFT 尾部的影响是**非线性**的（少算的 prefill 还会减少排队） | [模块 02](../modules/02_inference_engines/README.md) 第 4 节与实验 3 | vLLM `--enable-prefix-caching` / `--no-enable-prefix-caching`；SGLang `--disable-radix-cache` |
| B | token 预算（chunk 大小）是 TTFT 和 TPOT 之间的旋钮：预算越大，同批 decode 的 TPOT / ITL 尖刺越大，关掉 chunked prefill 时最严重；预算太小则 prefill 被饿死、TTFT 暴涨。vLLM 文档还指出大预算有利于 TTFT——模拟器没有分块开销，这一点在模拟里看不出来，正好用真机验证 | 模块 02 第 5 节与实验 1、2 | vLLM `--max-num-batched-tokens`、`--no-enable-chunked-prefill`；SGLang `--chunked-prefill-size`、`--enable-mixed-chunk` |
| C | PD 分离首先改善 ITL 尾部的隔离与可控性；它能否提升 SLO 下的 goodput，取决于 P : D 配比是否匹配负载——同构、卡少、不设 SLO 时不提升峰值吞吐 | 模块 02 第 9 节 | vLLM `--kv-transfer-config`（NixlConnector）；SGLang `--disaggregation-mode` |
| D | agent 多轮负载的命中率由复用距离决定：同时活跃的会话越多、两次使用之间写入的 KV 越多，前缀越容易在复用前被淘汰，TTFT 随之上升 | 模块 02 第 4.1 节、[模块 05](../modules/05_cache_prefill/README.md) | vLLM `--gpu-memory-utilization`（改变 KV 池大小）、多轮 benchmark 的 `--max-active-conversations` |

## 1. 准备

**硬件。** A、B 两个实验 1 张 80 GB 的卡（H100 / A100）就够；C 需要同一台机器上的 2 张卡（跨机需要 RDMA 网卡）。

**模型。** 下文用 `$MODEL` 表示。两家文档的示例都用 `meta-llama/Llama-3.1-8B-Instruct`（需要在 Hugging Face 上申请访问），换成任何 8B 级别的模型都可以；重点是**两边用同一个模型、同一个 dtype**。

**安装**（分别在两个虚拟环境里，避免依赖冲突）：

```bash
# vLLM（官方 quickstart 的写法，指定版本）
uv pip install vllm==0.30.0 --torch-backend=auto

# SGLang（官方安装文档的写法，指定版本）
uv pip install --prerelease=allow "sglang==0.5.20"
```

## 2. 测量纪律（先读这一节）

1. **缓存污染**：vLLM 文档专门警告过——对同一个 server 重复跑 `vllm bench serve` 会命中上一轮留下的前缀缓存；random 数据集在固定 `--seed`（默认 0）时每次生成的 prompt 一样。做法：每组实验之间重启 server，或换 `--seed`，或用 `vllm bench sweep serve`（会在两次运行之间重置缓存）。vLLM 的 `/reset_prefix_cache` 接口只有在 server 以 `VLLM_SERVER_DEV_MODE=1` 启动时才注册。SGLang 的 bench 有 `--flush-cache`（调 `/flush_cache`）。
2. **固定输出长度**：vLLM 加 `--ignore-eos`；SGLang 的 bench 默认就忽略 EOS（关掉用 `--disable-ignore-eos`）。
3. **长度分布的坑**：两家 `--random-range-ratio` 的语义**不一样**。vLLM 在 `[L·(1−r), L·(1+r)]` 里采样，默认 r = 0 即定长；SGLang 在 `[L·r, L]` 里采样，默认 r = 0 即 **1 到 L 均匀分布**。想要定长，SGLang 要显式写 `--random-range-ratio 1`。
4. **预热**：vLLM `--num-warmups N`（默认 0）；SGLang `--warmup-requests N`（默认 1）。
5. **保存原始数据**：vLLM `--save-result --save-detailed --result-dir DIR --label NAME`；SGLang `--output-file FILE.jsonl --output-details`。只记均值没有意义，至少要 p50 和 p99。
6. **指标口径**：两家的 TPOT 都是 `(端到端延迟 − TTFT) / (输出 token 数 − 1)`，按请求算再聚合；ITL 是相邻两次流式输出之间的间隔。开了投机解码时一次流式输出可能含多个 token，ITL 和 TPOT 会明显不同（vLLM 文档有图解）。模拟器的 TPOT 用的是同一个定义。

### 每次运行都要记下的东西

| 类别 | 记录项 | 从哪里拿 |
|---|---|---|
| 配置 | 引擎与版本、GPU 型号与数量、模型、dtype、所有非默认启动参数、bench 命令 | 自己记；vLLM 可用 `--metadata key=value` 写进结果 JSON |
| 延迟 | TTFT p50 / p99、TPOT p50 / p99、ITL p99 | bench 输出；vLLM 用 `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99` |
| 吞吐 | 输出 token 吞吐、请求吞吐；有 SLO 时记 goodput | bench 输出；vLLM `--goodput ttft:<ms> tpot:<ms>`（DistServe 的定义） |
| 缓存 | 前缀缓存命中率（按 token） | vLLM：`/metrics` 里 `vllm:prefix_cache_hits` 与 `vllm:prefix_cache_queries` 两个计数器（Prometheus 暴露时带 `_total` 后缀）跑前跑后的差值之比；SGLang：bench 加 `--cache-report`，打印 `Cache hit rate` |
| 显存压力 | 抢占次数、KV 使用率、排队请求数 | vLLM：`vllm:num_preemptions`、`vllm:kv_cache_usage_perc`、`vllm:num_requests_waiting`，以及周期性统计日志里的 `Preemptions: N`（官方调参文档里的 `PreemptionMode.RECOMPUTE` 样例日志来自 V0，v0.30.0 不再打印） |

## 3. 实验 A：前缀缓存与 TTFT

**思路。** 构造「P 个共享前缀 × 每个前缀 N 个请求」的负载，分别在开 / 关前缀缓存的 server 上跑，对比 TTFT，并用命中率去对照模拟器里的 `cached_tokens / prompt_tokens`。

**vLLM**（`prefix_repetition` 数据集：每个请求 = 某个共享前缀 + 随机后缀）：

```bash
# 服务端：V1 默认开启前缀缓存；对照组加 --no-enable-prefix-caching
vllm serve $MODEL --port 8000
vllm serve $MODEL --port 8000 --no-enable-prefix-caching

# 客户端：5 个前缀 × 20 个请求，前缀 4096 + 后缀 512 token，输出 128 token
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

**SGLang**（`generated-shared-prefix` 数据集：长 system prompt + 短问题）：

```bash
# 服务端：默认开启 radix cache；对照组加 --disable-radix-cache
python -m sglang.launch_server --model-path $MODEL --port 30000
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache

python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --model $MODEL \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 5 --gsp-prompts-per-group 20 \
  --gsp-system-prompt-len 4096 --gsp-question-len 512 --gsp-output-len 128 \
  --request-rate 4 --flush-cache --cache-report \
  --output-file results/A/sglang.jsonl --output-details
```

（v0.5.20 里 `sglang.bench_serving` 已标记为旧路径，会打印 FutureWarning，新路径是 `python -m sglang.benchmark.serving`，参数相同。）

**怎么读。**
- 理论命中率上限：每个前缀的第一个请求必然不命中，所以上限 ≈ (100 − 5) / 100 × 4096 / 4608 ≈ **84%**（按 token）。vLLM 还只缓存写满的块，再扣掉一点。实测明显低于这个值，通常是因为同一前缀的请求在第一个请求 prefill 完之前就到了——提高请求率就能看到命中率下降。
- 扫 `--request-rate`（如 1、2、4、8），画「开 / 关缓存的 TTFT p99 随请求率的曲线」。关缓存的曲线会更早拐上去——这就是结论 A 里的「非线性」。
- **和模拟器对照**：把实测命中率代入模块 02 实验 3 的 `cached_tokens`，比较「开 / 关缓存的 TTFT 比值」和「曲线拐点出现在哪个请求率附近」，**不要比较绝对毫秒数**。

## 4. 实验 B：chunked prefill 与 token 预算

**思路。** 一路「背景 decode 流」（短 prompt、长输出、固定并发）加一路「长 prompt 流」同时打到同一个 server 上；改变 token 预算，看背景流的 TPOT / ITL 尾部和长 prompt 的 TTFT 怎样此消彼长。两路流用两个终端同时跑。所有组都关掉前缀缓存，避免重复 prompt 命中缓存干扰结论。

**vLLM 服务端**（每次起一个）：

```bash
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --max-num-batched-tokens 2048
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --max-num-batched-tokens 8192
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --max-num-batched-tokens 16384
# 关闭 chunked prefill 时，预算必须 >= max_model_len，否则启动报错
vllm serve $MODEL --max-model-len 20480 --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 20480
```

**vLLM 客户端**（两个终端同时启动）：

```bash
# 终端 1：背景 decode 流，256 输入 / 1024 输出，并发 64
vllm bench serve --backend vllm --model $MODEL --dataset-name random \
  --random-input-len 256 --random-output-len 1024 \
  --num-prompts 640 --max-concurrency 64 --ignore-eos --seed 1 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
  --save-result --result-dir results/B --label bg_budget2048

# 终端 2：长 prompt 流，16000 输入 / 16 输出，0.5 req/s
vllm bench serve --backend vllm --model $MODEL --dataset-name random \
  --random-input-len 16000 --random-output-len 16 \
  --num-prompts 60 --request-rate 0.5 --ignore-eos --seed 2 \
  --percentile-metrics ttft,e2el --metric-percentiles 50,99 \
  --save-result --result-dir results/B --label long_budget2048
```

vLLM v0.30.0 的 bench 还有 `--probe-request-rate`：按给定速率额外发送只生成 1 个 token 的探测请求，绕过 `--max-concurrency`，单独报告它们的延迟——专门用来量化「主负载把无关请求卡住多久」，可以作为第三路测量。

**SGLang 服务端**（每次起一个；客户端把上面两条命令换成 `python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name random --random-range-ratio 1 ...`，输入输出长度、并发、请求率保持一致）：

```bash
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size 2048
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size 8192
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size -1   # 关闭分块
python -m sglang.launch_server --model-path $MODEL --port 30000 --disable-radix-cache --chunked-prefill-size 2048 --enable-mixed-chunk
```

SGLang 的 `random` 数据集会从 ShareGPT 的 token 空间采样，第一次运行会下载数据集；不想下载可以用 `--dataset-name random-ids`（随机 token id）。

**怎么读。**
- 画两条线：横轴是预算，纵轴分别是「背景流的 ITL p99」和「长 prompt 的 TTFT p50」。预期方向：预算越大，前者越高、后者越低；关闭分块时前者最高。
- 对照模块 02 实验 2 的表格：比较**方向**和**相对幅度**（例如预算从 2048 到 16384 时两个指标各变了几倍）。
- SGLang 的 `--enable-mixed-chunk` 这一组很有意思：默认不混合时，prefill 块和 decode 分开成批，背景流的 ITL 形态会和 vLLM 不同。模拟器只实现了「混合」这一种。

## 5. 实验 C：PD 分离

**定位先说清楚。** 2 张卡只能做 1 P + 1 D，这正是 DistServe 自己承认的不利场景（卡少、没有调配比的空间），所以 2 卡实验只用来**观察隔离效果**：长 prefill 还会不会拉高 ITL 尾部、TTFT 多了多少 KV 传输时间。要验证 goodput / 吞吐的结论，至少要 4–8 张卡，扫 P : D 配比（如 1:3、2:2、3:1）和输入输出长度比，结论以「goodput 随请求率变化的曲线」为主，而不是单点的总吞吐。下面先给 2 卡的命令，再给多实例的扩展方式。

**vLLM：NixlConnector，单机两卡。** 以下命令来自 vLLM 的 NixlConnector 使用指南；proxy 脚本在 vLLM 源码仓库的 `tests/` 目录下，需要先 `git clone --branch v0.30.0 https://github.com/vllm-project/vllm.git`。

```bash
uv pip install nixl

# prefill 实例（GPU 0）
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve $MODEL --port 8100 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# decode 实例（GPU 1）
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve $MODEL --port 8200 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# proxy：先发给 P，再发给 D
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200

# 基线：同样 2 张卡，数据并行
vllm serve $MODEL --port 8000 --data-parallel-size 2 --enforce-eager
```

文档示例带了 `--enforce-eager`（关闭 CUDA Graph），这会拖慢 decode；做性能对比时，**基线和 PD 两组的这类参数必须一致**（要么都加，要么都去掉）。

```bash
# 分别打 PD 的 proxy（8192）和基线（8000）
vllm bench serve --backend vllm --model $MODEL --port 8192 --dataset-name random \
  --random-input-len 8000 --random-output-len 256 \
  --num-prompts 300 --request-rate 2 --ignore-eos --seed 3 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --goodput ttft:2000 tpot:50 \
  --save-result --result-dir results/C --label pd_1p1d
```

**SGLang：Mooncake 或 NIXL 传输。** SGLang 文档的单机示例（Mooncake 后端需要 RDMA 网卡，`--disaggregation-ib-device` 填你的设备名）：

```bash
python -m sglang.launch_server --model-path $MODEL --disaggregation-mode prefill \
  --port 30000 --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server --model-path $MODEL --disaggregation-mode decode \
  --port 30001 --base-gpu-id 1 --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000

# 基线：同样 2 张卡的数据并行
python -m sglang.launch_server --model-path $MODEL --port 30000 --dp-size 2

# 压测 PD（bench 有专门的 --pd-separated 开关）
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 8000 --pd-separated \
  --dataset-name random --random-input-len 8000 --random-output-len 256 --random-range-ratio 1 \
  --num-prompts 300 --request-rate 2 --output-file results/C/sglang_pd.jsonl --output-details
```

`--disaggregation-transfer-backend` 在 v0.5.20 的可选值里有 `mooncake`（默认）和 `nixl` 等；没有 RDMA 时能否用 `nixl` 在单机上跑通，取决于 NIXL / UCX 的构建，本指南没有实测。

**扩展到多实例、扫 P : D 配比。** vLLM 的 toy proxy 的 `--prefiller-hosts / --prefiller-ports / --decoder-hosts / --decoder-ports` 都接受多个值，例如 4 张卡做 2 P + 2 D：

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost localhost --prefiller-ports 8100 8101 \
  --decoder-hosts localhost localhost --decoder-ports 8200 8201
```

每个实例用自己的 `CUDA_VISIBLE_DEVICES`、端口和 `VLLM_NIXL_SIDE_CHANNEL_PORT`（NixlConnector 文档要求同一台机器上每个 worker 的 side channel 端口不同）。SGLang 的 router 里 `--prefill` / `--decode` 是列表参数，重复写即可挂多个实例。

**怎么读。**
- 对每个配比扫 `--request-rate`（如 1、2、4、6），记录 TTFT p99、ITL p99、输出吞吐和 goodput（`--goodput ttft:<ms> tpot:<ms>`）；同时抓各实例 `/metrics` 里的 `vllm:num_requests_waiting`（P 和 D 哪边在排队）、NixlConnector 的传输指标（见 vLLM 指标文档的 NIXL 一节）和 GPU 利用率。
- 预期：PD 组的 ITL p99 更平稳。goodput 能否超过同卡数的 DP 基线，取决于配比是否匹配负载——8000 输入 / 256 输出这种长输入短输出的负载，P 不够时 P 侧排队会先爆（模块 02 第 9 节的配比心算）。vLLM 文档那句「Disaggregated prefill DOES NOT improve throughput」放在它的设计目标（分别调 TTFT 与 ITL、控制尾部 ITL）的语境里读；不要把 2 卡 1:1 的结果推广成「PD 分离没用」——DistServe 的 goodput 提升和 Splitwise 的同成本吞吐提升，都依赖合适的配比、规模或异构硬件。
- 算一下 KV 传输量：8000 token × 每 token KV 字节（Llama-3.1-8B 为 128 KiB）≈ 1 GiB / 请求，TP1 时全部走一张卡的链路。单机走 NVLink / PCIe 和跨机走 RDMA，TTFT 的增量会很不一样。
- 模拟器的 `codesign_sim.pd` 可以做同样的对照：模块 02 实验 4 在同样 32 张卡上比较共置 × 4 与 2P+2D / 3P+1D 在两种 SLO 下的 goodput。拿真机的「goodput 随请求率」曲线去对它的**方向**——严格 TPOT SLO 时配比合适的 P/D 胜、宽松 SLO 时共置胜、配比错时 P/D 在任何 SLO 下都输；模拟器对 P/D 的简化见第 8 节。

## 6. 实验 D：agent 多轮回放与复用距离

**思路。** 用多轮会话回放 agent 负载：每个会话有所有会话共享的公共前缀、自己独有的前缀，以及逐轮追加的用户消息和模型回复。改变「同时活跃的会话数」就改变了两次访问同一会话之间插进来的其他请求数（复用距离），观察按 token 的命中率和 TTFT 怎么变。

**vLLM：`benchmarks/multi_turn/benchmark_serving_multi_turn.py`**（在 v0.30.0 源码仓库里；按其 README 下载 `pg1184.txt` 用于生成合成会话）：

```bash
vllm serve $MODEL --port 8000                       # 想看容量的影响，用 --gpu-memory-utilization 调小 KV 池
cd vllm/benchmarks/multi_turn && pip install -r requirements.txt
# 编辑 generate_multi_turn.json：common_prefix_num_tokens（所有会话共享）、prefix_num_tokens（每个会话独有）、
# num_turns、每轮 num_tokens、输出 num_tokens
python benchmark_serving_multi_turn.py --model $MODEL --url http://localhost:8000 \
  --input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6 \
  --stats-json-output results/D/stats_active6.json
```

扫 `--max-active-conversations`（如 6、24、96、384）：客户端在活跃会话之间轮流发请求（默认 `--conversation-sampling round_robin`），活跃会话越多，同一会话两轮之间插入的其他请求越多。`--request-rate`（每个客户端的泊松到达率）可以模拟工具等待带来的间隔。每组前后抓 `/metrics` 算按 token 的命中率。

**SGLang：** `python -m sglang.bench_serving --backend sglang-oai-chat --dataset-name agentic-trace --dataset-path <trace.json> --cache-report` 按轮回放多轮 agent trace，把服务器的真实回复带进下一轮；trace 需要自备，格式见 v0.5.20 源码 `python/sglang/benchmark/datasets/agentic_trace.py` 的 docstring（OpenHands / SWE-smith 风格）。没有 trace 时，`generated-shared-prefix` 数据集的 `--gsp-num-turns` 也能生成共享系统前缀的多轮问题。

**怎么读。** 画「按 token 的命中率 / TTFT p50 / TTFT p99 随活跃会话数变化」的曲线。预期：活跃会话数乘以每个会话的 KV 超过 KV 池的空闲部分后，命中率开始下降、TTFT 陡增（模块 02 第 4.1 节的驻留时间公式）；把 KV 池调小，拐点左移。这一组对照 [模块 05](../modules/05_cache_prefill/README.md) 的缓存策略实验。

## 7. 和模拟器怎么对照

| 模拟器（`codesign_sim`） | vLLM v0.30.0 | SGLang v0.5.20 |
|---|---|---|
| `EngineConfig.max_batch_tokens` | `--max-num-batched-tokens` | `--chunked-prefill-size` |
| `chunked_prefill=True` 且 `max_batch_tokens` ≥ 最长 prompt（decode 优先、实际不分块，模块 02 实验 1 第 2 行） | `--no-enable-chunked-prefill`（此时预算必须 ≥ `--max-model-len`） | — |
| `chunked_prefill=False`（prefill 优先、这一步 decode 暂停，实验 1 第 1 行） | 无对应（V0 / Orca 式） | `--chunked-prefill-size -1`（默认不混合：有新 prefill 批时先跑它） |
| `Request.cached_tokens / prompt_tokens` | `Δvllm:prefix_cache_hits / Δvllm:prefix_cache_queries` | `--cache-report` 的 `Cache hit rate` |
| `EngineConfig.n_gpus`（设 `tp` 才计 TP 通信） | `--tensor-parallel-size` | `--tp-size` |
| `EngineConfig.kv_capacity_tokens` | 启动日志 `GPU KV cache size: N tokens` | 启动日志里 KV 池的 `#tokens: N` |
| `EngineConfig(kv_reserve="incremental")` 的 `preemptions` | `vllm:num_preemptions` | — |
| 实验 2 的闭环容量（64 个并发客户端） | `vllm bench serve --request-rate inf --max-concurrency 64` 的输出吞吐 | `python -m sglang.bench_serving --max-concurrency 64` 的输出吞吐 |
| `pd.simulate_pd(n_prefill, n_decode)` / `pd.simulate_colocated(n_instances)`（实验 4） | 实验 C 的多实例 proxy / `--data-parallel-size` 基线 | 实验 C 的 router / `--dp-size` 基线 |
| 结果 `ttft_p50` / `ttft_p99` / `tpot_p50` / `tpot_p99` / `itl_max` | bench 的 TTFT / TPOT / ITL 分位数 | bench 的 TTFT / TPOT / ITL 分位数 |

**对照三步。**
1. **标定**：用 `--max-concurrency 1` 跑两组单请求——长 prompt、1 个输出 token 的 TTFT ≈ 单次 prefill 时间；短 prompt、长输出的 TPOT ≈ batch 1 的 decode 步长。前者除以理论值 `2 × 参数量 × prompt 长度 / 峰值算力` 得到实测算力利用率，后者用 `权重字节 / TPOT / HBM 带宽` 得到实测带宽利用率。和模拟器默认假设（`step_time_s` 的 `mfu=0.6`、`mbu=0.8`）比一比。
2. **比相对量**：比「开 / 关缓存的 TTFT 比值」「预算翻倍时 TPOT p99 变化了几倍」「拐点出现在哪个请求率」，不比绝对毫秒数。
3. **解释差异**：对不上的地方逐条对照第 8 节，大多数偏差都能归到其中某一条。

## 8. 模拟器与真实系统的已知偏差

按 codesign_sim 当前版本（行为变化见 [API 文档](codesign-sim.md) 第 11 节）：

| 偏差 | 模拟器 | 真实系统 | 对结论的影响 |
|---|---|---|---|
| KV 分配 | 默认 `kv_reserve="oracle"`：准入时按 prompt + **真实输出长度**一次性预留（真实调度器拿不到这个信息），永不抢占；`"incremental"` 时按需增长，显存不够时抢占最后准入的序列并 recompute | 按块按需分配；vLLM V1 只有 recompute 抢占 | 默认模式看不到抢占尖刺；要对照抢占，用 `incremental` |
| 调度方式 | `chunked_prefill=False` 是 prefill 优先、decode 暂停；开启分块时 decode 优先且总是混合 | vLLM V1 始终 decode 优先 + 混合（关分块只是不切块）；SGLang 默认 prefill 优先、不混合，分块只限制每步 prefill 的大小（`--enable-mixed-chunk` 才混合） | 对照时选对行：vLLM 关分块 ↔ 大预算那一行；SGLang 默认 ↔ prefill 优先那一行（但 SGLang 还会分块） |
| 步耗时 | roofline：max(计算, 访存)，效率为常数 | kernel 效率随形状变化（tile 对齐、小 batch 效率低）；分块有额外开销（Sarathi-Serve 报告块 512 时最多约 25%） | 小预算时模拟器偏乐观 |
| 通信 | 默认不计；设 `tp` / `ep` 时加简化的带宽 + 延迟模型，不做 DeepSeek 式的同节点去重 + NVLink 转发 | 每层 2 次 all-reduce（TP）；MoE 有 all-to-all | 默认配置在多卡时偏乐观，TP 越大越明显 |
| 固定开销 | 默认不计调度、CUDA Graph / kernel launch、采样、反 token 化、HTTP（`step_overhead_s` 可补一个常数） | 都存在；客户端测到的 TTFT 有一个毫秒级的地板 | 短请求、小模型时相对误差最大；小预算要付更多次每步开销，所以真实的容量最优预算比模拟器（拐点附近）更大 |
| 前缀缓存 | `cached_tokens` 由调用方给定，按 token 精确；单实例 engine 不模拟淘汰（多实例 router 里有） | vLLM 只缓存写满的块；缓存可能在复用前被淘汰；hash 计算有开销 | 真实命中率 ≤ 名义共享比例 |
| P/D 分离 | `pd.py` 是最小模型：decode 侧按 prompt + 已生成 token 计入 KV 容量，不够时排队（或按抢占模式重算）；池内没有 DP / EP；传输与计算不争用链路；按层流水时只暴露约 1/层数 的传输 | 传输排队、P / D 并行度不同时的重新分片、链路争用 | 模拟器里的 P/D 偏乐观，高负载时尤甚 |
| 输出长度 | 固定 | 由 EOS 决定（除非 `--ignore-eos`） | 一定要用固定长度压测再对照 |

## 9. 参考（全部对应上面的版本）

- vLLM Benchmark CLI（`vllm bench serve`、数据集、指标定义、缓存污染警告）：<https://docs.vllm.ai/en/v0.30.0/benchmarking/cli.html>
- vLLM bench serve 参数定义源码：<https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/benchmarks/serve.py>、数据集参数：<https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/benchmarks/datasets/datasets.py>
- vLLM 优化与调参（抢占、chunked prefill、`max_num_batched_tokens`）：<https://docs.vllm.ai/en/v0.30.0/configuration/optimization.html>
- vLLM 默认 `max_num_batched_tokens` 按 GPU 显存选择（`get_batch_defaults`）：<https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py>
- vLLM 自动前缀缓存与设计文档：<https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching.html>、<https://docs.vllm.ai/en/v0.30.0/design/prefix_caching.html>
- vLLM 指标（Prometheus `/metrics`）：<https://docs.vllm.ai/en/v0.30.0/usage/metrics.html>；计数器定义：<https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/metrics/loggers.py>
- vLLM PD 分离与 NixlConnector：<https://docs.vllm.ai/en/v0.30.0/features/disagg_prefill.html>、<https://docs.vllm.ai/en/v0.30.0/features/nixl_connector_usage.html>
- vLLM 安装：<https://docs.vllm.ai/en/v0.30.0/getting_started/quickstart.html>
- SGLang Bench Serving 指南：<https://docs.sglang.io/docs/developer_guide/bench_serving>；参数定义源码：<https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/benchmark/serving.py>
- SGLang PD 分离：<https://docs.sglang.io/docs/advanced_features/pd_disaggregation>
- SGLang 服务端参数（`--chunked-prefill-size`、`--enable-mixed-chunk`、`--disable-radix-cache`、PD 参数）：<https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/arg_groups/fields>
- SGLang `--chunked-prefill-size` 按显存的默认值：<https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/arg_groups/memory_hook.py>
- SGLang 安装：<https://github.com/sgl-project/sglang/blob/v0.5.20/docs/docs/get-started/install.mdx>
- vLLM 多轮 benchmark（实验 D）：<https://github.com/vllm-project/vllm/tree/v0.30.0/benchmarks/multi_turn>；PD toy proxy：<https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py>
- SGLang agentic-trace 数据集格式：<https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/benchmark/datasets/agentic_trace.py>
- DistServe（goodput 的定义）：<https://arxiv.org/abs/2401.09670>；Sarathi-Serve（分块开销）：<https://arxiv.org/abs/2403.02310>

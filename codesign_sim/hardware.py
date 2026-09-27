"""GPU 规格表、互联常数与 roofline 模型。

所有数值都是 **dense**（非稀疏）算力：NVIDIA 规格页常把 2:4 结构化稀疏的数字当标称值，
这里一律除以 2 还原成 dense。每个条目的 ``source`` 字段给出核实过的来源 URL。

注意（常见误读）：H100 规格表里 BF16 "1,979 teraFLOPS" 带星号，表尾脚注是 "* With sparsity"，
所以 dense BF16 是 989 而不是 1,979；A100 表格同时列出 "312 TFLOPS | 624 TFLOPS*" 两列，可以对照口径。
FLOPs 口径全项目统一为"一次乘加 = 2 FLOPs"（与 NVIDIA 规格表、model.py 的 2·N 公式一致）。

单位约定：
- 算力 TFLOPS（1e12 FLOP/s）
- 显存带宽 TB/s（1e12 Byte/s）
- 容量 GB（1e9 Byte）
- NVLink / PCIe GB/s：每卡**双向**总带宽（单向 = 一半）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    bf16_tflops: float  # dense，非稀疏
    fp8_tflops: float | None  # dense；不支持则 None
    hbm_gb: float
    hbm_tbps: float  # TB/s
    nvlink_gbps: float  # 每卡 NVLink 双向总带宽 GB/s；没有则 0
    source: str  # 规格来源 URL


GPUS: dict[str, GPU] = {
    # A100 规格页：BF16 312 TFLOPS（624* 为稀疏），80GB HBM2e，SXM 2,039 GB/s，NVLink 600 GB/s；无 FP8
    "A100-80G": GPU(
        name="A100-80G",
        bf16_tflops=312.0,
        fp8_tflops=None,
        hbm_gb=80.0,
        hbm_tbps=2.039,
        nvlink_gbps=600.0,
        source="https://www.nvidia.com/en-us/data-center/a100/",
    ),
    # H100 规格页（SXM 列）：BF16 "1,979 teraFLOPS*"、FP8 "3,958 teraFLOPS*"，表尾脚注 "* With sparsity"
    # -> dense BF16 989、FP8 1,979；80GB，3.35 TB/s，NVLink 900 GB/s，PCIe Gen5 128 GB/s
    "H100-SXM": GPU(
        name="H100-SXM",
        bf16_tflops=989.0,
        fp8_tflops=1979.0,
        hbm_gb=80.0,
        hbm_tbps=3.35,
        nvlink_gbps=900.0,
        source="https://www.nvidia.com/en-us/data-center/h100/",
    ),
    # NVIDIA 没有公开的 H800 规格页。DeepSeek 论文（arXiv 2505.09343 §4.1）：H800 SXM 与 H100 同为 Hopper，
    # 只削减了 FP64 与 NVLink（900 -> 400 GB/s）；腾讯云实例文档列出 H800 BF16 989 TFLOPS、80GB、NVLink 400GB/s。
    # 因此 Tensor Core 算力与 HBM 取 H100 SXM 的 dense 值。
    "H800-SXM": GPU(
        name="H800-SXM",
        bf16_tflops=989.0,
        fp8_tflops=1979.0,
        hbm_gb=80.0,
        hbm_tbps=3.35,
        nvlink_gbps=400.0,
        source=(
            "https://arxiv.org/abs/2505.09343 (Sec 4.1: NVLink 400GB/s, otherwise H100-class); "
            "https://www.tencentcloud.com/document/product/1236/62812 (H800: BF16 989 TFLOPS, 80GB, NVLink 400GB/s); "
            "https://www.nvidia.com/en-us/data-center/h100/"
        ),
    ),
    # NVIDIA 没有公开的 H20 规格页，以下全部是"据报道 + 推断"：
    # - FP8 296 / FP16 148 TFLOPS、96GB HBM3、4.0TB/s 来自媒体转载的曝光参数（未说明 dense/sparse，本库按 dense 处理，属推断）
    # - 阿里云 gn8v/ebmgn8v 页面写"每卡 96 GB HBM3、4 TB/s、900 GB/s NVLINK"，但页面**没有写明型号**，
    #   按规格吻合推断为 H20；该页写的 FP32 是 39.5 TFLOPS，与媒体报道的 44 TFLOPS 不一致
    "H20": GPU(
        name="H20",
        bf16_tflops=148.0,
        fp8_tflops=296.0,
        hbm_gb=96.0,
        hbm_tbps=4.0,
        nvlink_gbps=900.0,
        source=(
            "reported + inferred, no official NVIDIA spec page: "
            "https://cloud.tencent.com/developer/news/1284500 (reported FP8 296 / FP16 148 TFLOPS, 96GB, 4.0TB/s; "
            "dense/sparse unstated, treated as dense); "
            "https://help.aliyun.com/zh/egs/gpu-accelerated-compute-optimized-instance-families "
            "(gn8v: 96GB HBM3, 4 TB/s, 900 GB/s NVLink; model name not stated, inferred to be H20)"
        ),
    ),
    # HGX B200（8 卡）：FP8 72 PF、FP16/BF16 36 PF，脚注 "Specification in Sparse. Dense is 1/2"；
    # 所以每卡 dense BF16 = 36/2/8 = 2.25 PF，FP8 = 72/2/8 = 4.5 PF；NVLink 每卡 1.8 TB/s。
    # DGX B200：1,440 GB 总显存、64 TB/s 总带宽 -> 每卡 180 GB、8 TB/s。
    "B200": GPU(
        name="B200",
        bf16_tflops=2250.0,
        fp8_tflops=4500.0,
        hbm_gb=180.0,
        hbm_tbps=8.0,
        nvlink_gbps=1800.0,
        source=(
            "https://www.nvidia.com/en-us/data-center/hgx/ ; "
            "https://www.nvidia.com/en-us/data-center/dgx-b200/"
        ),
    ),
}


# ---------------------------------------------------------------------------
# 互联常数（通信成本模型用；都可以在 EngineConfig 里覆盖）
# ---------------------------------------------------------------------------

# 每卡 400 Gb/s InfiniBand（ConnectX-7）≈ 50 GB/s 单向（DeepSeek ISCA'25，arXiv 2505.09343 §4.1）
IB_GBPS_PER_GPU = 50.0
# 链路可达带宽比例：H800 NVLink 单向 200 GB/s 实测约 160 GB/s 可用（同上 §4.1）
LINK_EFFICIENCY = 0.8
# 64B 小消息端到端延迟：NVLink 3.33 µs、IB 同 leaf 2.8 µs / 跨 leaf 3.7 µs（同上 Table 5）
NVLINK_LATENCY_S = 3.33e-6
IB_LATENCY_S = 3.7e-6
# PCIe Gen5 x16：H100 规格页写 128 GB/s（双向）-> 单向 64 GB/s；用作 HBM <-> 主机 DRAM 的 KV 搬运带宽
PCIE_GEN5_GBPS = 128.0


def peak_tflops(gpu: GPU, dtype: str = "bf16") -> float:
    """某精度下的 dense 峰值算力（TFLOPS）。dtype: "bf16" | "fp8"。"""
    if dtype == "bf16":
        return gpu.bf16_tflops
    if dtype == "fp8":
        if gpu.fp8_tflops is None:
            raise ValueError(f"{gpu.name} 不支持 FP8 Tensor Core")
        return gpu.fp8_tflops
    raise ValueError(f"未知 dtype: {dtype!r}（只支持 'bf16' / 'fp8'）")


def ridge_point(gpu: GPU, dtype: str = "bf16") -> float:
    """roofline 拐点（FLOPs/Byte）：算术强度高于它才是 compute-bound。

    TFLOPS / (TB/s) 的 1e12 恰好约掉，例如 H100 BF16：989 / 3.35 ≈ 295 FLOPs/Byte。
    """
    return peak_tflops(gpu, dtype) / gpu.hbm_tbps


def attainable_tflops(gpu: GPU, intensity: float, dtype: str = "bf16") -> float:
    """roofline：min(峰值算力, 算术强度 × 显存带宽)，单位 TFLOPS。"""
    if intensity < 0:
        raise ValueError("算术强度必须 >= 0")
    return min(peak_tflops(gpu, dtype), intensity * gpu.hbm_tbps)

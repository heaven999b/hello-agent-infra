"""hardware：规格表（dense 口径的内部一致性）、互联常数与 roofline 拐点。"""

import pytest

from codesign_sim import hardware
from codesign_sim.hardware import GPUS, attainable_tflops, peak_tflops, ridge_point


def test_required_gpus_present_with_sources():
    for name in ["A100-80G", "H100-SXM", "H800-SXM", "H20", "B200"]:
        gpu = GPUS[name]
        assert gpu.name == name
        assert "http" in gpu.source
        assert gpu.bf16_tflops > 0 and gpu.hbm_gb > 0 and gpu.hbm_tbps > 0
    assert "inferred" in GPUS["H20"].source  # H20 没有官方规格页，来源必须标明是推断


def test_dense_convention_is_consistent():
    """口径检查：全部是 dense。NVIDIA A100 表同时列 312 | 624*（* = with sparsity），
    H100 表只列带 * 的 1,979（BF16）/ 3,958（FP8），dense 是它们的一半。"""
    a100, h100 = GPUS["A100-80G"], GPUS["H100-SXM"]
    assert a100.bf16_tflops == pytest.approx(312)  # A100 表里 dense 那一列
    assert h100.bf16_tflops == pytest.approx(1979 / 2, rel=0.01)
    assert h100.fp8_tflops == pytest.approx(3958 / 2, rel=0.01)
    assert 3.0 < h100.bf16_tflops / a100.bf16_tflops < 3.5  # Hopper/Ampere dense BF16 约 3.2 倍（同口径才成立）
    for gpu in GPUS.values():  # FP8 dense 恰好是 BF16 dense 的 2 倍
        if gpu.fp8_tflops is not None:
            assert gpu.fp8_tflops == pytest.approx(2 * gpu.bf16_tflops, rel=0.01)
    # H800 与 H100 算力相同，只是 NVLink 砍到 400 GB/s
    assert GPUS["H800-SXM"].bf16_tflops == h100.bf16_tflops
    assert GPUS["H800-SXM"].nvlink_gbps == 400 and h100.nvlink_gbps == 900


def test_interconnect_constants():
    assert hardware.IB_GBPS_PER_GPU == 50.0  # 400 Gb/s CX7
    assert hardware.LINK_EFFICIENCY == pytest.approx(160 / 200)
    assert hardware.NVLINK_LATENCY_S < 1e-5 and hardware.PCIE_GEN5_GBPS == 128.0


def test_ridge_point_values():
    assert ridge_point(GPUS["H100-SXM"]) == pytest.approx(989 / 3.35)
    assert ridge_point(GPUS["A100-80G"]) == pytest.approx(312 / 2.039)
    # H20：算力砍得多、带宽反而更高 -> 拐点极低，decode 友好、prefill 不友好
    assert ridge_point(GPUS["H20"]) < 50 < ridge_point(GPUS["H100-SXM"])
    assert ridge_point(GPUS["H100-SXM"], "fp8") == pytest.approx(2 * ridge_point(GPUS["H100-SXM"]), rel=0.01)


def test_roofline_knee():
    gpu = GPUS["H100-SXM"]
    ridge = ridge_point(gpu)
    peak = peak_tflops(gpu)
    assert attainable_tflops(gpu, ridge) == pytest.approx(peak)
    assert attainable_tflops(gpu, ridge / 2) == pytest.approx(peak / 2)  # 拐点左边：带宽受限，线性
    assert attainable_tflops(gpu, ridge * 10) == pytest.approx(peak)  # 拐点右边：算力封顶
    assert attainable_tflops(gpu, 1.0) == pytest.approx(gpu.hbm_tbps)  # 强度 1 FLOP/B -> TFLOPS = TB/s


def test_fp8_unsupported_raises():
    with pytest.raises(ValueError):
        ridge_point(GPUS["A100-80G"], "fp8")

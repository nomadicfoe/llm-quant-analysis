"""Sanity checks for the simulated quantizer. Run: pytest -q"""
import torch

from src.quant import fake_quant_int4, fake_quant_int8, recon_error


def test_int8_error_small():
    w = torch.randn(64, 256, dtype=torch.float16)
    assert recon_error(w, "int8") < 0.02


def test_int4_error_larger_than_int8():
    w = torch.randn(64, 256, dtype=torch.float16)
    assert recon_error(w, "int4") > recon_error(w, "int8")
    assert recon_error(w, "int4") < 0.2


def test_int4_levels():
    w = torch.randn(8, 128)
    q = fake_quant_int4(w, 128)
    # each group should have at most 16 distinct dequantized values
    for row in q:
        assert len(torch.unique(row)) <= 16


def test_int8_preserves_shape_dtype():
    w = torch.randn(16, 300, dtype=torch.float16)
    assert fake_quant_int8(w).shape == w.shape
    assert fake_quant_int8(w).dtype == w.dtype


def test_int4_handles_non_multiple_of_group():
    w = torch.randn(16, 300, dtype=torch.float16)
    q = fake_quant_int4(w, 128)
    assert q.shape == w.shape
    assert torch.isfinite(q).all()

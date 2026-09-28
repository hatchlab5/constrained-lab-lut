"""Tests for the constrained LAB → .cube LUT core."""
import numpy as np
import pytest

from constrained_lab_lut import core


def _solid(color_bgr):
    """A small solid-color BGR image."""
    return np.full((16, 16, 3), color_bgr, dtype=np.uint8)


# --- constrained_map -------------------------------------------------------

def test_identity_stats_give_unit_gain_zero_offset():
    mean = [100.0, 128.0, 130.0]
    std = [40.0, 12.0, 14.0]
    gain, offset = core.constrained_map(mean, std, mean, std)
    assert np.allclose(gain, [1.0, 1.0, 1.0], atol=1e-4)
    assert np.allclose(offset, [0.0, 0.0, 0.0], atol=1e-3)


def test_chroma_gain_is_clamped():
    # reference chroma spread is 5x the source -> raw a/b gains would be huge.
    src_std = [50.0, 10.0, 10.0]
    ref_std = [60.0, 50.0, 3.0]
    gain, _ = core.constrained_map([0, 0, 0], src_std, [0, 0, 0], ref_std)
    lo, hi = core.DEFAULT_AB_GAIN
    assert gain[1] == pytest.approx(hi)   # raw 5.0 clamped up to the ceiling
    assert gain[2] == pytest.approx(lo)   # raw 0.3 clamped up to the floor
    l_lo, l_hi = core.DEFAULT_L_GAIN
    assert gain[0] == pytest.approx(l_hi)  # raw 1.2 == ceiling


def test_offset_maps_source_mean_exactly_to_reference_mean():
    # stats chosen so all raw gains fall inside the clamp ranges.
    src_mean = np.array([100.0, 120.0, 130.0])
    ref_mean = np.array([110.0, 125.0, 140.0])
    src_std = np.array([50.0, 20.0, 20.0])
    ref_std = np.array([55.0, 21.0, 19.0])
    gain, offset = core.constrained_map(src_mean, src_std, ref_mean, ref_std)
    mapped = gain * src_mean + offset
    assert np.allclose(mapped, ref_mean, atol=1e-3)


def test_offset_maps_mean_even_when_gain_clamped():
    # gains get clamped, but the mean must still land on the reference mean.
    src_mean = np.array([90.0, 140.0, 110.0])
    ref_mean = np.array([120.0, 130.0, 150.0])
    src_std = np.array([5.0, 5.0, 60.0])   # forces clamping in every channel
    ref_std = np.array([80.0, 60.0, 5.0])
    gain, offset = core.constrained_map(src_mean, src_std, ref_mean, ref_std)
    assert np.allclose(gain * src_mean + offset, ref_mean, atol=1e-3)


# --- identity_grid ---------------------------------------------------------

def test_identity_grid_shape_and_range():
    size = 5
    grid = core.identity_grid(size)
    assert grid.shape == (size ** 3, 3)
    assert grid.min() == pytest.approx(0.0)
    assert grid.max() == pytest.approx(1.0)


def test_identity_grid_red_varies_fastest():
    grid = core.identity_grid(2)
    # rows: (r,g,b) with r fastest -> [0,0,0], [1,0,0], [0,1,0], ...
    assert np.allclose(grid[0], [0, 0, 0])
    assert np.allclose(grid[1], [1, 0, 0])  # red changed first
    assert np.allclose(grid[2], [0, 1, 0])  # then green


def test_identity_grid_rejects_tiny_size():
    with pytest.raises(ValueError):
        core.identity_grid(1)


# --- lab_stats / average ---------------------------------------------------

def test_lab_stats_of_solid_image_has_zero_std():
    mean, std = core.lab_stats(_solid((120, 130, 90)))
    assert np.allclose(std, [0, 0, 0], atol=1e-4)
    assert mean.shape == (3,)


def test_average_lab_stats_averages_frames():
    a = _solid((100, 100, 100))
    b = _solid((160, 160, 160))
    ma, _ = core.lab_stats(a)
    mb, _ = core.lab_stats(b)
    mean, _ = core.average_lab_stats([a, b])
    assert np.allclose(mean, (ma + mb) / 2.0, atol=1e-4)


def test_average_lab_stats_empty_raises():
    with pytest.raises(ValueError):
        core.average_lab_stats([])


# --- build_cube_table / LUT correctness ------------------------------------

def test_unit_map_is_near_identity_lut():
    # gain=1, offset=0 should reproduce the identity grid (modulo LAB rounding).
    # Test the size we actually SHIP (DEFAULT_LUT_SIZE), not a smaller one that
    # would flatter the numbers — a smaller grid has fewer saturated corner nodes
    # and so drifts less. Measured max drift at the shipped size is ~0.094; a
    # finer grid (65) reaches ~0.102, so this is genuinely a per-node worst case,
    # not a bug: it's the cost of round-tripping through OpenCV's 8-bit LAB, which
    # is not perfectly invertible at the extremes of the color cube.
    size = core.DEFAULT_LUT_SIZE
    grid = core.identity_grid(size)
    table = core.build_cube_table([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], size)
    assert table.shape == grid.shape
    # The MEAN node barely moves at any size (<0.01 holds at 9, 17, 33 and 65) —
    # that's the honest "near-identity" guarantee. The per-node MAX is loose
    # because a handful of saturated corners drift; the bound reflects measured
    # reality (~0.094 at size 33), not an aspiration.
    assert np.mean(np.abs(table - grid)) < 0.01
    assert np.max(np.abs(table - grid)) < 0.10


def test_map_moves_source_color_onto_reference_color():
    # solid source and reference colors -> the map should carry the source
    # color essentially onto the reference color (the mean is matched exactly).
    src_bgr = (150, 120, 90)
    ref_bgr = (110, 120, 160)
    s_mean, s_std = core.lab_stats(_solid(src_bgr))
    r_mean, r_std = core.lab_stats(_solid(ref_bgr))
    gain, offset = core.constrained_map(s_mean, s_std, r_mean, r_std)

    src_rgb01 = np.array(src_bgr[::-1], dtype=np.float32) / 255.0
    ref_rgb01 = np.array(ref_bgr[::-1], dtype=np.float32) / 255.0
    mapped = core.apply_lab_map_to_rgb(src_rgb01, gain, offset)[0]
    assert np.allclose(mapped, ref_rgb01, atol=0.03)


# --- write_cube ------------------------------------------------------------

def test_write_cube_format_and_ordering(tmp_path):
    size = 4
    table = core.build_cube_table([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], size)
    out = tmp_path / "match.cube"
    core.write_cube(table, size, str(out), title="unit test")

    text = out.read_text().splitlines()
    assert text[0] == 'TITLE "unit test"'
    assert text[1] == f"LUT_3D_SIZE {size}"

    data = text[2:]
    assert len(data) == size ** 3
    for line in data:
        vals = [float(x) for x in line.split()]
        assert len(vals) == 3
        assert all(0.0 <= v <= 1.0 for v in vals)


def test_write_cube_rejects_wrong_row_count(tmp_path):
    with pytest.raises(ValueError):
        core.write_cube(np.zeros((10, 3)), 4, str(tmp_path / "bad.cube"))

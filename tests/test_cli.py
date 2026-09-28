"""End-to-end CLI test: build a real .cube from two synthetic frames."""
import numpy as np

import cv2  # noqa: F401  (import guard: skip cleanly if OpenCV is unavailable)
import pytest

from constrained_lab_lut import cli, core


def _write_solid(path, color_bgr):
    cv2.imwrite(str(path), np.full((32, 32, 3), color_bgr, dtype=np.uint8))


def test_cli_builds_valid_cube(tmp_path):
    src = tmp_path / "source.png"
    ref = tmp_path / "reference.png"
    out = tmp_path / "match.cube"
    _write_solid(src, (150, 120, 90))
    _write_solid(ref, (110, 120, 160))

    rc = cli.main(
        ["--source", str(src), "--reference", str(ref),
         "--output", str(out), "--size", "17"]
    )
    assert rc == 0
    assert out.exists()

    lines = out.read_text().splitlines()
    assert lines[1] == "LUT_3D_SIZE 17"
    assert len([l for l in lines[2:] if l.strip()]) == 17 ** 3


def test_cli_missing_reference_errors(tmp_path):
    src = tmp_path / "source.png"
    _write_solid(src, (128, 128, 128))
    with pytest.raises(SystemExit):
        cli.main(["--source", str(src), "--output", str(tmp_path / "x.cube")])

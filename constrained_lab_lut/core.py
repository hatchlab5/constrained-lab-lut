"""Core algorithm: constrained LAB linear color-match → 3D LUT.

Everything here is deterministic and depends only on NumPy + OpenCV. The map is
computed in OpenCV's 8-bit LAB space, where L, a and b all run 0-255 (a and b are
offset by 128). The map is linear per channel:

    out_lab = gain * in_lab + offset

with the chroma (a, b) gains clamped so the transform can only nudge saturation,
never explode it. `offset` is derived so the channel mean always maps exactly to
the reference mean regardless of clamping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:  # OpenCV is required at runtime but imported lazily so tests can be selective.
    import cv2
except ImportError:  # pragma: no cover - exercised only when cv2 is absent
    cv2 = None

DEFAULT_LUT_SIZE = 33
DEFAULT_L_GAIN: Tuple[float, float] = (0.85, 1.20)
DEFAULT_AB_GAIN: Tuple[float, float] = (0.90, 1.15)
_EPS = 1e-6
# When a video's frame count is unknown we can't seek, so we walk the stream and
# skip this many frames between the ones we keep — spreading samples across the
# clip instead of grabbing a burst from the opening seconds.
_UNKNOWN_LENGTH_FRAME_STRIDE = 30


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required for this operation. Install it with "
            "`pip install opencv-python`."
        )


def imread(path: str) -> np.ndarray:
    """Read an image as a BGR array, raising a clear error if it can't be read."""
    _require_cv2()
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def lab_stats(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of an image over the LAB channels (each length 3)."""
    _require_cv2()
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0)


def average_lab_stats(
    images: Iterable[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Average the per-frame LAB mean and std over several images.

    Averaging removes single-frame content bias: a LUT built from one busy frame
    is hostage to whatever happened to be in it.
    """
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    for image in images:
        mean, std = lab_stats(image)
        means.append(mean)
        stds.append(std)
    if not means:
        raise ValueError("no frames supplied to average_lab_stats")
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def sample_video_frames(path: str, n_frames: int = 10) -> List[np.ndarray]:
    """Sample up to ``n_frames`` roughly-even frames from a video as BGR arrays."""
    _require_cv2()
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    frames: List[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            indices = np.unique(
                np.linspace(0, total - 1, min(n_frames, total)).astype(int)
            )
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        else:  # unknown length: walk the stream, skipping ahead between grabs
            ok, frame = cap.read()
            while ok and len(frames) < n_frames:
                frames.append(frame)
                for _ in range(_UNKNOWN_LENGTH_FRAME_STRIDE):
                    ok, frame = cap.read()
                    if not ok:
                        break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames could be read from: {path}")
    return frames


def constrained_map(
    src_mean: Sequence[float],
    src_std: Sequence[float],
    ref_mean: Sequence[float],
    ref_std: Sequence[float],
    l_gain_range: Tuple[float, float] = DEFAULT_L_GAIN,
    ab_gain_range: Tuple[float, float] = DEFAULT_AB_GAIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive the clamped per-channel (gain, offset) LAB map.

    The gains are clamped; the offset is then computed from the clamped gains so
    the channel mean always maps exactly onto the reference mean.
    """
    src_mean = np.asarray(src_mean, dtype=np.float32)
    src_std = np.asarray(src_std, dtype=np.float32)
    ref_mean = np.asarray(ref_mean, dtype=np.float32)
    ref_std = np.asarray(ref_std, dtype=np.float32)

    gain = ref_std / (src_std + _EPS)
    gain[0] = np.clip(gain[0], *l_gain_range)   # L: exposure / contrast
    gain[1] = np.clip(gain[1], *ab_gain_range)  # a: green<->red chroma
    gain[2] = np.clip(gain[2], *ab_gain_range)  # b: blue<->yellow chroma

    offset = ref_mean - gain * src_mean
    return gain.astype(np.float32), offset.astype(np.float32)


def identity_grid(size: int = DEFAULT_LUT_SIZE) -> np.ndarray:
    """Return the identity RGB node grid as ``(size**3, 3)`` floats in [0, 1].

    Rows are ordered so that **red varies fastest**, which is the order a `.cube`
    file expects.
    """
    if size < 2:
        raise ValueError("LUT size must be >= 2")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # indexing='ij' with (b, g, r): flattening C-order makes r the fastest axis.
    bb, gg, rr = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)


def apply_lab_map_to_rgb(
    rgb01: np.ndarray, gain: Sequence[float], offset: Sequence[float]
) -> np.ndarray:
    """Apply the linear LAB map to RGB rows in [0, 1]; return mapped RGB rows."""
    _require_cv2()
    gain = np.asarray(gain, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    rgb01 = np.asarray(rgb01, dtype=np.float32).reshape(-1, 3)

    # RGB[0,1] -> BGR uint8 -> LAB
    bgr8 = np.clip(rgb01[:, ::-1] * 255.0, 0, 255).round().astype(np.uint8)
    bgr8 = bgr8.reshape(-1, 1, 3)
    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)

    lab = lab * gain + offset
    lab = np.clip(lab, 0, 255).round().astype(np.uint8).reshape(-1, 1, 3)

    out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32).reshape(-1, 3)
    out_rgb = out_bgr[:, ::-1] / 255.0
    return out_rgb


def build_cube_table(
    gain: Sequence[float], offset: Sequence[float], size: int = DEFAULT_LUT_SIZE
) -> np.ndarray:
    """Return the LUT node table: the mapped RGB values for every grid node."""
    return apply_lab_map_to_rgb(identity_grid(size), gain, offset)


def write_cube(
    rows: np.ndarray,
    size: int,
    path: str,
    title: str = "constrained LAB match",
) -> None:
    """Write a ``.cube`` 3D LUT. ``rows`` must be ``(size**3, 3)``, R fastest."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.shape != (size ** 3, 3):
        raise ValueError(
            f"expected {size**3} rows of 3 values, got {rows.shape}"
        )
    # The TITLE is wrapped in double quotes on a single header line, so an
    # embedded quote or newline would produce a malformed header. Collapse any
    # line breaks to spaces and swap double quotes for single ones so the header
    # is always exactly one valid line, whatever a caller passes for --title.
    safe_title = title.replace("\r", " ").replace("\n", " ").replace('"', "'")
    lines = [f'TITLE "{safe_title}"', f"LUT_3D_SIZE {size}"]
    lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in rows)
    Path(path).write_text("\n".join(lines) + "\n")

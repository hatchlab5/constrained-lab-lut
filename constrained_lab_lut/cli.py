"""Command-line interface for constrained-lab-lut."""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import core


def _collect_frames(
    images: Optional[Sequence[str]],
    videos: Optional[Sequence[str]],
    n_frames: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for path in images or []:
        frames.append(core.imread(path))
    for path in videos or []:
        frames.extend(core.sample_video_frames(path, n_frames))
    return frames


def _stats(
    images: Optional[Sequence[str]],
    videos: Optional[Sequence[str]],
    n_frames: int,
    role: str,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = _collect_frames(images, videos, n_frames)
    if not frames:
        raise SystemExit(f"error: no {role} frames given (use --{role} or --{role}-video)")
    return core.average_lab_stats(frames)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="constrained-lab-lut",
        description=(
            "Measure the LAB stats of a reference look and a source look, derive "
            "a constrained linear map, and bake it into a .cube 3D LUT."
        ),
    )
    p.add_argument("--source", "-s", action="append", metavar="IMG",
                   help="source (to-be-corrected) still frame; repeatable")
    p.add_argument("--source-video", action="append", metavar="VID",
                   help="source video to sample frames from; repeatable")
    p.add_argument("--reference", "-r", action="append", metavar="IMG",
                   help="reference (target look) still frame; repeatable")
    p.add_argument("--reference-video", action="append", metavar="VID",
                   help="reference video to sample frames from; repeatable")
    p.add_argument("--frames", type=int, default=10, metavar="N",
                   help="frames to sample per video (default: 10)")
    p.add_argument("--output", "-o", required=True, metavar="FILE.cube",
                   help="output .cube path")
    p.add_argument("--size", type=int, default=core.DEFAULT_LUT_SIZE, metavar="N",
                   help=f"LUT edge size (default: {core.DEFAULT_LUT_SIZE})")
    p.add_argument("--l-gain-min", type=float, default=core.DEFAULT_L_GAIN[0])
    p.add_argument("--l-gain-max", type=float, default=core.DEFAULT_L_GAIN[1])
    p.add_argument("--ab-gain-min", type=float, default=core.DEFAULT_AB_GAIN[0])
    p.add_argument("--ab-gain-max", type=float, default=core.DEFAULT_AB_GAIN[1])
    p.add_argument("--title", default="constrained LAB match",
                   help="TITLE written into the .cube header")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    src_mean, src_std = _stats(args.source, args.source_video, args.frames, "source")
    ref_mean, ref_std = _stats(
        args.reference, args.reference_video, args.frames, "reference"
    )

    gain, offset = core.constrained_map(
        src_mean, src_std, ref_mean, ref_std,
        l_gain_range=(args.l_gain_min, args.l_gain_max),
        ab_gain_range=(args.ab_gain_min, args.ab_gain_max),
    )

    table = core.build_cube_table(gain, offset, args.size)
    core.write_cube(table, args.size, args.output, title=args.title)

    print(f"LAB gain  = {np.round(gain, 3)}")
    print(f"LAB offset= {np.round(offset, 1)}")
    print(f"wrote {args.output}  (LUT_3D_SIZE {args.size})")
    # Use the explicit `file=` form so output paths containing ':' ',' or spaces
    # survive ffmpeg's filtergraph parsing (the bare `lut3d=path` shorthand breaks
    # on those characters).
    print(f"apply with:  ffmpeg -i in.mp4 -vf lut3d=file={args.output} out.mp4")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

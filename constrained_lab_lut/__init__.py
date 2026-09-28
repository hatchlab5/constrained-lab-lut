"""Constrained LAB linear color-match baked into a .cube 3D LUT.

Public API:
    lab_stats(image_bgr)                        -> (mean, std) in LAB
    average_lab_stats(images)                   -> mean/std averaged over frames
    constrained_map(src_mean, src_std,
                    ref_mean, ref_std, ...)     -> (gain, offset)
    identity_grid(size)                         -> RGB nodes, R varies fastest
    apply_lab_map_to_rgb(rgb01, gain, offset)   -> mapped RGB nodes
    build_cube_table(gain, offset, size)        -> LUT node table
    write_cube(rows, size, path, title)         -> write a .cube file
    imread(path)                                -> BGR image (raises on failure)
    sample_video_frames(path, n_frames)         -> list of BGR frames
"""
from .core import (
    DEFAULT_AB_GAIN,
    DEFAULT_L_GAIN,
    DEFAULT_LUT_SIZE,
    apply_lab_map_to_rgb,
    average_lab_stats,
    build_cube_table,
    constrained_map,
    identity_grid,
    imread,
    lab_stats,
    sample_video_frames,
    write_cube,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_AB_GAIN",
    "DEFAULT_L_GAIN",
    "DEFAULT_LUT_SIZE",
    "apply_lab_map_to_rgb",
    "average_lab_stats",
    "build_cube_table",
    "constrained_map",
    "identity_grid",
    "imread",
    "lab_stats",
    "sample_video_frames",
    "write_cube",
    "__version__",
]

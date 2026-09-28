# constrained-lab-lut

Build a **constrained** LAB color-match and bake it into a 33³ `.cube` 3D LUT you
apply at render time with `ffmpeg`.

Point it at a **reference** frame (the look you want) and a **source** frame (the
footage that's off), and it measures the mean and standard deviation of each
frame in LAB space, derives a per-channel linear map, **clamps the chroma gain so
the map can never oversaturate**, and writes a portable `.cube` file. From then on
the correction is a single deterministic `ffmpeg` filter — no Python at render
time, no per-frame processing, nothing to tune by eye.

Used in production video work to keep cutaway b-roll matched to interview footage
that was shot on a different camera or under different light, without hand-grading
every clip.

## Why "constrained"?

The classic Reinhard color transfer matches both the **mean** and the **full
standard deviation** of each channel. Matching the mean is the part you actually
want — it's the white-balance and exposure correction. Matching the *full* std is
where it goes wrong: if the source frame happens to have a narrow color spread
(say a mostly-uniform scene), scaling that spread up to a busier reference frame
blows up the chroma gain and drags neutral tones toward a color cast. A pale
garment picks up a magenta tint; skies go lurid.

The fix is small and decisive: **keep the mean shift, but clamp the per-channel
gain** so it can only nudge saturation, never explode it.

```
gain      = ref_std / source_std
gain[L]   = clamp(gain[L], 0.85, 1.20)   # exposure/contrast: modest
gain[a,b] = clamp(gain[a,b], 0.90, 1.15) # chroma: tightly constrained
offset    = ref_mean - gain * source_mean
```

The mean is always matched exactly (via the offset); the gain is only ever a
gentle correction. That single clamp is the whole idea.

## Install

Requires Python 3.8+, NumPy, and OpenCV. `ffmpeg` is only needed to *apply* the
LUT, not to build it.

```bash
pip install .
# or, for development:
pip install -e ".[test]"
```

## Usage

### 1. Build the LUT

From two still frames (extract them however you like — one from the look you want,
one from the footage to correct):

```bash
constrained-lab-lut \
    --source   drone_broll_frame.png \
    --reference interview_frame.png \
    --output   match.cube
```

Averaging over several frames removes single-frame content bias (recommended — a
LUT built from one busy frame is hostage to whatever was in it):

```bash
constrained-lab-lut \
    --source    broll_a.png --source broll_b.png --source broll_c.png \
    --reference ref_a.png   --reference ref_b.png \
    --output    match.cube
```

You can also point it straight at video files and let it sample frames evenly:

```bash
constrained-lab-lut \
    --source-video    cutaway.mp4 \
    --reference-video interview.mp4 \
    --frames 12 \
    --output match.cube
```

It prints the derived map so you can sanity-check it:

```
LAB gain  = [1.095 1.15  0.9  ]
LAB offset= [ -5.9 -15.6  18.7]
wrote match.cube  (LUT_3D_SIZE 33)
```

### 2. Apply it at render time

The LUT is a static file. Applying it is one filter — this is all the render
pipeline ever needs:

```bash
ffmpeg -i cutaway.mp4 -vf lut3d=file=match.cube out.mp4
```

Apply it only to the clips that need correcting; leave the reference-look footage
untouched (it's already the target).

### Tuning the clamp

The defaults are deliberately conservative. Loosen or tighten them per job:

```bash
constrained-lab-lut ... --ab-gain-min 0.85 --ab-gain-max 1.25 \
                        --l-gain-min 0.80 --l-gain-max 1.30
```

### LUT size

Default is 33³ (the common standard — smooth, ~1 MB). Drop to 17³ if size
matters, raise to 65³ only if you ever see banding.

## How it works

1. Convert each frame to LAB (OpenCV's 8-bit LAB, all channels 0–255).
2. Compute the mean and std of each channel; average across frames if given more
   than one.
3. Derive `gain` and `offset`, clamping the gains.
4. Apply that linear LAB map to an identity RGB grid of `size³` nodes and convert
   back to RGB.
5. Write the nodes as a `.cube` file in the order `ffmpeg` expects (red varies
   fastest).

The whole pipeline runs through OpenCV's **8-bit** LAB (uint8, channels 0–255),
so even an identity map (gain 1, offset 0) drifts by a small amount at the most
saturated nodes — a handful of corner nodes move by up to ~0.1, while the average
node barely moves at all. This is a deliberate simplicity tradeoff: 8-bit LAB
keeps the code short and the map a "gentle correction" rather than a precise
transform, and at the smooth 33³ grid it's well below anything visible in a
graded frame. If you needed bit-exact round-tripping you'd carry the map in
float32 — but that isn't what this tool is for.

One global LUT usually covers a whole set of clips cut from the same camera and
grade. Review the LUT once on a few stills; after that every render is
deterministic and zero-touch.

## License

MIT — see [LICENSE](LICENSE).

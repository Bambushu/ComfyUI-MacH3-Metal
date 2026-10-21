# Results

Measured on a MacBook Pro M5 Pro, 48 GB, 2026-10-21. ComfyUI 0.34.0, h3.c
`comfy-node` branch (integration + libh3.dylib), exact preset, 8 steps,
544x960, euler / linear, shift 12.

## Two-take chain inside ComfyUI

`example_workflows/h3c_chain_two_takes.json` as shipped: the bench prompt,
take A seed 77001 (124 frames, saves its context), take B seed 77002
continuing A's context file (124 frames requested, 102 delivered).

| Phase | Take A | Take B |
| --- | ---: | ---: |
| Text encoder | 53.6 s | 52.8 s |
| DiT load (int8 cache) | 8.3 s | 6.7 s |
| Denoise, 8 steps | 330.0 s | 423.9 s |
| Audio + video decode | 48.7 s | 43.9 s |

Whole queue item, both takes plus the join: 972.7 s.

The continuation's denoise is longer because its sequence carries the
pinned head of the previous take.

## Same frames as the h3 command line

Take A and take B from ComfyUI are bit-identical to the same takes rendered
with the `h3` binary (same prompt, seeds and settings): comparing the engine's
own MP4 from each path, every decoded frame's MD5 matches and PSNR is infinite
for both takes. (SaveVideo re-encodes, so its files differ by encoder only.)

## The join

`~/h3c-bench/mc/qa.py` on the two SaveVideo outputs:

| Measure | Value |
| --- | ---: |
| Correlation r, last frame of A vs first frame of B | 0.993 |
| Mean absolute difference at the join | 4.05 (+0.5 sd of the clips' own frame-to-frame motion) |
| Frame-to-frame motion inside the clips | mean 3.29, sd 1.50, max 8.12 |

At 100% the frames either side of the cut show the same face, earring, hair,
light and mouth movement.

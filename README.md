# MacH3 Metal

Every MiniMax-H3 mode, with its own sound, generated on a Mac inside ComfyUI.

MacH3 Metal is a node pack that runs MiniMax-H3 on
[h3.c](https://github.com/Bambushu/h3.c), a native C and Metal engine,
instead of ComfyUI's PyTorch path. One generation is picture and stereo sound
together: dialogue, foley and room tone come out of the same pass. It covers
text to video, first frame and first + last frame, ordered image, video and
audio references, and continuation: a take can pick up the last 22 frames and
0.917 s of sound of a previous take or any clip, so a shot can keep going
across prompts.

Apple silicon only. Measured on an M5 Pro with 48 GB.

## What is in the pack

| Node | Does |
| --- | --- |
| h3c Load Engine | Opens the engine. Reads model metadata only; weights are mapped per generation and freed after it. Optional `lora`: `turbo` for the 5-6 step speed lane on the same weights (see Turbo weights). |
| h3c Generate | Prompt, size, seconds or frames, steps, seed, sampler, scheduler, shift, speed preset, a **mode** and **continue** (off, last take, or the clip input). Optional first frame, last frame, ordered references and a motion context. Returns IMAGE frames, AUDIO, fps and a context for a next take. |
| h3c Reference Image / Video / Audio | Build the ordered reference list. The first image is `<Picture 1>` in the prompt, the first video `<Video 1>`, and so on. |
| h3c Load Motion Context | A `.h3ctx` from an earlier take (bit-exact) or any finished `.mp4` (45 s max). |
| h3c Motion Context From Video | Continue a clip that is already in the graph as a VIDEO. |
| h3c Settings | One set of size, steps, sampler, scheduler, shift and speed for several Generate nodes. |
| h3c Add to Shot | Builds a shot across runs: a take that continues the last one is added to the end, any other take starts a new shot. Plays the whole shot. |
| h3c Join Takes / Join All Takes | Takes in order as one IMAGE batch and one AUDIO track, for chains wired inside one graph. Join All Takes renders only the first N takes. |

The **mode** on h3c Generate is one of four: T2V (text to video), I2V (first
frame), FL2V (first + last frame) or R2V (references). Inputs the mode leaves out
are skipped and the loaders feeding them never run; in a workflow the pack's
frontend script greys out their groups too. **continue** (off, last take, clip
input) picks up where a take ended and works under every mode. The older
combined mode names (auto, references + first frame, continue + references, ...)
are still accepted from the API.

Frames and audio leave as ordinary IMAGE and AUDIO, so the stock CreateVideo
and SaveVideo nodes write the file.

### One workflow for everything

`example_workflows/macH3_metal_all_in_one.json` is one graph for every mode.
The control panel holds the Take (mode, continue, prompt, length, seed), the
Engine (lane) and Settings. Each mode's pictures have their own group, and picking a mode greys out the others: T2V (none),
I2V / FL2V (first frame, last frame), R2V (up to three references, people or
places; pictures 2 and 3 start bypassed, Ctrl+B turns one on) and a clip to
continue. Pick the mode, write the prompt, Run. To keep going, set Continue to
*last take*, write what happens next and Run again: the Shot node adds it to
the end. Continue works in every mode.

`macH3_metal.app.json` is the same graph for ComfyUI's App Mode: a form with
the mode, continue, pictures, prompt, length, seed, size, steps and lane, and
the shot as its output.

Both run as shipped: the pack copies its `h3c_sample_*` pictures, clip and
voice line into ComfyUI's input folder on first load (it never overwrites a
file).

### Single-purpose examples

Load from the workflow templates browser, or drag in from `example_workflows/`:

| File | Mode |
| --- | --- |
| `h3c_t2v.json` | Text to video with sound |
| `h3c_first_last_frame.json` | First frame, optional last frame |
| `h3c_references.json` | Identity picture as `<Picture 1>` |
| `h3c_reference_video_audio.json` | Ordered reference video and audio |

## What it is not

The engine runs the whole generation itself: text encoder, transformer and
both VAEs. It does not go through ComfyUI's model loader or latent graph.
That means:

- No LoRA loaders, no Comfy samplers, no SLA or other attention patches, no
  latent upscale. Use ComfyUI's native MiniMax-H3 support for those.
- It must not run while another H3 model is loaded. h3c Generate unloads
  every ComfyUI model before it starts and takes the same process lock as the
  `h3` command line, so a second h3c job waits instead of loading beside it.
- Mac only. The kernels are Metal; there is no CUDA path.

## Setup on a Mac

You need an Apple silicon Mac (48 GB or more is comfortable for 0.5 MP; see
the h3.c README for the low-memory modes), ffmpeg (`brew install ffmpeg`),
and the MiniMax-H3 weights.

```bash
git clone https://github.com/Bambushu/h3.c ~/h3c
make -C ~/h3c libh3.dylib
```

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3 --local-dir ~/h3c/MiniMax-H3
```

```bash
git clone https://github.com/Bambushu/ComfyUI-MacH3-Metal ~/ComfyUI/custom_nodes/ComfyUI-MacH3-Metal
```

Start ComfyUI as usual. The loader finds the engine at `~/h3c` and the
weights at `~/h3c/MiniMax-H3`. Point it elsewhere with the node's two fields
or with `H3C_DIR` and `H3C_MODEL_DIR`. The first generation builds the int8
transformer cache at `~/Library/Caches/h3c/dit` (18 GB, once).

The pack needs nothing beyond ComfyUI's own Python packages.

## Chaining takes

Turn on `save_context` on a take, connect its `context` output to the next
h3c Generate's `motion_context`, and keep the same references on both. The
continuation counts the 22 pinned frames in its length and delivers 22 fewer,
so a 5 s continuation adds about 4.1 s. Feed both takes into h3c Join Takes
for the whole shot, or, across runs, set Continue to *last take* and let h3c
Add to Shot collect them. Write each continuation's prompt so it opens on the state
the previous take ended in.

A take rendered without a context file can still be continued from its MP4.
The engine re-encodes the clip's tail (about 100 s at 544x960) and measured
the same join quality.

## Speed and quality

| Preset | Settings | Use |
| --- | --- | --- |
| exact | 50 DiT blocks, every step evaluated | Default, the measured lane below |
| balanced | 45 blocks, denoiser reuse 2 | Faster drafts |
| fast | 45 blocks, core reuse 4 | Faster drafts |
| aggressive | 40 blocks, reuse 3, token reduction | Composition previews only |

`preview` decodes one frame per denoising step into the progress bar. It
keeps a preview decoder resident, about 10 GiB more memory.

### Turbo weights (recommended speed lane)

The step-distillation adapter
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
runs on the same weights as the quality lane. Download
`minimax_h3_turbo_v4_step600_ema.safetensors` (780 MB) into `~/h3c/turbo_lora/`
(or ComfyUI's `models/loras`, or set `$H3C_TURBO_LORA`), set `lora` on h3c Load Engine
to `turbo`, and render at 5 or 6 steps with the `exact` preset. Leave `lora`
empty for the base weights. `lora` also takes a path to another DiT adapter,
with `lora_scale` as its strength.

h3c applies the adapter while it loads the weights, byte-identical to folding
it into a copy of the transformer, so there is no second 62 GB checkpoint. The
first turbo run builds its own int8 cache (18 GiB, beside the base one in
`~/Library/Caches/h3c/dit`); later runs map it like the base cache. It needs a
`libh3.dylib` built from an h3.c with `--lora`, and does not combine with SSD
streaming.

Same prompt and seed, 768x1344, 124 frames, M5 Pro 48 GB, inside ComfyUI:

| Weights | Steps | Wall |
| --- | ---: | ---: |
| original | 16 | 36 min |
| original | 6 | 15.4 min |
| turbo | 6 | 17.2 min (first run, weights read cold, folded copy) |

Turbo at 6 steps came closest to the 16-step take; the original weights at 6
steps keep the same shot but look softer. Tested on text to video; references
and continuation on turbo weights are not measured yet.

## Numbers

Engine, M5 Pro 48 GB, 124 frames (5.2 s), 8 steps, exact preset, from the
h3.c settings log:

| Shape | Denoise | Decode | Wall |
| --- | ---: | ---: | ---: |
| 544x960 | 297 s | 40 s | about 6.5 min |
| 768x1344 | 961 to 1015 s | 86 s | about 18.5 min |

Join quality, measured on the frame either side of the cut (correlation r,
1.0 is identical): 0.993 from a context file, 0.994 from the MP4, 0.969 with
an identity reference on both takes.

Inside ComfyUI through this pack: see `docs/results.md`.

## MacH3 Metal app

`app/` is a small web app on the Comfy SDK (Comfy API v2): pick a mode, write a
prompt, drop in pictures or frames, pick the size in megapixels, generate, and continue the take from where it
ended. It talks to ComfyUI through comfy-api-proxy, so the same code runs
against any Comfy API v2 surface that has this pack installed. See
`app/README.md`.

## License

MIT, Maikel Slomp.

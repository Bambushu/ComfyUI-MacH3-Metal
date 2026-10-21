# ComfyUI-h3c

MiniMax-H3 video with its own sound, generated on a Mac, inside ComfyUI.

The nodes drive [h3.c](https://github.com/Bambushu/h3.c), a native C and Metal
engine for the open-weight MiniMax-H3 model. One generation is picture and
stereo sound together: dialogue, foley and room tone come out of the same
pass. Takes chain: a continuation picks up the previous take's last 22
frames and 0.917 s of sound, so one shot keeps going across as many prompts
as you like, with the voice, the motion and the room carrying over.

Apple silicon only. Measured on an M5 Pro with 48 GB.

## What is in the pack

| Node | Does |
| --- | --- |
| h3c Load Engine | Opens the engine. Reads model metadata only; weights are mapped per generation and freed after it. |
| h3c Generate | Prompt, size, seconds or frames, steps, seed, sampler, scheduler, shift, speed preset. Optional first frame, last frame, ordered references and a motion context. Returns IMAGE frames, AUDIO, fps and a context for the next take. |
| h3c Reference Image / Video / Audio | Build the ordered reference list. The first image is `<Picture 1>` in the prompt, the first video `<Video 1>`, and so on. |
| h3c Load Motion Context | A `.h3ctx` from an earlier take (bit-exact) or any finished `.mp4` (45 s max). |
| h3c Motion Context From Video | Continue a clip that is already in the graph as a VIDEO. |
| h3c Join Takes | A take and its continuation as one IMAGE batch and one AUDIO track. |
| h3c Join All Takes | Takes 1 to N as one shot; only asks for (and so only renders) the first N. |
| h3c Mode | The control panel: one mode choice drives the on/off switches below. |
| h3c Settings | One set of size, steps, sampler, scheduler, shift and speed for every take. |
| h3c Optional Image | On/off switch for a first or last frame. |

Reference and context nodes carry an `enabled` switch too. A switch that is
off skips its input entirely: the loader feeding it never runs.

Frames and audio leave as ordinary IMAGE and AUDIO, so the stock CreateVideo
and SaveVideo nodes write the file.

### h3c Studio: everything in one graph

`example_workflows/h3c_studio_all_in_one.json` carries every feature in
colored groups: Control, Continue a clip, First / last frame, References,
Take 1 to 3, Join and save. The h3c Mode node is the control panel: pick text
to video, first frame, first + last frame or references, switch "continue an
existing clip" on or off, and set how many chained takes to render. Inputs
that are switched off never run, and takes past the chosen number never
render, in the GUI and headless alike.

`h3c_studio.app.json` is the same graph for ComfyUI's App Mode: a form with
the mode, takes, identity picture, the three prompts, lengths, seeds and size,
and the finished shot as its output.

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
| `h3c_chain_two_takes.json` | Take A, take B continuing A, and the joined take |
| `h3c_continue_video.json` | Continue any finished MP4 |

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
git clone https://github.com/Bambushu/ComfyUI-h3c ~/ComfyUI/custom_nodes/ComfyUI-h3c
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
for the whole shot. Write each continuation's prompt so it opens on the state
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

## Continuous Take Studio

`app/` is a small web app on the Comfy SDK (Comfy API v2): prompt, identity
pictures, length, generate, then "continue the take" to chain the next prompt
onto the last one, with the joined take playing on the right. It talks to
ComfyUI through comfy-api-proxy, so the same code would run against any Comfy
API v2 surface that has this pack installed. See `app/README.md`.

## License

MIT, Maikel Slomp.

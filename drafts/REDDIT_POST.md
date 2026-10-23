TITLE:
MacH3 Metal: MiniMax H3 with sound, running natively on a Mac inside ComfyUI (T2V, I2V, FL2V, R2V + continuation)

SUBREDDIT: r/comfyui (also fits r/StableDiffusion)

FLAIR: Workflow Included

VIDEO: mach3_metal_demo_v6.mp4 (1920x1080, 55.7 s, 29 MB)

BODY:
I wanted MiniMax H3 on my MacBook without needing to rent a GPU, but the PyTorch way in ComfyUI was too slow to be fun. So I wrapped h3.c, a native C and Metal engine for H3, in a ComfyUI node pack. Picture and sound come out of one pass, and the frames and audio leave as standard IMAGE and AUDIO, so the stock video nodes save them.

Everything in the video was rendered using this pack on my M5 Pro (48 GB). I only ran the hiker selfie near the end using the h3c command line directly. I used MOSS-Image pictures for the synthetic people, and I cut the final video in Remotion.

What is included:

- All H3 modes are in one workflow: text to video, first frame, first + last frame, and references (up to 3 people or places). Just pick the mode and switch off the groups you don't need.
- Continue: select "last take." The next take starts from the last 22 frames and 0.9 s of sound from the previous one. You write a few seconds, render, then write what happens next. The Shot node joins them, so the trailhead conversation in the video is three takes played as one.
- Two lanes use the same weights. The quality lane uses the released model at 12 steps. The speed lane loads the Turbo LoRA (larryvrh) and runs at 6 steps, so you don't need a second 62 GB checkpoint on disk.
- I also included an App Mode version and a small web app on the Comfy SDK that drives the nodes through the Comfy API.

Here are some numbers from my Mac, for a 3 s clip (the cat and the wine glass, using the same prompt and seed):

| | 0.5 MP (544x960) | 1.0 MP (768x1344) |
|---|---|---|
| Speed, 6 steps | 3:25 | 7:05 |
| Quality, 12 steps | 6:14 | 12:15 |

Memory was surprising. ComfyUI peaked at 7.7 to 9.1 GB across those four renders because the 18 GB of int8 weights are mapped from the SSD instead of loaded into RAM. Also, per step, it is about 4x faster than the PyTorch + GGUF route in ComfyUI on the same Mac (that ran 188 s/step at 768x1376 for 3 s; h3c is around 45 to 50 s/step at 768x1344).

You need an Apple silicon Mac. I have only tested this on my 48 GB M5 Pro so far; it should run on 24 GB and up on M1 or newer, but I haven't measured that. You also need ffmpeg, the MiniMax H3 weights, and the h3.c engine built as a library. The first run builds an 18 GB int8 cache, one per lane.

Limits, to be straight: it's Mac only (Metal). It runs outside of ComfyUI's latent graph, so you won't find LoRA loader nodes, Comfy samplers, or latent upscale on this path. The engine has its own LoRA field instead.

Repo, MIT, with the workflows and setup steps: https://github.com/Bambushu/ComfyUI-MacH3-Metal

Credits:
- MiniMax H3 by MiniMax
- h3.c by antirez (https://github.com/antirez/h3.c). My fork with the Comfy library build: https://github.com/Bambushu/h3.c
- Turbo LoRA by larryvrh: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- comfy-api-proxy and the Comfy SDK for the app

#OpenSourceHiggsfield

TITLE:
MacH3 Metal: MiniMax H3 with sound, running natively on Apple silicon in ComfyUI

SUBREDDIT: r/comfyui (also fits r/StableDiffusion)

FLAIR: Workflow Included

VIDEO: mach3_metal_demo_v6.mp4 (1920x1080, 55.7 s, 29 MB)

BODY:
This is MacH3 Metal, a ComfyUI node pack and workflows that run MiniMax H3 natively on Apple silicon. H3 is an open video model that makes picture and its own sound in one pass. The attached demo was rendered with it on my M5 Pro (48 GB), except the hiker selfie near the end, which I ran from the h3c command line directly. Synthetic people in it are from MOSS-Image pictures, and the video was edited in Remotion.

I wanted H3 on my MacBook without renting a GPU. The PyTorch route in ComfyUI was too slow to be fun.

Features:
- All H3 modes in one workflow: text to video, first frame, first and last frame, references (up to 3 people or places). Pick the mode, switch off the groups you do not need.
- Continue: set to "last take"; the next take starts from the last 22 frames and 0.9 s of sound of the previous one. Write a few seconds, render, write what happens next. The Shot node joins them; the trailhead conversation in the video is three takes played as one.
- Two lanes on the same weights: Quality is the released model at 12 steps; Speed loads larryvrh's Turbo LoRA at 6 steps. So no second 62 GB checkpoint on disk.
- Frames and audio come out as standard IMAGE and AUDIO, so the stock video nodes save them.
- Also an App Mode version, and a small web app on the Comfy SDK that drives the nodes through the Comfy API.

Numbers for a 3 s clip, cat knocking a wine glass, same prompt and seed:

| | 0.5 MP (544x960) | 1.0 MP (768x1344) |
|---|---|---|
| Speed, 6 steps | 3:25 | 7:05 |
| Quality, 12 steps | 6:14 | 12:15 |

ComfyUI peaked at 7.7 to 9.1 GB across those four renders. The 18 GB of int8 weights are mapped from the SSD instead of loaded into RAM.

Per step it is about 4x faster than the PyTorch + GGUF route in ComfyUI on the same Mac. That route ran 188 s/step at 768x1376 for 3 s; h3c is around 45 to 50 s/step at 768x1344.

Needs: an Apple silicon Mac. I only tested it on my 48 GB M5 Pro. 24 GB and up on M1 or newer should work but is untested. You also need ffmpeg, the MiniMax H3 weights, and the h3.c engine built as a library. First run builds an 18 GB int8 cache, one per lane.

Limits: Mac only (Metal). It runs outside ComfyUI's latent graph, so no LoRA loader nodes, Comfy samplers or latent upscale on this path. The engine has its own LoRA field instead.

Mac owners, especially on 24 GB, post your times if you try it. Happy to answer questions.

Repo (MIT, workflows and setup): https://github.com/Bambushu/ComfyUI-MacH3-Metal

Credits:
- MiniMax H3 by MiniMax
- h3.c by antirez: https://github.com/antirez/h3.c. Comfy library build fork: https://github.com/Bambushu/h3.c
- Turbo LoRA by larryvrh: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- comfy-api-proxy and the Comfy SDK for the app

#OpenSourceHiggsfield

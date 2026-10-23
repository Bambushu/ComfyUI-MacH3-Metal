TITLE:
I made ComfyUI nodes that run MiniMax H3 (video + its own sound in one pass) on Apple Silicon

SUBREDDIT: r/comfyui (also fits r/StableDiffusion)

FLAIR: Workflow Included

VIDEO: mach3_metal_demo_v6.mp4 (1920x1080, 55.7 s, 29 MB)

BODY:
MacH3 Metal is a ComfyUI node pack plus workflows that run MiniMax H3 natively on Apple silicon. H3 is the open video model that makes the picture and its own sound in one pass. The demo video attached to this post is made with it.

I built it because I wanted H3 on my MacBook without renting a GPU. The PyTorch route in ComfyUI was too slow to be fun. So the pack wraps h3.c, antirez's native C + Metal engine for H3, into nodes. Frames and audio come out as standard IMAGE and AUDIO, so the stock video nodes save them.

What's in it:

- All H3 modes in one workflow: text to video, first frame, first + last frame, references (up to 3 people or places). Pick the mode, switch off the groups you don't need.
- Continue: set it to "last take" and the next take starts from the last 22 frames and 0.9 s of sound of the previous one. Write a few seconds, render, write what happens next. The Shot node joins the takes. The trailhead conversation in the video is three takes played as one.
- Two lanes on the same weights: Quality is the released model at 12 steps, Speed loads larryvrh's Turbo LoRA and runs 6 steps. No second 62 GB checkpoint on disk.
- An App Mode version too, plus a small web app on the Comfy SDK that drives the nodes through the Comfy API.

Times for a 3 s clip (cat knocking a wine glass, same prompt and seed), on my 48 GB M5 Pro:

| | 0.5 MP (544x960) | 1.0 MP (768x1344) |
|---|---|---|
| Speed, 6 steps | 3:25 | 7:05 |
| Quality, 12 steps | 6:14 | 12:15 |

ComfyUI peaked at 7.7 to 9.1 GB across those four renders, because the 18 GB of int8 weights are mapped from the SSD instead of loaded into RAM.

Per step this is about 4x faster than the PyTorch + GGUF route in ComfyUI on the same Mac. For reference, that route ran 188 s/step at 768x1376 for a 3 s clip, and h3c sits around 45 to 50 s/step at 768x1344.

Needs an Apple silicon Mac, ffmpeg, the MiniMax H3 weights and h3.c built as a library. First run builds an 18 GB int8 cache, one per lane.

Limits: Mac only, it's Metal. The engine runs outside ComfyUI's latent graph, so no LoRA loader nodes, Comfy samplers or latent upscale on this path. It has its own LoRA field instead. Also, I've only tested this on my 48 GB M5 Pro. 24 GB and up on M1 or newer should work, but that is untested.

About the video: everything in it was rendered with the pack, except the hiker selfie near the end, which I ran from the h3c command line directly. The people are synthetic, from MOSS-Image pictures. Edited in Remotion.

Mac owners, especially on 24 GB: if you run this, post your times. Happy to answer questions.

Repo, MIT, workflows and setup: https://github.com/Bambushu/ComfyUI-MacH3-Metal

Credits:
- MiniMax H3 by MiniMax
- h3.c by antirez: https://github.com/antirez/h3.c (my fork with the Comfy library build: https://github.com/Bambushu/h3.c)
- Turbo LoRA by larryvrh: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- comfy-api-proxy and the Comfy SDK for the app

#OpenSourceHiggsfield (entered in the Open Source Higgsfield challenge)

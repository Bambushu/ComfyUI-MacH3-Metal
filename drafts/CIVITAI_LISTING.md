TYPE: Workflows
NAME: MacH3 Metal: native MiniMax H3 with sound on Mac
BASE MODEL: MiniMax H3 (pick "Other" if not listed)
TAGS: comfyui, minimax h3, mac, apple silicon, metal, video, audio, text to video, image to video, continuation
FILE: mach3_metal_civitai.zip (macH3_metal_all_in_one.json, macH3_metal.app.json, the 4 single-mode example workflows, README)
COVER: mach3_metal_demo_v6.mp4 (1920x1080, 29 MB), plus comfy_aio.png (the workflow) and app_page.png as extra showcase images
RESOURCES USED (link on the page): MiniMax H3 (MiniMax), Turbo LoRA by larryvrh (Hugging Face, not on Civitai)

DESCRIPTION:
MiniMax H3 on a Mac, without the PyTorch route. This is a node pack plus workflows built on h3.c, a native C and Metal engine for H3. One pass gives you the picture and its own sound.

**Heads up:** you need the MacH3 Metal node pack and the h3.c engine installed (links below). The workflow alone **won't run on stock ComfyUI**, and it **won't run on NVIDIA cards**.

**Start here:** open `macH3_metal_all_in_one.json`. It opens on text to video with a short prompt already in, so you can just press Run. On first load the pack copies its sample pictures and clip into ComfyUI's input folder.

**How it works:** pick a mode on the Take (T2V, I2V, FL2V or R2V with up to 3 references), write the prompt, press Run. To keep going, set Continue to "last take" in Settings, write what happens next, run again, and the Shot node joins the takes into one clip. Settings also has the size in megapixels and the lane: Quality (12 steps) or Speed (the Turbo LoRA at 6 steps, on the same weights).

**On my M5 Pro (48 GB), a 3 s clip:** at 0.5 MP, 3:25 on Speed and 6:14 on Quality. At 1.0 MP, 7:05 on Speed and 12:15 on Quality. ComfyUI peaked under 10 GB of memory on all four, because the 18 GB of int8 weights stream from the SSD.

**What you need:** an Apple silicon Mac (tested on a 48 GB M5 Pro; 24 GB+ on M1 or newer should work but is untested), ffmpeg, the MiniMax H3 weights, the h3.c engine built with `make libh3.dylib`, and the node pack. The first run builds an 18 GB int8 cache once. Full steps are in the repo README.

Credits:
- MiniMax H3 by MiniMax
- h3.c by antirez: https://github.com/antirez/h3.c (my fork with the library build: https://github.com/Bambushu/h3.c)
- Turbo LoRA by larryvrh: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora

Source, MIT license and updates: https://github.com/Bambushu/ComfyUI-MacH3-Metal
I entered this in the Open Source Higgsfield challenge. #OpenSourceHiggsfield

TYPE: Workflows
NAME: MacH3 Metal: MiniMax H3 for Apple Silicon (Native Sound & Video)
BASE MODEL: MiniMax H3 (pick "Other" if not listed)
TAGS: comfyui, minimax h3, mac, apple silicon, metal, video, audio, text to video, image to video, continuation
FILE: mach3_metal_civitai.zip (macH3_metal_all_in_one.json, macH3_metal.app.json, the 4 single-mode example workflows, README)
COVER: mach3_metal_demo_v6.mp4 (1920x1080, 29 MB), plus comfy_aio.png (the workflow) and app_page.png as extra showcase images
RESOURCES USED (link on the page): MiniMax H3 (MiniMax), Turbo LoRA by larryvrh (Hugging Face, not on Civitai)

DESCRIPTION:
***
### ⚠️ IMPORTANT WARNING ⚠️
This workflow requires the MacH3 Metal node pack and the h3.c engine installed. It will not run on stock ComfyUI or on NVIDIA cards.

### 🛠️ What It Does
I built this because I wanted MiniMax H3 running natively on my Mac without needing the PyTorch route. This is a node pack plus workflows, based on h3.c, a native C and Metal engine for H3. One pass gives you both the picture and its own sound.

### 🚀 How to Start
1.  **Install Dependencies:** You must install the MacH3 Metal node pack and the h3.c engine (links are below).
2.  **Run Workflow:** Start with `macH3_metal_all_in_one.json`. It opens ready for text to video with a short prompt, so you can just press Run.
3.  **First Load:** The node pack copies its sample pictures and clip into ComfyUI's input folder on the first run.

### ⚙️ Usage & Settings
*   **Modes:** Select a mode on the Take node (T2V, I2V, FL2V, or R2V with up to 3 references).
*   **Continuation:** To continue a shot, set Continue to "last take" in Settings, write what happens next, run again, and the Shot node joins the takes into one clip.
*   **Settings:** You can adjust the size in megapixels and the lane: Quality (12 steps) or Speed (using the Turbo LoRA at 6 steps, same weights).

### ⏱️ Performance Notes (M5 Pro, 48 GB)
On my M5 Pro (48 GB), here are the times for a 3 second clip:
*   0.5 MP takes: 3:25 on Speed, 6:14 on Quality.
*   1.0 MP takes: 7:05 on Speed, 12:15 on Quality.

ComfyUI peaked under 10 GB of memory on all four tests because the 18 GB of int8 weights stream from the SSD.

### 🍎 System Requirements
*   Apple silicon Mac (Tested on 48 GB M5 Pro).
*   ffmpeg.
*   MiniMax H3 weights.
*   h3.c engine built with `make libh3.dylib`.
*   The node pack.

*Note: 24 GB+ on M1 or newer should work, but this is untested.*

The first run builds an 18 GB int8 cache. Full steps are in the repo README.

### 🔗 Credits & Source
*   MiniMax H3 by MiniMax
*   h3.c by antirez: https://github.com/antirez/h3.c (my fork with the library build: https://github.com/Bambushu/h3.c)
*   Turbo LoRA by larryvrh: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora

Source, MIT license, and updates are available here: https://github.com/Bambushu/ComfyUI-MacH3-Metal
I entered this in the Open Source Higgsfield challenge. #OpenSourceHiggsfield

# Social post drafts, for Mike to edit and post himself

Attach the demo video. Nothing has been posted.

## X

MacH3 Metal: MiniMax H3 on a MacBook, inside ComfyUI. Text to video, first and
last frame, references, and takes that continue each other. Picture and sound
in one pass, no cloud.

Node pack + Comfy SDK app, open source: [repo link]

#OpenSourceHiggsfield

## Reddit (r/comfyui)

**Title:** MacH3 Metal: every MiniMax H3 mode running natively on Apple silicon in ComfyUI

I built a node pack that runs MiniMax H3 through h3.c, a C and Metal engine, so
a Mac renders H3 video with its own sound inside ComfyUI: text to video, first
and last frame, image/video/audio references and continuation, all from one
workflow with a mode dropdown. Frames and audio come out as normal IMAGE and
AUDIO, so the stock video nodes save them.

The part I like most is continuation. Every take saves its final latents, and the
next take starts from the last 22 frames and 0.9 s of sound. Write a few
seconds, render, write what happens next, keep going: it plays as one take,
same voice, same room.

Numbers on an M5 Pro with 48 GB: a 5 s take at 544x960 takes about 7.5
minutes, the frames match the engine's command line bit for bit, and the
join between takes measures r 0.993.

There is also a small app on the Comfy SDK that drives the same nodes.

Repo, example workflows and setup: [repo link]

Mac only (Metal). It runs outside Comfy's latent graph, so no LoRA loaders or
latent upscale on this path.

#OpenSourceHiggsfield

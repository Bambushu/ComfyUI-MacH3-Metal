# #OpenSourceHiggsfield submission, draft for Mike

Form: https://form.typeform.com/to/V1dSQrkS (closes Friday 10/30, 9am PT).
Nothing here has been submitted. Fill the bracketed parts, tick the
agreements yourself, and send it when you are happy. Delete this `drafts/`
folder before the repo goes public, or keep it private.

## Tell us about yourself

| Question | Answer |
| --- | --- |
| Name | Maikel Slomp |
| Comfy app email | [your Comfy account email] |
| Country | [your call] |
| Social handles | [X / IG / LI / YT handles] |
| Discord username | [ ] |
| Reddit username | [ ] |
| How familiar were you with ComfyUI before this challenge? | I build custom nodes or contribute in other ways |

## Tell us about your project

**Project name/title:** h3c Continuous Take Studio

**One-line description (100 characters or less):**
MiniMax H3 picture and sound made natively on a Mac in ComfyUI, chained into one continuous take

## Submission links

| Field | Value |
| --- | --- |
| GitHub repo URL | [https://github.com/Bambushu/ComfyUI-h3c once public] |
| Demo video URL | [YouTube or X link] |
| Live link to try it | Leave empty: it runs on the user's own Apple silicon Mac, there is no hosted endpoint |
| Social post link | [link to the post below] |

## Which parts of ComfyUI did you use?

Comfy API, Comfy SDK

## List all generative models used

MiniMax H3 (open weights, FL2VA and Ref2VA), run by the h3.c engine. MOSS-Image
SFT for the two synthetic identity pictures in the demo.

## Other tools or techniques used

h3.c, a C and Metal inference engine for MiniMax H3 (MIT; my fork adds motion
context chaining, a resident int8 transformer cache and the shared library
this pack loads). comfy-api-proxy for the local Comfy API v2. ffmpeg. The
community MiniMax H3 Prompt Composer for the demo prompts.

## Which category or categories does your submission best fit?

- Most Innovative
- Open Weights Only
- Built with MiniMax H3 Bonus

## Is your stack fully open?

Yes. MiniMax H3 is open weights, h3.c and this pack are MIT, ComfyUI is
GPL, comfy-sdk and comfy-api-proxy are open source, and nothing calls a closed
API. Everything runs on the user's own Mac.

## Agreements

Tick these yourself: public repo with license; lawful and SFW with no
unlicensed IP or real likenesses (both demo people are synthetic MOSS-Image
pictures); permission to feature; only submission.

## Anything else you want ComfyUI or our judges to know?

ComfyUI-h3c is a custom node pack that runs MiniMax H3 on Apple silicon
through a native C and Metal engine instead of ComfyUI's PyTorch path. It runs
the whole model outside the latent graph and hands back ordinary IMAGE and
AUDIO, so the stock video nodes save the result. One generation is picture and
stereo sound together, on a 48 GB laptop.

The new part is chaining. Each take saves its final latents; the next take
pins the last 22 frames and 0.917 s of sound at its head and keeps going, so
a shot can be written a few seconds at a time and still play as one take,
voice and room included. The Continuous Take Studio app does exactly that
through the Comfy SDK, passing each take's context file to the next job as an
output asset.

Measured on an M5 Pro, 48 GB, 544x960, 8 steps: a 5.2 s take renders in
about 7.5 minutes inside ComfyUI, and the frames are bit-identical to the
engine's command line. The cut between two chained takes measures r 0.993 on
the frames either side.

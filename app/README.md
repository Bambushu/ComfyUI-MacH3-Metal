# MacH3 Metal app

MiniMax-H3 with sound on a Mac, from a browser tab.

Pick a mode at the top (text to video, first frame, first + last, references;
references also takes an optional start and end frame), write the prompt, drop
in the pictures it asks for, pick portrait, landscape or square and 0.2 to 1.0
megapixels, and generate. To keep going, write what happens next and press
"Continue the take": the next generation starts from the last take's final
frames and sound, and the joined shot plays on the right.

Every generation is a ComfyUI job submitted with the Comfy SDK
([comfy-sdk](https://pypi.org/project/comfy-sdk/)) to a Comfy API v2 surface.
Locally that is [comfy-api-proxy](https://pypi.org/project/comfy-api-proxy/)
in front of a ComfyUI with the MacH3 Metal nodes installed. The graph is
`workflows/take_api.json`, exported from `workflows/take_ui.json`; the app only
sets inputs and the mode, prunes the loaders the mode does not use, and passes
the previous take's context file (an output asset) to the next job.

## Run it

```bash
cd app && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/comfy-api-proxy run --comfyui http://127.0.0.1:8188 --port 8191
```

```bash
COMFY_BASE_URL=http://127.0.0.1:8191 .venv/bin/python app.py
```

Open http://127.0.0.1:8766. The proxy's usual port is 8189; any free port
works as long as `COMFY_BASE_URL` matches.

Each shot lands in its own `app/session/<time>/` folder. The joined take is a stream copy of the takes
in order: each continuation already left out the frames it shares with the
take before it.

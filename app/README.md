# Continuous Take Studio

One continuous shot, written a few seconds at a time.

Write what happens, add up to two identity pictures, pick a length and
generate. Then write what happens next and press "Continue the take": the next
generation starts from the last take's final frames and sound, and the joined
shot plays on the right.

Every generation is a ComfyUI job submitted with the Comfy SDK
([comfy-sdk](https://pypi.org/project/comfy-sdk/)) to a Comfy API v2 surface.
Locally that is [comfy-api-proxy](https://pypi.org/project/comfy-api-proxy/)
in front of a ComfyUI with ComfyUI-h3c installed. The graph is
`workflows/take_api.json`, exported from `workflows/take_ui.json`; the app only
sets inputs, prunes unused reference slots and passes the previous take's
context file (an output asset) to the next job.

## Run it

```bash
cd app && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/comfy-api-proxy run --comfyui http://127.0.0.1:8188 --port 8191
```

```bash
COMFY_BASE_URL=http://127.0.0.1:8191 .venv/bin/python studio.py
```

Open http://127.0.0.1:8766. The proxy's usual port is 8189; any free port
works as long as `COMFY_BASE_URL` matches.

Takes land in `app/session/`. The joined take is a stream copy of the takes
in order: each continuation already left out the frames it shares with the
take before it.

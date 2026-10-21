"""Continuous Take Studio: a small web app on the Comfy SDK (Comfy API v2).

Write a prompt, add identity pictures, generate a take; then "continue the
take" chains the next prompt onto the last one with h3c motion context and
shows the joined result. Every generation is a ComfyUI job submitted through
comfy-sdk to a Comfy API v2 surface (locally: comfy-api-proxy in front of a
ComfyUI that has ComfyUI-h3c installed).

    COMFY_BASE_URL=http://127.0.0.1:8191 python studio.py   # then open :8766
"""

from __future__ import annotations

import base64
import copy
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from comfy_sdk import Comfy
from comfy_sdk.events import Progress, StatusChange

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("STUDIO_DIR", HERE / "session"))
GRAPH = json.loads((HERE / "workflows" / "take_api.json").read_text())
IDS, TEMPLATE = GRAPH["ids"], GRAPH["api"]
FFMPEG = os.environ.get("H3_FFMPEG") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


class Studio:
    def __init__(self) -> None:
        os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8191")
        self.client = Comfy()  # self-hosted proxy: no API key
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.takes: list[dict] = []
        self.ref_assets: list = []
        self.status = "idle"
        self.progress = 0.0
        self.error = ""
        self.joined = None
        # One folder per shot, so a new first take never overwrites the last shot.
        self.dir = WORK / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)

    def state(self) -> dict:
        return {"status": self.status, "progress": round(self.progress, 3), "error": self.error,
                "takes": [{k: t[k] for k in ("n", "prompt", "seed", "seconds", "video", "wall")}
                          for t in self.takes],
                "joined": self.joined}

    # -- one take -------------------------------------------------------------
    def build(self, req: dict, continuing: bool):
        wf = self.client.workflows.from_json(copy.deepcopy(TEMPLATE))
        gen = IDS["gen"]
        for field in ("prompt", "seconds", "steps", "seed", "width", "height"):
            if field in req:
                wf.set_input(gen, field, req[field])
        wf.set_input(gen, "save_context", True)
        wf.set_input(gen, "filename_prefix", f"h3c/studio/take{len(self.takes) + 1:02d}")
        wf.set_input(IDS["sv"], "filename_prefix", f"h3c/studio/take{len(self.takes) + 1:02d}")
        # Identity pictures: fill the two reference slots in order, drop unused
        # ones (removing a LoadImage, then its reference node, re-links the chain).
        for slot, (img, ref) in enumerate((("img1", "ref1"), ("img2", "ref2"))):
            if slot < len(self.ref_assets):
                wf.set_input(IDS[img], "image", self.ref_assets[slot])
            else:
                wf.remove_node(IDS[img])
                wf.remove_node(IDS[ref])
        if continuing:
            # The previous take's .h3ctx, an output asset of its Generate node.
            wf.set_input(IDS["ctx"], "path",
                         {"__type": "core/ASSET", "info": {"id": self.takes[-1]["context_id"]}})
        else:
            wf.remove_node(IDS["ctx"])
        return wf

    def run_take(self, req: dict) -> None:
        continuing = bool(req.get("continue")) and bool(self.takes)
        try:
            if not continuing:
                self.reset()
                for i, data_url in enumerate(req.get("refs", [])[:2]):
                    header, b64 = data_url.split(",", 1)
                    ext = ".png" if "png" in header else ".jpg"
                    path = self.dir / f"ref{i + 1}{ext}"
                    path.write_bytes(base64.b64decode(b64))
                    self.ref_assets.append(self.client.assets.from_file(path))
            self.status, self.progress, self.error = "running", 0.0, ""
            started = time.time()
            job = self.client.submit(self.build(req, continuing))
            for event in job.events():
                if isinstance(event, Progress):
                    # The long node is h3c Generate; its own bar (step/steps) is the
                    # engine's phase-weighted progress. event.value counts nodes.
                    if event.steps and event.steps > 1 and event.step is not None:
                        self.progress = event.step / event.steps
                elif isinstance(event, StatusChange):
                    self.status = event.status
            job.result()
            n = len(self.takes) + 1
            video = self.dir / f"take{n:02d}.mp4"
            job.get_outputs(IDS["sv"])[0].to_file(video)
            context = [o for o in job.get_outputs(IDS["gen"]) if o.name.endswith(".h3ctx")][0]
            self.takes.append({"n": n, "prompt": req["prompt"], "seed": req.get("seed"),
                               "seconds": req.get("seconds"), "video": video.name,
                               "context_id": context.id, "wall": round(time.time() - started, 1)})
            self.join()
            self.status, self.progress = "done", 1.0
        except Exception as exc:  # surfaced in the page
            traceback.print_exc()
            self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"

    def join(self) -> None:
        """Each continuation already dropped the frames it shares with the take
        before, so the joined take is a stream-copy concat of the files."""
        if len(self.takes) < 2:
            self.joined = self.takes[0]["video"]
            return
        listing = self.dir / "concat.txt"
        listing.write_text("".join(f"file '{self.dir / t['video']}'\n" for t in self.takes))
        out = self.dir / f"joined_{len(self.takes):02d}.mp4"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", str(out)], check=True)
        self.joined = out.name


STUDIO = Studio()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(STUDIO.state()).encode(), "application/json")
        elif self.path.startswith("/media/"):
            path = STUDIO.dir / os.path.basename(self.path.split("?")[0])
            if path.is_file():
                self._send(200, path.read_bytes(), "video/mp4")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/take":
            return self._send(404, b"not found", "text/plain")
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if STUDIO.status in ("running", "pending", "in_progress"):
            return self._send(409, b'{"error":"a take is already running"}', "application/json")
        if not str(req.get("prompt", "")).strip():
            return self._send(400, b'{"error":"prompt is empty"}', "application/json")
        STUDIO.status = "running"
        threading.Thread(target=STUDIO.run_take, args=(req,), daemon=True).start()
        self._send(202, b'{"ok":true}', "application/json")

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("STUDIO_PORT", "8766"))
    print(f"Continuous Take Studio on http://127.0.0.1:{port}  (Comfy API v2 at {os.environ['COMFY_BASE_URL']})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

"""MacH3 Metal app: MiniMax-H3 on a Mac through the Comfy SDK (Comfy API v2).

Pick a mode, write a prompt, drop in the pictures it asks for, pick the frame
shape and megapixels, generate; "continue the take" then
chains the next prompt onto the last take with motion context and shows the
joined result. Every generation is a ComfyUI job submitted through comfy-sdk
to a Comfy API v2 surface (locally: comfy-api-proxy in front of a ComfyUI
with the MacH3 Metal nodes installed).

    COMFY_BASE_URL=http://127.0.0.1:8191 python app.py   # then open :8766
"""

from __future__ import annotations

import base64
import copy
import re
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
WORK = Path(os.environ.get("MACH3_APP_DIR", HERE / "session"))
GRAPH = json.loads((HERE / "workflows" / "take_api.json").read_text())
IDS, TEMPLATE = GRAPH["ids"], GRAPH["api"]
FFMPEG = os.environ.get("H3_FFMPEG") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def spoken_line(prompt: str) -> str:
    """The dialogue of a composed H3 prompt (inside <d>...</d>), else its start."""
    lines = re.findall(r"<d>(?:\[[^\]]*\]\s*)?(.*?)</d>", prompt, re.S)
    return " ".join(lines) if lines else prompt.strip()[:140]


# mode -> (first frame, last frame, identity pictures)
MODES = {
    "text to video": (False, False, False),
    "first frame": (True, False, False),
    "first + last frame": (True, True, False),
    "references": (False, False, True),
    "references + first frame": (True, False, True),
    "references + first + last frame": (True, True, True),
}


# lane -> h3c Load Engine lora and the default step count. Both lanes run on the
# same base weights; "speed" applies the Turbo step-distillation LoRA at load.
LANES = {"quality": ("", 12), "speed": ("turbo", 6)}


def save_data_url(data_url: str, path_stem: Path) -> Path:
    header, b64 = data_url.split(",", 1)
    path = path_stem.with_suffix(".png" if "png" in header else ".jpg")
    path.write_bytes(base64.b64decode(b64))
    return path


class MacH3App:
    def __init__(self) -> None:
        os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8191")
        self.client = Comfy()  # self-hosted proxy: no API key
        self.lock = threading.Lock()
        self.job = None
        self.reset()

    def reset(self) -> None:
        self.takes: list[dict] = []
        self.ref_assets: list = []
        self.status = "idle"
        self.progress = 0.0
        self.error = ""
        self.joined = None
        self.current = None
        self.refs: list[str] = []
        # One folder per shot, so a new first take never overwrites the last shot.
        self.dir = WORK / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)

    def state(self) -> dict:
        return {"status": self.status, "progress": round(self.progress, 3), "error": self.error,
                "takes": [{k: t[k] for k in ("n", "line", "seed", "seconds", "video", "wall",
                                             "lane", "steps")}
                          for t in self.takes],
                "current": self.current, "refs": self.refs, "joined": self.joined}

    # -- one take -------------------------------------------------------------
    def build(self, req: dict, continuing: bool, frames: dict):
        """Set the inputs of the shipped take graph and prune what the mode does
        not use, so nothing unneeded is validated, uploaded or loaded."""
        wf = self.client.workflows.from_json(copy.deepcopy(TEMPLATE))
        gen = IDS["gen"]
        lora, steps = LANES[req.get("lane", "quality")]
        req.setdefault("steps", steps)
        wf.set_input(IDS["load"], "lora", req.get("lora", lora))
        for field in ("prompt", "seconds", "steps", "seed", "width", "height", "reference_image_size", "speed"):
            if field in req:
                wf.set_input(gen, field, req[field])
        if req.get("model_dir"):  # another weights folder (base weights: lanes add the LoRA)
            wf.set_input(IDS["load"], "model_dir", req["model_dir"])
        n = len(self.takes) + 1
        wf.set_input(gen, "save_context", True)
        wf.set_input(gen, "filename_prefix", f"MacH3/app/take{n:02d}")
        wf.set_input(IDS["sv"], "filename_prefix", f"MacH3/app/take{n:02d}")
        if continuing:
            mode = "continue + references" if self.ref_assets else "continue"
            # The previous take's .h3ctx, an output asset of its Generate node.
            wf.set_input(IDS["ctx"], "path",
                         {"__type": "core/ASSET", "info": {"id": self.takes[-1]["context_id"]}})
        else:
            mode = req.get("mode", "text to video")
            wf.remove_node(IDS["ctx"])
        wf.set_input(gen, "mode", mode)
        for key in ("first", "last"):
            if key in frames and not continuing:
                wf.set_input(IDS[key], "image", frames[key])
            else:
                wf.remove_node(IDS[key])
        # Reference pictures: fill the three reference slots in order, drop unused
        # ones (removing a LoadImage, then its reference node, re-links the chain).
        uses_refs = continuing or MODES[mode][2]
        for slot, (img, ref) in enumerate((("img1", "ref1"), ("img2", "ref2"), ("img3", "ref3"))):
            if uses_refs and slot < len(self.ref_assets):
                wf.set_input(IDS[img], "image", self.ref_assets[slot])
            else:
                wf.remove_node(IDS[img])
                wf.remove_node(IDS[ref])
        return wf

    def run_take(self, req: dict) -> None:
        continuing = bool(req.get("continue")) and bool(self.takes)
        if continuing:  # motion context only continues a take of the same frame size
            req.update(width=self.takes[-1]["width"], height=self.takes[-1]["height"])
        try:
            mode = req.get("mode", "text to video")
            if not continuing and mode not in MODES:
                raise ValueError(f"unknown mode {mode!r}")
            if req.get("lane", "quality") not in LANES:
                raise ValueError(f"unknown lane {req['lane']!r}")
            frames = {}
            if not continuing:
                needs = dict(zip(("first", "last", "refs"), MODES[mode]))
                for key, what in (("first", "a first frame"), ("last", "a last frame"),
                                  ("refs", "at least one identity picture")):
                    if needs[key] and not req.get(key):
                        raise ValueError(f"{mode} needs {what}")
                self.reset()
                for i, data_url in enumerate(req.get("refs", [])[:3] if needs["refs"] else []):
                    path = save_data_url(data_url, self.dir / f"ref{i + 1}")
                    self.ref_assets.append(self.client.assets.from_file(path))
                    self.refs.append(path.name)
                for key in ("first", "last"):
                    if needs[key]:
                        frames[key] = self.client.assets.from_file(
                            save_data_url(req[key], self.dir / f"{key}_frame"))
            self.status, self.progress, self.error = "running", 0.0, ""
            self.current = {"n": len(self.takes) + 1, "line": spoken_line(req["prompt"]),
                            "seconds": req.get("seconds"), "seed": req.get("seed"),
                            "lane": req.get("lane", "quality")}
            started = time.time()
            job = self.job = self.client.submit(self.build(req, continuing, frames))
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
            self.takes.append({"n": n, "line": spoken_line(req["prompt"]), "seed": req.get("seed"),
                               "seconds": req.get("seconds"), "video": video.name,
                               "lane": req.get("lane", "quality"), "steps": req["steps"],
                               "context_id": context.id, "wall": round(time.time() - started, 1),
                               "width": req.get("width", 544), "height": req.get("height", 960)})
            self.join()
            self.current = None
            self.status, self.progress = "done", 1.0
        except Exception as exc:  # surfaced in the page
            traceback.print_exc()
            self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.job = None

    def join(self) -> None:
        """Each continuation already dropped the frames it shares with the take
        before, so the takes join end to end. Each file is decoded on its own and
        its sound cut to its own video length: a stream-copy concat would keep
        every take's AAC priming and slide the sound later at each join."""
        if len(self.takes) < 2:
            self.joined = self.takes[0]["video"]
            return
        args, chains, pads = [], "", ""
        for i, t in enumerate(self.takes):
            path = self.dir / t["video"]
            frames = int(subprocess.run(
                [FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                 str(path)], check=True, capture_output=True, text=True).stdout.strip())
            secs = frames / 24
            args += ["-i", str(path)]
            chains += (f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim=0:{secs},"
                       f"asetpts=PTS-STARTPTS,apad=whole_dur={secs}[a{i}];")
            pads += f"[v{i}][a{i}]"
        out = self.dir / f"joined_{len(self.takes):02d}.mp4"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", *args, "-filter_complex",
                        f"{chains}{pads}concat=n={len(self.takes)}:v=1:a=1[v][a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "14",
                        "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                        str(out)], check=True)
        self.joined = out.name


APP = MacH3App()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        """Serve byte ranges: Safari's <video> only plays and seeks with 206 replies."""
        ctype = "video/mp4" if path.suffix == ".mp4" else "image/png" if path.suffix == ".png" else "image/jpeg"
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            start, end = (int(m[1]), int(m[2] or end)) if m[1] else (max(size - int(m[2]), 0), end)
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if m and (m[1] or m[2]) else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m and (m[1] or m[2]):
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0 and (chunk := f.read(min(left, 1 << 20))):
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(APP.state()).encode(), "application/json")
        elif self.path.startswith("/media/"):
            path = APP.dir / os.path.basename(self.path.split("?")[0])
            if path.is_file():
                self._send_file(path)
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path == "/api/cancel":
            job = APP.job
            if job is not None:
                job.cancel()  # ComfyUI interrupts h3c Generate between denoise steps
            return self._send(202 if job else 409, b'{"ok":true}' if job else b'{"error":"nothing is running"}',
                              "application/json")
        if self.path != "/api/take":
            return self._send(404, b"not found", "text/plain")
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if APP.status in ("running", "pending", "in_progress"):
            return self._send(409, b'{"error":"a take is already running"}', "application/json")
        if not str(req.get("prompt", "")).strip():
            return self._send(400, b'{"error":"prompt is empty"}', "application/json")
        APP.status = "running"
        threading.Thread(target=APP.run_take, args=(req,), daemon=True).start()
        self._send(202, b'{"ok":true}', "application/json")

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("MACH3_APP_PORT", "8766"))
    print(f"MacH3 Metal app on http://127.0.0.1:{port}  (Comfy API v2 at {os.environ['COMFY_BASE_URL']})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

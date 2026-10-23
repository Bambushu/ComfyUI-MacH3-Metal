"""ComfyUI nodes for h3.c, a native C/Metal MiniMax-H3 engine on Apple silicon.

The engine runs the whole generation itself (text encoder, DiT, both VAEs)
outside ComfyUI's model and latent graph. These nodes hand it a prompt and
files and take back IMAGE frames and AUDIO, so the stock video nodes
(CreateVideo, SaveVideo) finish the job.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time
import wave

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths

from . import h3c_engine as h3c

ENGINE = "H3C_ENGINE"
REFS = "H3C_REFS"
CONTEXT = "H3C_CONTEXT"
SETTINGS = "H3C_SETTINGS"
CATEGORY = "h3c"

# Share of the progress bar per engine phase (denoise dominates the wall).
PHASE_WEIGHTS = [
    ("tokenizer", 0.0, 0.01), ("text encoder", 0.01, 0.06),
    ("Qwen vision", 0.01, 0.06), ("video VAE encoder", 0.06, 0.10),
    ("audio VAE encoder", 0.06, 0.10), ("denoise", 0.10, 0.85),
    ("audio VAE", 0.85, 0.88), ("video VAE", 0.88, 0.97), ("FFmpeg", 0.97, 1.0),
]
BAR = 1000


def _work_dir(sub: str = "h3c") -> str:
    path = os.path.join(folder_paths.get_temp_directory(), sub)
    os.makedirs(path, exist_ok=True)
    return path


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def _save_image(image: torch.Tensor, name: str) -> str:
    """First frame of an IMAGE batch to a PNG the engine can read."""
    if image.shape[0] > 1:
        print(f"h3c: {name}: got {image.shape[0]} frames, using only the first one "
              f"(use h3c Reference Video for a clip)")
    array = (image[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    path = os.path.join(_work_dir(), f"{name}-{_stamp()}.png")
    Image.fromarray(array).save(path)
    return path


def _write_wav(audio: dict, path: str) -> str:
    """AUDIO dict to a 16-bit WAV at path."""
    waveform = audio["waveform"][0].clamp(-1, 1).cpu().numpy()  # [C, T]
    pcm = (waveform.T * 32767.0).astype("<i2")
    with wave.open(path, "wb") as out:
        out.setnchannels(pcm.shape[1])
        out.setsampwidth(2)
        out.setframerate(int(audio["sample_rate"]))
        out.writeframes(pcm.tobytes())
    return path


def _save_audio(audio: dict, name: str) -> str:
    """AUDIO dict to a 16-bit WAV in the work folder; the engine resamples references itself."""
    return _write_wav(audio, os.path.join(_work_dir(), f"{name}-{_stamp()}.wav"))


def _save_video(video, name: str) -> str:
    path = os.path.join(_work_dir(), f"{name}-{_stamp()}.mp4")
    video.save_to(path)
    return path


def _read_audio(mp4: str, sample_rate: int, frames: int, fps: int) -> dict:
    """The engine's soundtrack, trimmed or padded to exactly the delivered
    frames so consecutive takes concatenate on the frame grid."""
    ffmpeg = os.environ.get("H3_FFMPEG", "ffmpeg")
    try:
        raw = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", mp4, "-vn", "-ac", "2",
             "-ar", str(sample_rate), "-f", "f32le", "-"],
            check=True, capture_output=True).stdout
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg not found ({ffmpeg}): brew install ffmpeg, or set H3_FFMPEG") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg could not read the engine's audio from {mp4}: "
                           f"{exc.stderr.decode(errors='replace').strip()[-300:]}") from None
    pcm = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy()  # [2, T]
    want = int(round(frames * sample_rate / fps))
    if pcm.shape[1] >= want:
        pcm = pcm[:, :want]
    else:
        pcm = np.pad(pcm, ((0, 0), (0, want - pcm.shape[1])))
    return {"waveform": torch.from_numpy(pcm).unsqueeze(0), "sample_rate": sample_rate}


def _resolve_comfy_path(name: str) -> str:
    name = os.path.expanduser(name.strip())
    if name.endswith(" [output]") or name.endswith(" [input]"):
        name = name.rsplit(" [", 1)[0]
    candidates = [name] if os.path.isabs(name) else [
        os.path.join(folder_paths.get_output_directory(), name),
        os.path.join(folder_paths.get_input_directory(), name)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"h3c: no motion context file at {name}")


class H3cLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "engine_dir": ("STRING", {"default": "", "tooltip":
                "Folder with libh3.dylib and h3_shaders.metal (an h3.c checkout after "
                "`make libh3.dylib`). Empty = $H3C_DIR or ~/h3c."}),
            "model_dir": ("STRING", {"default": "", "tooltip":
                "MiniMax-H3 weights folder with FL2VA/ and Ref2VA/. Empty = $H3C_MODEL_DIR, "
                "else <engine_dir>/MiniMax-H3."}),
            "keep_resident": ("BOOLEAN", {"default": False, "tooltip":
                "Keep the prepared DiT and VAE decoder in memory between runs. Off frees "
                "everything after each take (the engine's one-shot lifetime)."}),
        }, "optional": {
            "lora": ("STRING", {"default": "", "tooltip":
                "Empty = base weights (quality lane). \"turbo\" = the Turbo step-distillation "
                "LoRA (speed lane: render at 5 or 6 steps), found via $H3C_TURBO_LORA, "
                "<engine_dir>/turbo_lora/, ~/h3c/turbo_lora/, the model folder or models/loras. Or a path to "
                "another DiT LoRA. Applied while the weights load; its int8 cache (18 GiB) "
                "is built once on the first run."}),
            "lora_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                "tooltip": "LoRA strength. 1 for the Turbo adapter."}),
        }}

    RETURN_TYPES = (ENGINE,)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Open the h3.c engine. Reads model metadata only; weights map per generation."

    def load(self, engine_dir, model_dir, keep_resident, lora="", lora_scale=1.0):
        engine_dir = engine_dir.strip() or h3c.default_engine_dir()
        model_dir = model_dir.strip() or h3c.default_model_dir(os.path.expanduser(engine_dir))
        return (h3c.Engine(engine_dir, model_dir, keep_resident,
                           lora=lora.strip() or None, lora_scale=lora_scale),)


def _append(refs, item):
    return (list(refs or []) + [item],)


class H3cReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)},
                "optional": {"references": (REFS,)}}

    RETURN_TYPES = (REFS,)
    RETURN_NAMES = ("references",)
    FUNCTION = "add"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Append an identity/look reference. References are numbered in chain "
                   "order per kind: the first image is <Picture 1> in the prompt.")

    def add(self, image, references=None):
        return _append(references, h3c.Reference(h3c.REFERENCE_IMAGE, _save_image(image, "ref")))


class H3cReferenceVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",),
                             "include_audio": ("BOOLEAN", {"default": True})},
                "optional": {"replacement_audio": ("AUDIO",), "references": (REFS,)}}

    RETURN_TYPES = (REFS,)
    RETURN_NAMES = ("references",)
    FUNCTION = "add"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Append a reference video (<Video N>), with its own soundtrack, a "
                   "replacement soundtrack, or silent.")

    def add(self, video, include_audio, replacement_audio=None, references=None):
        path = _save_video(video, "refvid")
        if replacement_audio is not None:
            ref = h3c.Reference(h3c.REFERENCE_VIDEO_AUDIO, path,
                                _save_audio(replacement_audio, "refvid-audio"))
        else:
            ref = h3c.Reference(h3c.REFERENCE_VIDEO, path, None, include_audio)
        return _append(references, ref)


class H3cReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)},
                "optional": {"references": (REFS,)}}

    RETURN_TYPES = (REFS,)
    RETURN_NAMES = ("references",)
    FUNCTION = "add"
    CATEGORY = CATEGORY
    DESCRIPTION = "Append a standalone reference audio clip (<Audio N>, up to 45 s)."

    def add(self, audio, references=None):
        return _append(references, h3c.Reference(h3c.REFERENCE_AUDIO, _save_audio(audio, "refaud")))


class H3cContextFromVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = (CONTEXT,)
    RETURN_NAMES = ("motion_context",)
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Continue any finished clip (45 s max): the engine re-encodes its last 22 "
                   "frames and their sound. A .h3ctx from a take is bit-exact and skips that.")

    def make(self, video):
        return ({"path": _save_video(video, "context"), "kind": "video"},)


class H3cLoadContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "", "tooltip":
                "A .h3ctx written by a take with save_context, or an .mp4. Absolute, or "
                "relative to ComfyUI's output then input folder (an API output asset "
                "resolves to such a relative name)."})}}

    RETURN_TYPES = (CONTEXT,)
    RETURN_NAMES = ("motion_context",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, path):
        path = _resolve_comfy_path(path)
        return ({"path": path, "kind": "h3ctx" if path.endswith(".h3ctx") else "video"},)


# Generate modes: which optional inputs a take uses.
# The four modes the dropdown offers; each reads only its own inputs.
MAIN_MODES = {
    "T2V · text to video": set(),
    "I2V · first frame": {"first_frame"},
    "FL2V · first + last frame": {"first_frame", "last_frame"},
    "R2V · references": {"references"},
}
# Older names, still accepted (the app and earlier workflows send them).
GENERATE_MODES = {
    **MAIN_MODES,
    "auto": {"first_frame", "last_frame", "references", "motion_context"},
    "text to video": set(),
    "first frame": {"first_frame"},
    "first + last frame": {"first_frame", "last_frame"},
    "references": {"references"},
    "references + first frame": {"references", "first_frame"},
    "references + first + last frame": {"references", "first_frame", "last_frame"},
    "continue": {"motion_context"},
    "continue + references": {"motion_context", "references"},
    "continue + references + last frame": {"motion_context", "references", "last_frame"},
}

CONTINUE_FROM = ["off", "last take", "clip input"]
# The newest take per filename_prefix, for continue_from = "last take".
_LAST_TAKES: dict = {}


def _wanted(mode: str, continue_from: str = "off") -> set:
    """Inputs a take reads. Continuing drops the first frame (the previous take's
    tail replaces it) and keeps the mode's references and last frame."""
    wanted = set(GENERATE_MODES.get(mode, GENERATE_MODES["auto"]))
    if continue_from != "off":
        wanted -= {"first_frame", "motion_context"}
        if continue_from == "clip input":
            wanted.add("motion_context")
    return wanted


def _last_take_context(filename_prefix: str) -> dict:
    if filename_prefix in _LAST_TAKES:
        return dict(_LAST_TAKES[filename_prefix])
    # After a restart: the newest context file this prefix saved, and its size.
    pattern = os.path.join(folder_paths.get_output_directory(), f"{filename_prefix}_ctx_*.h3ctx")
    found = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not found:
        raise ValueError("h3c: there is no take to continue yet. Run one with Continue off "
                         "(save_context on) first.")
    last = {"path": found[-1], "kind": "h3ctx"}
    try:
        with open(found[-1] + ".json") as f:
            last["size"] = tuple(json.load(f)["size"])
    except (OSError, ValueError, KeyError):
        pass
    return last


class H3cGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (ENGINE,),
                "mode": (list(MAIN_MODES), {"default": "T2V · text to video", "tooltip":
                    "Which inputs this take uses. Inputs the mode leaves out are skipped, and "
                    "the loaders feeding them never run (in the all-in-one their groups grey out)."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 544, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 960, "min": 64, "max": 2048, "step": 32}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 15.0, "step": 0.25,
                    "tooltip": "Length at 24 fps. A continuation includes the 22 pinned frames "
                               "(0.917 s) here and delivers that much less."}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 362,
                    "tooltip": "0 = use seconds. The engine aligns to its 17-frame grid (5..362)."}),
                "steps": ("INT", {"default": 8, "min": 2, "max": 100,
                    "tooltip": "8 is the measured M5 Pro production lane; the engine's own default is 20."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True}),
                "sampler": (h3c.SAMPLERS, {"default": "euler"}),
                "scheduler": (h3c.SCHEDULERS, {"default": "linear"}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Video sigma shift. 0 = the released 12. Audio stays at 3."}),
                "speed": (list(h3c.SPEED_PRESETS), {"default": "exact"}),
                "save_context": ("BOOLEAN", {"default": True, "tooltip":
                    "Write this take's final latents (.h3ctx) so a later take continues it bit-exact."}),
                "preview": ("BOOLEAN", {"default": False, "tooltip":
                    "Decode one frame per denoise step into the progress bar. Costs about "
                    "10 GiB more unified memory and some decode time."}),
                "filename_prefix": ("STRING", {"default": "h3c/take"}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"lazy": True}),
                "last_frame": ("IMAGE", {"lazy": True}),
                "references": (REFS, {"lazy": True}),
                "motion_context": (CONTEXT, {"lazy": True, "tooltip":
                    "Continue a previous take. Replaces first_frame; references and last_frame ride along."}),
                "reference_image_size": (["match", "max"], {"default": "max", "tooltip":
                    "Size reference pictures are encoded at. max = their own size up to 2048 px "
                    "on the short side, never upscaled (the most detail, more memory and time "
                    "for big pictures); match = scaled down to the video's size."}),
                "settings": (SETTINGS, {"tooltip":
                    "From h3c Settings: overrides this node's size, steps, sampler, scheduler, "
                    "shift, speed, preview, save_context and reference size, so every take of "
                    "a chain shares one set."}),
                "continue_from": (CONTINUE_FROM, {"default": "off", "tooltip":
                    "off = a new shot. last take = pick up where the previous take ended. "
                    "Works in every mode: the previous take's last 22 frames and 0.9 s of "
                    "sound replace the first frame (I2V/FL2V first frames are not used), "
                    "references and a last frame still apply, and T2V becomes a plain "
                    "continuation. clip input = continue the clip in the Continue a clip group."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", CONTEXT)
    RETURN_NAMES = ("images", "audio", "fps", "context")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Generate video with native sound on h3.c (Apple silicon). T2V, first/last "
                   "frame, Ref2VA references and motion-context continuation.")

    @classmethod
    def IS_CHANGED(cls, continue_from="off", filename_prefix="h3c/take", **_):
        # A "last take" continuation depends on which take is last, which ComfyUI's
        # cache cannot see: key the cache on it.
        if continue_from != "last take":
            return ""
        try:
            return _last_take_context(filename_prefix)["path"]
        except ValueError:
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, mode):
        # The dropdown shows MAIN_MODES; the older names stay valid.
        return True if mode in GENERATE_MODES else f"h3c: unknown mode {mode!r}"

    def check_lazy_status(self, mode="auto", continue_from="off", **inputs):
        # A connected input that has not run yet arrives as None; an unconnected
        # one is absent, and must not be requested.
        wanted = _wanted(mode, continue_from)
        return [k for k in wanted if k in inputs and inputs[k] is None]

    def generate(self, engine, prompt, width, height, seconds, frames, steps, seed,
                 sampler, scheduler, shift, speed, save_context, preview, filename_prefix,
                 first_frame=None, last_frame=None, references=None, motion_context=None,
                 reference_image_size="max", mode="auto", settings=None, continue_from="off"):
        wanted = _wanted(mode, continue_from)
        first_frame = first_frame if "first_frame" in wanted else None
        last_frame = last_frame if "last_frame" in wanted else None
        references = references if "references" in wanted else None
        motion_context = motion_context if "motion_context" in wanted else None
        if settings:
            width, height, steps = settings["width"], settings["height"], settings["steps"]
            sampler, scheduler, shift = settings["sampler"], settings["scheduler"], settings["shift"]
            speed, preview = settings["speed"], settings["preview"]
            save_context = settings["save_context"]
            reference_image_size = settings["reference_image_size"]
        turbo = bool(settings) and settings.get("lane", "").startswith("Speed")
        if continue_from == "last take":
            last = _last_take_context(filename_prefix)
            if "size" in last and (width, height) != last["size"]:
                print(f"h3c: continuing at the last take's size {last['size'][0]}x{last['size'][1]}")
                width, height = last["size"]
            motion_context = {"path": last["path"], "kind": last["kind"]}
        if not prompt.strip():
            raise ValueError("h3c: the prompt is empty")
        if mode != "auto":  # say what is missing before anything loads
            missing = [what for name, what, value in (
                ("first_frame", "a first frame", first_frame),
                ("last_frame", "a last frame", last_frame),
                ("references", "a reference picture", references),
                ("motion_context", "a clip to continue (motion_context)", motion_context))
                if name in wanted and not value]
            if missing:
                raise ValueError(f"h3c: {mode} needs {' and '.join(missing)}; connect it or "
                                 f"switch the mode")
        if motion_context is not None and first_frame is not None:
            raise ValueError("h3c: motion_context replaces first_frame; pick a mode that uses one of them")
        # Nothing else may hold an H3 (or any) model while this runs.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        stamp = _stamp()
        out_dir = folder_paths.get_output_directory()
        context_path = None
        if save_context:
            # "_ctx_" keeps SaveVideo's counter scan (which parses the digits after
            # "<prefix>_") from reading the seed as a file counter.
            context_path = os.path.join(out_dir, f"{filename_prefix}_ctx_{seed}_{stamp}.h3ctx")
            os.makedirs(os.path.dirname(context_path), exist_ok=True)
        mp4 = os.path.join(_work_dir(), f"take-{seed}-{stamp}.mp4")

        request = h3c.Request(
            prompt=prompt, width=width, height=height,
            frames=frames if frames > 0 else h3c.frames_for_seconds(seconds),
            steps=steps, seed=seed, sampler=sampler, scheduler=scheduler, shift=shift,
            speed=speed, output_path=mp4,
            first_frame=_save_image(first_frame, "first") if first_frame is not None else None,
            last_frame=_save_image(last_frame, "last") if last_frame is not None else None,
            motion_context=motion_context["path"] if motion_context else None,
            save_context=context_path, references=list(references or []),
            reference_image_size=reference_image_size, preview=preview, turbo=turbo)

        bar = comfy.utils.ProgressBar(BAR)
        state = {"images": None, "last": -1}

        def update(*args):
            """ComfyUI's progress hook raises InterruptProcessingException on a cancel. These
            callbacks run inside the engine through ctypes, which swallows the exception and
            returns 0 (keep going), so catch it here and return 1 to stop the engine."""
            try:
                bar.update_absolute(*args)
            except comfy.model_management.InterruptProcessingException:
                return True
            return comfy.model_management.processing_interrupted()

        def on_progress(phase, done, total):
            for name, lo, hi in PHASE_WEIGHTS:
                if phase.startswith(name) and total > 0:
                    value = int(BAR * (lo + (hi - lo) * min(done, total) / total))
                    if value > state["last"]:
                        state["last"] = value
                        return update(value, BAR)
                    break
            return comfy.model_management.processing_interrupted()

        def on_frame(w, h, stride, rgb, index, count, step, steps_):
            rows = np.ctypeslib.as_array(rgb, shape=(h * stride,)).reshape(h, stride)
            pixels = rows[:, : w * 3].reshape(h, w, 3)
            if step >= 0:  # a denoising preview
                value = max(state["last"], int(BAR * (0.10 + 0.75 * (step + 1) / max(steps_, 1))))
                return update(value, BAR, ("JPEG", Image.fromarray(pixels.copy()), 512))
            if state["images"] is None:
                state["images"] = np.empty((count, h, w, 3), dtype=np.uint8)
            state["images"][index] = pixels
            return False

        print(f"h3c: {request.frames} frames {width}x{height}, {steps} steps, seed {seed}, "
              f"{len(request.references)} refs, context {'in' if motion_context else 'none'}")
        started = time.time()
        try:
            result = engine.generate(request, on_frame, on_progress)
        except h3c.H3cError:
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise
        comfy.model_management.throw_exception_if_processing_interrupted()
        print(f"h3c: delivered {result.frames} frames in {time.time() - started:.1f} s")
        bar.update_absolute(BAR, BAR)
        if state["images"] is None:
            raise RuntimeError("h3c: the engine finished but delivered no frames")

        images = torch.from_numpy(state["images"]).float().div_(255.0)
        audio = _read_audio(mp4, result.sample_rate, result.frames, result.fps)
        context = ({"path": context_path, "kind": "h3ctx"} if context_path
                   else {"path": mp4, "kind": "video"})
        if context_path:
            with open(context_path + ".json", "w") as f:  # lets a restart continue at this size
                json.dump({"size": [width, height]}, f)
        _LAST_TAKES[filename_prefix] = dict(context, size=(width, height))
        # Lineage for h3c Add to Shot: this take, and the take it continues.
        context.update(take_id=context["path"], continues=motion_context is not None,
                       parent=motion_context["path"] if motion_context else None)
        ui = {}
        if context_path:
            # Listed as a file output: the Comfy API v2 proxy exposes it as an
            # asset, so an API client can pass it to h3c Load Motion Context.
            rel = os.path.relpath(context_path, out_dir)
            ui["files"] = [{"filename": os.path.basename(rel),
                            "subfolder": os.path.dirname(rel), "type": "output"}]
        return {"ui": ui, "result": (images, audio, float(result.fps), context)}


class H3cJoinTakes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images_a": ("IMAGE",), "audio_a": ("AUDIO",),
                             "images_b": ("IMAGE",), "audio_b": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "join"
    CATEGORY = CATEGORY
    DESCRIPTION = ("A take followed by its continuation. The continuation already dropped the "
                   "frames it shares with A, so this is a plain cut on the frame grid.")

    def join(self, images_a, audio_a, images_b, audio_b):
        if images_a.shape[1:] != images_b.shape[1:]:
            raise ValueError(f"h3c: takes differ in size {tuple(images_a.shape[1:])} vs "
                             f"{tuple(images_b.shape[1:])}")
        rate = audio_a["sample_rate"]
        if audio_b["sample_rate"] != rate:
            raise ValueError("h3c: takes differ in sample rate")
        wave_a, wave_b = audio_a["waveform"], audio_b["waveform"]
        if wave_a.shape[1] != wave_b.shape[1]:
            raise ValueError("h3c: takes differ in channel count")
        return (torch.cat([images_a, images_b], dim=0),
                {"waveform": torch.cat([wave_a, wave_b], dim=2), "sample_rate": rate})



MEGAPIXELS = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
ASPECTS = {"9:16 portrait": 9 / 16, "16:9 landscape": 16 / 9, "1:1 square": 1.0}
MAX_PIXELS = 768 * 1344  # the largest canvas the released weights take
LANES = ["Quality", "Speed (Turbo)"]


def size_for(megapixels: str, aspect: str) -> tuple[int, int]:
    """Width and height for a megapixel count (1 MP = 1024 x 1024, as in ComfyUI)
    and an aspect, on the engine's 32-pixel grid and within its pixel cap."""
    pixels, ratio = float(megapixels) * 1024 * 1024, ASPECTS[aspect]
    r32 = lambda v: max(64, int(round(v / 32)) * 32)
    w, h = r32((pixels * ratio) ** 0.5), r32((pixels / ratio) ** 0.5)
    while w * h > MAX_PIXELS:
        if w > h:
            w -= 32
        elif h > w:
            h -= 32
        else:
            w, h = w - 32, h - 32
    return w, h


class H3cSettings:
    # Generate fields this node takes over; Generate hides them while it is wired in.
    OVERRIDES = ("width", "height", "steps", "sampler", "scheduler", "shift", "speed",
                 "save_context", "preview", "reference_image_size")

    @classmethod
    def INPUT_TYPES(cls):
        g = H3cGenerate.INPUT_TYPES()
        req, opt = g["required"], g["optional"]
        spec = {"megapixels": (MEGAPIXELS, {"default": "0.5", "tooltip":
                    "Video size in megapixels (1 MP = 1024 x 1024), on the engine's 32-pixel grid. "
                    "1.0 (768 x 1344 portrait) is the largest canvas the weights take."}),
                "aspect": (list(ASPECTS), {"default": "9:16 portrait"}),
                "lane": (LANES, {"default": LANES[0], "tooltip":
                    "Quality = the released weights (12-16 steps). Speed = the same weights with "
                    "the Turbo LoRA applied while they load (5-6 steps; its first take builds an "
                    "18 GiB cache once). Switching the lane sets the steps to match."})}
        for k in ("steps", "sampler", "scheduler", "shift", "speed", "save_context", "preview"):
            spec[k] = req[k]
        spec["reference_image_size"] = opt["reference_image_size"]
        return {"required": spec}

    RETURN_TYPES = (SETTINGS,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = "Size, steps and sampling for the take(s) it is wired to."

    def make(self, megapixels, aspect, **kwargs):
        width, height = size_for(megapixels, aspect)
        return (dict(kwargs, width=width, height=height),)



class H3cJoinAll:
    @classmethod
    def INPUT_TYPES(cls):
        lazy = {"lazy": True}
        return {"required": {"takes": ("INT", {"default": 1, "min": 1, "max": 3}),
                             "images_1": ("IMAGE", lazy), "audio_1": ("AUDIO", lazy)},
                "optional": {"images_2": ("IMAGE", lazy), "audio_2": ("AUDIO", lazy),
                             "images_3": ("IMAGE", lazy), "audio_3": ("AUDIO", lazy)}}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "join"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Takes 1 to N of a chain as one shot. Takes past N are never rendered: "
                   "this node only asks for the ones it needs.")

    def check_lazy_status(self, takes=1, **inputs):
        # Request only connected inputs (present in **inputs), never absent ones.
        return [f"{kind}_{n}" for n in range(1, takes + 1) for kind in ("images", "audio")
                if f"{kind}_{n}" in inputs and inputs[f"{kind}_{n}"] is None]

    def join(self, takes, **inputs):
        images, audio = inputs["images_1"], inputs["audio_1"]
        for n in range(2, takes + 1):
            images, audio = H3cJoinTakes().join(images, audio, inputs[f"images_{n}"],
                                                inputs[f"audio_{n}"])
        return (images, audio)


_SHOTS: dict = {}  # filename_prefix -> the shot being built: {"dir", "format", "takes"}


def _load_shot(filename_prefix: str):
    """The newest shot on disk for this prefix (after a restart)."""
    base = os.path.join(folder_paths.get_output_directory(), filename_prefix)
    for folder in sorted(glob.glob(base + "_[0-9][0-9][0-9][0-9][0-9]"), reverse=True):
        try:
            with open(os.path.join(folder, "shot.json")) as f:
                shot = json.load(f)
            shot["dir"] = folder
            return shot
        except (OSError, ValueError):
            continue
    return None


def _shot_ui(shot: dict) -> dict:
    rel = os.path.relpath(shot["dir"] + ".mp4", folder_paths.get_output_directory())
    return {"ui": {"images": [{"filename": os.path.basename(rel),
                               "subfolder": os.path.dirname(rel), "type": "output"}],
                   "animated": (True,)}}


class H3cAddToShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "images": ("IMAGE",), "audio": ("AUDIO",),
                    "fps": ("FLOAT", {"default": 24.0, "forceInput": True}),
                    "context": (CONTEXT, {"tooltip":
                        "From h3c Generate: which take this is and which take it continues. "
                        "A take that continues this shot's last take is added to the end; any "
                        "other take starts a new shot."}),
                    "filename_prefix": ("STRING", {"default": "MacH3/shot"})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "add"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Builds the shot across runs: a take that continues the shot's last take is "
                   "added to the end, any other take starts a new shot. Plays the whole shot.")

    def add(self, images, audio, fps, context, filename_prefix):
        ffmpeg = os.environ.get("H3_FFMPEG", "ffmpeg")
        take_id, parent = context.get("take_id") or context.get("path"), context.get("parent")
        shot = _SHOTS.get(filename_prefix) or _load_shot(filename_prefix)
        if shot and any(t["id"] == take_id for t in shot["takes"]):
            return _shot_ui(shot)  # this take is already in the shot (a cached rerun)
        frames, h, w = images.shape[0], images.shape[1], images.shape[2]
        fmt = {"width": int(w), "height": int(h), "fps": float(fps),
               "rate": int(audio["sample_rate"]), "channels": int(audio["waveform"].shape[1])}
        appending = bool(shot and shot["takes"] and context.get("continues")
                         and parent == shot["takes"][-1]["id"])
        if appending and shot["format"] != fmt:
            raise ValueError(f"h3c: this take ({w}x{h}, {fps} fps) does not match the shot it "
                             f"continues ({shot['format']['width']}x{shot['format']['height']}, "
                             f"{shot['format']['fps']} fps)")
        if not appending:
            base = os.path.join(folder_paths.get_output_directory(), filename_prefix)
            os.makedirs(os.path.dirname(base), exist_ok=True)
            n = 1
            while os.path.exists(f"{base}_{n:05}"):
                n += 1
            shot = {"dir": f"{base}_{n:05}", "format": fmt, "takes": []}
            os.makedirs(shot["dir"])
        k = len(shot["takes"]) + 1
        video = os.path.join(shot["dir"], f"take{k:02d}.mp4")
        wav = os.path.join(shot["dir"], f"take{k:02d}.wav")
        pixels = (images.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", video],
                       input=pixels.tobytes(), check=True, capture_output=True)
        _write_wav(audio, wav)
        takes = shot["takes"] + [{"id": take_id, "video": os.path.basename(video),
                                  "wav": os.path.basename(wav), "frames": int(frames)}]
        # Video segments share one encoder setup and format, so they stream-copy; the
        # sound is joined as PCM (each take's audio is exactly its frames long), so no
        # AAC priming gap lands at a join. The shot is written aside and swapped in.
        listing = os.path.join(shot["dir"], "segments.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{t['video']}'\n" for t in takes)
        joined_wav = os.path.join(shot["dir"], "shot.wav")
        with wave.open(joined_wav, "wb") as dst:
            dst.setnchannels(fmt["channels"])
            dst.setsampwidth(2)
            dst.setframerate(fmt["rate"])
            for t in takes:
                with wave.open(os.path.join(shot["dir"], t["wav"])) as src:
                    dst.writeframes(src.readframes(src.getnframes()))
        final = shot["dir"] + ".mp4"
        partial = shot["dir"] + ".partial.mp4"
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", listing, "-i", joined_wav, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", partial],
                       check=True, capture_output=True)
        os.replace(partial, final)
        shot = dict(shot, takes=takes)
        manifest = os.path.join(shot["dir"], "shot.json")
        with open(manifest + ".tmp", "w") as f:
            json.dump({"format": shot["format"], "takes": takes}, f, indent=1)
        os.replace(manifest + ".tmp", manifest)
        _SHOTS[filename_prefix] = shot
        total = sum(t["frames"] for t in takes)
        print(f"h3c: shot {os.path.basename(final)}: {len(takes)} takes, "
              f"{total} frames ({total / fps:.1f} s)")
        return _shot_ui(shot)


class H3cPicture:
    """A reference picture in one node: pick or upload the image, chain the next one."""

    @classmethod
    def INPUT_TYPES(cls):
        files = sorted(f for f in os.listdir(folder_paths.get_input_directory())
                       if os.path.isfile(os.path.join(folder_paths.get_input_directory(), f)))
        return {"required": {"image": (files, {"image_upload": True})},
                "optional": {"references": (REFS, {"tooltip":
                    "The pictures before this one. Chain order sets the numbering: the first "
                    "picture is <Picture 1> in the prompt."})}}

    RETURN_TYPES = (REFS,)
    RETURN_NAMES = ("references",)
    FUNCTION = "add"
    CATEGORY = CATEGORY
    DESCRIPTION = ("A reference picture (a person, a place, a look). Bypass it (Ctrl+B) to "
                   "leave it out; the pictures after it keep their place in the chain.")

    @classmethod
    def IS_CHANGED(cls, image, **_):
        path = folder_paths.get_annotated_filepath(image)
        return os.path.getmtime(path) if os.path.exists(path) else ""

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"h3c: picture not found: {image}"
        return True

    def add(self, image, references=None):
        path = folder_paths.get_annotated_filepath(image)
        return _append(references, h3c.Reference(h3c.REFERENCE_IMAGE, path))


NODE_CLASS_MAPPINGS = {
    "H3cLoader": H3cLoader,
    "H3cGenerate": H3cGenerate,
    "H3cReferenceImage": H3cReferenceImage,
    "H3cReferenceVideo": H3cReferenceVideo,
    "H3cReferenceAudio": H3cReferenceAudio,
    "H3cContextFromVideo": H3cContextFromVideo,
    "H3cLoadContext": H3cLoadContext,
    "H3cJoinTakes": H3cJoinTakes,
    "H3cSettings": H3cSettings,
    "H3cJoinAll": H3cJoinAll,
    "H3cAddToShot": H3cAddToShot,
    "H3cPicture": H3cPicture,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3cLoader": "h3c Load Engine",
    "H3cGenerate": "h3c Generate",
    "H3cReferenceImage": "h3c Reference Image",
    "H3cReferenceVideo": "h3c Reference Video",
    "H3cReferenceAudio": "h3c Reference Audio",
    "H3cContextFromVideo": "h3c Motion Context From Video",
    "H3cLoadContext": "h3c Load Motion Context",
    "H3cJoinTakes": "h3c Join Takes",
    "H3cSettings": "h3c Settings",
    "H3cJoinAll": "h3c Join All Takes",
    "H3cAddToShot": "h3c Add to Shot",
    "H3cPicture": "h3c Picture",
}

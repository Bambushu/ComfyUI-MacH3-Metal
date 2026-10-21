"""ComfyUI nodes for h3.c, a native C/Metal MiniMax-H3 engine on Apple silicon.

The engine runs the whole generation itself (text encoder, DiT, both VAEs)
outside ComfyUI's model and latent graph. These nodes hand it a prompt and
files and take back IMAGE frames and AUDIO, so the stock video nodes
(CreateVideo, SaveVideo) finish the job.
"""

from __future__ import annotations

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
    array = (image[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    path = os.path.join(_work_dir(), f"{name}-{_stamp()}.png")
    Image.fromarray(array).save(path)
    return path


def _save_audio(audio: dict, name: str) -> str:
    """AUDIO dict to a 16-bit WAV; the engine resamples references itself."""
    waveform = audio["waveform"][0].clamp(-1, 1).cpu().numpy()  # [C, T]
    pcm = (waveform.T * 32767.0).astype("<i2")
    path = os.path.join(_work_dir(), f"{name}-{_stamp()}.wav")
    with wave.open(path, "wb") as out:
        out.setnchannels(pcm.shape[1])
        out.setsampwidth(2)
        out.setframerate(int(audio["sample_rate"]))
        out.writeframes(pcm.tobytes())
    return path


def _save_video(video, name: str) -> str:
    path = os.path.join(_work_dir(), f"{name}-{_stamp()}.mp4")
    video.save_to(path)
    return path


def _read_audio(mp4: str, sample_rate: int, frames: int, fps: int) -> dict:
    """The engine's soundtrack, trimmed or padded to exactly the delivered
    frames so consecutive takes concatenate on the frame grid."""
    ffmpeg = os.environ.get("H3_FFMPEG", "ffmpeg")
    raw = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", mp4, "-vn", "-ac", "2",
         "-ar", str(sample_rate), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
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
        }}

    RETURN_TYPES = (ENGINE,)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Open the h3.c engine. Reads model metadata only; weights map per generation."

    def load(self, engine_dir, model_dir, keep_resident):
        engine_dir = engine_dir.strip() or h3c.default_engine_dir()
        model_dir = model_dir.strip() or h3c.default_model_dir(os.path.expanduser(engine_dir))
        return (h3c.Engine(engine_dir, model_dir, keep_resident),)


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


class H3cGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (ENGINE,),
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
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "references": (REFS,),
                "motion_context": (CONTEXT, {"tooltip":
                    "Continue a previous take. Replaces first_frame; references and last_frame ride along."}),
                "reference_image_size": (["match", "max"], {"default": "match"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", CONTEXT)
    RETURN_NAMES = ("images", "audio", "fps", "context")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Generate video with native sound on h3.c (Apple silicon). T2V, first/last "
                   "frame, Ref2VA references and motion-context continuation.")

    def generate(self, engine, prompt, width, height, seconds, frames, steps, seed,
                 sampler, scheduler, shift, speed, save_context, preview, filename_prefix,
                 first_frame=None, last_frame=None, references=None, motion_context=None,
                 reference_image_size="match"):
        if not prompt.strip():
            raise ValueError("h3c: the prompt is empty")
        if motion_context is not None and first_frame is not None:
            raise ValueError("h3c: motion_context replaces first_frame; connect one of them")
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
            reference_image_size=reference_image_size, preview=preview)

        bar = comfy.utils.ProgressBar(BAR)
        state = {"images": None, "last": -1}

        def on_progress(phase, done, total):
            for name, lo, hi in PHASE_WEIGHTS:
                if phase.startswith(name) and total > 0:
                    value = int(BAR * (lo + (hi - lo) * min(done, total) / total))
                    if value > state["last"]:
                        state["last"] = value
                        bar.update_absolute(value, BAR)
                    break
            return comfy.model_management.processing_interrupted()

        def on_frame(w, h, stride, rgb, index, count, step, steps_):
            rows = np.ctypeslib.as_array(rgb, shape=(h * stride,)).reshape(h, stride)
            pixels = rows[:, : w * 3].reshape(h, w, 3)
            if step >= 0:  # a denoising preview
                value = max(state["last"], int(BAR * (0.10 + 0.75 * (step + 1) / max(steps_, 1))))
                bar.update_absolute(value, BAR, ("JPEG", Image.fromarray(pixels.copy()), 512))
                return comfy.model_management.processing_interrupted()
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
        print(f"h3c: delivered {result.frames} frames in {time.time() - started:.1f} s")
        bar.update_absolute(BAR, BAR)

        images = torch.from_numpy(state["images"]).float().div_(255.0)
        audio = _read_audio(mp4, result.sample_rate, result.frames, result.fps)
        context = ({"path": context_path, "kind": "h3ctx"} if context_path
                   else {"path": mp4, "kind": "video"})
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


NODE_CLASS_MAPPINGS = {
    "H3cLoader": H3cLoader,
    "H3cGenerate": H3cGenerate,
    "H3cReferenceImage": H3cReferenceImage,
    "H3cReferenceVideo": H3cReferenceVideo,
    "H3cReferenceAudio": H3cReferenceAudio,
    "H3cContextFromVideo": H3cContextFromVideo,
    "H3cLoadContext": H3cLoadContext,
    "H3cJoinTakes": H3cJoinTakes,
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
}

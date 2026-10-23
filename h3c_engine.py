"""ctypes binding for libh3.dylib (h3.c, a native C/Metal MiniMax-H3 engine).

Only the flat API (h3_flat.h) is used, so nothing here mirrors the
positional h3_params struct. No ComfyUI imports: this module also drives the
engine from plain Python and from the tests.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Callable, Optional

REFERENCE_IMAGE = 1
REFERENCE_VIDEO = 2
REFERENCE_AUDIO = 3
REFERENCE_VIDEO_AUDIO = 4

SAMPLERS = ["euler", "res"]
SCHEDULERS = ["linear", "simple", "beta", "beta57"]
FPS = 24
# A continuation pins this many frames of the previous take at its head and
# trims them from what it delivers.
MOTION_CONTEXT_FRAMES = 22

# Speed presets: the README's validated rows. "exact" is the engine default.
# denoise_reuse and core_reuse are mutually exclusive in the engine.
SPEED_PRESETS = {
    "exact": dict(denoise_reuse=1, dit_layers=50, core_reuse=1, token_reduction=0),
    "balanced (layers 45, reuse 2)": dict(denoise_reuse=2, dit_layers=45, core_reuse=1, token_reduction=0),
    "fast (layers 45, core reuse 4)": dict(denoise_reuse=1, dit_layers=45, core_reuse=4, token_reduction=0),
    "aggressive (layers 40, reuse 3, token reduction)": dict(
        denoise_reuse=3, dit_layers=40, core_reuse=1, token_reduction=1),
}

FRAME_FN = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p)
PROGRESS_FN = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


class H3cError(RuntimeError):
    pass


def frames_for_seconds(seconds: float) -> int:
    """The CLI's --seconds rule: round(seconds * 24). The engine aligns it."""
    return max(5, int(round(seconds * FPS)))


def default_engine_dir() -> str:
    return os.path.expanduser(os.environ.get("H3C_DIR", "~/h3c"))


def default_model_dir(engine_dir: str) -> str:
    env = os.environ.get("H3C_MODEL_DIR")
    if env:
        return os.path.expanduser(env)
    for name in ("MiniMax-H3", "fl2va_as_r2v"):
        candidate = os.path.join(engine_dir, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(engine_dir, "MiniMax-H3")


TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"


def find_turbo_lora(engine_dir: str, model_dir: str) -> str:
    """The Turbo step-distillation adapter (larryvrh/MiniMax-H3-Turbo-Lora):
    $H3C_TURBO_LORA, else the first TURBO_LORA beside the engine, in
    ~/h3c/turbo_lora, the model folder or in ComfyUI's models/loras."""
    env = os.environ.get("H3C_TURBO_LORA")
    if env:
        return os.path.expanduser(env)
    folders = [os.path.join(engine_dir, "turbo_lora"), engine_dir,
               os.path.expanduser("~/h3c/turbo_lora"), model_dir]
    try:
        import folder_paths  # inside ComfyUI only
        folders += folder_paths.get_folder_paths("loras")
    except Exception:
        pass
    for folder in folders:
        candidate = os.path.join(os.path.expanduser(folder), TURBO_LORA)
        if os.path.isfile(candidate):
            return candidate
    raise H3cError(
        f"{TURBO_LORA} not found in {', '.join(folders)}. Download it from "
        "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora or set $H3C_TURBO_LORA.")


def ensure_ffmpeg_env() -> None:
    """The engine shells out to ffmpeg. A GUI-launched ComfyUI often lacks
    Homebrew on PATH, so point H3_FFMPEG at a real binary."""
    if os.environ.get("H3_FFMPEG"):
        return
    found = shutil.which("ffmpeg")
    for candidate in (found, "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if candidate and os.path.exists(candidate):
            os.environ["H3_FFMPEG"] = candidate
            bindir = os.path.dirname(candidate)
            if bindir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            return


def _bind(lib: ctypes.CDLL) -> None:
    c_void_p, c_char_p, c_int = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    sig = {
        "h3_flat_version": (c_char_p, []),
        "h3_load_dir": (c_void_p, [c_char_p]),
        "h3_free": (None, [c_void_p]),
        "h3_last_error": (c_char_p, [c_void_p]),
        "h3_cache_set_enabled": (None, [c_void_p, c_int]),
        "h3_cache_clear": (None, [c_void_p]),
        "h3_result_free": (None, [c_void_p]),
        "h3_flat_params_new": (c_void_p, []),
        "h3_flat_params_free": (None, [c_void_p]),
        "h3_flat_params_set_int": (c_int, [c_void_p, c_char_p, c_int]),
        "h3_flat_params_set_u64": (c_int, [c_void_p, c_char_p, ctypes.c_uint64]),
        "h3_flat_params_set_double": (c_int, [c_void_p, c_char_p, ctypes.c_double]),
        "h3_flat_params_set_string": (c_int, [c_void_p, c_char_p, c_char_p]),
        "h3_flat_params_add_reference": (c_int, [c_void_p, c_int, c_char_p, c_char_p, c_int]),
        "h3_flat_params_set_callbacks": (None, [c_void_p, FRAME_FN, PROGRESS_FN, c_void_p]),
        "h3_flat_generate": (c_void_p, [c_void_p, c_char_p, c_void_p, c_char_p]),
        "h3_flat_result_get": (ctypes.c_int64, [c_void_p, c_char_p]),
        "h3_flat_lock_acquire": (c_void_p, []),
        "h3_flat_lock_release": (None, [c_void_p]),
    }
    for name, (restype, argtypes) in sig.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes


_LIBS: dict[str, ctypes.CDLL] = {}


def load_library(engine_dir: str) -> ctypes.CDLL:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise H3cError("h3c runs on Apple silicon Macs only (Metal).")
    engine_dir = os.path.abspath(os.path.expanduser(engine_dir))
    if engine_dir in _LIBS:
        return _LIBS[engine_dir]
    path = os.path.join(engine_dir, "libh3.dylib")
    if not os.path.exists(path):
        raise H3cError(f"{path} not found. Build it with: make -C {engine_dir} libh3.dylib")
    if not os.path.exists(os.path.join(engine_dir, "h3_shaders.metal")):
        raise H3cError(f"h3_shaders.metal not found beside {path}")
    lib = ctypes.CDLL(path)
    _bind(lib)
    _LIBS[engine_dir] = lib
    return lib


@dataclass
class Reference:
    kind: int
    path: str
    audio_path: Optional[str] = None
    include_embedded_audio: bool = False


@dataclass
class Request:
    prompt: str
    width: int = 544
    height: int = 960
    frames: int = 120
    steps: int = 8
    seed: int = 42
    sampler: str = "euler"
    scheduler: str = "linear"
    shift: float = 0.0           # 0 = the engine's released 12
    speed: str = "exact"
    output_path: Optional[str] = None
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None
    motion_context: Optional[str] = None
    save_context: Optional[str] = None
    references: list[Reference] = field(default_factory=list)
    reference_image_size: str = "match"
    preview: bool = False
    turbo: bool = False          # the Speed lane: the Turbo LoRA for this generation


@dataclass
class Result:
    width: int
    height: int
    frames: int
    fps: int
    sample_rate: int
    seed: int


FrameSink = Callable[[int, int, int, "ctypes.POINTER(ctypes.c_uint8)", int, int, int, int], bool]
ProgressSink = Callable[[str, int, int], bool]


class Engine:
    """One h3_ctx. h3_load_dir only reads metadata; weights map per run.

    lora: a DiT adapter applied while the weights load (W + scale * B @ A), or
    "turbo" for the Turbo step-distillation adapter (sample at 5-6 steps).
    None keeps the base weights. The engine builds a separate int8 cache for
    it once, beside the base one."""

    def __init__(self, engine_dir: str, model_dir: str, keep_resident: bool = False,
                 lora: Optional[str] = None, lora_scale: float = 1.0):
        ensure_ffmpeg_env()
        self.engine_dir = os.path.abspath(os.path.expanduser(engine_dir))
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        if not os.path.isdir(self.model_dir):
            raise H3cError(f"model directory not found: {self.model_dir}")
        self.lib = load_library(self.engine_dir)
        self.ctx = self.lib.h3_load_dir(self.model_dir.encode())
        if not self.ctx:
            raise H3cError(f"h3_load_dir failed for {self.model_dir}")
        if lora == "turbo":
            lora = find_turbo_lora(self.engine_dir, self.model_dir)
        if lora:
            lora = os.path.abspath(os.path.expanduser(lora))
            if not os.path.isfile(lora):
                raise H3cError(f"LoRA not found: {lora}")
            if not lora_scale:
                raise H3cError("lora_scale must be nonzero")
        self.lora = lora or None
        self.lora_scale = float(lora_scale)
        self.keep_resident = keep_resident
        self.lib.h3_cache_set_enabled(self.ctx, 1 if keep_resident else 0)

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.h3_free(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def release_cache(self) -> None:
        if self.ctx:
            self.lib.h3_cache_clear(self.ctx)

    def _abs(self, path: Optional[str]) -> Optional[bytes]:
        return os.path.abspath(path).encode() if path else None

    def generate(self, request: Request, on_frame: Optional[FrameSink] = None,
                 on_progress: Optional[ProgressSink] = None) -> Result:
        lib = self.lib
        p = lib.h3_flat_params_new()
        if not p:
            raise H3cError("out of memory")
        try:
            def set_int(name, value):
                if not lib.h3_flat_params_set_int(p, name.encode(), int(value)):
                    raise H3cError(f"engine rejected parameter {name}")

            set_int("width", request.width)
            set_int("height", request.height)
            set_int("frames", request.frames)
            set_int("steps", request.steps)
            set_int("sampler", SAMPLERS.index(request.sampler))
            set_int("scheduler", SCHEDULERS.index(request.scheduler))
            for name, value in SPEED_PRESETS[request.speed].items():
                set_int(name, value)
            set_int("reference_image_size", 1 if request.reference_image_size == "max" else 0)
            set_int("preview_denoise", 1 if (request.preview and on_frame) else 0)
            lib.h3_flat_params_set_u64(p, b"seed", int(request.seed) & 0xFFFFFFFFFFFFFFFF)
            lib.h3_flat_params_set_double(p, b"video_shift", float(request.shift))
            for name in ("output_path", "first_frame", "last_frame",
                         "motion_context", "save_context"):
                lib.h3_flat_params_set_string(p, name.encode(), self._abs(getattr(request, name)))
            lora, lora_scale = self.lora, self.lora_scale
            if request.turbo:
                lora, lora_scale = find_turbo_lora(self.engine_dir, self.model_dir), 1.0
            if lora:
                if not (lib.h3_flat_params_set_string(p, b"lora_path", lora.encode())
                        and lib.h3_flat_params_set_double(p, b"lora_scale", lora_scale)):
                    raise H3cError("this libh3.dylib has no LoRA support; rebuild it "
                                   f"from an h3.c with --lora: make -C {self.engine_dir} libh3.dylib")
            for ref in request.references:
                if not lib.h3_flat_params_add_reference(
                        p, ref.kind, self._abs(ref.path), self._abs(ref.audio_path),
                        1 if ref.include_embedded_audio else 0):
                    raise H3cError(f"cannot add reference {ref.path}")

            def frame_cb(w, h, stride, rgb, idx, count, step, steps, _opaque):
                try:
                    return 1 if (on_frame and on_frame(w, h, stride, rgb, idx, count, step, steps)) else 0
                except Exception:
                    return 1

            def progress_cb(phase, done, total, _opaque):
                try:
                    return 1 if (on_progress and on_progress(phase.decode(), done, total)) else 0
                except Exception:
                    return 1

            # Keep the C trampolines alive for the whole call.
            frame_fn = FRAME_FN(frame_cb) if on_frame else ctypes.cast(None, FRAME_FN)
            progress_fn = PROGRESS_FN(progress_cb) if on_progress else ctypes.cast(None, PROGRESS_FN)
            lib.h3_flat_params_set_callbacks(p, frame_fn, progress_fn, None)

            lock = lib.h3_flat_lock_acquire()
            if not lock:
                raise H3cError("cannot acquire the h3 process lock")
            try:
                result = lib.h3_flat_generate(self.ctx, request.prompt.encode(), p,
                                              self.engine_dir.encode())
            finally:
                lib.h3_flat_lock_release(lock)
            if not result:
                err = lib.h3_last_error(self.ctx)
                raise H3cError(err.decode() if err else "h3_generate failed")
            try:
                get = lambda key: int(lib.h3_flat_result_get(result, key.encode()))
                return Result(get("width"), get("height"), get("frames"), get("fps"),
                              get("sample_rate"), get("seed"))
            finally:
                lib.h3_result_free(result)
        finally:
            lib.h3_flat_params_free(p)

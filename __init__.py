"""ComfyUI-h3c: MiniMax-H3 video with native sound on Apple silicon, via h3.c."""
import os
import shutil

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def _install_sample_inputs() -> None:
    """Copy the h3c_sample_* files into ComfyUI's input folder so the example
    workflows run as shipped. Never overwrites a file that is already there."""
    try:
        import folder_paths
        src = os.path.join(os.path.dirname(__file__), "example_inputs")
        dst = folder_paths.get_input_directory()
        for name in os.listdir(src):
            target = os.path.join(dst, name)
            if name.startswith("h3c_sample_") and not os.path.exists(target):
                shutil.copy2(os.path.join(src, name), target)
    except Exception as exc:  # samples are a convenience, never a load failure
        print(f"h3c: could not install sample inputs: {exc}")


_install_sample_inputs()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

"""Switch, mode, settings and join logic without the engine. Needs a ComfyUI
checkout for its imports: COMFYUI_DIR=~/ComfyUI python -m unittest tests.test_nodes
(run from the folder that holds this pack, so it imports as a package)."""
import importlib
import os
import sys
import unittest

COMFY = os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/ComfyUI"))
HAVE_COMFY = os.path.isfile(os.path.join(COMFY, "folder_paths.py"))


@unittest.skipUnless(HAVE_COMFY, "set COMFYUI_DIR to a ComfyUI checkout")
class NodeLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, COMFY)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        pack = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.n = importlib.import_module(pack + ".nodes")  # hyphens are fine here
        import torch
        cls.torch = torch

    def clip(self, frames, value, samples):
        t = self.torch
        return (t.full((frames, 8, 8, 3), value),
                {"waveform": t.full((1, 2, samples), value), "sample_rate": 32000})

    def test_switches_ask_only_when_on(self):
        n = self.n
        self.assertEqual(n.H3cOptionalImage().check_lazy_status(True, None), ["image"])
        self.assertEqual(n.H3cOptionalImage().check_lazy_status(False, None), [])
        self.assertIsNone(n.H3cOptionalImage().pick(None, False)[0])
        self.assertEqual(n.H3cReferenceImage().check_lazy_status(False, None), [])
        self.assertEqual(n.H3cReferenceAudio().check_lazy_status(True, None), ["audio"])
        self.assertEqual(n.H3cReferenceImage().add(None, False, ["x"]), (["x"],))
        self.assertEqual(n.H3cReferenceVideo().add(None, True, False, None, None), ([],))
        self.assertEqual(n.H3cContextFromVideo().make(None, False), (None,))

    def test_mode_table(self):
        pick = self.n.H3cMode().pick
        self.assertEqual(pick("text to video", False, 1), (False, False, False, False, 1))
        self.assertEqual(pick("references + first + last frame", False, 3), (True, True, True, False, 3))
        # continuing a clip replaces the first frame
        self.assertEqual(pick("first + last frame", True, 2), (False, True, False, True, 2))

    def test_join_all_asks_for_exactly_n_takes(self):
        j = self.n.H3cJoinAll()
        a, aa = self.clip(4, 0.1, 100)
        self.assertEqual(j.check_lazy_status(1, images_1=a, audio_1=aa), [])
        self.assertEqual(j.check_lazy_status(2, images_1=a, audio_1=aa), ["images_2", "audio_2"])
        b, ba = self.clip(3, 0.2, 75)
        c, ca = self.clip(2, 0.3, 50)
        images, audio = j.join(3, images_1=a, audio_1=aa, images_2=b, audio_2=ba,
                               images_3=c, audio_3=ca)
        self.assertEqual(images.shape[0], 9)
        self.assertEqual(audio["waveform"].shape[2], 225)
        images, audio = j.join(1, images_1=a, audio_1=aa)
        self.assertEqual(images.shape[0], 4)

    def test_settings_carry_every_generate_field(self):
        spec = self.n.H3cSettings.INPUT_TYPES()["required"]
        defaults = {k: (v[1].get("default") if len(v) > 1 and isinstance(v[1], dict) else None)
                    for k, v in spec.items()}
        defaults["sampler"], defaults["scheduler"], defaults["speed"] = "euler", "linear", "exact"
        defaults["reference_image_size"] = "match"
        settings = self.n.H3cSettings().make(**defaults)[0]
        for key in ("width", "height", "steps", "sampler", "scheduler", "shift", "speed",
                    "preview", "save_context", "reference_image_size"):
            self.assertIn(key, settings)


if __name__ == "__main__":
    unittest.main()

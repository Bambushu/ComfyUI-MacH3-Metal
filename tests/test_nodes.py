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

    def test_generate_mode_requests_only_connected_inputs_it_uses(self):
        g = self.n.H3cGenerate()
        # connected-but-pending inputs arrive as None; unconnected ones are absent
        pending = dict(first_frame=None, last_frame=None, references=None, motion_context=None)
        self.assertEqual(g.check_lazy_status("text to video", **pending), [])
        self.assertEqual(sorted(g.check_lazy_status("first + last frame", **pending)),
                         ["first_frame", "last_frame"])
        self.assertEqual(sorted(g.check_lazy_status("continue + references", **pending)),
                         ["motion_context", "references"])
        self.assertEqual(len(g.check_lazy_status("auto", **pending)), 4)
        # nothing connected: never request an absent input
        self.assertEqual(g.check_lazy_status("auto"), [])
        self.assertEqual(g.check_lazy_status("references", first_frame=None), [])
        # already evaluated: not requested again
        self.assertEqual(g.check_lazy_status("first frame", first_frame="img"), [])

    def test_join_all_asks_for_exactly_n_takes(self):
        j = self.n.H3cJoinAll()
        a, aa = self.clip(4, 0.1, 100)
        self.assertEqual(j.check_lazy_status(1, images_1=a, audio_1=aa), [])
        self.assertEqual(j.check_lazy_status(2, images_1=a, audio_1=aa, images_2=None, audio_2=None),
                         ["images_2", "audio_2"])
        # take 2 asked for but not connected: never requested
        self.assertEqual(j.check_lazy_status(2, images_1=a, audio_1=aa), [])
        b, ba = self.clip(3, 0.2, 75)
        c, ca = self.clip(2, 0.3, 50)
        images, audio = j.join(3, images_1=a, audio_1=aa, images_2=b, audio_2=ba,
                               images_3=c, audio_3=ca)
        self.assertEqual(images.shape[0], 9)
        self.assertEqual(audio["waveform"].shape[2], 225)
        images, audio = j.join(1, images_1=a, audio_1=aa)
        self.assertEqual(images.shape[0], 4)

    def test_continue_option_works_under_any_mode(self):
        g = self.n.H3cGenerate()
        pending = dict(first_frame=None, last_frame=None, references=None, motion_context=None)
        # continuing the last take: the first frame is replaced, refs and last frame stay
        self.assertEqual(sorted(g.check_lazy_status("references + first + last frame", "last take", **pending)),
                         ["last_frame", "references"])
        self.assertEqual(g.check_lazy_status("text to video", "last take", **pending), [])
        # continuing a clip also reads the motion_context input
        self.assertEqual(sorted(g.check_lazy_status("references", "clip input", **pending)),
                         ["motion_context", "references"])

    def test_add_to_shot_appends_continuations_and_restarts_otherwise(self):
        import tempfile, subprocess
        import folder_paths
        out = tempfile.mkdtemp()
        folder_paths.set_output_directory(out)
        folder_paths.set_temp_directory(tempfile.mkdtemp())
        add = self.n.H3cAddToShot().add
        a, aa = self.clip(24, 0.2, 32000)
        b, ba = self.clip(48, 0.8, 64000)
        r1 = add(a, aa, 24.0, {"continues": False}, "t/shot")
        r2 = add(b, ba, 24.0, {"continues": True}, "t/shot")
        shot = os.path.join(out, r2["ui"]["images"][0]["subfolder"], r2["ui"]["images"][0]["filename"])
        self.assertEqual(r1["ui"]["images"][0]["filename"], r2["ui"]["images"][0]["filename"])
        probe = lambda kind: subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", kind, "-count_frames", "-show_entries",
             "stream=nb_read_frames,duration", "-of", "csv=p=0", shot],
            capture_output=True, text=True).stdout.strip()
        self.assertEqual(probe("v").split(",")[-1], "72")          # 24 + 48 frames
        self.assertAlmostEqual(float(probe("a").split(",")[0]), 3.0, delta=0.03)
        r3 = add(a, aa, 24.0, {"continues": False}, "t/shot")       # a new shot
        self.assertNotEqual(r3["ui"]["images"][0]["filename"], r2["ui"]["images"][0]["filename"])

    def test_settings_carry_every_generate_field(self):
        spec = self.n.H3cSettings.INPUT_TYPES()["required"]
        defaults = {k: (v[1].get("default") if len(v) > 1 and isinstance(v[1], dict) else None)
                    for k, v in spec.items()}
        defaults["sampler"], defaults["scheduler"], defaults["speed"] = "euler", "linear", "exact"
        defaults["reference_image_size"] = "max"
        settings = self.n.H3cSettings().make(**defaults)[0]
        for key in ("width", "height", "steps", "sampler", "scheduler", "shift", "speed",
                    "preview", "save_context", "reference_image_size"):
            self.assertIn(key, settings)


if __name__ == "__main__":
    unittest.main()

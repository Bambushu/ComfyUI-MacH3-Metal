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

    def test_add_to_shot_follows_lineage_survives_restart_and_is_idempotent(self):
        import tempfile, subprocess
        import folder_paths
        out = tempfile.mkdtemp()
        folder_paths.set_output_directory(out)
        folder_paths.set_temp_directory(tempfile.mkdtemp())
        add = self.n.H3cAddToShot().add
        a, aa = self.clip(24, 0.2, 32000)
        b, ba = self.clip(48, 0.8, 64000)
        name = lambda r: r["ui"]["images"][0]["filename"]
        frames = lambda r: subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries",
             "stream=nb_read_frames", "-of", "csv=p=0",
             os.path.join(out, r["ui"]["images"][0]["subfolder"], name(r))],
            capture_output=True, text=True).stdout.strip()
        r1 = add(a, aa, 24.0, {"take_id": "t1", "continues": False}, "t/shot")
        r2 = add(b, ba, 24.0, {"take_id": "t2", "continues": True, "parent": "t1"}, "t/shot")
        self.assertEqual(name(r1), name(r2))
        self.assertEqual(frames(r2), "72")                       # 24 + 48
        r2b = add(b, ba, 24.0, {"take_id": "t2", "continues": True, "parent": "t1"}, "t/shot")
        self.assertEqual(frames(r2b), "72")                      # a rerun adds nothing
        # continues, but not this shot's last take: a new shot
        r3 = add(a, aa, 24.0, {"take_id": "t3", "continues": True, "parent": "elsewhere"}, "t/shot")
        self.assertNotEqual(name(r3), name(r2))
        # after a restart the shot on disk is picked up again
        self.n._SHOTS.clear()
        r4 = add(b, ba, 24.0, {"take_id": "t4", "continues": True, "parent": "t3"}, "t/shot")
        self.assertEqual((name(r4), frames(r4)), (name(r3), "72"))
        # a continuation in another size is refused
        c = self.torch.full((24, 16, 8, 3), 0.5)
        with self.assertRaises(ValueError):
            add(c, aa, 24.0, {"take_id": "t5", "continues": True, "parent": "t4"}, "t/shot")

    def test_last_take_keys_the_cache_and_missing_pictures_fail_early(self):
        g = self.n.H3cGenerate
        self.n._LAST_TAKES["p/take"] = {"path": "/x/one.h3ctx", "kind": "h3ctx"}
        self.assertEqual(g.IS_CHANGED(continue_from="last take", filename_prefix="p/take"), "/x/one.h3ctx")
        self.assertEqual(g.IS_CHANGED(continue_from="off", filename_prefix="p/take"), "")
        args = dict(engine=None, prompt="x", width=544, height=960, seconds=5.0, frames=0, steps=6,
                    seed=1, sampler="euler", scheduler="linear", shift=0.0, speed="exact",
                    save_context=False, preview=False, filename_prefix="p/take")
        for mode in ("I2V · first frame", "R2V · references"):
            with self.assertRaisesRegex(ValueError, "needs"):
                g().generate(mode=mode, **args)

    def test_dropdown_shows_four_modes_and_old_names_still_validate(self):
        g = self.n.H3cGenerate
        self.assertEqual(len(g.INPUT_TYPES()["required"]["mode"][0]), 4)
        for name in ("R2V · references", "continue + references", "auto"):
            self.assertIs(g.VALIDATE_INPUTS(name), True)
        self.assertIsInstance(g.VALIDATE_INPUTS("nope"), str)

    def test_settings_size_from_megapixels_and_aspect(self):
        size = self.n.size_for
        self.assertEqual(size("0.5", "9:16 portrait"), (544, 960))
        self.assertEqual(size("1.0", "9:16 portrait"), (768, 1344))   # capped
        self.assertEqual(size("1.0", "16:9 landscape"), (1344, 768))
        self.assertEqual(size("1.0", "1:1 square"), (992, 992))
        settings = self.n.H3cSettings().make(megapixels="0.8", aspect="9:16 portrait",
                                             lane="Speed (Turbo)", steps=6,
                                             sampler="euler", scheduler="linear", shift=0.0,
                                             speed="exact", save_context=True, preview=False,
                                             reference_image_size="max")[0]
        self.assertEqual((settings["width"], settings["height"]), (672, 1216))
        for key in self.n.H3cSettings.OVERRIDES:
            self.assertIn(key, settings)
        self.assertEqual(settings["lane"], "Speed (Turbo)")

if __name__ == "__main__":
    unittest.main()

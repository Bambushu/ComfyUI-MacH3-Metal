"""Binding checks that need the dylib but no weights in memory: h3_load_dir
only reads metadata, and an invalid request fails in the engine's own
validation before anything is mapped. Run: python -m unittest discover tests
with H3C_DIR pointing at an h3.c checkout built with `make libh3.dylib`."""

import os
import platform
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h3c_engine as h3c  # noqa: E402

ENGINE_DIR = h3c.default_engine_dir()
HAVE_LIB = (platform.system() == "Darwin" and platform.machine() == "arm64"
            and os.path.exists(os.path.join(ENGINE_DIR, "libh3.dylib")))


class FramesTest(unittest.TestCase):
    def test_seconds_rule_matches_cli(self):
        self.assertEqual(h3c.frames_for_seconds(5.0), 120)
        self.assertEqual(h3c.frames_for_seconds(0.1), 5)
        self.assertEqual(h3c.frames_for_seconds(4.47), 107)

    def test_presets_never_combine_reuse_kinds(self):
        for name, preset in h3c.SPEED_PRESETS.items():
            self.assertFalse(preset["denoise_reuse"] > 1 and preset["core_reuse"] > 1, name)


@unittest.skipUnless(HAVE_LIB, "libh3.dylib not built or not Apple silicon")
class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = h3c.Engine(ENGINE_DIR, h3c.default_model_dir(ENGINE_DIR))

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_engine_validation_reaches_python(self):
        cwd = os.getcwd()
        with self.assertRaisesRegex(h3c.H3cError, "denoising steps"):
            self.engine.generate(h3c.Request(prompt="x", steps=1))
        self.assertEqual(os.getcwd(), cwd)

    def test_bad_frames_rejected(self):
        with self.assertRaisesRegex(h3c.H3cError, "frames"):
            self.engine.generate(h3c.Request(prompt="x", frames=400))

    def test_callbacks_and_references_marshal(self):
        seen = []
        with self.assertRaises(h3c.H3cError):
            self.engine.generate(
                h3c.Request(prompt="x", steps=1, preview=True,
                            references=[h3c.Reference(h3c.REFERENCE_IMAGE, "/nonexistent.png")]),
                on_frame=lambda *a: False, on_progress=lambda *a: seen.append(a) or False)


if __name__ == "__main__":
    unittest.main()

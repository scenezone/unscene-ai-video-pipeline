"""Offline checks for bounded caption batches, model cleanup, and cache validity."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.prompt_generation.prompt_generator import PromptGenerator

spec = importlib.util.spec_from_file_location("caption_runner", ROOT / "tests/test_caption_generator.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CaptionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clothing = self.root / "clothing"
        self.objects = self.root / "objects"
        self.clothing.mkdir()
        self.objects.mkdir()
        self.output = io.StringIO()
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(contextlib.redirect_stdout(self.output))

    def generator(self, **kwargs):
        return PromptGenerator(self.clothing, self.objects, **kwargs)

    def configure_runner(self):
        for name, value in {
            "CLOTHING_DIR": self.clothing,
            "OBJECT_DIR": self.objects,
            "CACHE_FILE": self.root / "cache.json",
            "PROMPT_OUTPUT_FILE": self.root / "prompt.json",
        }.items():
            self.contexts.enter_context(patch.object(runner, name, value))

    def test_batches_visit_nested_images_and_close_before_next_batch(self):
        nested = self.clothing / "nested"
        nested.mkdir()
        for number in range(5):
            Image.new("RGB", (4, 4)).save(nested / f"{number}.png")
        generator = self.generator(batch_size=2)
        seen = []
        sizes = []

        def extract(images):
            for previous in seen:
                with self.assertRaises(ValueError):
                    previous.getpixel((0, 0))
            sizes.append(len(images))
            seen.extend(images)
            return ["shirt", "shirt"]

        with patch.object(generator, "extract_tags", side_effect=extract):
            self.assertEqual(generator.extract_folder_tags(self.clothing), ["shirt"])
        self.assertEqual(sizes, [2, 2, 1])
        for image in seen:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_empty_folders_do_not_load_models(self):
        generator = self.generator()
        with patch.object(generator, "_load_florence") as load, patch.object(generator.ollama_client, "chat") as chat:
            self.assertEqual(generator.generate_prompt(), {"clothing_prompt": "", "object_prompt": ""})
        load.assert_not_called()
        chat.assert_not_called()

    def test_florence_is_released_before_ollama(self):
        generator = self.generator()
        generator.florence_model = Mock()
        generator.processor = Mock()

        def normalize(tags):
            self.assertIsNone(generator.florence_model)
            self.assertIsNone(generator.processor)
            self.assertEqual(tags, ["shirt"])
            return ["shirt"]

        with patch.object(generator, "extract_folder_tags", side_effect=[["person", "shirt"], ["bag"]]), \
                patch.object(generator, "process_clothing_tags_with_llm", side_effect=normalize):
            self.assertEqual(generator.generate_prompt(), {"clothing_prompt": "shirt", "object_prompt": "bag"})

    def test_inference_failure_closes_images_and_releases_models(self):
        generator = self.generator()
        generator.florence_model = Mock()
        generator.processor = Mock()
        image = Image.new("RGB", (4, 4))
        with patch.object(generator, "image_paths", return_value=["image.png"]), \
                patch.object(generator, "_open_images", return_value=[image]), \
                patch.object(generator, "extract_tags", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                generator.generate_prompt()
        self.assertIsNone(generator.florence_model)
        self.assertIsNone(generator.processor)
        with self.assertRaises(ValueError):
            image.getpixel((0, 0))

    def test_invalid_batch_size_fails_early(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self.generator(batch_size=0)

    def test_fingerprint_detects_changes_additions_and_deletions(self):
        self.configure_runner()
        image = self.clothing / "shirt.png"
        image.write_bytes(b"first")
        initial = runner.input_fingerprint()
        image.write_bytes(b"other")  # Same size, different content.
        changed = runner.input_fingerprint()
        self.assertNotEqual(initial, changed)
        added = self.objects / "bag.JPG"
        added.write_bytes(b"bag")
        self.assertNotEqual(changed, runner.input_fingerprint())
        added.unlink()
        self.assertEqual(changed, runner.input_fingerprint())

    def test_missing_input_folder_is_not_cached_as_empty(self):
        self.configure_runner()
        self.objects.rmdir()
        with self.assertRaises(FileNotFoundError):
            runner.input_fingerprint()

    def test_invalid_cache_is_ignored(self):
        self.configure_runner()
        self.assertIsNone(runner.load_cached_prompt("key"))
        for value in ("{", "null", "[]", '{"fingerprint":"key","prompt_data":{}}'):
            runner.CACHE_FILE.write_text(value)
            self.assertIsNone(runner.load_cached_prompt("key"))

    def test_cached_run_skips_model_imports_and_preserves_output_schema(self):
        self.configure_runner()
        data = {"clothing_prompt": "shirt", "object_prompt": "bag"}
        runner.CACHE_FILE.write_text(json.dumps({"fingerprint": runner.input_fingerprint(), "prompt_data": data}))
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name in ("torch", "services.prompt_generation.prompt_generator"):
                raise AssertionError("Cached run imported model dependencies")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            runner.main([])
        self.assertEqual(json.loads(runner.PROMPT_OUTPUT_FILE.read_text()), data)
        self.assertIn("Using cached prompts", self.output.getvalue())

    def test_auto_device_caches_success_and_refresh_recomputes(self):
        self.configure_runner()
        data = {"clothing_prompt": "shirt", "object_prompt": "bag"}
        with patch("services.prompt_generation.prompt_generator.PromptGenerator") as factory, \
                patch("torch.cuda.is_available", return_value=True):
            factory.return_value.generate_prompt.return_value = data
            runner.main([])
            self.assertEqual(factory.call_args.kwargs["device"], "cuda")
            self.assertEqual(factory.call_args.kwargs["batch_size"], 1)
            runner.main([])
            self.assertEqual(factory.call_count, 1)
            runner.main(["--refresh"])
            self.assertEqual(factory.call_count, 2)
        self.assertEqual(runner.load_cached_prompt(runner.input_fingerprint()), data)

    def test_cpu_fallback_limits_threads_and_failed_run_does_not_cache(self):
        self.configure_runner()
        with patch("services.prompt_generation.prompt_generator.PromptGenerator") as factory, \
                patch("torch.cuda.is_available", return_value=False), \
                patch("torch.set_num_threads") as threads:
            factory.return_value.generate_prompt.side_effect = RuntimeError("failed")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                runner.main(["--cpu-threads", "2"])
            self.assertEqual(factory.call_args.kwargs["device"], "cpu")
            threads.assert_called_once_with(2)
        self.assertFalse(runner.CACHE_FILE.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from server.config import load_env_file
from server.mix import render_mix
from server.store import JobNotFound, JobStore
from worker import separate_v4
from worker.separate_v4 import STEM_NAMES, inference_settings


class JobStoreTests(unittest.TestCase):
    def test_create_update_list_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = JobStore(Path(temporary), retention_hours=24)
            job, metadata = store.create("song.mp3", ".mp3")
            self.assertTrue(job.is_dir())
            self.assertEqual(metadata["status"], "uploading")
            updated = store.update(metadata["jobId"], status="queued", progress=2)
            self.assertEqual(updated["progress"], 2)
            self.assertEqual(store.list()[0]["jobId"], metadata["jobId"])
            store.delete(metadata["jobId"])
            with self.assertRaises(JobNotFound):
                store.read(metadata["jobId"])


class MixTests(unittest.TestCase):
    def test_render_mix_applies_levels_and_writes_pcm24(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rate = 8_000
            length = rate
            tone = np.full((length, 2), 0.05, dtype=np.float32)
            for name in STEM_NAMES:
                sf.write(root / f"{name}.wav", tone, rate, subtype="FLOAT")
            levels = {name: 0.0 for name in STEM_NAMES}
            levels["vocals"] = 1.0
            output = render_mix(root, levels)
            data, output_rate = sf.read(output, dtype="float32", always_2d=True)
            self.assertEqual(output_rate, rate)
            self.assertEqual(len(data), length)
            self.assertAlmostEqual(float(data.mean()), 0.05, places=3)

    def test_render_mix_rejects_invalid_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in STEM_NAMES:
                sf.write(root / f"{name}.wav", np.zeros((10, 2)), 8_000)
            with self.assertRaises(ValueError):
                render_mix(root, {"vocals": 2.0})


class InferenceSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {
            name: os.environ.pop(name, None)
            for name in ("STEMFLOW_MDXC_OVERLAP", "STEMFLOW_USE_AUTOCAST")
        }

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_defaults_match_the_documented_pipeline(self) -> None:
        self.assertEqual(
            inference_settings(), {"mdxcOverlap": 8, "autocast": True}
        )

    def test_environment_overrides_are_applied(self) -> None:
        os.environ["STEMFLOW_MDXC_OVERLAP"] = "16"
        os.environ["STEMFLOW_USE_AUTOCAST"] = "false"
        self.assertEqual(
            inference_settings(), {"mdxcOverlap": 16, "autocast": False}
        )

    def test_autocast_accepts_common_truthy_spellings(self) -> None:
        for value, expected in (
            ("1", True),
            ("TRUE", True),
            (" on ", True),
            ("0", False),
            ("no", False),
            ("", False),
        ):
            os.environ["STEMFLOW_USE_AUTOCAST"] = value
            self.assertIs(inference_settings()["autocast"], expected, value)


class OfflineAssetTests(unittest.TestCase):
    """Nothing may silently reach the network to fetch a model."""

    def setUp(self) -> None:
        self.saved = os.environ.pop("STEMFLOW_MODEL_DIR", None)

    def tearDown(self) -> None:
        if self.saved is None:
            os.environ.pop("STEMFLOW_MODEL_DIR", None)
        else:
            os.environ["STEMFLOW_MODEL_DIR"] = self.saved

    def populate(self, directory: Path) -> None:
        for name in separate_v4.required_model_files():
            (directory / name).write_bytes(b"x")

    def test_model_directory_is_absolute_and_drive_independent(self) -> None:
        default = separate_v4.model_directory()
        self.assertTrue(default.is_absolute(), default)
        self.assertTrue(default.drive, f"expected a drive letter in {default}")

    def test_model_directory_honours_the_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["STEMFLOW_MODEL_DIR"] = temporary
            self.assertEqual(
                separate_v4.model_directory(), Path(temporary).resolve()
            )

    def test_required_files_include_configs_and_download_index(self) -> None:
        required = separate_v4.required_model_files()
        self.assertIn("download_checks.json", required)
        self.assertIn("BS-Roformer-SW.yaml", required)
        self.assertIn("model_bs_roformer_ep_317_sdr_12.9755.yaml", required)
        self.assertIn(separate_v4.VOCAL_MODEL, required)
        self.assertIn(separate_v4.SIX_STEM_MODEL, required)

    def test_complete_directory_reports_nothing_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.populate(directory)
            self.assertEqual(separate_v4.missing_model_files(directory), ())
            separate_v4.verify_offline_assets(directory)  # must not raise

    def test_a_missing_config_is_reported_not_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.populate(directory)
            (directory / "BS-Roformer-SW.yaml").unlink()
            self.assertEqual(
                separate_v4.missing_model_files(directory), ("BS-Roformer-SW.yaml",)
            )
            with self.assertRaises(FileNotFoundError) as caught:
                separate_v4.verify_offline_assets(directory)
            self.assertIn("BS-Roformer-SW.yaml", str(caught.exception))
            self.assertIn("STEMFLOW_MODEL_DIR", str(caught.exception))

    def test_a_missing_download_index_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.populate(directory)
            (directory / "download_checks.json").unlink()
            with self.assertRaises(FileNotFoundError):
                separate_v4.verify_offline_assets(directory)


class EnvFileTests(unittest.TestCase):
    """A path the user fills into .env has to actually take effect."""

    KEY = "STEMFLOW_MDXC_OVERLAP"

    def setUp(self) -> None:
        self.saved = os.environ.pop(self.KEY, None)

    def tearDown(self) -> None:
        if self.saved is None:
            os.environ.pop(self.KEY, None)
        else:
            os.environ[self.KEY] = self.saved

    def write_env(self, directory: Path, body: str) -> Path:
        target = directory / ".env"
        target.write_text(body, encoding="utf-8")
        return target

    def test_values_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = self.write_env(Path(temporary), f"{self.KEY}=16\n")
            self.assertTrue(load_env_file(target))
            self.assertEqual(inference_settings()["mdxcOverlap"], 16)

    def test_a_real_environment_variable_wins(self) -> None:
        os.environ[self.KEY] = "32"
        with tempfile.TemporaryDirectory() as temporary:
            target = self.write_env(Path(temporary), f"{self.KEY}=16\n")
            load_env_file(target)
            self.assertEqual(inference_settings()["mdxcOverlap"], 32)

    def test_a_missing_file_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(load_env_file(Path(temporary) / "absent.env"))

    def test_start_script_does_not_pin_tool_paths(self) -> None:
        """Pinning them would outrank vendored assets and silently shadow them."""
        script = (
            separate_v4.PROJECT_ROOT / "scripts" / "start-api.ps1"
        ).read_text(encoding="utf-8")
        for name in ("STEMFLOW_SEPARATOR_EXE", "STEMFLOW_FFMPEG_EXE"):
            self.assertNotIn(f"$env:{name} =", script, name)


class ResolutionOrderTests(unittest.TestCase):
    """Project-owned assets must win over ambient PATH and legacy paths."""

    def setUp(self) -> None:
        self.saved = os.environ.pop("STEMFLOW_SEPARATOR_EXE", None)

    def tearDown(self) -> None:
        if self.saved is None:
            os.environ.pop("STEMFLOW_SEPARATOR_EXE", None)
        else:
            os.environ["STEMFLOW_SEPARATOR_EXE"] = self.saved

    def test_environment_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tool = Path(temporary) / "chosen.exe"
            tool.write_bytes(b"x")
            os.environ["STEMFLOW_SEPARATOR_EXE"] = str(tool)
            self.assertEqual(
                separate_v4.audio_separator_executable(), str(tool.resolve())
            )

    def test_a_broken_override_is_an_explicit_error(self) -> None:
        os.environ["STEMFLOW_SEPARATOR_EXE"] = r"C:\nope\missing.exe"
        with self.assertRaises(FileNotFoundError):
            separate_v4.audio_separator_executable()

    def test_bundled_beats_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundled = Path(temporary) / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe"
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"x")
            resolved = separate_v4.resolve_executable(
                "STEMFLOW_UNSET_FOR_TEST",
                (bundled,),
                # A command that certainly exists on PATH, to prove it loses.
                "cmd" if os.name == "nt" else "sh",
            )
            self.assertEqual(resolved, str(bundled))

    def test_path_is_used_when_nothing_is_bundled(self) -> None:
        command = "cmd" if os.name == "nt" else "sh"
        resolved = separate_v4.resolve_executable(
            "STEMFLOW_UNSET_FOR_TEST", (), command
        )
        self.assertIn(command, resolved.casefold())

    def test_nothing_anywhere_is_an_explicit_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            separate_v4.resolve_executable(
                "STEMFLOW_UNSET_FOR_TEST", (), "stemflow-no-such-command"
            )
        self.assertIn("STEMFLOW_UNSET_FOR_TEST", str(caught.exception))

    def test_bundled_candidates_expand_globs(self) -> None:
        candidates = separate_v4.bundled_candidates("vendor/ffmpeg/bin/ffmpeg.exe")
        self.assertEqual(
            candidates,
            (separate_v4.PROJECT_ROOT / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe",),
        )

    def test_a_half_built_models_dir_does_not_shadow_a_working_one(self) -> None:
        """An empty project models/ must not hide the legacy install."""
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "models"
            empty.mkdir()
            complete = Path(temporary) / "complete"
            complete.mkdir()
            for name in separate_v4.required_model_files():
                (complete / name).write_bytes(b"x")
            # missing_model_files is what model_directory uses to make this call.
            self.assertTrue(separate_v4.missing_model_files(empty))
            self.assertEqual(separate_v4.missing_model_files(complete), ())


class RetentionTests(unittest.TestCase):
    def test_zero_retention_leaves_a_null_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = JobStore(Path(temporary), retention_hours=0)
            _, metadata = store.create("song.mp3", ".mp3")
            self.assertIsNone(metadata["expiresAt"])
            self.assertEqual(store.cleanup_expired(), 0)
            self.assertTrue(store.read(metadata["jobId"]))

    def test_zero_retention_protects_jobs_created_while_it_was_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            enabled = JobStore(root, retention_hours=24)
            _, metadata = enabled.create("song.mp3", ".mp3")
            self.assertIsNotNone(metadata["expiresAt"])
            enabled.update(
                metadata["jobId"],
                expiresAt="2000-01-01T00:00:00+00:00",
                status="completed",
            )
            disabled = JobStore(root, retention_hours=0)
            self.assertEqual(disabled.cleanup_expired(), 0)
            self.assertTrue(disabled.read(metadata["jobId"]))
            # The same stale job is still collected when retention is on.
            self.assertEqual(enabled.cleanup_expired(), 1)
            with self.assertRaises(JobNotFound):
                enabled.read(metadata["jobId"])


class ResidualInvariantTests(unittest.TestCase):
    """The six stems must sum back to the source once written to disk."""

    RATE = 8_000
    SOURCE = 0.6
    STEM = 0.1  # five model stems, so `other` should land on 0.1

    #: Stems the stubbed separator omits, mimicking its near-silent skip.
    omit: tuple[str, ...] = ()

    def fake_separator(self, source, output, model, sample_rate, *_args) -> None:
        output.mkdir(parents=True, exist_ok=True)
        names = (
            ("vocals", "instrumental")
            if model == separate_v4.VOCAL_MODEL
            else separate_v4.INSTRUMENT_STEMS
        )
        for name in names:
            if name in self.omit:
                continue
            sf.write(
                output / f"track_({name})_model.wav",
                np.full((self.RATE, 2), self.STEM, dtype=np.float32),
                sample_rate,
                subtype="FLOAT",
            )

    def run_pipeline(self, temporary: Path):
        """Run separate_job with the separator and encoder stubbed out."""
        patches = {
            "run_separator": self.fake_separator,
            "encode_preview": lambda src, dst: dst.write_bytes(b""),
        }
        original = {name: getattr(separate_v4, name) for name in patches}
        for name, replacement in patches.items():
            setattr(separate_v4, name, replacement)
        self.addCleanup(
            lambda: [setattr(separate_v4, n, v) for n, v in original.items()]
        )

        # Stub the model directory so the test never depends on this machine
        # actually having the checkpoints installed.
        models = temporary / "models"
        models.mkdir()
        for name in separate_v4.required_model_files():
            (models / name).write_bytes(b"x")
        saved = os.environ.get("STEMFLOW_MODEL_DIR")
        os.environ["STEMFLOW_MODEL_DIR"] = str(models)
        self.addCleanup(
            lambda: os.environ.__setitem__("STEMFLOW_MODEL_DIR", saved)
            if saved is not None
            else os.environ.pop("STEMFLOW_MODEL_DIR", None)
        )

        store = JobStore(temporary, retention_hours=0)
        job, metadata = store.create("song.wav", ".wav")
        sf.write(
            job / "original" / "source.wav",
            np.full((self.RATE, 2), self.SOURCE, dtype=np.float32),
            self.RATE,
            subtype="FLOAT",
        )
        return job, separate_v4.separate_job(job, metadata, lambda *_: None, None)

    def summed_stems(self, job: Path) -> np.ndarray:
        total = np.zeros((self.RATE, 2))
        for name in STEM_NAMES:
            data, _ = sf.read(
                job / "stems" / f"{name}.wav", dtype="float64", always_2d=True
            )
            total += data
        return total

    def test_stems_reconstruct_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job, completed = self.run_pipeline(Path(temporary))

            np.testing.assert_allclose(self.summed_stems(job), self.SOURCE, atol=1e-6)
            other, _ = sf.read(
                job / "stems" / "other.wav", dtype="float64", always_2d=True
            )
            np.testing.assert_allclose(other, self.STEM, atol=1e-6)

            separation = completed["separation"]
            self.assertLess(separation["reconstructionError"], 1e-6)
            self.assertEqual(separation["vocalModel"], separate_v4.VOCAL_MODEL)
            self.assertEqual(separation["mdxcOverlap"], inference_settings()["mdxcOverlap"])
            self.assertEqual(separation["silentStems"], [])
            self.assertTrue((job / "exports" / "stems.zip").is_file())
            self.assertFalse((job / "_stage1").exists())

    def test_a_silent_stem_is_zero_filled_not_an_error(self) -> None:
        """audio-separator omits near-silent stems; a track with no piano is
        ordinary input and must still produce six stems that sum correctly."""
        self.omit = ("piano", "guitar")
        with tempfile.TemporaryDirectory() as temporary:
            job, completed = self.run_pipeline(Path(temporary))

            for name in self.omit:
                data, _ = sf.read(
                    job / "stems" / f"{name}.wav", dtype="float64", always_2d=True
                )
                self.assertEqual(float(np.max(np.abs(data))), 0.0, name)

            self.assertCountEqual(
                completed["separation"]["silentStems"], list(self.omit)
            )
            # The invariant must survive the substitution.
            np.testing.assert_allclose(self.summed_stems(job), self.SOURCE, atol=1e-6)
            self.assertLess(completed["separation"]["reconstructionError"], 1e-6)
            for name in STEM_NAMES:
                self.assertTrue((job / "stems" / f"{name}.wav").is_file(), name)

    def test_a_silent_instrumental_skips_the_second_pass(self) -> None:
        """An a-cappella source leaves no instrumental for stage 2 to split."""
        self.omit = ("instrumental",)
        with tempfile.TemporaryDirectory() as temporary:
            job, completed = self.run_pipeline(Path(temporary))

            self.assertCountEqual(
                completed["separation"]["silentStems"],
                list(separate_v4.INSTRUMENT_STEMS),
            )
            np.testing.assert_allclose(self.summed_stems(job), self.SOURCE, atol=1e-6)
            # other absorbs everything the vocal pass did not claim.
            other, _ = sf.read(
                job / "stems" / "other.wav", dtype="float64", always_2d=True
            )
            np.testing.assert_allclose(
                other, self.SOURCE - self.STEM, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()

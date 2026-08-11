from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from omegaconf import DictConfig


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "export"
    / "locked_magpie_restore.py"
)
SPEC = importlib.util.spec_from_file_location("locked_magpie_restore", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load locked_magpie_restore")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REMOTE_CODEC_ID = "nvidia/nemo-nano-codec-locked"


class FakeAudioCodecModel:
    from_pretrained_calls = 0
    config_restore_calls = 0
    override_restore_calls = 0
    restored_use_scl_loss = False
    config_has_use_scl_loss = True

    @classmethod
    def reset(cls) -> None:
        cls.from_pretrained_calls = 0
        cls.config_restore_calls = 0
        cls.override_restore_calls = 0
        cls.restored_use_scl_loss = False
        cls.config_has_use_scl_loss = True

    @classmethod
    def from_pretrained(cls, model_id: str):
        cls.from_pretrained_calls += 1
        raise AssertionError(f"remote codec resolution was attempted: {model_id}")

    @classmethod
    def restore_from(
        cls,
        path: str,
        *,
        return_config: bool = False,
        strict: bool | None = None,
        override_config_path: DictConfig | None = None,
    ):
        if return_config:
            cls.config_restore_calls += 1
            config = {}
            if cls.config_has_use_scl_loss:
                config["use_scl_loss"] = True
            return DictConfig(config)
        if strict is not False or override_config_path is None:
            raise AssertionError("local codec restore requires the explicit config")
        if override_config_path.get("use_scl_loss") is not False:
            raise AssertionError("local codec restore did not disable speaker loss")
        cls.override_restore_calls += 1
        return SimpleNamespace(
            use_scl_loss=cls.restored_use_scl_loss,
            path=path,
        )


class FakeMagpieModel:
    embedded_codec_id = REMOTE_CODEC_ID

    @classmethod
    def restore_from(
        cls,
        model_path: str,
        *,
        return_config: bool = False,
        override_config_path: DictConfig | None = None,
        map_location: str | None = None,
    ):
        if return_config:
            return DictConfig({"codecmodel_path": cls.embedded_codec_id})
        if override_config_path is None:
            raise AssertionError("model construction requires an override config")
        codec_reference = override_config_path.get("codecmodel_path")
        if codec_reference.startswith("nvidia/"):
            codec_model = FakeAudioCodecModel.from_pretrained(codec_reference)
        else:
            codec_config = FakeAudioCodecModel.restore_from(
                codec_reference,
                return_config=True,
            )
            if "use_scl_loss" in codec_config:
                codec_config.use_scl_loss = False
            codec_model = FakeAudioCodecModel.restore_from(
                codec_reference,
                strict=False,
                override_config_path=codec_config,
            )
        return SimpleNamespace(
            cfg=override_config_path,
            _codec_helper=SimpleNamespace(codec_model=codec_model),
            model_path=model_path,
            map_location=map_location,
        )


class LockedMagpieRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAudioCodecModel.reset()
        FakeMagpieModel.embedded_codec_id = REMOTE_CODEC_ID
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model = self.root / "model.nemo"
        self.model.write_bytes(b"locked-model")
        self.codec = self.root / "codec.nemo"
        self.codec_payload = b"locked-local-codec"
        self.codec.write_bytes(self.codec_payload)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def locked_codec(self, *, sha256: str | None = None):
        return MODULE.LockedCodec(
            path=self.codec,
            model_id=REMOTE_CODEC_ID,
            sha256=sha256 or hashlib.sha256(self.codec_payload).hexdigest(),
            size_bytes=len(self.codec_payload),
        )

    def restore_with_fake_classes(self):
        with mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
            return_value=FakeAudioCodecModel,
        ), mock.patch.object(
            MODULE,
            "_magpie_model_class",
            return_value=FakeMagpieModel,
        ):
            return MODULE.load_locked_magpie_model(
                self.model,
                self.locked_codec(),
            )

    def test_local_codec_is_injected_before_model_construction(self) -> None:
        model = self.restore_with_fake_classes()

        self.assertEqual(FakeAudioCodecModel.from_pretrained_calls, 0)
        self.assertEqual(FakeAudioCodecModel.config_restore_calls, 2)
        self.assertEqual(FakeAudioCodecModel.override_restore_calls, 1)
        self.assertEqual(
            model.cfg.get("codecmodel_path"),
            str(self.codec.resolve()),
        )
        self.assertEqual(model.map_location, "cpu")

    def test_codec_hash_mismatch_fails_before_model_restore(self) -> None:
        with mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
        ) as codec_class, mock.patch.object(
            MODULE,
            "_magpie_model_class",
        ) as model_class:
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                MODULE.load_locked_magpie_model(
                    self.model,
                    self.locked_codec(sha256="0" * 64),
                )
        codec_class.assert_not_called()
        model_class.assert_not_called()

    def test_embedded_codec_id_mismatch_is_rejected(self) -> None:
        FakeMagpieModel.embedded_codec_id = "nvidia/different-codec"
        with mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
            return_value=FakeAudioCodecModel,
        ), mock.patch.object(
            MODULE,
            "_magpie_model_class",
            return_value=FakeMagpieModel,
        ):
            with self.assertRaisesRegex(RuntimeError, "embedded codec"):
                MODULE.load_locked_magpie_model(
                    self.model,
                    self.locked_codec(),
                )
        self.assertEqual(FakeAudioCodecModel.from_pretrained_calls, 0)
        self.assertEqual(FakeAudioCodecModel.override_restore_calls, 0)

    def test_missing_explicit_use_scl_loss_is_rejected(self) -> None:
        FakeAudioCodecModel.config_has_use_scl_loss = False
        with mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
            return_value=FakeAudioCodecModel,
        ), mock.patch.object(
            MODULE,
            "_magpie_model_class",
            return_value=FakeMagpieModel,
        ) as model_class:
            with self.assertRaisesRegex(RuntimeError, "use_scl_loss"):
                MODULE.load_locked_magpie_model(
                    self.model,
                    self.locked_codec(),
                )
        model_class.assert_not_called()
        self.assertEqual(FakeAudioCodecModel.from_pretrained_calls, 0)
        self.assertEqual(FakeAudioCodecModel.override_restore_calls, 0)

    def test_non_boolean_use_scl_loss_is_rejected(self) -> None:
        original_restore = FakeAudioCodecModel.restore_from

        def restore_with_non_boolean(path: str, **arguments):
            if arguments.get("return_config"):
                FakeAudioCodecModel.config_restore_calls += 1
                return DictConfig({"use_scl_loss": "true"})
            return original_restore(path, **arguments)

        with mock.patch.object(
            FakeAudioCodecModel,
            "restore_from",
            side_effect=restore_with_non_boolean,
        ), mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
            return_value=FakeAudioCodecModel,
        ), mock.patch.object(
            MODULE,
            "_magpie_model_class",
            return_value=FakeMagpieModel,
        ) as model_class:
            with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                MODULE.load_locked_magpie_model(
                    self.model,
                    self.locked_codec(),
                )
        model_class.assert_not_called()

    def test_codec_size_mismatch_fails_before_model_restore(self) -> None:
        codec = MODULE.LockedCodec(
            path=self.codec,
            model_id=REMOTE_CODEC_ID,
            sha256=hashlib.sha256(self.codec_payload).hexdigest(),
            size_bytes=len(self.codec_payload) + 1,
        )
        with mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
        ) as codec_class, mock.patch.object(
            MODULE,
            "_magpie_model_class",
        ) as model_class:
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                MODULE.load_locked_magpie_model(self.model, codec)
        codec_class.assert_not_called()
        model_class.assert_not_called()

    def test_training_only_speaker_loss_is_rejected(self) -> None:
        FakeAudioCodecModel.restored_use_scl_loss = True
        with mock.patch.object(
            MODULE,
            "_audio_codec_model_class",
            return_value=FakeAudioCodecModel,
        ), mock.patch.object(
            MODULE,
            "_magpie_model_class",
            return_value=FakeMagpieModel,
        ):
            with self.assertRaisesRegex(RuntimeError, "speaker loss"):
                MODULE.load_locked_magpie_model(
                    self.model,
                    self.locked_codec(),
                )
        self.assertEqual(FakeAudioCodecModel.from_pretrained_calls, 0)
        self.assertEqual(FakeAudioCodecModel.override_restore_calls, 1)


if __name__ == "__main__":
    unittest.main()

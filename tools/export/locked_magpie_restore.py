"""Restore the locked Magpie model without any remote codec resolution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, open_dict

if TYPE_CHECKING:
    from nemo.collections.tts.models import MagpieTTSModel


@dataclass(frozen=True)
class LockedCodecRestoreReceipt:
    embedded_codec_model_id: str
    codec_model_sha256: str
    codec_model_size_bytes: int
    codec_resolution: str = "authenticated_local_file"
    use_scl_loss: bool = False
    network_resolution: bool = False

    def to_json(self) -> dict[str, str | int | bool]:
        return {
            "embedded_codec_model_id": self.embedded_codec_model_id,
            "codec_model_sha256": self.codec_model_sha256,
            "codec_model_size_bytes": self.codec_model_size_bytes,
            "codec_resolution": self.codec_resolution,
            "use_scl_loss": self.use_scl_loss,
            "network_resolution": self.network_resolution,
        }


@dataclass(frozen=True)
class LockedCodec:
    """Authenticated local NanoCodec input required by Magpie restore."""

    path: Path
    model_id: str
    sha256: str
    size_bytes: int

    def restore_receipt(self) -> LockedCodecRestoreReceipt:
        return LockedCodecRestoreReceipt(
            embedded_codec_model_id=self.model_id,
            codec_model_sha256=self.sha256,
            codec_model_size_bytes=self.size_bytes,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _magpie_model_class():
    from nemo.collections.tts.models import MagpieTTSModel

    return MagpieTTSModel


def _audio_codec_model_class():
    from nemo.collections.tts.models import AudioCodecModel

    return AudioCodecModel


def _authenticate_codec(codec: LockedCodec) -> Path:
    path = codec.path.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"locked NanoCodec is not a file: {path}")
    if not codec.model_id.startswith("nvidia/"):
        raise RuntimeError(
            "locked NanoCodec model_id must be the exact packaged NVIDIA "
            "reference"
        )
    if path.stat().st_size != codec.size_bytes:
        raise RuntimeError(
            "locked NanoCodec size mismatch: "
            f"expected={codec.size_bytes}, actual={path.stat().st_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != codec.sha256:
        raise RuntimeError(
            "locked NanoCodec SHA-256 mismatch: "
            f"expected={codec.sha256}, actual={actual_sha256}"
        )
    return path


def load_locked_magpie_model(
    model_path: Path,
    codec: LockedCodec,
) -> MagpieTTSModel:
    """Restore Magpie with the authenticated local codec and no HTTP branch.

    The accepted Magpie archive names its NanoCodec by an ``nvidia/...`` model
    ID.  Passing that config to the constructor selects NeMo's
    ``from_pretrained`` branch and can perform an external HTTP fetch.  We
    first authenticate the separately accepted local codec, require the model
    archive to contain the exact locked model ID, and replace that one field
    before model construction.  The local NeMo branch also disables the
    training-only speaker contrastive-loss module before restoring the codec.
    """

    resolved_model = model_path.resolve(strict=True)
    resolved_codec = _authenticate_codec(codec)
    codec_config = _audio_codec_model_class().restore_from(
        str(resolved_codec),
        return_config=True,
    )
    if not isinstance(codec_config, DictConfig):
        raise RuntimeError(
            "locked NanoCodec restore_config must be an OmegaConf DictConfig"
        )
    if "use_scl_loss" not in codec_config:
        raise RuntimeError(
            "locked NanoCodec config does not explicitly declare "
            "use_scl_loss"
        )
    if type(codec_config.get("use_scl_loss")) is not bool:
        raise RuntimeError(
            "locked NanoCodec config use_scl_loss must be a boolean"
        )
    model_class = _magpie_model_class()
    config = model_class.restore_from(
        str(resolved_model),
        return_config=True,
    )
    if not isinstance(config, DictConfig):
        raise RuntimeError(
            "locked Magpie restore_config must be an OmegaConf DictConfig"
        )
    embedded_codec = config.get("codecmodel_path")
    if embedded_codec != codec.model_id:
        raise RuntimeError(
            "Magpie embedded codec reference differs from the oracle lock: "
            f"expected={codec.model_id!r}, actual={embedded_codec!r}"
        )
    with open_dict(config):
        config.codecmodel_path = str(resolved_codec)

    model = model_class.restore_from(
        str(resolved_model),
        override_config_path=config,
        map_location="cpu",
    )
    configured_codec = model.cfg.get("codecmodel_path")
    if configured_codec != str(resolved_codec):
        raise RuntimeError(
            "restored Magpie model did not retain the authenticated local "
            "NanoCodec path"
        )
    codec_model = model._codec_helper.codec_model
    if codec_model.use_scl_loss is not False:
        raise RuntimeError(
            "restored NanoCodec retained the training-only speaker loss"
        )
    return model

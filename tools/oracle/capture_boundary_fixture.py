#!/usr/bin/env python3
"""Capture portable Sofia boundary tensors from the locked PyTorch oracle.

Run this with the accepted NeMo checkout and isolated environment on AGX Thor.
The output contains raw little-endian tensors plus a checksummed JSON manifest.
It is intended for TensorRT component parity tests and is not a model bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyopenjtalk
import torch
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.models.magpietts import ContextTensorsOutput
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType
from nemo.collections.tts.modules.streaming_codec import (
    CausalCodecStreamingDecoder,
    CausalCodecStreamingState,
    _iter_named_hifigan_state_tensors,
    preallocate_causal_codec_lengths,
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    chunk_text_for_inference,
    get_tokenizer_for_language,
)

from verify_oracle_lock import (
    FileExpectation,
    require_file,
    require_model_configs,
    require_source_checkout,
    sha256_file,
)
from validate_boundary_fixture import validate_boundary_fixture


TARGET_DTYPE = torch.bfloat16
DEFAULT_TEXT = "あ、こんにちは。今日は何をしようか？"
DEFAULT_LANGUAGE = "ja"
SOFIA_INDEX = 4
DEFAULT_LOCAL_AR_SEED = 20260729
JAPANESE_TOKENIZER_NAME = "japanese_phoneme"
TOKEN_TABLE_CANONICALIZATION = (
    "utf8_json_sorted_keys_compact_rows_in_local_id_order"
)
DICTIONARY_MANIFEST_CANONICALIZATION = (
    "utf8_json_sorted_keys_compact_rows_in_path_order"
)
TEXT_TOKEN_DTYPE_NAME = "int32"
IMPORTED_NEMO_SOURCES = {
    "nemo.collections.tts.models.magpietts": (
        "nemo/collections/tts/models/magpietts.py"
    ),
    "nemo.collections.tts.modules.magpietts_modules": (
        "nemo/collections/tts/modules/magpietts_modules.py"
    ),
    "nemo.collections.tts.modules.streaming_codec": (
        "nemo/collections/tts/modules/streaming_codec.py"
    ),
    "nemo.collections.tts.parts.utils.tts_dataset_utils": (
        "nemo/collections/tts/parts/utils/tts_dataset_utils.py"
    ),
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    path: str
    dtype: str
    shape: list[int]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class FrontendTokenRecord:
    effective_local_id: int
    global_id: int
    local_id: int
    token: str


@dataclass(frozen=True)
class FrontendGoldenContract:
    raw_text: str
    normalized_text: str
    g2p_tokens: list[str]
    local_token_ids: list[int]
    global_token_ids_with_eos: list[int]


@dataclass(frozen=True)
class DictionaryFileRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class OpenJTalkDictionaryContract:
    directory_name: str
    manifest_canonicalization: str
    manifest_sha256: str
    files: list[DictionaryFileRecord]


@dataclass(frozen=True)
class PyOpenJTalkContract:
    version: str
    source_tag: str
    source_revision: str
    open_jtalk_submodule_revision: str
    dictionary: OpenJTalkDictionaryContract


@dataclass(frozen=True)
class JapaneseFrontendContract:
    tokenizer_name: str
    global_offset: int
    local_vocabulary_size: int
    global_pad_token_id: int
    pyopenjtalk: PyOpenJTalkContract
    token_table_canonicalization: str
    token_table_sha256: str
    token_table: list[FrontendTokenRecord]
    golden: FrontendGoldenContract


@dataclass(frozen=True)
class FrontendContract:
    aggregate_vocabulary_size: int
    text_embedding_rows: int
    bos_token_id: int
    eos_token_id: int
    text_token_dtype: str
    dependencies: dict[str, str]
    japanese: JapaneseFrontendContract


class TensorWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: list[TensorRecord] = []
        self.names: set[str] = set()

    @staticmethod
    def _payload(tensor: torch.Tensor) -> tuple[str, bytes]:
        value = tensor.detach().contiguous().cpu()
        if value.dtype == torch.bfloat16:
            return "bf16", value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
        dtype_names = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.int64: "int64",
            torch.int32: "int32",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        dtype_name = dtype_names.get(value.dtype)
        if dtype_name is None:
            raise TypeError(f"unsupported fixture dtype: {value.dtype}")
        numpy_value = value.numpy()
        if numpy_value.dtype.itemsize > 1:
            numpy_value = numpy_value.astype(numpy_value.dtype.newbyteorder("<"), copy=False)
        return dtype_name, numpy_value.tobytes(order="C")

    def write(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.names:
            raise RuntimeError(f"duplicate tensor name: {name}")
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in name):
            raise ValueError(f"unsafe tensor name: {name!r}")
        self.names.add(name)
        dtype_name, payload = self._payload(tensor)
        relative_path = Path("tensors") / f"{name}.bin"
        output_path = self.root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        self.records.append(
            TensorRecord(
                name=name,
                path=relative_path.as_posix(),
                dtype=dtype_name,
                shape=list(tensor.shape),
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--local-ar-seed",
        type=int,
        default=DEFAULT_LOCAL_AR_SEED,
    )
    return parser.parse_args()


def verify_inputs(args: argparse.Namespace) -> dict:
    if args.language != DEFAULT_LANGUAGE:
        raise RuntimeError(
            f"boundary fixture language must be {DEFAULT_LANGUAGE!r}, "
            f"got {args.language!r}"
        )
    lock = json.loads(args.lock.resolve(strict=True).read_text(encoding="utf-8"))
    model_lock = lock["model"]
    codec_lock = lock["codec"]
    source_lock = lock["oracle_source"]
    acceptance_lock = lock["acceptance"]
    require_file(
        args.model,
        FileExpectation(model_lock["sha256"], model_lock["size_bytes"]),
        "Magpie model",
    )
    require_model_configs(
        args.model,
        model_lock["config_member_sha256"],
        model_lock["active_config_policy"],
        model_lock["active_config_sha256"],
    )
    require_file(
        args.codec_model,
        FileExpectation(codec_lock["sha256"], codec_lock["size_bytes"]),
        "NanoCodec model",
    )
    require_file(
        args.acceptance_receipt,
        FileExpectation(
            acceptance_lock["receipt_sha256"],
            acceptance_lock["receipt_size_bytes"],
        ),
        "acceptance receipt",
    )
    require_source_checkout(
        args.speech_root,
        source_lock["base_revision"],
        source_lock["files"],
    )
    require_imported_nemo_source(args.speech_root)
    return lock


def require_imported_nemo_source(speech_root: Path) -> None:
    root = speech_root.resolve(strict=True)
    for module_name, expected_relative_path in IMPORTED_NEMO_SOURCES.items():
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(f"required NeMo module was not imported: {module_name}")
        module_file = module.__file__
        if module_file is None:
            raise RuntimeError(f"imported NeMo module has no file: {module_name}")
        actual_path = Path(module_file).resolve(strict=True)
        expected_path = (root / expected_relative_path).resolve(strict=True)
        if actual_path != expected_path:
            raise RuntimeError(
                f"imported NeMo source mismatch for {module_name}: "
                f"expected {expected_path}, got {actual_path}"
            )


def canonical_token_table_sha256(records: list[FrontendTokenRecord]) -> str:
    payload = json.dumps(
        [asdict(record) for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_dictionary_manifest_sha256(
    records: list[DictionaryFileRecord],
) -> str:
    payload = json.dumps(
        [asdict(record) for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_pyopenjtalk_contract(japanese_lock: dict) -> PyOpenJTalkContract:
    pyopenjtalk_lock = japanese_lock["pyopenjtalk"]
    version = importlib.metadata.version("pyopenjtalk")
    if version != pyopenjtalk_lock["version"]:
        raise RuntimeError(
            f"pyopenjtalk version mismatch: "
            f"expected {pyopenjtalk_lock['version']}, got {version}"
        )

    dictionary_lock = pyopenjtalk_lock["dictionary"]
    if (
        dictionary_lock["manifest_canonicalization"]
        != DICTIONARY_MANIFEST_CANONICALIZATION
    ):
        raise RuntimeError(
            "unsupported OpenJTalk dictionary manifest canonicalization: "
            f"{dictionary_lock['manifest_canonicalization']}"
        )
    unresolved_root = Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR))
    if unresolved_root.is_symlink():
        raise RuntimeError(
            f"OpenJTalk dictionary must not be a symbolic link: {unresolved_root}"
        )
    dictionary_root = unresolved_root.resolve(strict=True)
    if not dictionary_root.is_dir():
        raise RuntimeError(
            f"OpenJTalk dictionary is not a directory: {dictionary_root}"
        )
    if dictionary_root.name != dictionary_lock["directory_name"]:
        raise RuntimeError(
            "OpenJTalk dictionary directory mismatch: "
            f"expected {dictionary_lock['directory_name']!r}, "
            f"got {dictionary_root.name!r}"
        )

    expected_names: set[str] = set()
    records: list[DictionaryFileRecord] = []
    for locked_file in dictionary_lock["files"]:
        relative_path = locked_file["path"]
        if (
            not relative_path
            or Path(relative_path).name != relative_path
            or "/" in relative_path
            or "\\" in relative_path
        ):
            raise RuntimeError(
                f"unsafe OpenJTalk dictionary file path: {relative_path!r}"
            )
        if relative_path in expected_names:
            raise RuntimeError(
                f"duplicate OpenJTalk dictionary file: {relative_path}"
            )
        expected_names.add(relative_path)
        file_path = dictionary_root / relative_path
        if file_path.is_symlink():
            raise RuntimeError(
                f"OpenJTalk dictionary file must not be a symbolic link: "
                f"{relative_path}"
            )
        require_file(
            file_path,
            FileExpectation(
                locked_file["sha256"],
                locked_file["size_bytes"],
            ),
            f"OpenJTalk dictionary {relative_path}",
        )
        records.append(
            DictionaryFileRecord(
                path=relative_path,
                size_bytes=locked_file["size_bytes"],
                sha256=locked_file["sha256"],
            )
        )
    records.sort(key=lambda record: record.path)
    actual_entries = {entry.name for entry in dictionary_root.iterdir()}
    if actual_entries != expected_names:
        raise RuntimeError(
            "OpenJTalk dictionary entries mismatch: "
            f"missing={sorted(expected_names - actual_entries)}, "
            f"extra={sorted(actual_entries - expected_names)}"
        )
    manifest_sha256 = canonical_dictionary_manifest_sha256(records)
    if manifest_sha256 != dictionary_lock["manifest_sha256"]:
        raise RuntimeError(
            "OpenJTalk dictionary manifest SHA-256 mismatch: "
            f"expected {dictionary_lock['manifest_sha256']}, "
            f"got {manifest_sha256}"
        )

    return PyOpenJTalkContract(
        version=version,
        source_tag=pyopenjtalk_lock["source_tag"],
        source_revision=pyopenjtalk_lock["source_revision"],
        open_jtalk_submodule_revision=(
            pyopenjtalk_lock["open_jtalk_submodule_revision"]
        ),
        dictionary=OpenJTalkDictionaryContract(
            directory_name=dictionary_root.name,
            manifest_canonicalization=DICTIONARY_MANIFEST_CANONICALIZATION,
            manifest_sha256=manifest_sha256,
            files=records,
        ),
    )


def require_frontend_contract(
    model: MagpieTTSModel,
    lock: dict,
) -> FrontendContract:
    frontend_lock = lock["frontend"]
    japanese_lock = frontend_lock["japanese"]
    tokenizer = model.tokenizer
    if tokenizer.vocab_size != frontend_lock["aggregate_vocabulary_size"]:
        raise RuntimeError(
            "aggregate tokenizer vocabulary mismatch: "
            f"expected {frontend_lock['aggregate_vocabulary_size']}, "
            f"got {tokenizer.vocab_size}"
        )
    if model.text_embedding.num_embeddings != frontend_lock["text_embedding_rows"]:
        raise RuntimeError(
            "text embedding row count mismatch: "
            f"expected {frontend_lock['text_embedding_rows']}, "
            f"got {model.text_embedding.num_embeddings}"
        )
    if model.bos_id != frontend_lock["bos_token_id"]:
        raise RuntimeError(
            f"text BOS mismatch: expected {frontend_lock['bos_token_id']}, "
            f"got {model.bos_id}"
        )
    if model.eos_id != frontend_lock["eos_token_id"]:
        raise RuntimeError(
            f"text EOS mismatch: expected {frontend_lock['eos_token_id']}, "
            f"got {model.eos_id}"
        )
    if frontend_lock["text_token_dtype"] != TEXT_TOKEN_DTYPE_NAME:
        raise RuntimeError(
            f"unsupported locked text token dtype: "
            f"{frontend_lock['text_token_dtype']}"
        )

    tokenizer_name = japanese_lock["tokenizer_name"]
    if tokenizer_name != JAPANESE_TOKENIZER_NAME:
        raise RuntimeError(
            f"unsupported locked Japanese tokenizer name: {tokenizer_name}"
        )
    if tokenizer_name not in tokenizer.tokenizers:
        raise RuntimeError(f"Japanese tokenizer is missing: {tokenizer_name}")
    japanese_tokenizer = tokenizer.tokenizers[tokenizer_name]
    global_offset = tokenizer.tokenizer_offsets[tokenizer_name]
    if global_offset != japanese_lock["global_offset"]:
        raise RuntimeError(
            f"Japanese tokenizer offset mismatch: "
            f"expected {japanese_lock['global_offset']}, got {global_offset}"
        )
    local_vocabulary_size = len(japanese_tokenizer.tokens)
    if local_vocabulary_size != japanese_lock["local_vocabulary_size"]:
        raise RuntimeError(
            "Japanese tokenizer vocabulary mismatch: "
            f"expected {japanese_lock['local_vocabulary_size']}, "
            f"got {local_vocabulary_size}"
        )
    if tokenizer.num_tokens_per_tokenizer[tokenizer_name] != local_vocabulary_size:
        raise RuntimeError("aggregate Japanese tokenizer length is inconsistent")
    global_pad_token_id = tokenizer.tokenizer_pad_ids[tokenizer_name]
    if global_pad_token_id != japanese_lock["global_pad_token_id"]:
        raise RuntimeError(
            f"Japanese global pad mismatch: "
            f"expected {japanese_lock['global_pad_token_id']}, "
            f"got {global_pad_token_id}"
        )

    pyopenjtalk_contract = require_pyopenjtalk_contract(japanese_lock)
    if (
        japanese_lock["token_table_canonicalization"]
        != TOKEN_TABLE_CANONICALIZATION
    ):
        raise RuntimeError(
            "unsupported Japanese token table canonicalization: "
            f"{japanese_lock['token_table_canonicalization']}"
        )
    token_table = [
        FrontendTokenRecord(
            effective_local_id=japanese_tokenizer._token2id[token],
            global_id=global_offset + local_id,
            local_id=local_id,
            token=token,
        )
        for local_id, token in enumerate(japanese_tokenizer.tokens)
    ]
    token_table_sha256 = canonical_token_table_sha256(token_table)
    if token_table_sha256 != japanese_lock["token_table_sha256"]:
        raise RuntimeError(
            "Japanese token table SHA-256 mismatch: "
            f"expected {japanese_lock['token_table_sha256']}, "
            f"got {token_table_sha256}"
        )
    long_vowel_mark_effective_id = japanese_tokenizer._token2id["ー"]
    if (
        long_vowel_mark_effective_id
        != japanese_lock["long_vowel_mark_effective_local_token_id"]
    ):
        raise RuntimeError(
            "Japanese long-vowel token resolution mismatch: "
            f"expected "
            f"{japanese_lock['long_vowel_mark_effective_local_token_id']}, "
            f"got {long_vowel_mark_effective_id}"
        )

    golden_lock = japanese_lock["golden"]
    raw_text = golden_lock["raw_text"]
    normalized_text = japanese_tokenizer.text_preprocessing_func(raw_text)
    g2p_tokens = japanese_tokenizer.g2p(normalized_text)
    local_token_ids = japanese_tokenizer.encode(raw_text)
    global_token_ids_with_eos = tokenizer.encode(
        raw_text,
        tokenizer_name=tokenizer_name,
    ) + [model.eos_id]
    if g2p_tokens != golden_lock["g2p_tokens"]:
        raise RuntimeError(
            f"Japanese golden G2P mismatch: "
            f"expected {golden_lock['g2p_tokens']}, got {g2p_tokens}"
        )
    if local_token_ids != golden_lock["local_token_ids"]:
        raise RuntimeError(
            f"Japanese golden local token IDs mismatch: "
            f"expected {golden_lock['local_token_ids']}, got {local_token_ids}"
        )
    if global_token_ids_with_eos != golden_lock["global_token_ids_with_eos"]:
        raise RuntimeError(
            "Japanese golden global token IDs mismatch: "
            f"expected {golden_lock['global_token_ids_with_eos']}, "
            f"got {global_token_ids_with_eos}"
        )

    dependency_packages = {
        "transformers": "transformers",
        "tokenizers": "tokenizers",
        "huggingface_hub": "huggingface-hub",
        "omegaconf": "omegaconf",
    }
    dependencies = {"nemo": str(sys.modules["nemo"].__version__)}
    for lock_name, package_name in dependency_packages.items():
        dependencies[lock_name] = importlib.metadata.version(package_name)
    for dependency_name, actual_version in dependencies.items():
        expected_version = frontend_lock["dependencies"][dependency_name]
        if actual_version != expected_version:
            raise RuntimeError(
                f"frontend dependency mismatch for {dependency_name}: "
                f"expected {expected_version}, got {actual_version}"
            )

    return FrontendContract(
        aggregate_vocabulary_size=tokenizer.vocab_size,
        text_embedding_rows=model.text_embedding.num_embeddings,
        bos_token_id=model.bos_id,
        eos_token_id=model.eos_id,
        text_token_dtype=TEXT_TOKEN_DTYPE_NAME,
        dependencies=dependencies,
        japanese=JapaneseFrontendContract(
            tokenizer_name=tokenizer_name,
            global_offset=global_offset,
            local_vocabulary_size=local_vocabulary_size,
            global_pad_token_id=global_pad_token_id,
            pyopenjtalk=pyopenjtalk_contract,
            token_table_canonicalization=TOKEN_TABLE_CANONICALIZATION,
            token_table_sha256=token_table_sha256,
            token_table=token_table,
            golden=FrontendGoldenContract(
                raw_text=raw_text,
                normalized_text=normalized_text,
                g2p_tokens=g2p_tokens,
                local_token_ids=local_token_ids,
                global_token_ids_with_eos=global_token_ids_with_eos,
            ),
        ),
    )


def seed_everything(seed: int) -> None:
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_bf16_modules(model: MagpieTTSModel) -> dict[str, torch.nn.Module]:
    if model.model_type != "decoder_ce":
        raise RuntimeError(f"expected decoder_ce, got {model.model_type}")
    if model.local_transformer_type != LocalTransformerType.AR:
        raise RuntimeError(f"expected local autoregression, got {model.local_transformer_type}")
    if not isinstance(model.baked_context_embedding, torch.nn.Embedding):
        raise RuntimeError("Sofia baked context embedding is missing")
    return {
        "main_decoder": model.decoder,
        "final_projection": model.final_proj,
        "text_embedding": model.text_embedding,
        "text_encoder": model.encoder,
        "baked_context_embedding": model.baked_context_embedding,
        "audio_embeddings": model.audio_embeddings,
        "audio_in_projection": model.audio_in_projection,
        "local_transformer": model.local_transformer,
        "local_transformer_in_projection": model.local_transformer_in_projection,
        "local_transformer_audio_out_projection": model.local_transformer_audio_out_projection,
        "local_transformer_out_projections": model.local_transformer_out_projections,
    }


def load_model(model_path: Path) -> MagpieTTSModel:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fixture generation is forbidden")
    model = MagpieTTSModel.restore_from(str(model_path.resolve(strict=True)), map_location="cpu")
    model.eval()
    model.to("cuda")
    for module in selected_bf16_modules(model).values():
        module.to(dtype=TARGET_DTYPE)
    codec = model._codec_helper.codec_model
    floating_codec_dtypes = {parameter.dtype for parameter in codec.parameters() if parameter.is_floating_point()}
    if floating_codec_dtypes != {torch.float32}:
        raise RuntimeError(f"NanoCodec must remain FP32, got {floating_codec_dtypes}")
    model.inference_parameters.use_LT_kv_cache = True
    return model


def prepare_batch(
    model: MagpieTTSModel,
    text: str,
    language: str,
) -> dict[str, torch.Tensor | int]:
    tokenizer_name = get_tokenizer_for_language(
        language,
        list(model.tokenizer.tokenizers.keys()),
        language_tokenizer_map=model.cfg.get("language_to_tokenizer_mapping", None),
    )
    chunks, chunk_lengths, _ = chunk_text_for_inference(
        text=text,
        language=language,
        tokenizer_name=tokenizer_name,
        text_tokenizer=model.tokenizer,
        eos_token_id=model.eos_id,
    )
    if len(chunks) != 1:
        raise RuntimeError(f"fixture text must tokenize to one chunk, got {len(chunks)}")
    if chunks[0].dtype != torch.int32:
        raise RuntimeError(
            f"text token dtype mismatch: expected torch.int32, got {chunks[0].dtype}"
        )
    return {
        "text": chunks[0].unsqueeze(0).to(model.device),
        "text_lens": torch.tensor([chunk_lengths[0]], device=model.device, dtype=torch.long),
        "speaker_indices": SOFIA_INDEX,
    }


def prior_for_decoder(
    model: MagpieTTSModel,
    prior: torch.Tensor | None,
) -> torch.Tensor | list[torch.Tensor | None] | None:
    layers = model.inference_parameters.apply_prior_to_layers
    if layers is None:
        return prior
    result: list[torch.Tensor | None] = [None for _ in range(model.cfg.decoder.n_layers)]
    for layer in layers:
        result[layer] = prior
    return result


def write_decoder_state(
    writer: TensorWriter,
    prefix: str,
    session,
    valid_self_positions: int,
) -> None:
    for index, layer in enumerate(session.transformer_state.layers):
        self_state = layer.self_attention
        writer.write(
            f"{prefix}.layer_{index:02d}.self_key",
            self_state.key[:, :valid_self_positions],
        )
        writer.write(
            f"{prefix}.layer_{index:02d}.self_value",
            self_state.value[:, :valid_self_positions],
        )
        writer.write(
            f"{prefix}.layer_{index:02d}.self_mask",
            self_state.key_mask[:, :valid_self_positions],
        )
        cross_state = layer.cross_attention
        if cross_state is None or cross_state.key is None or cross_state.value is None:
            raise RuntimeError(f"decoder layer {index} has no initialized cross-attention state")
        writer.write(f"{prefix}.layer_{index:02d}.cross_key", cross_state.key)
        writer.write(f"{prefix}.layer_{index:02d}.cross_value", cross_state.value)


def capture_decoder_boundaries(
    model: MagpieTTSModel,
    batch: dict[str, torch.Tensor | int],
    writer: TensorWriter,
    local_ar_seed: int,
) -> tuple[dict[str, int | float | str | list[int]], torch.Tensor]:
    chunk_state = model.create_chunk_state(batch_size=1)
    current_chunk_len = copy.deepcopy(batch["text_lens"].detach())
    prepared_batch, max_text_len = model._prepare_chunked_text_tensors(
        chunk_state,
        copy.deepcopy(batch),
        current_chunk_len,
        True,
        model.device,
    )
    prepared_text = prepared_batch["text"]
    if (
        not isinstance(prepared_text, torch.Tensor)
        or prepared_text.dtype != torch.int32
    ):
        actual_dtype = (
            str(prepared_text.dtype)
            if isinstance(prepared_text, torch.Tensor)
            else type(prepared_text).__name__
        )
        raise RuntimeError(
            f"prepared text token dtype mismatch: "
            f"expected torch.int32, got {actual_dtype}"
        )
    context: ContextTensorsOutput = model.prepare_context_tensors(prepared_batch)
    model._update_context_from_history(
        chunk_state,
        context,
        current_chunk_len,
        max_text_len,
        True,
        prepared_batch["text_lens"],
        1,
    )
    if not isinstance(context.cond, torch.Tensor) or not isinstance(context.cond_mask, torch.Tensor):
        raise RuntimeError("decoder_ce conditioning must be tensor-valued")
    if context.additional_decoder_input is None or context.additional_decoder_mask is None:
        raise RuntimeError("Sofia baked prefix is missing")
    (
        dummy_cond,
        dummy_cond_mask,
        dummy_prefix,
        dummy_prefix_mask,
        _,
    ) = model.prepare_dummy_cond_for_cfg(
        context.cond,
        context.cond_mask,
        context.additional_decoder_input,
        context.additional_decoder_mask,
    )
    if (
        not isinstance(dummy_cond, torch.Tensor)
        or not isinstance(dummy_cond_mask, torch.Tensor)
        or dummy_prefix is None
        or dummy_prefix_mask is None
    ):
        raise RuntimeError("Sofia unconditional CFG tensors are missing")
    if dummy_cond_mask.ndim != 2 or dummy_cond_mask.shape[1] == 0:
        raise RuntimeError(
            "unconditional CFG condition mask must have shape [batch, text]"
        )
    expected_dummy_cond_mask = torch.zeros_like(dummy_cond_mask)
    expected_dummy_cond_mask[:, 0] = True
    if not torch.equal(dummy_cond_mask, expected_dummy_cond_mask):
        true_indices = torch.nonzero(
            dummy_cond_mask[0],
            as_tuple=False,
        ).flatten().tolist()
        raise RuntimeError(
            "unconditional CFG condition mask mismatch: "
            f"expected true index [0], got {true_indices}"
        )

    writer.write("input.text_token_ids", prepared_batch["text"])
    writer.write("input.text_lengths", prepared_batch["text_lens"])
    writer.write("text.mask", context.text_mask)
    writer.write("text.embedded", context.text_embedded)
    writer.write("text.condition", context.text_encoder_out)
    writer.write("cfg.conditional_condition", context.cond)
    writer.write("cfg.conditional_mask", context.cond_mask)
    writer.write("cfg.unconditional_condition", dummy_cond)
    writer.write("cfg.unconditional_mask", dummy_cond_mask)
    writer.write("cfg.sofia_prefix", context.additional_decoder_input)
    writer.write("cfg.sofia_prefix_mask", context.additional_decoder_mask)
    writer.write("cfg.unconditional_prefix", dummy_prefix)
    writer.write("cfg.unconditional_prefix_mask", dummy_prefix_mask)

    audio_codes = torch.full(
        (1, model.num_audio_codebooks, model.frame_stacking_factor),
        model.audio_bos_id,
        device=model.device,
        dtype=torch.long,
    )
    audio_mask = torch.ones((1, 1), device=model.device, dtype=torch.bool)
    audio_embedding = model._embed_audio_step(audio_codes)
    writer.write("prefill.audio_bos_codes", audio_codes)
    writer.write("prefill.audio_bos_embedding", audio_embedding)
    session = model._create_incremental_decoder_session(
        context_tensors=context,
        use_cfg=True,
        cfg_scale=model.inference_parameters.cfg_scale,
        dummy_cond=dummy_cond,
        dummy_cond_mask=dummy_cond_mask,
        dummy_additional_decoder_input=dummy_prefix,
        dummy_addition_dec_mask=dummy_prefix_mask,
        batch_size=1,
        device=audio_embedding.device,
        dtype=audio_embedding.dtype,
    )
    first_logits, first_alignment, first_hidden = model._run_chunked_forward_with_cfg(
        session=session,
        audio_codes_embedded=audio_embedding,
        audio_codes_mask=audio_mask,
        attn_prior=prior_for_decoder(model, None),
        project_code_logits=True,
    )
    if first_logits is None or first_alignment is None:
        raise RuntimeError("prefill did not produce logits and alignment")
    writer.write("prefill.logits", first_logits)
    writer.write("prefill.alignment", first_alignment)
    writer.write("prefill.hidden", first_hidden)
    write_decoder_state(writer, "prefill.state", session, session.next_position)

    alignment_scratch = model._create_first_chunk_alignment_scratch(
        text_lens=context.text_lens,
        last_attended_timesteps=chunk_state.last_attended_timesteps,
        effective_batch_size=2,
        text_length=max_text_len,
        dtype=context.text_encoder_out.dtype,
        status_capacity=4,
    )
    next_prior = model._compute_first_chunk_alignment_prior(
        first_alignment,
        alignment_scratch,
    )
    writer.write("step_000.next_prior", next_prior)
    writer.write("step_000.attended", alignment_scratch.attended)
    writer.write("step_000.attention_counters", alignment_scratch.counters)

    random_state = model._lt_helper.create_random_state(
        actual_batch_size=1,
        device=model.device,
        seed=local_ar_seed,
    )
    writer.write("local_ar.initial_seed", random_state.seed)
    writer.write("local_ar.initial_counters", random_state.counters)
    sampled_codes = model._lt_helper.sample_autoregressive(
        dec_output=first_hidden[:, -1, :],
        output_codes=audio_codes,
        temperature=model.inference_parameters.temperature,
        topk=model.inference_parameters.topk,
        unfinished_items={0: True},
        finished_items={},
        use_cfg=True,
        cfg_scale=model.inference_parameters.cfg_scale,
        forbid_audio_eos=True,
        random_state=random_state,
    )
    writer.write("local_ar.step_000.codes", sampled_codes)
    writer.write("local_ar.step_000.counters", random_state.counters)

    second_embedding = model._embed_audio_step(sampled_codes)
    second_logits, second_alignment, second_hidden = model._run_chunked_forward_with_cfg(
        session=session,
        audio_codes_embedded=second_embedding,
        audio_codes_mask=audio_mask,
        attn_prior=prior_for_decoder(model, next_prior),
        project_code_logits=True,
    )
    if second_logits is None or second_alignment is None:
        raise RuntimeError("one-step decoder did not produce logits and alignment")
    writer.write("step_001.audio_embedding", second_embedding)
    writer.write("step_001.logits", second_logits)
    writer.write("step_001.alignment", second_alignment)
    writer.write("step_001.hidden", second_hidden)
    write_decoder_state(writer, "step_001.state", session, session.next_position)
    return (
        {
            "text_tokens": max_text_len,
            "prefill_positions": 218,
            "next_position_after_step_001": session.next_position,
            "cfg_scale": float(model.inference_parameters.cfg_scale),
            "temperature": float(model.inference_parameters.temperature),
            "top_k": int(model.inference_parameters.topk),
            "attention_lookahead": int(
                model.inference_parameters.attention_prior_lookahead_window
            ),
            "attention_sink_threshold": int(
                model.inference_parameters.attention_sink_threshold
            ),
            "text_token_dtype": TEXT_TOKEN_DTYPE_NAME,
            "unconditional_condition_mask_true_indices": [0],
            "local_ar_seed": local_ar_seed,
        },
        sampled_codes.clone(),
    )


def capture_codec_boundaries(
    model: MagpieTTSModel,
    batch: dict[str, torch.Tensor | int],
    writer: TensorWriter,
    local_ar_seed: int,
    expected_first_codes: torch.Tensor,
) -> dict[str, int | list[int]]:
    seed_everything(local_ar_seed)
    with torch.inference_mode():
        generated = model.generate_speech(
            copy.deepcopy(batch),
            chunk_state=model.create_chunk_state(batch_size=1),
            end_of_text=[True],
            beginning_of_text=True,
            local_ar_seed=local_ar_seed,
            use_cfg=True,
            use_local_transformer_for_inference=True,
        )
    valid_frames = int(generated.predicted_codes_lens[0].item())
    if valid_frames < 4:
        raise RuntimeError(f"generation produced fewer than four frames: {valid_frames}")
    codes = generated.predicted_codes[:, :, :valid_frames]
    if codes.size(2) < model.frame_stacking_factor:
        raise RuntimeError(
            "complete generation is shorter than one Local AR frame stack"
        )
    generated_first_codes = codes[:, :, : model.frame_stacking_factor]
    if not torch.equal(generated_first_codes, expected_first_codes):
        mismatch_count = int(
            torch.count_nonzero(generated_first_codes != expected_first_codes)
        )
        raise RuntimeError(
            "complete generation does not use the declared local_ar_seed: "
            f"first-stack mismatches={mismatch_count}"
        )
    writer.write("generation.codes", codes)
    writer.write("generation.code_lengths", generated.predicted_codes_lens)

    frame_counts: list[int] = []
    remaining = valid_frames
    first = min(4, remaining)
    frame_counts.append(first)
    remaining -= first
    while remaining:
        frame_count = min(8, remaining)
        frame_counts.append(frame_count)
        remaining -= frame_count

    decoder = CausalCodecStreamingDecoder(
        codec_model=model._codec_helper.codec_model,
        codec_converter=model._codec_helper.codec_converter,
    )
    lengths = preallocate_causal_codec_lengths(
        batch_size=1,
        max_codec_frames=8,
        samples_per_frame=decoder.samples_per_frame,
        device=model.device,
    )
    state = CausalCodecStreamingState()
    first_frame = 0
    pcm_chunks: list[torch.Tensor] = []
    for sequence, frame_count in enumerate(frame_counts):
        next_frame = first_frame + frame_count
        pcm, pcm_lengths = decoder.decode_new(
            codes[:, :, first_frame:next_frame],
            state,
            lengths=lengths[frame_count - 1],
        )
        writer.write(f"codec.chunk_{sequence:03d}.codes", codes[:, :, first_frame:next_frame])
        writer.write(f"codec.chunk_{sequence:03d}.pcm", pcm)
        writer.write(f"codec.chunk_{sequence:03d}.pcm_lengths", pcm_lengths)
        pcm_chunks.append(pcm)
        capture_state = sequence == 0 or sequence == 1 or sequence == len(frame_counts) - 1
        if capture_state:
            if state.decoder is None:
                raise RuntimeError("codec state was not initialized")
            for state_name, tensor in _iter_named_hifigan_state_tensors(state.decoder):
                safe_state_name = state_name.replace("_", "-")
                writer.write(
                    f"codec.chunk_{sequence:03d}.state.{safe_state_name}",
                    tensor,
                )
        first_frame = next_frame
    writer.write("codec.complete_pcm", torch.cat(pcm_chunks, dim=-1))
    return {
        "local_ar_seed": local_ar_seed,
        "valid_codec_frames": valid_frames,
        "frame_schedule": frame_counts,
        "sample_rate_hz": int(model._codec_helper.codec_model.sample_rate),
        "samples_per_frame": int(decoder.samples_per_frame),
    }


def write_manifest(
    root: Path,
    lock: dict,
    args: argparse.Namespace,
    writer: TensorWriter,
    frontend_contract: FrontendContract,
    decoder_contract: dict[str, int | float | str | list[int]],
    codec_contract: dict[str, int | list[int]],
    elapsed_seconds: float,
) -> None:
    manifest = {
        "schema_version": 1,
        "fixture_id": "magpie-v2607-sofia-ja-boundaries-v1",
        "oracle_lock_sha256": sha256_file(args.lock.resolve(strict=True)),
        "model_sha256": lock["model"]["sha256"],
        "codec_model_sha256": lock["codec"]["sha256"],
        "source_bundle_sha256": lock["oracle_source"]["optimized_source_bundle_sha256"],
        "acceptance_receipt_sha256": lock["acceptance"]["receipt_sha256"],
        "text": args.text,
        "language": args.language,
        "speaker": "Sofia",
        "speaker_index": SOFIA_INDEX,
        "local_ar_seed": args.local_ar_seed,
        "frontend_contract": asdict(frontend_contract),
        "decoder_contract": decoder_contract,
        "codec_contract": codec_contract,
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_compute_capability": list(torch.cuda.get_device_capability()),
            "elapsed_seconds": elapsed_seconds,
        },
        "tensors": [asdict(record) for record in writer.records],
    }
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    (root / "manifest.json").write_bytes(payload)
    (root / "manifest.json.sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n",
        encoding="ascii",
    )


def main() -> int:
    if sys.byteorder != "little":
        raise RuntimeError(f"only little-endian fixture hosts are supported, got {sys.byteorder}")
    args = parse_args()
    output = args.output.absolute()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    lock = verify_inputs(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    started = time.perf_counter()
    try:
        seed_everything(args.local_ar_seed)
        # NanoCodec parity is defined in IEEE FP32.  PyTorch enables TF32 for
        # cuDNN convolutions by default on NVIDIA GPUs; leaving that implicit
        # produced a fixture that no longer represented the accepted
        # no-TF32 TensorRT build contract.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_model(args.model)
        frontend_contract = require_frontend_contract(model, lock)
        batch = prepare_batch(model, args.text, args.language)
        writer = TensorWriter(staging)
        with torch.inference_mode():
            decoder_contract, expected_first_codes = capture_decoder_boundaries(
                model,
                batch,
                writer,
                args.local_ar_seed,
            )
        codec_contract = capture_codec_boundaries(
            model,
            batch,
            writer,
            args.local_ar_seed,
            expected_first_codes,
        )
        torch.cuda.synchronize(model.device)
        write_manifest(
            staging,
            lock,
            args,
            writer,
            frontend_contract,
            decoder_contract,
            codec_contract,
            time.perf_counter() - started,
        )
        validate_boundary_fixture(staging, args.lock)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

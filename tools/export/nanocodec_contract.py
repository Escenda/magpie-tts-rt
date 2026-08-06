"""Canonical v1 NanoCodec persistent-state and binding registry.

This module is intentionally independent of PyTorch and NeMo.  Exporters must
prove that the loaded model enumerates this exact registry.  The manifest
schema, C++ parser, and checked fixture are generated or validated against the
same order; no runtime component derives state from an opaque aggregate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


STAGE_CHANNELS = (432, 216, 108, 54, 27)
STAGE_OVERLAPS = (8, 8, 4, 2, 2)
BRANCH_KERNELS = (3, 7, 11)
BLOCK_DILATIONS = (1, 3, 5)


@dataclass(frozen=True)
class CanonicalCodecStateBinding:
    logical_name: str
    dtype: str
    shape: tuple[int, int, int]
    initial_output_binding: str
    steady_input_binding: str
    steady_output_binding: str
    tail_input_binding: str
    tail_output_binding: str

    def manifest_record(self) -> dict:
        record = asdict(self)
        record["shape"] = list(self.shape)
        return record


def make_binding(
    logical_name: str,
    shape: tuple[int, int, int],
) -> CanonicalCodecStateBinding:
    return CanonicalCodecStateBinding(
        logical_name=logical_name,
        dtype="fp32",
        shape=shape,
        initial_output_binding=f"state_out.{logical_name}",
        steady_input_binding=f"state_in.{logical_name}",
        steady_output_binding=f"state_out.{logical_name}",
        tail_input_binding=f"state_in.{logical_name}",
        tail_output_binding=f"state_out.{logical_name}",
    )


def canonical_state_bindings() -> tuple[CanonicalCodecStateBinding, ...]:
    result = [
        make_binding("pre_conv.input_history", (1, 32, 6)),
    ]
    result.extend(
        make_binding(
            f"upsample_convs.{stage_index}.pending_overlap",
            (1, channels, overlap),
        )
        for stage_index, (channels, overlap) in enumerate(
            zip(STAGE_CHANNELS, STAGE_OVERLAPS)
        )
    )
    for stage_index, channels in enumerate(STAGE_CHANNELS):
        for branch_index, kernel_size in enumerate(BRANCH_KERNELS):
            skip_history = kernel_size - 1
            for block_index, dilation in enumerate(BLOCK_DILATIONS):
                prefix = (
                    f"residual_layers.{stage_index}.branches.{branch_index}."
                    f"residual_blocks.{block_index}"
                )
                result.append(
                    make_binding(
                        f"{prefix}.input_conv.input_history",
                        (1, channels, (kernel_size - 1) * dilation),
                    )
                )
                result.append(
                    make_binding(
                        f"{prefix}.skip_conv.input_history",
                        (1, channels, skip_history),
                    )
                )
    result.append(
        make_binding("post_conv.input_history", (1, 27, 2))
    )
    bindings = tuple(result)
    if len(bindings) != 97:
        raise AssertionError(
            f"canonical NanoCodec state count changed: {len(bindings)}"
        )
    names = tuple(binding.logical_name for binding in bindings)
    if len(set(names)) != len(names):
        raise AssertionError("canonical NanoCodec state names are duplicated")
    return bindings


CANONICAL_STATE_BINDINGS = canonical_state_bindings()

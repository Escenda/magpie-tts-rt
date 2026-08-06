# Stateful NanoCodec export

`tools/export/export_nanocodec.py` is the complete P1 gate for the accepted
NanoCodec. It does not decode an accumulated prefix. Each invocation accepts
only one new chunk and explicitly replaces all persistent causal state.

The gate performs these operations in order:

1. authenticate the Magpie model, NanoCodec, optimized NeMo checkout,
   acceptance receipt, oracle lock, and boundary fixture;
2. require the fixture's IEEE FP32 policy and validate every fixture byte;
3. materialize the 97 weight-normalized decoder convolutions;
4. prove that the loaded model has the exact 97-state canonical registry;
5. reproduce the fixture schedule and compare tail frame counts 1 through 8
   against the locked NeMo stateful decoder;
6. export initial-4, steady-8, and dynamic tail-1-through-8 ONNX graphs;
7. build strongly typed TensorRT plans with TF32 disabled;
8. deserialize and inspect every plan binding, dtype, shape, and profile;
9. execute plan parity for initial, steady, and all eight tail sizes;
10. execute the complete initial-4 → steady-8 → terminal-1-through-8 state
    chain and measure per-state drift plus concatenated PCM error;
11. record non-isolated Thor latency and atomically publish only after every
    preceding gate passes.

PyTorch functional-wrapper parity uses predeclared FP32 tolerances
`atol=1e-5, rtol=1e-5`. TensorRT parity uses
`atol=2e-4, rtol=2e-4`. Passing every authenticated component gate produces
an `accepted` stateful NanoCodec receipt. The receipt records max, mean, p99,
relative error, and SNR for every compared PCM/state tensor; the exporter
never derives or widens a tolerance from a run.

This acceptance is scoped to the stateful NanoCodec component: exact plan
contracts, initial/steady/tail state transitions, all terminal frame counts,
and complete chained PCM parity. Multi-utterance sequence evidence,
long-duration drift, and isolated performance remain release-level evidence;
the component receipt does not claim those results.

The 92 concatenation buffers used by the PyTorch implementation are workspace,
not session state. Only the 92 causal histories and five pending overlaps
cross an engine boundary. `tools/export/nanocodec_contract.py` is the checked
canonical registry; `tools/export/sync_nanocodec_manifest_contract.py` keeps
the JSON schema and parser fixture synchronized with it.

Example:

```bash
python3 tools/export/export_nanocodec.py \
  --lock reference/oracle-lock.json \
  --speech-root /path/to/accepted/NeMo \
  --model /path/to/magpie_tts_multilingual_357m.nemo \
  --codec-model /path/to/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  --acceptance-receipt /path/to/final_native_bf16_streaming_acceptance.json \
  --fixture /path/to/magpie-v2607-sofia-ja-boundaries-v1 \
  --trtexec /usr/bin/trtexec \
  --output /new/path/nanocodec-v1
```

The output path must not exist. Build logs, layer introspection, ONNX files,
plans, the exact state contract, hashes, parity metrics, and latency metrics
are staged together. Any failure removes staging and leaves the destination
absent.

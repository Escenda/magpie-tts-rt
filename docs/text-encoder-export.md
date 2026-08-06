# Text Encoder export

The first TensorRT component is the six-layer Text Encoder. Its public engine
contract is:

- `text_token_ids`: `[1,T] INT32`;
- `text_mask`: `[1,T] BOOL`;
- `text_condition`: `[1,T,768] BF16`; and
- profile `text_1_512`, with `T=1/64/512` for min/opt/max.

The ONNX file is not exported from an arbitrary restored model. The exporter
first verifies the model, NanoCodec, all accepted NeMo source files, repeated
active model configuration, and acceptance receipt against
`reference/oracle-lock.json`. It then requires an authenticated boundary
fixture made from that exact lock.

The accepted fixture contains INT32 token IDs, and those exact IDs and its mask
are the trace inputs. A caller with wider integer IDs must range-check before
converting them at its boundary; the engine does not perform a narrowing
conversion. Before ONNX export, the BF16 PyTorch output must be bit-identical
to `text.condition` in the fixture.
A fixture generated from another lock, a source imported from another NeMo
checkout, or a numerically different Text Encoder stops the export.

## Export

Run the tool in the accepted isolated Python environment. The environment must
make its CUDA, NVPL, and PyTorch shared libraries resolvable:

```bash
python tools/export/export_text_encoder.py \
  --lock reference/oracle-lock.json \
  --speech-root /path/to/accepted/Speech \
  --model /path/to/magpie_tts_multilingual_357m.nemo \
  --codec-model /path/to/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  --acceptance-receipt /path/to/final_native_bf16_streaming_acceptance.json \
  --fixture /path/to/authenticated/magpie-v2607-sofia-ja-boundaries-v1 \
  --output /new/path/text-encoder-export
```

The output path must not exist. Export happens in a sibling staging directory
and is published with Linux `renameat2(RENAME_NOREPLACE)` only after:

1. PyTorch fixture parity passes;
2. the legacy TorchScript ONNX exporter completes at opset 20;
3. ONNX structural checking, exact MagpieTTS-RT custom-node counts, and
   complex-type checks pass;
4. names, dtypes, dynamic dimensions, and embedded weights match the fixed
   contract; and
5. the exporter receipt and its checksum are complete.

There is no dynamo exporter, external-weight, CPU, alternate-source, alternate
device, or existing-output fallback.

PyTorch 2.11's legacy exporter incorrectly lowers implicit BOOL-to-BF16
arithmetic promotion to ONNX `Cast(to=15)`, which is COMPLEX128. The export
wrapper therefore converts the BOOL mask to exact BF16 zero/one values with
`Where` before calling the unchanged six-layer encoder. The mandatory
bit-exact PyTorch fixture comparison verifies that this explicit lowering does
not change the accepted output.

The wrapper also lowers all 13 LayerNorm operations, six tanh-GELU
operations, and six self-attention Softmax operations to the authenticated
plugin ABI. Their accepted Text shapes are respectively `[1,T,768]`,
`[1,3072,T]`, and `[1,12,T,T]`, with `T=1..512`. The same plugin library is
authenticated by SHA-256 and loaded before both Text Encoder and Local AR
plans are deserialized.

ONNX 1.22's optional full shape-inference pass does not implement BF16 Conv
and reports it as unsupported. The exporter does not relabel or widen the
accepted BF16 network to satisfy that tool limitation. It runs the standard
ONNX checker, verifies all external names/dtypes/shapes itself, permits only
the locked `magpie_tts_rt` LayerNorm, tanh-GELU, and Softmax nodes, rejects
every other non-standard domain and complex cast/weight, and leaves TensorRT
parsing plus sequence parity as an independent required gate.

## Build the parity TensorRT plan

The initial parity plan uses TensorRT 10.16, BF16, and explicitly disables
TF32:

The build tool registers the authenticated plugin in-process before the ONNX
parser runs, creates a strongly typed TensorRT network, disables TF32, and
adds the exact `1/64/512` optimization profile. A subprocess-only `trtexec`
build is not used because loading the shared library alone does not invoke the
MagpieTTS-RT explicit registration API.

Engine build success alone is not component acceptance. Plan I/O and the
complete optimization profile must be inspected by name. Canonical-fixture
numeric parity is only a candidate gate; the plan cannot enter a model bundle
until three predeclared Japanese fixtures preserve the complete generated
codec sequence and EOS boundary exactly.

The ONNX export receipt is `accepted` only as structural export evidence after
locked-source authentication, bit-exact PyTorch fixture parity, and ONNX
contract validation. The built plan receipt remains
`measured-not-accepted`, even after it authenticates that export, plugin,
profile, binding contract, and all four numerical diagnostics below. A
separate no-replace promotion receipt is produced only after the complete
Text Encoder → Main Decoder → Local AR loop is exact for all three fixtures.

`build_text_encoder_plan.py` performs that complete gate and atomically
publishes the plan, captured build log, and checksummed receipt:

```bash
python tools/export/build_text_encoder_plan.py \
  --lock reference/oracle-lock.json \
  --fixture /path/to/authenticated/magpie-v2607-sofia-ja-boundaries-v1 \
  --export /path/to/text-encoder-export \
  --plugin /path/to/libmagpie_tts_rt_plugins.so \
  --tensorrt-python-path /usr/lib/python3.12/dist-packages \
  --output /new/path/text-encoder-plan
```

It deserializes the generated plan with TensorRT, requires exactly one profile
and exactly the three named bindings, and checks both inputs have the complete
`1/64/512` profile. It then executes the fixture on Thor and requires:

- maximum absolute error at most `0.125`;
- mean absolute error at most `0.00125`;
- 99th-percentile absolute error at most `0.006`; and
- cosine similarity at least `0.99998`.

These thresholds reject a grossly changed graph, precision route, or
conditioning tensor, but do not authorize semantic acceptance. The receipt
records the measured values, plan and build-log hashes, authenticated plugin,
exact build switches, inspected I/O/profile, TensorRT/CUDA environment, and a
short device-only latency diagnostic. Exact generated sequences, rather than
these floating-point thresholds, are the acceptance boundary.

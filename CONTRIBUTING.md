# Contributing

MagpieTTS-RT is latency-sensitive, stateful inference software. A change is not
accepted merely because it builds or produces audible output.

Every inference change must state:

- the exact engine or runtime boundary it changes;
- tensor names, dtypes, shapes, state ownership, and lifetime;
- the reference oracle and model revision used;
- numerical, token, alignment, EOS, waveform, and performance results;
- failure behavior and why it cannot silently change synthesis semantics.

Do not add:

- implicit CPU, PyTorch, eager, or alternate-engine fallback;
- default values for missing manifest fields;
- unbounded work or audio queues;
- process-global last-error state;
- CUDA, TensorRT, C++, or STL types to the C ABI;
- engine plans, model weights, generated audio, or developer-local paths.

The runtime must reject incompatible or unverifiable assets before creating a
synthesis session.

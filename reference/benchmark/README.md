# Japanese runtime benchmark corpus

`thor-ja-v1-source.json` is the reviewed source of the Japanese acceptance
corpus. It pins every source utterance and deterministic `uint32` random seed.
It contains 108 normal cases and one cancellation case. The normal cases cover
short replies, fillers, conversation, punctuation, numbers and units, Latin
abbreviations, spatial language, robot instructions, and long replies.

The tracked source intentionally does not contain prepared token IDs. Generate
the strict JSONL consumed by `mtt-runtime-benchmark` with the accepted
Japanese frontend:

```bash
python3 -m tools.frontend.generate_benchmark_corpus \
  --source /absolute/path/to/magpie-tts-rt/reference/benchmark/thor-ja-v1-source.json \
  --oracle-lock /absolute/path/to/magpie-tts-rt/reference/oracle-lock.json \
  --frontend-contract /absolute/path/to/accepted/frontend-contract.json \
  --output /absolute/path/to/new/thor-ja-v1.jsonl
```

The generator validates the frontend contract against `oracle-lock.json`,
computes the tokenizer identity, prepares the global token IDs, and computes
each UTF-8 source-text SHA-256. The tokenizer identity must equal the identity
pinned by the source file. Input files must be absolute canonical regular
files. The output must be a new absolute path; an existing file is never
replaced.

The generated JSONL has this exact order:

1. one `header` record;
2. exactly 108 `case` records;
3. one final `cancel_case` record.

Do not hand-edit the generated JSONL. Review text or seed changes in the source
JSON, update its corpus ID and identity pin when required, and generate a new
output file.

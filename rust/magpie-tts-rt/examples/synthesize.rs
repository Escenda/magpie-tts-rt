use std::env;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use magpie_tts_rt::{InferenceWorker, RuntimeConfig, SynthesisEvent, WorkerConfig, WorkerError};

const WAV_HEADER_BYTES: u32 = 44;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;
const PCM_CHANNELS: u16 = 1;
const PCM_BITS_PER_SAMPLE: u16 = 32;

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Debug)]
struct Arguments {
    bundle: String,
    manifest_sha256: [u8; 32],
    token_ids: Vec<i64>,
    random_seed: u64,
    output_wav: PathBuf,
    cuda_device: i32,
    warmup_runs: u32,
}

#[derive(Clone, Copy, Debug)]
struct SynthesisMetrics {
    request_start: Duration,
    ttfa: Duration,
    total: Duration,
    audio_samples: u64,
    sample_rate_hz: u32,
    chunks: u64,
}

struct WavWriter {
    path: PathBuf,
    output: BufWriter<File>,
    sample_rate_hz: Option<u32>,
    sample_count: u64,
    complete: bool,
}

impl WavWriter {
    fn create(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut output = BufWriter::new(file);
        output.write_all(&[0_u8; WAV_HEADER_BYTES as usize])?;
        Ok(Self {
            path: path.to_path_buf(),
            output,
            sample_rate_hz: None,
            sample_count: 0,
            complete: false,
        })
    }

    fn write_samples(
        &mut self,
        sample_rate_hz: u32,
        samples: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        match self.sample_rate_hz {
            None => self.sample_rate_hz = Some(sample_rate_hz),
            Some(expected) if expected == sample_rate_hz => {}
            Some(_) => return Err(CliError("sample rate changed between chunks".into()).into()),
        }
        let incoming = u64::try_from(samples.len())?;
        let next_count = self
            .sample_count
            .checked_add(incoming)
            .ok_or_else(|| CliError("WAV sample count overflowed".into()))?;
        let data_bytes = next_count
            .checked_mul(u64::from(PCM_BITS_PER_SAMPLE / 8))
            .ok_or_else(|| CliError("WAV byte count overflowed".into()))?;
        if data_bytes > u64::from(u32::MAX - (WAV_HEADER_BYTES - 8)) {
            return Err(CliError("WAV output exceeds the RIFF32 size limit".into()).into());
        }
        for sample in samples {
            self.output.write_all(&sample.to_le_bytes())?;
        }
        self.sample_count = next_count;
        Ok(())
    }

    fn finish(mut self) -> Result<(), Box<dyn Error>> {
        let sample_rate_hz = self
            .sample_rate_hz
            .ok_or_else(|| CliError("no PCM was produced".into()))?;
        let data_bytes = u32::try_from(self.sample_count * u64::from(PCM_BITS_PER_SAMPLE / 8))?;
        let riff_bytes = data_bytes
            .checked_add(WAV_HEADER_BYTES - 8)
            .ok_or_else(|| CliError("RIFF size overflowed".into()))?;
        let bytes_per_sample = PCM_BITS_PER_SAMPLE / 8;
        let byte_rate = sample_rate_hz
            .checked_mul(u32::from(PCM_CHANNELS))
            .and_then(|value| value.checked_mul(u32::from(bytes_per_sample)))
            .ok_or_else(|| CliError("WAV byte rate overflowed".into()))?;
        let block_align = PCM_CHANNELS
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| CliError("WAV block alignment overflowed".into()))?;

        self.output.flush()?;
        self.output.seek(SeekFrom::Start(0))?;
        self.output.write_all(b"RIFF")?;
        self.output.write_all(&riff_bytes.to_le_bytes())?;
        self.output.write_all(b"WAVE")?;
        self.output.write_all(b"fmt ")?;
        self.output.write_all(&16_u32.to_le_bytes())?;
        self.output
            .write_all(&WAV_FORMAT_IEEE_FLOAT.to_le_bytes())?;
        self.output.write_all(&PCM_CHANNELS.to_le_bytes())?;
        self.output.write_all(&sample_rate_hz.to_le_bytes())?;
        self.output.write_all(&byte_rate.to_le_bytes())?;
        self.output.write_all(&block_align.to_le_bytes())?;
        self.output.write_all(&PCM_BITS_PER_SAMPLE.to_le_bytes())?;
        self.output.write_all(b"data")?;
        self.output.write_all(&data_bytes.to_le_bytes())?;
        self.output.flush()?;
        self.output.get_ref().sync_all()?;
        self.complete = true;
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        if !self.complete {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn usage() -> &'static str {
    concat!(
        "usage: synthesize \\\n",
        "  --bundle PATH --manifest-sha256 HEX --tokens-file PATH \\\n",
        "  --seed UINT32 --output-wav PATH ",
        "[--cuda-device INDEX] [--warmup-runs COUNT]"
    )
}

fn option_value(
    values: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, CliError> {
    values
        .next()
        .ok_or_else(|| CliError(format!("{option} requires a value")))
}

fn parse_arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut bundle = None;
    let mut manifest_sha256 = None;
    let mut tokens_file = None;
    let mut random_seed = None;
    let mut output_wav = None;
    let mut cuda_device = 0_i32;
    let mut warmup_runs = 0_u32;
    let mut values = env::args().skip(1);
    while let Some(option) = values.next() {
        match option.as_str() {
            "--bundle" => bundle = Some(option_value(&mut values, &option)?),
            "--manifest-sha256" => {
                manifest_sha256 = Some(parse_sha256(&option_value(&mut values, &option)?)?)
            }
            "--tokens-file" => {
                tokens_file = Some(PathBuf::from(option_value(&mut values, &option)?))
            }
            "--seed" => random_seed = Some(option_value(&mut values, &option)?.parse::<u32>()?),
            "--output-wav" => output_wav = Some(PathBuf::from(option_value(&mut values, &option)?)),
            "--cuda-device" => cuda_device = option_value(&mut values, &option)?.parse::<i32>()?,
            "--warmup-runs" => warmup_runs = option_value(&mut values, &option)?.parse::<u32>()?,
            "--help" | "-h" => return Err(CliError(usage().into()).into()),
            _ => return Err(CliError(format!("unknown option {option}\n{}", usage())).into()),
        }
    }
    let bundle = bundle.ok_or_else(|| CliError(format!("missing --bundle\n{}", usage())))?;
    let manifest_sha256 = manifest_sha256
        .ok_or_else(|| CliError(format!("missing --manifest-sha256\n{}", usage())))?;
    let tokens_file =
        tokens_file.ok_or_else(|| CliError(format!("missing --tokens-file\n{}", usage())))?;
    let random_seed =
        random_seed.ok_or_else(|| CliError(format!("missing --seed\n{}", usage())))?;
    let output_wav =
        output_wav.ok_or_else(|| CliError(format!("missing --output-wav\n{}", usage())))?;
    let token_ids = read_token_ids(&tokens_file)?;
    Ok(Arguments {
        bundle,
        manifest_sha256,
        token_ids,
        random_seed: u64::from(random_seed),
        output_wav,
        cuda_device,
        warmup_runs,
    })
}

fn parse_sha256(value: &str) -> Result<[u8; 32], Box<dyn Error>> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CliError("manifest SHA-256 must contain exactly 64 hex digits".into()).into());
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)?;
    }
    Ok(digest)
}

fn read_token_ids(path: &Path) -> Result<Vec<i64>, Box<dyn Error>> {
    let mut input = String::new();
    File::open(path)?.read_to_string(&mut input)?;
    let normalized = input.replace(',', " ");
    if normalized
        .chars()
        .any(|character| !character.is_ascii_whitespace() && !character.is_ascii_digit())
    {
        return Err(CliError(
            "token file may contain only non-negative decimal IDs, commas, and whitespace".into(),
        )
        .into());
    }
    let tokens = normalized
        .split_ascii_whitespace()
        .map(str::parse::<i64>)
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err(CliError("token file contains no IDs".into()).into());
    }
    Ok(tokens)
}

fn run_request(
    worker: &InferenceWorker,
    token_ids: &[i64],
    random_seed: u64,
    mut wav: Option<&mut WavWriter>,
) -> Result<SynthesisMetrics, Box<dyn Error>> {
    let request_started_at = Instant::now();
    let stream = worker.synthesize(token_ids.to_vec(), random_seed)?;
    let request_start = request_started_at.elapsed();
    let mut first_audio_at = None;
    let mut expected_sequence = 0_u64;
    let mut expected_sample_index = 0_u64;
    let mut sample_rate_hz = None;
    let mut final_seen = false;

    loop {
        match stream.recv()? {
            SynthesisEvent::Audio(chunk) => {
                let now = Instant::now();
                first_audio_at.get_or_insert(now);
                if chunk.sequence != expected_sequence {
                    return Err(CliError(format!(
                        "audio sequence mismatch: expected {expected_sequence}, got {}",
                        chunk.sequence
                    ))
                    .into());
                }
                if chunk.first_sample_index != expected_sample_index {
                    return Err(CliError(format!(
                        "audio sample gap: expected {expected_sample_index}, got {}",
                        chunk.first_sample_index
                    ))
                    .into());
                }
                if chunk.first != (expected_sequence == 0) {
                    return Err(CliError("FIRST flag does not match sequence zero".into()).into());
                }
                if final_seen {
                    return Err(CliError("received audio after FINAL".into()).into());
                }
                expected_sequence = expected_sequence
                    .checked_add(1)
                    .ok_or_else(|| CliError("audio sequence overflowed".into()))?;
                expected_sample_index = expected_sample_index
                    .checked_add(u64::try_from(chunk.samples.len())?)
                    .ok_or_else(|| CliError("audio sample position overflowed".into()))?;
                match sample_rate_hz {
                    None => sample_rate_hz = Some(chunk.sample_rate_hz),
                    Some(expected) if expected == chunk.sample_rate_hz => {}
                    Some(_) => {
                        return Err(CliError("sample rate changed between chunks".into()).into());
                    }
                }
                final_seen = chunk.final_chunk;
                if let Some(output) = wav.as_mut() {
                    output.write_samples(chunk.sample_rate_hz, &chunk.samples)?;
                }
            }
            SynthesisEvent::Completed => break,
            SynthesisEvent::Cancelled => {
                return Err(CliError("request was cancelled".into()).into());
            }
            SynthesisEvent::Failed(error) => return Err(error.into()),
            SynthesisEvent::RuntimeError(error) => return Err(error.into()),
        }
    }
    let total = request_started_at.elapsed();
    let first_audio_at =
        first_audio_at.ok_or_else(|| CliError("request completed without PCM".into()))?;
    if !final_seen {
        return Err(CliError("request completed without a FINAL audio chunk".into()).into());
    }
    Ok(SynthesisMetrics {
        request_start,
        ttfa: first_audio_at.duration_since(request_started_at),
        total,
        audio_samples: expected_sample_index,
        sample_rate_hz: sample_rate_hz
            .ok_or_else(|| CliError("request completed without a sample rate".into()))?,
        chunks: expected_sequence,
    })
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_metrics(prefix: &str, metrics: SynthesisMetrics) {
    let audio_seconds = metrics.audio_samples as f64 / f64::from(metrics.sample_rate_hz);
    let realtime_factor = metrics.total.as_secs_f64() / audio_seconds;
    println!(
        "{prefix}.request_start_ms={:.3}",
        milliseconds(metrics.request_start)
    );
    println!("{prefix}.ttfa_ms={:.3}", milliseconds(metrics.ttfa));
    println!("{prefix}.total_ms={:.3}", milliseconds(metrics.total));
    println!("{prefix}.audio_samples={}", metrics.audio_samples);
    println!("{prefix}.audio_seconds={audio_seconds:.6}");
    println!("{prefix}.chunks={}", metrics.chunks);
    println!("{prefix}.rtf={realtime_factor:.6}");
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments()?;
    let runtime = RuntimeConfig::new(arguments.cuda_device)?;
    let worker_config =
        WorkerConfig::new(runtime, arguments.bundle.clone(), arguments.manifest_sha256)?;
    let load_started_at = Instant::now();
    let worker = InferenceWorker::spawn(worker_config)?;
    println!(
        "runtime_load_ms={:.3}",
        milliseconds(load_started_at.elapsed())
    );

    for index in 0..arguments.warmup_runs {
        let metrics = run_request(&worker, &arguments.token_ids, arguments.random_seed, None)?;
        print_metrics(&format!("warmup_{index}"), metrics);
    }

    let mut wav = WavWriter::create(&arguments.output_wav)?;
    let metrics = run_request(
        &worker,
        &arguments.token_ids,
        arguments.random_seed,
        Some(&mut wav),
    )?;
    wav.finish()?;
    worker
        .shutdown()
        .map_err(|error: WorkerError| -> Box<dyn Error> { Box::new(error) })?;
    print_metrics("measured", metrics);
    println!("output_wav={}", arguments.output_wav.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("synthesis failed: {error}");
        std::process::exit(1);
    }
}

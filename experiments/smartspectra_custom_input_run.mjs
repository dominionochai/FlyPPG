import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import {
  SmartSpectraSDK,
  ProcessingStatus,
  breathingMetrics,
  cardioMetrics,
  PixelFormat,
} from '@smartspectra/node-sdk';
import { decodeMetrics } from '@smartspectra/node-sdk/messages';

const [inputPath, outputDir = 'smartspectra-custom-input'] = process.argv.slice(2);
const apiKey = process.env.SMARTSPECTRA_API_KEY;

if (!apiKey) throw new Error('SMARTSPECTRA_API_KEY is not set');
if (!inputPath || !fs.existsSync(inputPath)) {
  throw new Error('Usage: node experiments/smartspectra_custom_input_run.mjs <processed-video> [output-dir]');
}

function runProbe(args) {
  const result = spawnSync('ffprobe', args, { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr || 'ffprobe failed');
  return JSON.parse(result.stdout);
}

function parseRate(value) {
  const [n, d] = String(value).split('/').map(Number);
  return d ? n / d : n;
}

const probe = runProbe([
  '-v', 'error', '-select_streams', 'v:0',
  '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate',
  '-of', 'json', inputPath,
]);
const stream = probe.streams?.[0];
if (!stream) throw new Error('No video stream found');

const width = Number(stream.width);
const height = Number(stream.height);
const avgFps = parseRate(stream.avg_frame_rate);\nconst fallbackFps = parseRate(stream.r_frame_rate);\nconst fps = Number.isFinite(avgFps) && avgFps > 0 ? avgFps : fallbackFps;
if (!width || !height || !Number.isFinite(fps) || fps <= 0) {
  throw new Error('Invalid video dimensions or frame rate');
}

fs.mkdirSync(outputDir, { recursive: true });
const packets = [];
const validations = [];
const errors = [];
let runningResolve;
const running = new Promise((resolve) => { runningResolve = resolve; });

const sdk = new SmartSpectraSDK({
  apiKey,
  requestedMetrics: [...breathingMetrics, ...cardioMetrics],
});

sdk.on('processingStatus', (status) => {
  if (status === ProcessingStatus.kRunning) runningResolve();
});
sdk.on('metrics', (buf, timestampUs) => {
  packets.push({ timestamp_us: timestampUs, metrics: decodeMetrics(buf) });
});
sdk.on('validationStatus', (code, timestampUs, hint) => {
  validations.push({ code, timestamp_us: timestampUs, hint });
});
sdk.on('error', (code, message, retryable) => {
  errors.push({ code, message, retryable });
  process.stderr.write(`SmartSpectra error ${code}: ${message} (retryable=${retryable})\n`);
});

const ffmpeg = spawn('ffmpeg', [
  '-v', 'error', '-i', inputPath, '-map', '0:v:0',
  '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
], { stdio: ['ignore', 'pipe', 'pipe'] });

let ffmpegError = '';
ffmpeg.stderr.setEncoding('utf8');
ffmpeg.stderr.on('data', (chunk) => { ffmpegError += chunk; });

async function* readRgbFrames(readable, frameBytes) {
  let pending = Buffer.alloc(0);
  for await (const chunk of readable) {
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
    while (pending.length >= frameBytes) {
      yield pending.subarray(0, frameBytes);
      pending = pending.subarray(frameBytes);
    }
  }
  if (pending.length) throw new Error(`Decoder ended with ${pending.length} trailing bytes (expected a full frame)`);
}

let frameCount = 0;
const frameBytes = width * height * 3;
const frameIntervalMs = 1000 / fps;

try {
  sdk.useCustomInput();
  sdk.start();
  await Promise.race([
    running,
    new Promise((_, reject) => setTimeout(() => reject(new Error('SDK did not enter running state')), 30000)),
  ]);

  for await (const frame of readRgbFrames(ffmpeg.stdout, frameBytes)) {
    const timestampUs = Math.round(frameCount * 1_000_000 / fps);
    const accepted = sdk.sendFrame(
      frame, width, height, width * 3, PixelFormat.kRGB, timestampUs,
    );
    if (!accepted) throw new Error(`SmartSpectra rejected input frame ${frameCount}`);
    frameCount += 1;
    await new Promise((resolve) => setTimeout(resolve, frameIntervalMs));
  }

  const exit = await ffmpegClosed;
  if (exit !== 0) throw new Error(`ffmpeg exited ${exit}: ${ffmpegError}`);
  await sdk.stopAsync();
} finally {
  if (ffmpeg.exitCode === null) ffmpeg.kill('SIGTERM');
  await sdk.destroy();
}

const pulsePackets = packets.filter((p) => p.metrics?.cardio?.pulseRate?.length);
const breathingPackets = packets.filter((p) => p.metrics?.breathing?.rate?.length);
const summary = {
  input: path.basename(inputPath),
  input_mode: 'useCustomInput + sendFrame',
  width, height, fps,
  frames_submitted: frameCount,
  packet_count: packets.length,
  pulse_packet_count: pulsePackets.length,
  breathing_packet_count: breathingPackets.length,
  validation_event_count: validations.length,
  error_count: errors.length,
  pulse_rate_samples: pulsePackets.flatMap((p) => p.metrics.cardio.pulseRate),
  breathing_rate_samples: breathingPackets.flatMap((p) => p.metrics.breathing.rate),
};

fs.writeFileSync(path.join(outputDir, 'metrics.json'), JSON.stringify({
  summary, packets, validations, errors,
}, null, 2));
fs.writeFileSync(path.join(outputDir, 'summary.json'), JSON.stringify(summary, null, 2));
process.stdout.write(JSON.stringify(summary, null, 2) + '\n');

if (!frameCount) throw new Error('No video frames were submitted');
if (!summary.pulse_packet_count) process.exitCode = 1;

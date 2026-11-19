import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { SmartSpectraSDK, PixelFormat, breathingMetrics, cardioMetrics } from '@smartspectra/node-sdk';
import { decodeMetrics } from '@smartspectra/node-sdk/messages';

const [videoPath, outputDir = 'smartspectra-custom-input'] = process.argv.slice(2);
const apiKey = process.env.SMARTSPECTRA_API_KEY;
if (!apiKey) throw new Error('SMARTSPECTRA_API_KEY is required');
if (!videoPath || !fs.existsSync(videoPath)) throw new Error('Usage: node experiments/smartspectra_custom_input_run.mjs <video> [output-dir]');

function probeVideo(file) {
  const result = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate', '-of', 'json', file], { encoding: 'utf8' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr || 'ffprobe failed');
  const stream = JSON.parse(result.stdout).streams?.[0];
  if (!stream) throw new Error('Input contains no video stream');
  const rate = (value) => { const [n, d = '1'] = String(value).split('/'); return Number(n) / Number(d); };
  const width = Number(stream.width);
  const height = Number(stream.height);
  const fps = rate(stream.avg_frame_rate) || rate(stream.r_frame_rate);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('Invalid video dimensions');
  if (!Number.isFinite(fps) || fps <= 0) throw new Error('Invalid video frame rate');
  return { width, height, fps };
}

async function* readRgbFrames(readable, frameSize) {
  let frame = Buffer.allocUnsafe(frameSize);
  let filled = 0;
  for await (const chunk of readable) {
    let offset = 0;
    while (offset < chunk.length) {
      const copied = chunk.copy(frame, filled, offset);
      filled += copied;
      offset += copied;
      if (filled === frameSize) {
        yield frame;
        frame = Buffer.allocUnsafe(frameSize);
        filled = 0;
      }
    }
  }
  if (filled !== 0) throw new Error(`Decoder ended with an incomplete frame (${filled}/${frameSize} bytes)`);
}

const { width, height, fps } = probeVideo(videoPath);
const stride = width * 3;
const frameSize = stride * height;
const packets = [];
const validations = [];
const errors = [];
const sdk = new SmartSpectraSDK({ apiKey, requestedMetrics: [...breathingMetrics, ...cardioMetrics] });
sdk.on('metrics', (buffer, timestampUs) => packets.push({ timestamp_us: timestampUs, metrics: decodeMetrics(buffer) }));
sdk.on('validationStatus', (code, timestampUs, hint) => validations.push({ code, timestamp_us: timestampUs, hint }));
sdk.on('error', (code, message, retryable) => {
  errors.push({ code, message, retryable });
  process.stderr.write(`SmartSpectra error ${code}: ${message} (retryable=${retryable})\n`);
});

const decoder = spawn('ffmpeg', ['-v', 'error', '-i', videoPath, '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], { stdio: ['ignore', 'pipe', 'pipe'] });
let decoderError = '';
decoder.stderr.setEncoding('utf8');
decoder.stderr.on('data', (chunk) => { decoderError += chunk; });
const decoderClosed = new Promise((resolve) => decoder.once('close', resolve));
let framesSubmitted = 0;

try {
  sdk.useCustomInput();
  sdk.start();
  for await (const frame of readRgbFrames(decoder.stdout, frameSize)) {
    const timestampUs = Math.round(framesSubmitted * 1_000_000 / fps);
    if (!sdk.sendFrame(frame, width, height, stride, PixelFormat.kRGB, timestampUs)) {
      throw new Error(`SmartSpectra rejected frame ${framesSubmitted}`);
    }
    framesSubmitted += 1;
  }
  const exitCode = await decoderClosed;
  if (exitCode !== 0) throw new Error(decoderError || `ffmpeg exited with code ${exitCode}`);
  await sdk.stopAsync();
} catch (error) {
  decoder.kill('SIGTERM');
  throw error;
} finally {
  await sdk.destroy();
}

const pulsePackets = packets.filter(({ metrics }) => metrics?.cardio?.pulseRate?.length);
const breathingPackets = packets.filter(({ metrics }) => metrics?.breathing?.rate?.length);
const summary = {
  input: path.basename(videoPath), input_mode: 'custom_rgb_frames', sdk_version: SmartSpectraSDK.version,
  width, height, fps, frames_submitted: framesSubmitted, packet_count: packets.length,
  pulse_packet_count: pulsePackets.length, breathing_packet_count: breathingPackets.length,
  validation_event_count: validations.length, error_count: errors.length,
  pulse_rate_samples: pulsePackets.flatMap(({ metrics }) => metrics.cardio.pulseRate),
  breathing_rate_samples: breathingPackets.flatMap(({ metrics }) => metrics.breathing.rate),
};
fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(path.join(outputDir, 'metrics.json'), JSON.stringify({ summary, packets, validations, errors }, null, 2));
process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
if (framesSubmitted === 0 || pulsePackets.length === 0) process.exitCode = 1;
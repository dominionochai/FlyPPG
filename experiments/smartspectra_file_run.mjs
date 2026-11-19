import fs from 'node:fs';
import path from 'node:path';
import {
  SmartSpectraSDK,
  ProcessingStatus,
  breathingMetrics,
  cardioMetrics,
} from '@smartspectra/node-sdk';
import { decodeMetrics } from '@smartspectra/node-sdk/messages';

const apiKey = process.env.SMARTSPECTRA_API_KEY;
const videoPath = process.argv[2];
const outPath = process.argv[3] ?? 'smartspectra-results';

if (!apiKey) throw new Error('SMARTSPECTRA_API_KEY is not set');
if (!videoPath || !fs.existsSync(videoPath)) {
  throw new Error(`Video file not found: ${videoPath ?? '(no path supplied)'}`);
}

fs.mkdirSync(outPath, { recursive: true });
const packets = [];
const validations = [];
const errors = [];
let started = false;
let settled = false;
let timeout;

const sdk = new SmartSpectraSDK({
  apiKey,
  requestedMetrics: [...breathingMetrics, ...cardioMetrics],
});

const finish = () => {
  if (settled) return;
  settled = true;
  clearTimeout(timeout);
  fs.writeFileSync(
    path.join(outPath, 'metrics.json'),
    JSON.stringify({ video: path.basename(videoPath), packets, validations, errors }, null, 2),
  );
};

sdk.on('metrics', (buf, timestampUs) => {
  const decoded = decodeMetrics(buf);
  packets.push({ timestamp_us: timestampUs, metrics: decoded });
});
sdk.on('validationStatus', (code, timestampUs, hint) => {
  validations.push({ code, timestamp_us: timestampUs, hint });
});
sdk.on('error', (code, message, retryable) => {
  errors.push({ code, message, retryable });
  console.error(`SmartSpectra error ${code}: ${message} (retryable=${retryable})`);
});
const done = new Promise((resolve) => {
  sdk.on('processingStatus', (status) => {
    console.log('SmartSpectra processing status:', status);
    if (status === ProcessingStatus.kRunning) started = true;
    if (started && (status === ProcessingStatus.kIdle || status === ProcessingStatus.kError)) {
      finish();
      resolve();
    }
  });
});

timeout = setTimeout(async () => {
  errors.push({ code: 'TIMEOUT', message: 'SDK did not finish video playback within 180 seconds' });
  finish();
  try { await sdk.stop(); } catch {}
}, 180_000);

sdk.useFile(videoPath);
sdk.start();
await done;
await sdk.destroy();
finish();

const pulsePackets = packets.filter((p) => p.metrics?.cardio?.pulseRate?.length);
const breathingPackets = packets.filter((p) => p.metrics?.breathing?.rate?.length);
const summary = {
  video: path.basename(videoPath),
  packet_count: packets.length,
  pulse_packet_count: pulsePackets.length,
  breathing_packet_count: breathingPackets.length,
  validation_event_count: validations.length,
  error_count: errors.length,
  pulse_rate_samples: pulsePackets.flatMap((p) => p.metrics.cardio.pulseRate),
  breathing_rate_samples: breathingPackets.flatMap((p) => p.metrics.breathing.rate),
};
fs.writeFileSync(path.join(outPath, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));

if (!summary.pulse_packet_count) {
  console.error('No pulse-rate packets were produced.');
  process.exitCode = 1;
}

// SmartSpectra file-processing runner (documented Node.js SDK API only).
//   node smartspectra_file_run.mjs <video> [outDir] [--metrics=cardio|breathing+cardio] [--label=NAME] [--timestamps=FILE] [--log-level=debug|info|warning] [--interframe-delay-ms=N] [--max-duration-ms=N]
// Defaults are unchanged from the original runner: pulse-only request (cardioMetrics), useFile(video) with no playback options.
// Additions are diagnostic only: per-run counters, status-transition log, Long->Number JSON conversion, and a summary.json that is
// ALWAYS written (even on SDK error/timeout). Exit code is 1 whenever no pulse-rate packet was produced.
import fs from 'node:fs';
import path from 'node:path';
import {
  SmartSpectraSDK,
  ProcessingStatus,
  ValidationCode,
  SmartSpectraLogLevel,
  breathingMetrics,
  cardioMetrics,
} from '@smartspectra/node-sdk';
// `/messages` decodeMetrics always uses the bundled Metrics class (SmartSpectra's own headless-CI doc uses this import).
import { decodeMetrics } from '@smartspectra/node-sdk/messages';

const argv = process.argv.slice(2);
const flags = Object.fromEntries(argv.filter((a) => a.startsWith('--')).map((a) => { const [k, ...v] = a.slice(2).split('='); return [k, v.join('=') || 'true']; }));
const positional = argv.filter((a) => !a.startsWith('--'));
const videoPath = positional[0];
const outPath = positional[1] ?? 'smartspectra-results';
const label = flags.label ?? path.basename(videoPath ?? 'unknown');
const metricsMode = flags.metrics ?? 'cardio';
const logLevelName = flags['log-level'];
const apiKey = process.env.SMARTSPECTRA_API_KEY;

if (!apiKey) throw new Error('SMARTSPECTRA_API_KEY is not set');
if (!videoPath || !fs.existsSync(videoPath)) throw new Error(`Video file not found: ${videoPath ?? '(no path supplied)'}`);
if (!['cardio', 'breathing+cardio'].includes(metricsMode)) throw new Error(`--metrics must be cardio or breathing+cardio (got ${metricsMode})`);
if (logLevelName && !['debug', 'info', 'warning'].includes(logLevelName)) throw new Error('--log-level must be debug, info or warning');
fs.mkdirSync(outPath, { recursive: true });

const requestedMetrics = metricsMode === 'cardio' ? [...cardioMetrics] : [...breathingMetrics, ...cardioMetrics];
const codeName = Object.fromEntries(Object.entries(ValidationCode).map(([k, v]) => [v, k]));

// protobufjs/long.js Long values JSON-encode as {low, high, unsigned}; convert so downstream scoring sees plain numbers.
const longToNumber = (_k, v) => {
  if (v && typeof v === 'object' && !Array.isArray(v) && 'low' in v && 'high' in v && Object.keys(v).length <= 3) {
    const high = v.unsigned ? v.high >>> 0 : v.high;
    return high * 4294967296 + (v.low >>> 0);
  }
  return v;
};

const packets = [];
const validations = [];
const errors = [];
const statusLog = [];
const t0 = Date.now();
let started = false;
let settled = false;
let timeout;
let resolveDone;

const logLevel = { debug: SmartSpectraLogLevel.kDebug, info: SmartSpectraLogLevel.kInfo, warning: SmartSpectraLogLevel.kWarning }[logLevelName];
const sdk = new SmartSpectraSDK({ apiKey, requestedMetrics, ...(logLevel === undefined ? {} : { logLevel }) });

const summarize = (extra = {}) => {
  const pulsePackets = packets.filter((p) => p.metrics?.cardio?.pulseRate?.length);
  const breathingPackets = packets.filter((p) => p.metrics?.breathing?.rate?.length);
  const pulseSamples = pulsePackets.flatMap((p) => p.metrics.cardio.pulseRate);
  const byCode = validations.reduce((acc, v) => { const k = `${v.code} ${codeName[v.code] ?? '?'}: ${v.hint}`; acc[k] = (acc[k] ?? 0) + 1; return acc; }, {});
  return {
    label,
    video: path.basename(videoPath),
    requested_metrics: metricsMode,
    status: pulsePackets.length ? 'PULSE_PRODUCED' : 'INVALID_NO_OUTPUT',
    packet_count: packets.length,
    packets_with_cardio_object: packets.filter((p) => p.metrics?.cardio).length,
    pulse_packet_count: pulsePackets.length,
    pulse_samples_total: pulseSamples.length,
    pulse_samples_nonzero_confidence: pulseSamples.filter((s) => s.confidence > 0).length,
    breathing_packet_count: breathingPackets.length,
    validation_event_count: validations.length,
    validation_summary: Object.entries(byCode).map(([message, count]) => ({ message, count })).sort((a, b) => b.count - a.count),
    first_validation_timestamp_us: validations[0]?.timestamp_us ?? null,
    last_validation_timestamp_us: validations.at(-1)?.timestamp_us ?? null,
    error_count: errors.length,
    errors,
    status_transitions: statusLog,
    wall_seconds: (Date.now() - t0) / 1000,
    pulse_rate_samples: pulseSamples,
    breathing_rate_samples: breathingPackets.flatMap((p) => p.metrics.breathing.rate),
    ...extra,
  };
};

const finish = (extra = {}) => {
  if (settled) return;
  settled = true;
  clearTimeout(timeout);
  fs.writeFileSync(path.join(outPath, 'metrics.json'), JSON.stringify({ video: path.basename(videoPath), packets, validations, errors }, longToNumber, 2));
  fs.writeFileSync(path.join(outPath, 'summary.json'), JSON.stringify(summarize(extra), longToNumber, 2));
};

sdk.on('metrics', (buf, timestampUs) => {
  packets.push({ timestamp_us: timestampUs, metrics: decodeMetrics(buf) });
});
sdk.on('validationStatus', (code, timestampUs, hint) => {
  validations.push({ code, timestamp_us: timestampUs, hint });
});
sdk.on('error', (code, message, retryable) => {
  errors.push({ code, message, retryable });
  console.error(`SmartSpectra error ${code}: ${message} (retryable=${retryable})`);
});
const done = new Promise((resolve) => {
  resolveDone = resolve;
  sdk.on('processingStatus', (status) => {
    statusLog.push({ status, wall_ms: Date.now() - t0 });
    console.log('SmartSpectra processing status:', status);
    if (status === ProcessingStatus.kRunning) started = true;
    // kError is terminal even if the session never reached kRunning (Starting -> Error); kIdle is terminal only after a run.
    if (status === ProcessingStatus.kError || (started && status === ProcessingStatus.kIdle)) resolve();
  });
});

timeout = setTimeout(async () => {
  errors.push({ code: 'TIMEOUT', message: 'SDK did not finish video playback within 600 seconds' });
  try { await sdk.stopAsync(); } catch {}
  resolveDone();
}, 600_000);

try {
  // Documented: useFile(videoPath, { timestampsPath?, interframeDelayMs?, startOffsetMs?, maxDurationMs? }); all exposed here as opt-in flags.
  const fileOptions = {};
  if (flags.timestamps) fileOptions.timestampsPath = flags.timestamps;
  if (flags['interframe-delay-ms']) fileOptions.interframeDelayMs = Number(flags['interframe-delay-ms']);   // documented: throttle so file playback is not frame-dropped
  if (flags['max-duration-ms']) fileOptions.maxDurationMs = Number(flags['max-duration-ms']);               // documented: stop after this much content
  if (Object.keys(fileOptions).length) sdk.useFile(videoPath, fileOptions); else sdk.useFile(videoPath);
  sdk.start();
  await done;
  finish();                                   // write results BEFORE teardown so a stuck destroy() cannot lose them
  await Promise.race([sdk.destroy(), new Promise((r) => setTimeout(r, 20_000))]);
} catch (err) {
  errors.push({ code: 'RUNNER_EXCEPTION', message: String(err?.message ?? err) });
  finish();
}

const summary = JSON.parse(fs.readFileSync(path.join(outPath, 'summary.json'), 'utf8'));
console.log(JSON.stringify({ ...summary, pulse_rate_samples: `(${summary.pulse_rate_samples.length} samples)`, breathing_rate_samples: `(${summary.breathing_rate_samples.length} samples)` }, null, 2));
if (!summary.pulse_packet_count) {
  console.error('No pulse-rate packets were produced. Status: INVALID_NO_OUTPUT');
  process.exitCode = 1;
}
process.exit(process.exitCode ?? 0);   // native worker threads can keep the event loop alive after destroy()

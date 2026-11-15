# FlyPPG - final report (everything achievable without SmartSpectra / real data)

**Scope warning:** all numbers are on a synthetic face-video generator (global head translation, illumination drift/steps, noise, JPEG, low light) with a stand-in rPPG extractor (ROI green mean -> bandpass -> spectral peak). **Nothing here is a SmartSpectra or real-data result.** Success = window error <= 5 bpm; 6 clips/condition; readout tuned on separate seeds.

## Findings
1. **Illumination-step cancellation works and is nearly free.** A spatially uniform ON/OFF transient is read as an illumination step and cancelled: illumination MAE 1.69 -> 0.12 bpm, SNR 2.1 -> 9.9 dB, 0.03 ms/frame. Classical stabilisation + the same step fix ties it (the step fix is what matters there).
2. **Connectome-derived T4/T5 stabiliser (v2) matches classical phase-correlation stabilisation but does not beat it.** MAE (bpm): motion 0.13 vs 0.14; motion+illum 0.14 vs 0.14; compressed 0.13 vs 0.12; low-light 0.12 vs 0.14 (differences ~0.01-0.02 bpm, within noise at n=6). Both beat raw (0.24/0.83/0.30/0.38) and simple smoothing.
   - Key design change v1->v2: spatially pooled T4/T5 outputs (global + 4 quadrants + centre, x8 subtypes) -> 0.71 -> 0.94 cross-validated velocity correlation.
3. **Cost:** FlyPPG v2 10.5 ms/frame, ~1.1 GB peak (Python, 96x96) vs classical 1.4 ms/frame, ~0.13 GB. Step-only 0.03 ms/frame.
4. **Frame-quality flagging / efficiency: not demonstrated.** Fly motion speed predicts bad windows with AUC 0.75 vs classical speed 0.77 vs plain SNR 0.80 (48 windows - weak evidence). No compute saving can be claimed without SmartSpectra's real per-frame cost.
5. **Connectome link:** T4a-d/T5a-d input cell types, signs, synapse counts and hex offsets come from the flyvis fib25-fib19 connectome. Assumptions (not from the connectome): fast/slow time constants, subtractive-inhibition form, pixel pitch; velocity readout is fitted (linear, +-6 frame lookahead ~0.2 s). Pretrained flyvis weights were not used. Raw-weight direction selectivity index 0.70.

## Verdict against the brief's success criterion
"Measurably better than unmodified SmartSpectra AND conventional preprocessing": **met vs raw and simple conventional smoothing; NOT met vs a competent classical stabilisation baseline; unverified on SmartSpectra/real data.** Honest headline: bio-inspired illumination-step cancellation is the one clear, cheap win; the T4/T5 pathway is a working but non-superior motion sensor.

## Remaining (needs a human / hardware)
SmartSpectra integration; UBFC/PURE/MMPD runs; SmartSpectra per-frame cost for any efficiency claim; real FlyWire/hemibrain neurons for the 3D model.

## Artifacts
`flyppg.py` v0 harness; `fly_conn.py` connectome T4/T5 bank; `readout*.py` readout; `eval2.py`/`eval3.py` benchmarks; `analysis.py`; `demo_video.py`/`demo.mp4`; `build_brain.py`, `brain.html` (2D), `brain3d.html` (3D, real JRC2018F template mesh); `results_v1.json`, `results_v2.json`, `analysis.json`.

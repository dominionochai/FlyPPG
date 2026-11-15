# FlyPPG

Fruit-fly-visual-circuit-inspired preprocessing for camera-based rPPG (SmartSpectra).
**Status: v1 prototype on synthetic video with a proxy extractor (not SmartSpectra yet). Nothing here is a SmartSpectra result.**

## What the v1 benchmark shows (6 videos/condition, held-out seeds)
- ON/OFF step compensation (spatially uniform transient = illumination step): illumination MAE 1.69 -> 0.12 bpm, ~0.04 ms/frame. Classical stabilisation + the same step fix matches it.
- Connectome-derived T4/T5 motion stabiliser beats raw and simple smoothing under motion, but **classical phase-correlation stabilisation + step beats it in every condition** (e.g. motion+illum 0.14 vs 0.20 bpm) and is ~7x cheaper (13.8 vs ~2 ms/frame; ~1.1 GB peak vs ~0.13 GB).
- So the "beats conventional preprocessing" bar is NOT met against a strong conventional baseline. Real data + real SmartSpectra still required.

## Files
- `flyppg.py` v0 harness/preprocessors; `fly_conn.py` connectome-derived T4/T5 bank (weights from flyvis `fib25-fib19_v2.2.json`; fast/slow time constants, subtractive inhibition, pixel pitch are assumptions); `readout.py`/`eval2.py` readout + v1 benchmark; `build_brain.py` + `brain.html` live fly-brain visual driven by the real activations.
- `results_v1.json`, `flyppg_v1_results.png`, screenshots `brain_*.png`.
Caveats: readout fit on synthetic motion, uses +-8 frame lookahead (~0.27 s), flyvis pretrained weights not used.

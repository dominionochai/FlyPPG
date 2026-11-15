# FlyPPG - report v3 (after the three improvements)

**Scope:** synthetic face video + stand-in rPPG extractor (ROI green mean -> bandpass -> spectral peak). **Not a SmartSpectra or real-data result.** 6 clips/condition (18 windows/cell), readout trained on separate seeds. Success = window error <= 5 bpm.

## The three improvements and what they actually delivered
1. **Illumination hardening - delivered.** v2's step-cancellation FAILED on realistic lighting (auto-exposure steps that settle + mains flicker + drift: MAE 45.99 bpm, worse than raw 21.48). Replacing it with **surround (background-referenced) divisive normalisation** (lamina-style centre-surround gain control) gives **0.18 bpm** (SNR -3.5 -> 9.1 dB). Caveat: any baseline given the same surround normalisation gets the same 0.18 - the win is the normalisation, not fly-specific motion processing.
2. **Faster, local motion path - partly delivered.** Connectome T4/T5 bank rewritten as per-frame correlation kernels (numerically equivalent, corr ~1.0): **19.5 -> 1.1 ms/frame (18x), 1.1 GB -> 160 MB (7x)**, now cheaper than the classical dense-flow stabiliser (2.3 ms). A *local* (3x3-cell) velocity readout handles rotation/zoom (0.65 bpm vs v2 2.15, raw 1.38). It does **not** handle non-rigid motion: 6.24 bpm vs classical dense optical flow 0.28.
3. **Harder benchmark - delivered, and it was humbling.** Added rotation, zoom, non-rigid deformation, spatially varying pulse amplitude/phase, face-edge/ROI effects, AE + flicker, JPEG. It exposed that v2 only worked on easy translation: v2 breaks on AE/flicker (46 bpm) and rotation.

## MAE (bpm), lower is better
| condition | raw | classical dense-flow+surround | FlyPPG v3 | FlyPPG v2 |
|---|---|---|---|---|
| rigid motion | 7.83 | 0.38 | 0.65 | 1.92 |
| rotation+zoom | 1.38 | 0.25 | 0.65 | 2.15 |
| non-rigid | 15.30 | 0.28 | 6.24 | 13.14 |
| hard motion (all) | 15.83 | 12.83 | 13.36 | 15.58 |
| AE+flicker+drift | 21.48 | 0.18 | 0.18 | 45.99 |
| everything (+JPEG) | 18.10 | 16.87 | 11.44 | 16.52 |

## Honest verdict
- Strong, cheap, general result: **surround-normalised illumination** (fixes a failure mode that breaks raw video and my own earlier method).
- The connectome T4/T5 pathway is now fast and decent on rigid/rotation/zoom, but **classical dense optical flow is better on every motion condition except 'everything'** (where all methods fail: 11-17 bpm, ~40% success). It's best-of-bad there, not a win.
- **Unsolved by anyone, incl. FlyPPG:** combined hard motion (rotation + non-rigid + translation) - 13-16 bpm error. That is the real research problem, and the brief's "better than conventional" bar is **not met** against classical dense flow.
- Differences of ~0.1-0.3 bpm are within noise at this sample size.

## Still needs a human / hardware
SmartSpectra integration, UBFC/PURE/MMPD runs, SmartSpectra per-frame cost (needed for any efficiency claim), real FlyWire/hemibrain neurons for the 3D model.

## Files
`flyppg3.py` (generator, surround norm, fast fly bank, local readout, classical baselines, benchmark); `flyppg.py`, `fly_conn.py` (connectome T4/T5 weights from flyvis fib25-fib19), `eval*.py`/`results_v*.json` earlier versions, `build_brain.py` + `brain3d.html` (3D real-mesh visual).

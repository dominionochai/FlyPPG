# FlyPPG - report (READ FIRST: the lighting-fix plan did not survive testing)

**Scope:** synthetic video + proxy extractors. **Not a SmartSpectra or real-data result.**

## Update: failure tests and a stronger extractor (supersedes the v3 claims below)
Earlier "wins" were measured against a weak extractor (green-channel ROI mean). Adding the standard chrominance method **POS** (Wang et al. 2017) as the baseline changes the picture:

| lighting scenario (8 clips, no motion) | raw green | POS only | surround-norm (green) | surround-norm + POS |
|---|---|---|---|---|
| shared lighting (AE + flicker + drift) | 26.4 | 0.24 | 0.14 | 0.24 |
| face-only lighting (screen glow) | 26.4 | 0.24 | **26.5 (fails)** | 0.24 |
| partial coupling (50%) | 26.4 | 0.24 | **26.3 (fails)** | 0.24 |
| background-only distractor | 0.10 | 0.20 | **19.0 (fails)** | 0.20 |
| background clutter + shared lighting | 26.4 | 0.24 | **34.4 (worse than raw)** | 0.24 |
| coloured lighting (shared / face-only) | 26.1 | 0.17 | 26.1 / 26.1 | 0.25 / 0.19 |
| two independent lights (ambient + screen) | 27.9 | 0.16 | 24.2 | 0.13 |

(MAE bpm; success rate is 100% for every POS arm.)

**Findings**
1. **Background-referenced normalisation is fragile**: it breaks whenever face and background lighting are decoupled (screen glow, bg distractors, bg clutter).
2. **A standard chrominance extractor already solves every lighting scenario in this simulator.** Surround-norm adds at most ~0.1 bpm on top of POS and only in some cases. The simulator's lighting is mostly slow, so this may be too forgiving (no specular glare, shadows, spatial gradients) - not proof POS is robust in reality.
3. **With POS as the extractor, the motion benchmark shows no FlyPPG advantage.** Raw video already gets 0.33-0.45 bpm, 100% success, in all rigid/rotation/non-rigid/hard-motion/lighting conditions; every arm (classical stabilisation, FlyPPG v2/v3) ties within ~0.1 bpm. Only in "everything" (JPEG q30 + all disturbances) do methods differ, and all fail there (8-13 bpm, 55-67% success); **simple conventional smoothing is best (7.9 bpm), FlyPPG v3 worst (13.1)**.
4. Therefore: **"the contribution is the lighting fix" is NOT supported**, and no FlyPPG component beats a strong baseline in simulation. Standing result: against a green-mean extractor the fly-style pieces help; against POS they don't.

## What this means for the project
- The decisive test is raw SmartSpectra on real, disturbed video. If SmartSpectra already behaves like POS, there is little left for preprocessing to fix in lighting or mild motion.
- The only regime where preprocessing mattered at all is **heavy compression / noise / low light**. A biologically grounded, untested candidate: lamina-style **light-adaptive spatial pooling** (neural superposition: pool more photoreceptors when it is dim). Not built.
- Differences of ~0.1 bpm are within noise; 18 windows per cell.

## Update 2: light-adaptive pooling (lamina-style) - hypothesis not supported
Tested dim/noisy and JPEG-compressed clips (no motion, 6 clips each, POS and green extractors). Adaptive spatial+temporal pooling did **not** help: with POS on dim clips raw video was best (0.20 vs 0.34 bpm). Reason: the ROI mean already pools thousands of pixels, so extra pooling adds nothing against white noise. Added noise (up to sigma 0.04) barely hurt at all. The only large errors came from JPEG quantisation (q15 with POS: 22 bpm; q30 with green: 20 bpm), which behaves erratically across quality levels and extractors and is not fixed by any preprocessing tried. n=6, so treat as noise-level evidence. `lowlight_test.py`, `results_lowlight.json`.
Net: in simulation, nothing in FlyPPG beats a plain POS pipeline. Next informative step is real SmartSpectra output on real disturbed video.

---
(Earlier v3 report follows; read it with the caveat above.)


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

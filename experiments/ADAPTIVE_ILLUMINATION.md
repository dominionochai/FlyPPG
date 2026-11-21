# Adaptive spatial illumination experiment

**Status: implemented as an experimental preprocessing arm; not yet benchmarked for rPPG accuracy.**

This experiment adds two arms to `ss_preprocess.py`:

- `spatial`: bounded local illumination compensation.
- `combined_lighting`: existing gated background-referenced temporal correction plus spatial compensation.

## Method

At the 96x96 analysis resolution, the implementation estimates a slowly varying luminance field with a broad Gaussian filter. It compares each location with the median illumination in the central face-region proxy, computes an achromatic gain map, clips gain to 0.80–1.25, and smooths the map before applying it to the full-resolution RGB frame.

The correction is deliberately conservative. It does not use histogram equalisation or independent per-channel contrast stretching, both of which can distort the subtle colour variations used by rPPG. It can still amplify sensor noise in dark regions and is not expected to recover clipped highlights or detail absent from the source.

## Run

```bash
python ss_preprocess.py --in videos --out processed --arms control,surround,spatial,combined_lighting
```

The output videos must be evaluated with the same downstream extractor and settings. Compare at minimum:

1. Original video + POS.
2. Surround correction + POS.
3. Spatial correction + POS.
4. Combined correction + POS.

Use held-out clips and report BPM error, valid-output coverage, and performance on normally lit clips as well as uneven/dim/flickering conditions. Do not infer improvement from visual appearance alone.

## Claim boundary

The biological motivation is inspired by local gain control in fly visual processing. This implementation is a conventional Gaussian illumination-field estimator with bounded gain; it is not a validated reconstruction of lamina circuitry. No accuracy or robustness improvement is claimed until the benchmark is run.

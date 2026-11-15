# FlyPPG

Fruit-fly-visual-circuit-inspired preprocessing for camera-based rPPG (SmartSpectra).

**Status: v0 prototype, synthetic benchmark, proxy extractor (not SmartSpectra yet).**
No improvement is claimed beyond what `results.txt` shows.

- `flyppg.py` – synthetic data, preprocessors (raw / conventional / FlyPPG + ablations), proxy rPPG, evaluation
- `tune.py` – gate-scale sweep on separate tuning seeds
- `results.txt`, `results.json`, `flyppg_results.png` – v0 results

Run: `python flyppg.py 6`

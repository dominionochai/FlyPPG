# SmartSpectra head-to-head harness (run on YOUR machine)

Measures whether FlyPPG preprocessing changes SmartSpectra's pulse-rate accuracy on real video. SmartSpectra authenticates online and needs real faces, so this cannot run in the Claude sandbox.

## Pre-registered protocol (do not change after seeing results)
- **Arms** (all re-encoded identically with x264 crf 12, so compression is never a confound): `control` (re-encode only), `conventional` (blur + 5-frame mean), `classical` (Farneback dense-flow stabilisation), `surround` (gated background-referenced gain), `fly_stab` (connectome T4/T5 local stabiliser), `flyppg` (surround + fly_stab).
- **Primary metric:** paired MAE difference vs `control` per video, bootstrap 95% CI. An arm "wins" only if its whole CI is below 0.
- **No tuning on these videos.** `fly_stab`'s readout was trained on synthetic video only - treat it as experimental.
- **Include hard clips and matched clean clips**: head motion, lighting changes (auto-exposure, screen glow), low light, heavy compression. On clean footage expect ties.
- Need 30-60 s clips (rates start after ~12 s), MP4/H.264 preferred, plus a reference HR per clip.

## Steps (PowerShell)
```powershell
pip install numpy scipy opencv-python-headless imageio-ffmpeg
# 1. Build the SDK's minimal_example per cpp/README.md in https://github.com/Presage-Security/SmartSpectra
# 2. Put clips in .\videos and a ref.csv:  video,hr_bpm   (video = filename without extension)
python ss_preprocess.py --in videos --out processed
$env:SMARTSPECTRA_API_KEY = "<your key>"      # never committed; masked in logs
python ss_run.py --exe .\build\samples\minimal_example\minimal_example.exe --dir processed --out ss_out
python ss_score.py --out ss_out --ref ref.csv
Remove-Item Env:SMARTSPECTRA_API_KEY
```
Run `ss_preprocess.py` from the repo root (needs `connectome/` and `readout_W3.npy`). `fake_ss.py` is a stand-in used only to test the plumbing (`--exe python fake_ss.py`).

## Caveats
- `ss_score.py` parses the JSON that `minimal_example` prints ("Got edge metrics at ... microseconds: {...}") and looks for `pulse_rate` entries. The exact field names are unverified against the real SDK - if no readings are found, open one `ss_out/<arm>/<clip>.txt` and adjust `readings()`.
- Per-video estimate = median of the later half of SDK readings vs the clip's mean reference HR.
- Simulation so far (see REPORT.md): against a POS extractor FlyPPG shows no advantage. Do not expect a win on easy footage.

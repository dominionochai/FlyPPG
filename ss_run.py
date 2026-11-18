"""Run the SmartSpectra C++ minimal_example headlessly on every processed video. API key is read from env SMARTSPECTRA_API_KEY (never stored).
Usage: python ss_run.py --exe path\\to\\minimal_example --dir processed --out ss_out"""
import os, sys, argparse, subprocess
ap = argparse.ArgumentParser(); ap.add_argument('--exe', required=True, nargs='+'); ap.add_argument('--dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--timeout', type=int, default=600); a = ap.parse_args()
key = os.environ.get('SMARTSPECTRA_API_KEY'); assert key, 'set SMARTSPECTRA_API_KEY'
for arm in sorted(os.listdir(a.dir)):
    os.makedirs(os.path.join(a.out, arm), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(a.dir, arm))):
        out = os.path.join(a.out, arm, os.path.splitext(f)[0] + '.txt')
        if os.path.exists(out): continue
        r = subprocess.run(a.exe + [f'--api_key={key}', f'--input_video_path={os.path.join(a.dir, arm, f)}', '--requested_metrics=pulse_rate'], capture_output=True, text=True, timeout=a.timeout)
        open(out, 'w').write(r.stdout.replace(key, '***') + '\n--STDERR--\n' + r.stderr.replace(key, '***') + f'\n--EXIT {r.returncode}--\n'); print(arm, f, 'exit', r.returncode, flush=True)

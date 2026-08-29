import sys
from tippytop.agent.sandbox import run_solution

VALID_CSV = (
    "import argparse\n"
    "a = argparse.ArgumentParser()\n"
    "a.add_argument('--data_dir'); a.add_argument('--split'); a.add_argument('--out')\n"
    "args = a.parse_args()\n"
    "open(args.out, 'w').write('row_id,user_id,video_id,score\\n0,0,0,1.0\\n')\n"
)
CRASH = "raise RuntimeError('boom')\n"
HANG = "import time\ntime.sleep(30)\n"


def test_success_writes_output(tmp_path):
    res = run_solution(VALID_CSV, iter_dir=tmp_path / "it", data_dir=tmp_path,
                       split="valid", timeout_s=30, python_exe=sys.executable)
    assert res.ok and res.out_path is not None and res.out_path.exists()
    assert res.returncode == 0 and not res.timed_out


def test_crash_captures_traceback(tmp_path):
    res = run_solution(CRASH, iter_dir=tmp_path / "it", data_dir=tmp_path,
                       split="valid", timeout_s=30, python_exe=sys.executable)
    assert not res.ok and res.out_path is None
    assert "RuntimeError" in res.stderr and res.returncode != 0


def test_timeout_kills_without_hanging(tmp_path):
    res = run_solution(HANG, iter_dir=tmp_path / "it", data_dir=tmp_path,
                       split="valid", timeout_s=2, python_exe=sys.executable)
    assert res.timed_out and not res.ok
    assert res.wall_s < 20      # killed promptly, did not run the full 30s sleep

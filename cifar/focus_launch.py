import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cifar.variants import focus_rows, focus_sequences


def main():
    index = int(sys.argv[1])
    seqs = focus_sequences()
    if index < 0 or index >= len(seqs):
        raise SystemExit(f"index {index} outside 0..{len(seqs) - 1}")
    rows = focus_rows()
    for j in seqs[index]:
        spec = rows[j]
        cmd = [
            sys.executable,
            "-u",
            str(_ROOT / "cifar" / "train.py"),
            "--group",
            "focus",
            "--index",
            str(j),
            "--epochs",
            str(spec["epochs"]),
            "--patience",
            "10",
            "--val-every",
            "5",
            "--batch-size",
            "64",
            "--lr",
            "2e-5",
            "--lbd",
            "100",
            "--seed",
            "0",
            "--out-root",
            str(_ROOT / "experiments" / "cifar" / "focus"),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=_ROOT)


if __name__ == "__main__":
    main()

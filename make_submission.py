"""Regenerate the submission CSV from the data: one command, same file every time.

    python make_submission.py              # writes Octo.csv in the repo root
    python make_submission.py some/path.csv

Method: src/algo/segment_control (see its README and docs/Octo.md).
The first run builds the ball model and the feature caches under .local/cache (~10-20 min),
later runs reuse them (~3 min).
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "algo", "segment_control"))
import segment_control  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "Octo.csv")
WORKERS = 4


def main(path=DEFAULT_OUT):
    segment_control.run(dict(segment_control.DEFAULTS), WORKERS, "Octo")
    shutil.copyfile(os.path.join(segment_control.RESULTS_DIR, "Octo.csv"), path)
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main(*sys.argv[1:2])

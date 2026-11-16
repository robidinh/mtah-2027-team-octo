"""DFL / IDSSE open data: 7 Bundesliga matches with per-frame ball status and possession.

Another set of OTHER matches for the test bench (RULES.md section 2). Unlike the Metrica sample
games, the answers here come straight from the tracking, the way the challenge's ground truth is
made: every ball frame carries BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away).
No event-to-label conversion needed.

    python src/common/dfl.py               # parse all downloaded matches into data/cache/ and summarise

Data (CC-BY 4.0, credit the DFL and cite Bassek et al. 2025, doi.org/10.1038/s41597-025-04505-y),
mirrored at huggingface.co/datasets/pysport/idsse-data. Put the XML files in data/external/idsse/.

Parsed once per match (~1 min each) and cached as npz next to our own cache.
"""

import glob
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import load  # noqa: E402

# Params
DFL_DIR = os.path.join(ROOT, "data", "external", "idsse")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
HALF_GAP = 250              # ungraded frames inserted between the halves so nothing spans them
FPS = 25

FRAMESET = re.compile(r'<FrameSet [^>]*GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"')
BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def match_ids():
    paths = glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(re.search(r"DFL-MAT-(\w+)\.xml", p).group(1) for p in paths)


def file_for(kind, match_id):
    return glob.glob(os.path.join(DFL_DIR, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match_id)))[0]


def read_meta(match_id):
    """(home team id, pitch (length, width))."""
    with open(file_for("02_01_matchinformation", match_id), encoding="utf-8") as handle:
        text = handle.read()
    home = re.search(r'<Team TeamId="([^"]+)"[^>]*Role="home"', text).group(1)
    length = float(re.search(r'PitchX="([\d.]+)"', text).group(1))
    width = float(re.search(r'PitchY="([\d.]+)"', text).group(1))
    return home, (length, width)


def parse(match_id):
    """Raw per-object tracks: {(half, team, person): (frames, x, y[, possession, status])}."""
    tracks = {}
    key, rows = None, []

    def flush():
        if key is not None and rows:
            tracks[key] = np.array(rows, dtype=np.float64)

    with open(file_for("04_03_positions_raw_observed", match_id), encoding="utf-8") as handle:
        for line in handle:
            if "<Frame " in line:
                n, x, y = FRAME.search(line).groups()
                row = [int(n), float(x), float(y)]
                if key[1] == "BALL":
                    flags = BALL_FLAGS.search(line)
                    row += [int(flags.group(1)), int(flags.group(2))] if flags else [0, 0]
                rows.append(row)
            elif "<FrameSet " in line:
                flush()
                key, rows = FRAMESET.search(line).groups(), []
    flush()
    return tracks


def build(match_id):
    """The same dict load.load_tracking() returns, plus 'gt' (H / A / D / X per frame)."""
    home, pitch = read_meta(match_id)
    tracks = parse(match_id)
    halves = ("firstHalf", "secondHalf")

    # Frame index: 1st half, a short ungraded gap, 2nd half.
    first, offset = {}, {}
    position = 0
    for half in halves:
        frames = np.concatenate([t[:, 0] for (h, _, _), t in tracks.items() if h == half])
        first[half] = int(frames.min())
        offset[half] = position
        position += int(frames.max()) - first[half] + 1 + HALF_GAP
    n = position

    people = sorted({(team, person) for (_, team, person) in tracks if team not in ("BALL", "referee")})
    slot = {p: k for k, p in enumerate(people)}
    player_xy = np.full((n, len(people), 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    gt = np.full(n, "X", dtype="<U1")
    half_of = np.zeros(n, dtype=np.int8)
    centre = np.array(pitch) / 2.0          # DFL coordinates are centred on the kickoff spot

    for (half, team, person), t in tracks.items():
        if team == "referee":
            continue
        index = t[:, 0].astype(np.int64) - first[half] + offset[half]
        xy = t[:, 1:3] + centre
        if team == "BALL":
            ball_xy[index] = xy
            possession, status = t[:, 3], t[:, 4]
            gt[index] = np.where(status == 0, "D", np.where(possession == 1, "H", "A"))
            half_of[index] = halves.index(half) + 1
        else:
            player_xy[index, slot[(team, person)]] = xy

    team = np.array([load.HOME if t == home else load.AWAY for t, _ in people], dtype=np.int8)
    return {
        "player_xy": player_xy,
        "ball_xy": ball_xy,
        "team": team,
        "goalkeeper": np.zeros(len(people), dtype=bool),
        "half": half_of,
        "graded": half_of != 0,
        "pitch": pitch,
        "kickoff_2nd": None,
        "gt": gt,
    }


def load_match(match_id):
    """Cached build(); speeds are computed from positions, like for the Metrica sample games."""
    import bench  # here to avoid a circular import: bench imports this module

    path = os.path.join(CACHE_DIR, "idsse_{}.npz".format(match_id))
    if os.path.isfile(path):
        with np.load(path) as cached:
            data = {k: cached[k] for k in cached.files}
    else:
        started = time.time()
        data = build(match_id)
        data["player_speed"] = bench.player_speed(data["player_xy"]).astype(np.float32)
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        np.savez(path, **{k: v for k, v in data.items() if k not in ("pitch", "kickoff_2nd")})
        print("  parsed {} in {:.0f} s".format(match_id, time.time() - started))
    data["pitch"] = tuple(read_meta(match_id)[1])
    data["kickoff_2nd"] = None
    return data


if __name__ == "__main__":
    for match_id in match_ids():
        data = load_match(match_id)
        gt, graded = data["gt"], data["graded"]
        seen = ~np.isnan(data["ball_xy"][:, 0])
        visible = (~np.isnan(data["player_xy"][:, :, 0])).sum(axis=1)
        print("{}: {:.0f} min, D {:.1f}%, H {:.1f}%, ball seen live {:.0f}% / dead {:.0f}%, "
              "players visible {:.1f}".format(
                  match_id, graded.sum() / FPS / 60.0, 100.0 * (gt[graded] == "D").mean(),
                  100.0 * (gt[graded] == "H").mean(), 100.0 * seen[graded & (gt != "D")].mean(),
                  100.0 * seen[gt == "D"].mean(), visible[graded].mean()))

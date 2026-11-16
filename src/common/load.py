"""Load the tactical tracking and Smart Tagging into per-frame numpy arrays.

Everything is indexed by VIDEO frame (0 .. 199318), the same index as a submission row, so no
other module has to think about the ATD's frame_id = video_frame + 1.

    python src/common/load.py              # build the cache (once, ~1 min) and print a summary
    python src/common/load.py --rebuild    # force a re-read of the raw file

The raw ATD is parsed once and cached in data/cache/ (gitignored with the rest of data/). Later
loads take a couple of seconds. The cache is rebuilt automatically when the raw file changes.

What load_tracking() returns, all in METRES (x along the pitch 0..105.3, y across it 0..68):

    player_xy     (n, 46, 2) float32   NaN where the slot is not tracked on that frame
    player_speed  (n, 46)    float32   m/s, as given in the file
    ball_xy       (n, 2)     float32   NaN where the ball is not tracked
    team          (46,)      int8      0 = home (Barcelona), 1 = away (Athletic)
    goalkeeper    (46,)      bool      slots the metadata marks as goalkeepers
    half          (n,)       int8      1, 2, or 0 outside the graded halves
    graded        (n,)       bool      half != 0
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
# Bump when the cached arrays change meaning, so an old cache is never silently reused.
CACHE_VERSION = 1
N_SLOTS = 46
VALUES_PER_LINE = 1 + (N_SLOTS + 1) * 3  # frame_id, 46 players x (x, y, speed), ball (x, y, speed)
CHUNK_LINES = 20000  # parse in chunks to keep peak memory low

HOME, AWAY = 0, 1
# Smart Tagging spells Athletic "Atletic Club"; N/A means no team.
TAG_TEAMS = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


# -- tracking ----------------------------------------------------------------------------------

def read_slots(meta_path):
    """(team, goalkeeper) arrays over the 46 slots, from the ATD metadata XML.

    The first team in the file is home, the FIFA/EPTS convention (Barcelona here).
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    team, goalkeeper = [], []
    for player in root.iter("Player"):
        team.append(HOME if player.get("teamId") == team_ids[0] else AWAY)
        params = {p.findtext("Name"): p.findtext("Value")
                  for p in player.iter("ProviderParameter")}
        goalkeeper.append(params.get("position_type") == "Goalkeeper")
    if len(team) != N_SLOTS:
        raise ValueError("{}: expected {} slots, found {}".format(meta_path, N_SLOTS, len(team)))
    return np.array(team, dtype=np.int8), np.array(goalkeeper, dtype=bool)


def parse_raw(raw_path, n_frames):
    """(n_frames, VALUES_PER_LINE) float32 table straight from the raw ATD text.

    Every line is "frame_id:x,y,s;x,y,s;...:x,y,s", so turning ':' and ';' into ',' leaves a flat
    list of numbers that numpy parses in C. Rows are placed by frame_id, so the check below that
    every frame appears exactly once is what guarantees the video-frame indexing.
    """
    table = np.full((n_frames, VALUES_PER_LINE), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)

    def flush(lines):
        text = ",".join(lines).replace(":", ",").replace(";", ",")
        values = np.fromstring(text, sep=",", dtype=np.float64)
        if values.size != len(lines) * VALUES_PER_LINE:
            raise ValueError("{}: a line does not have {} values".format(
                raw_path, VALUES_PER_LINE))
        block = values.reshape(len(lines), VALUES_PER_LINE)
        video_frames = block[:, 0].astype(np.int64) - 1  # frame_id counts from 1
        if video_frames.min() < 0 or video_frames.max() >= n_frames:
            raise ValueError("{}: frame_id outside 1..{}".format(raw_path, n_frames))
        if seen[video_frames].any():
            raise ValueError("{}: duplicate frame_id".format(raw_path))
        seen[video_frames] = True
        table[video_frames] = block

    lines = []
    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) == CHUNK_LINES:
                flush(lines)
                lines = []
    if lines:
        flush(lines)

    if not seen.all():
        raise ValueError("{}: {} video frames have no record".format(
            raw_path, int((~seen).sum())))
    return table


def frame_halves(n_frames, game=GAME):
    """(n,) int8: 1 or 2 inside a graded half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def build_tracking(game=GAME):
    """Parse the raw files into the arrays described in the module docstring."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    table = parse_raw(raw_path, n_frames)
    groups = table[:, 1:].reshape(n_frames, N_SLOTS + 1, 3)  # 46 players + ball, (x, y, speed)
    scale = np.array([length, width], dtype=np.float32)

    team, goalkeeper = read_slots(meta_path)
    half = frame_halves(n_frames, game)
    return {
        "player_xy": groups[:, :N_SLOTS, :2] * scale,
        "player_speed": np.ascontiguousarray(groups[:, :N_SLOTS, 2]),
        "ball_xy": groups[:, N_SLOTS, :2] * scale,
        "team": team,
        "goalkeeper": goalkeeper,
        "half": half,
        "graded": half != 0,
    }


def cache_path(game=GAME):
    return os.path.join(CACHE_DIR, "{}_tactical_tracking.npz".format(game))


def source_stamp(game=GAME):
    """Size and modification time of the raw files, stored in the cache to detect changes."""
    stamp = [CACHE_VERSION]
    for path in require(["atd_meta", "atd_raw"], game):
        info = os.stat(path)
        stamp += [info.st_size, int(info.st_mtime)]
    return np.array(stamp, dtype=np.int64)


def load_tracking(game=GAME, rebuild=False):
    """The tracking arrays, from the cache when it is current, otherwise parsed and cached."""
    path = cache_path(game)
    stamp = source_stamp(game)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as cached:
            if np.array_equal(cached["stamp"], stamp):
                data = {key: cached[key] for key in cached.files if key != "stamp"}
                return with_game_info(data, game)

    print("parsing tracking (once, ~1 min) ...")
    started = time.time()
    data = build_tracking(game)
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    np.savez(path, stamp=stamp, **data)
    print("  done in {:.0f} s, cached to {}".format(
        time.time() - started, os.path.relpath(path, ROOT)))
    return with_game_info(data, game)


def with_game_info(data, game=GAME):
    """Add the per-game constants the detectors need (kept out of the cache on purpose).

    pitch        (length, width) in metres
    kickoff_2nd  video frame of the 2nd-half kickoff when grading starts before it, else None
    """
    data["pitch"] = tuple(GAMES[game]["pitch"])
    kickoff = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]
    second = np.flatnonzero(data["half"] == 2)
    data["kickoff_2nd"] = kickoff if len(second) and second[0] < kickoff else None
    return data


# -- smart tagging -----------------------------------------------------------------------------

def load_tags(game=GAME):
    """Smart Tagging as a DataFrame, one row per instance, frames on the video timeline.

    Columns: code, start_frame, end_frame (exclusive), team ('H', 'A' or None), type, half.
    team is mapped from the club name; N/A and missing both become None.
    """
    rows = []
    for event in load_smart_tagging.load(game):
        tags = event["tags"]
        rows.append({
            "code": (event["code"] or "").strip().upper(),
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "team": TAG_TEAMS.get((tags.get("Team") or "").strip().lower()),
            "type": tags.get("Type"),
            "half": tags.get("Half"),
        })
    return pd.DataFrame(rows).sort_values("start_frame", kind="stable").reset_index(drop=True)


# -- summary -----------------------------------------------------------------------------------

def summary(data, tags):
    graded = data["graded"]
    ball_seen = ~np.isnan(data["ball_xy"][:, 0])
    tracked = (~np.isnan(data["player_xy"][:, :, 0])).sum(axis=1)
    print("frames            {}  ({} graded)".format(len(graded), int(graded.sum())))
    print("slots             {} home, {} away, goalkeepers {}".format(
        int((data["team"] == HOME).sum()), int((data["team"] == AWAY).sum()),
        (np.flatnonzero(data["goalkeeper"]) + 1).tolist()))
    print("ball seen         {:.1f}% of all frames, {:.1f}% of graded".format(
        100.0 * ball_seen.mean(), 100.0 * ball_seen[graded].mean()))
    print("players tracked   mean {:.1f} per graded frame".format(tracked[graded].mean()))
    print("tags              {} instances, team H {} / A {} / none {}".format(
        len(tags), int((tags["team"] == "H").sum()), int((tags["team"] == "A").sum()),
        int(tags["team"].isna().sum())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache and re-parse")
    args = parser.parse_args()
    summary(load_tracking(rebuild=args.rebuild), load_tags())

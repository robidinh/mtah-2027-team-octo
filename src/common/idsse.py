import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import ROOT, CACHE_DIR, BALL, HOME, AWAY, GAME, GAMES  # noqa: E402

DFL_DIR = os.path.join(ROOT, "data", "external", "idsse")
FRAME_SET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)">(.*?)</FrameSet>',
                       re.S)
PLAYER_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
BALL_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"[^>]*? BallPossession="(\d)" BallStatus="(\d)"')


def match_ids():
    return sorted(re.search(r"(DFL-MAT-\w+)\.xml$", path).group(1)
                  for path in glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*.xml")))


def load_match(match_id):
    # One IDSSE (DFL, Bundesliga 2022/23) match in the same layout as tracking.load_tracking.
    # - tracking[frame, slot, (x, y, speed)], slots 0..45 players padded with NaN, slot 46 (BALL) the ball
    # - x, y rescaled from the DFL pitch (centre origin) to the competition pitch GAMES[GAME]["pitch"],
    #   so the algorithms' out-of-field test applies unchanged; speed is NaN
    # - row index is DFL frame N minus the first N of the match, the halftime gap stays as NaN rows
    # - team[slot] is HOME or AWAY, from the match information home team id; referees are left out
    # - gt[frame]: provider labels, H / A when BallStatus is 1 and BallPossession is 1 / 2, D when BallStatus
    #   is 0, X where the ball has no frame
    # - halves: [(half, first_frame, last_frame)] in row indices
    # parsed arrays are cached as .npz under .local/cache
    cache_path = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match_id))
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["tracking"], cached["team"], cached["gt"], [tuple(half) for half in cached["halves"]]

    info_path = glob.glob(os.path.join(DFL_DIR, "DFL_02_01_matchinformation_*_{}.xml".format(match_id)))[0]
    with open(info_path, encoding="utf-8") as handle:
        home_id = re.search(r'HomeTeamId="([^"]*)"', handle.read()).group(1)
    positions_path = glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*_{}.xml".format(match_id)))[0]
    with open(positions_path, encoding="utf-8") as handle:
        text = handle.read()
    pitch = re.search(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"', text)
    pitch_x, pitch_y = float(pitch.group(1)), float(pitch.group(2))

    # person frame sets are split per half, a person keeps one slot over both halves
    sets = FRAME_SET.findall(text)
    del text

    first_n, last_n = None, None
    ball_parts = []
    for section, team_id, person, body in sets:
        if team_id == "BALL":
            values = np.array(BALL_FRAME.findall(body), dtype=np.float64)
            ball_parts.append((section, values))
            first_n = values[0, 0] if first_n is None else min(first_n, values[0, 0])
            last_n = values[-1, 0] if last_n is None else max(last_n, values[-1, 0])
    n_frames = int(last_n - first_n) + 1
    length, width = GAMES[GAME]["pitch"]
    tracking = np.full((n_frames, BALL + 1, 3), np.nan, dtype=np.float32)
    gt = np.full(n_frames, "X", dtype="<U1")
    halves = []
    for half, (section, values) in enumerate(sorted(ball_parts, key=lambda part: part[1][0, 0]), start=1):
        rows = (values[:, 0] - first_n).astype(np.int64)
        tracking[rows, BALL, 0] = (values[:, 1] + pitch_x / 2) * length / pitch_x
        tracking[rows, BALL, 1] = (values[:, 2] + pitch_y / 2) * width / pitch_y
        gt[rows] = np.where(values[:, 4] == 0, "D", np.where(values[:, 3] == 1, "H", "A"))
        halves.append((half, int(rows[0]), int(rows[-1])))

    slots = {}
    team = np.zeros(BALL, dtype=np.int8)
    for section, team_id, person, body in sets:
        if team_id in ("BALL", "referee"):
            continue
        if person not in slots:
            if len(slots) == BALL:
                raise ValueError("{}: more than {} players".format(match_id, BALL))
            slots[person] = len(slots)
            team[slots[person]] = HOME if team_id == home_id else AWAY
        values = np.array(PLAYER_FRAME.findall(body), dtype=np.float64)
        rows = (values[:, 0] - first_n).astype(np.int64)
        keep = (rows >= 0) & (rows < n_frames)
        tracking[rows[keep], slots[person], 0] = (values[keep, 1] + pitch_x / 2) * length / pitch_x
        tracking[rows[keep], slots[person], 1] = (values[keep, 2] + pitch_y / 2) * width / pitch_y
    # unused padding slots are AWAY with no positions, they never count as nearest
    team[len(slots):] = AWAY

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, tracking=tracking, team=team, gt=gt, halves=np.array(halves))
    return tracking, team, gt, halves


if __name__ == '__main__':
    for match_id in sys.argv[1:] or match_ids():
        tracking, team, gt, halves = load_match(match_id)
        labels, counts = np.unique(gt, return_counts=True)
        print(match_id, tracking.shape, "players", int(np.sum(np.any(~np.isnan(tracking[:, :BALL, 0]), axis=0))),
              "halves", halves, dict(zip(labels.tolist(), counts.tolist())))

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, ".local", "cache")
OUT_DIR = os.path.join(ROOT, "output")
BALL = 46
HOME, AWAY = 0, 1


def load_tracking(source="tactical", game=GAME):
    # tracking[video_frame, slot, (x, y, speed)]
    # - slots 0..45 are the 46 player track slots in metadata order, slot 46 (BALL) is the ball
    # - x, y in metres from the pitch corner, speed in m/s, NaN where not tracked
    # - row index is the video frame (ATD frame_id - 1)
    # team[slot] is HOME or AWAY for the 46 player slots
    # parsed array is cached as .npy under .local/cache, delete it to re-parse
    paths = asset_paths(game=game, source=source)
    root = ET.parse(paths["atd_meta"]).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    team = np.array([HOME if player.get("teamId") == team_ids[0] else AWAY
                     for player in root.iter("Player")], dtype=np.int8)

    n_frames = source_frames(source, game)
    n_slots = len(team) + 1
    cache_path = os.path.join(CACHE_DIR, "{}_{}_atd.npy".format(game, source))
    if os.path.exists(cache_path):
        tracking = np.load(cache_path)
    else:
        with open(paths["atd_raw"]) as handle:
            text = handle.read()
        # every line is "frame_id:x,y,s;...;x,y,s:x,y,s", so all separators become commas
        flat = np.fromstring(text.replace(":", ",").replace(";", ",").replace("\n", ","),
                             dtype=np.float64, sep=",")
        records = flat.reshape(-1, 1 + n_slots * 3)
        if records.shape[0] != n_frames:
            raise ValueError("{} records, expected {}".format(records.shape[0], n_frames))
        tracking = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
        tracking[records[:, 0].astype(np.int64) - 1] = records[:, 1:].reshape(-1, n_slots, 3)
        length, width = GAMES[game]["pitch"]
        tracking[:, :, 0] *= length
        tracking[:, :, 1] *= width
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_path, tracking)
    return tracking, team

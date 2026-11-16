"""Test bench: score the pipeline on OTHER matches, the Metrica open sample games.

The only legal way to measure the method on full matches (RULES.md section 2: data about other
games is allowed). The sample games come with human event data; the possession labels below are
derived from those events by code, never by hand.

    git clone https://github.com/metrica-sports/sample-data data/external/metrica-sample
    (DFL matches: see src/common/dfl.py)
    python src/common/bench.py

Possession labels from events, following the challenge's definition:

  * dead ball: from the last ball event before a set piece until the set piece itself (the ball
    went out, a foul, a goal ... and play restarts with the set piece);
  * a team gains possession with an action that shows control -- a pass, a shot, a set piece, a
    recovery or a carry (game 3 only). A recovery immediately followed by a ball lost on the
    same frame is a single touch, not control, and does not count;
  * between those actions the ball stays with the last team (loose balls, passes in flight);
  * a half is graded from its kickoff to the end of its last event.

Differences to BAR-ATH to keep in mind when reading the scores: the sample tracking has stable
player identities and all 22 players on every frame, and no speed column (computed here).
Game 3 is in the FIFA EPTS format and is read with kloppy; it takes ~40 s, games 1-2 a few.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "algo", "src_combine"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import load  # noqa: E402
import src_combine as combine  # noqa: E402
import baseline  # noqa: E402
import possession_metrics as pm  # noqa: E402
import dfl  # noqa: E402

# Params
SAMPLE_DIR = os.path.join(ROOT, "data", "external", "metrica-sample", "data")
GAMES = (1, 2, 3)
EPTS_GAMES = (3,)           # FIFA EPTS + JSON format, the others are CSV
PITCH = (105.0, 68.0)       # metres, from the sample data README
FPS = 25
SPEED_SMOOTH = 5            # frames: moving average on player speed from positions
CONTROL_TYPES = ("PASS", "SHOT", "SET PIECE", "RECOVERY", "CARRY")  # CARRY: game 3 only
BASELINE_WINDOW = 50        # the official floor, baseline_smooth50

TEAM_CODE = {"Home": load.HOME, "Away": load.AWAY}


def game_path(game, kind):
    return os.path.join(SAMPLE_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_{1}".format(game, kind))


def read_team(game, side):
    """(positions (n, players, 2) in metres, ball (n, 2) in metres) from one CSV team file."""
    table = pd.read_csv(game_path(game, "RawTrackingData_{}_Team.csv".format(side)), skiprows=2)
    xy = table.to_numpy(dtype=np.float64)[:, 3:].reshape(len(table), -1, 2) * np.array(PITCH)
    return xy[:, :-1], xy[:, -1]


def read_csv_game(game):
    """Games 1 and 2: Metrica CSV. (players, ball, team per slot, events)."""
    home, ball = read_team(game, "Home")
    away, _ = read_team(game, "Away")
    team = [load.HOME] * home.shape[1] + [load.AWAY] * away.shape[1]
    events = pd.read_csv(game_path(game, "RawEventsData.csv"))
    events = pd.DataFrame({
        "Team": events["Team"], "Type": events["Type"], "Period": events["Period"],
        "start_frame": events["Start Frame"],
        "end_frame": np.where(events["End Frame"] > 0, events["End Frame"], events["Start Frame"]),
    })
    return np.concatenate([home, away], axis=1), ball, team, events


def read_epts_game(game):
    """Game 3: FIFA EPTS tracking + JSON events, read with kloppy. Same output as read_csv_game."""
    from kloppy import metrica  # only needed here

    dataset = metrica.load_tracking_epts(
        meta_data=game_path(game, "metadata.xml"), raw_data=game_path(game, "tracking.txt"),
        coordinates="metrica")
    home_id = next(t.team_id for t in dataset.metadata.teams if str(t.ground) == "home")
    players = [p for t in dataset.metadata.teams for p in t.players]
    team = [load.HOME if p.team.team_id == home_id else load.AWAY for p in players]

    frame = dataset.to_df(engine="pandas")
    index = frame["frame_id"].to_numpy() - 1
    n = index.max() + 1
    xy = np.full((n, len(players), 2), np.nan)
    for k, player in enumerate(players):
        for axis, name in enumerate("xy"):
            column = "{}_{}".format(player.player_id, name)
            if column in frame:
                xy[index, k, axis] = frame[column].to_numpy(dtype=np.float64)
    ball = np.full((n, 2), np.nan)
    ball[index] = frame[["ball_x", "ball_y"]].to_numpy(dtype=np.float64)

    with open(game_path(game, "events.json")) as handle:
        raw = json.load(handle)["data"]
    events = pd.DataFrame({
        "Team": ["Home" if e["team"]["id"] == home_id else "Away" for e in raw],
        "Type": [e["type"]["name"] for e in raw],
        "Period": [e["period"] for e in raw],
        "start_frame": [e["start"]["frame"] for e in raw],
        "end_frame": [e["end"]["frame"] or e["start"]["frame"] for e in raw],
    })
    scale = np.array(PITCH)
    return xy * scale, ball * scale, team, events


def player_speed(xy):
    """m/s per player per frame from consecutive positions, lightly smoothed."""
    speed = np.full(xy.shape[:2], np.nan)
    speed[1:] = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1)) * FPS
    kernel = np.ones(SPEED_SMOOTH) / SPEED_SMOOTH
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filled = np.where(np.isnan(speed), 0.0, speed)
        counts = np.apply_along_axis(np.convolve, 0, ~np.isnan(speed), kernel, "same")
        sums = np.apply_along_axis(np.convolve, 0, filled, kernel, "same")
        return np.where(np.isnan(speed), np.nan, sums / counts)


def load_game(game):
    """The same dict load.load_tracking() returns, plus the events."""
    reader = read_epts_game if game in EPTS_GAMES else read_csv_game
    players, ball, team, events = reader(game)
    players = players.astype(np.float32)
    # The files count frames from 1, our arrays from 0.
    events["start"] = events.pop("start_frame") - 1
    events["end"] = events.pop("end_frame") - 1

    half = np.zeros(len(ball), dtype=np.int8)
    for number in (1, 2):
        in_half = events[events["Period"] == number]
        half[in_half["start"].min():in_half["end"].max() + 1] = number

    team = np.array(team, dtype=np.int8)
    data = {
        "player_xy": players,
        "player_speed": player_speed(players).astype(np.float32),
        "ball_xy": ball.astype(np.float32),
        "team": team,
        "goalkeeper": np.zeros(len(team), dtype=bool),
        "half": half,
        "graded": half != 0,
        "pitch": PITCH,
        "kickoff_2nd": None,
    }
    return data, events


def event_labels(events, half):
    """'H' / 'A' / 'D' / 'X' per frame from the events (see module docstring)."""
    n = len(half)
    owner = np.full(n, -1, dtype=np.int8)
    events = events[events["Type"] != "CARD"].sort_values(["start", "end"], kind="stable")

    # Control actions, minus single touches (recovery + ball lost on the same frame).
    lost = set(zip(events.loc[events["Type"] == "BALL LOST", "Team"],
                   events.loc[events["Type"] == "BALL LOST", "start"]))
    control = events[events["Type"].isin(CONTROL_TYPES)]
    control = control[~((control["Type"] == "RECOVERY")
                        & np.array([(t, s) in lost
                                    for t, s in zip(control["Team"], control["start"])]))]
    for start, team_name in zip(control["start"], control["Team"]):
        owner[start:] = TEAM_CODE[team_name]

    labels = np.where(owner == load.HOME, "H", np.where(owner == load.AWAY, "A", "X"))

    # Dead: from the last ball event before each set piece until the set piece.
    starts = events["start"].to_numpy()
    ends = events["end"].to_numpy()
    for start in events.loc[events["Type"] == "SET PIECE", "start"]:
        before = ends[starts < start]
        if len(before):
            dead_from = min(before.max(), start)
            labels[dead_from:start] = "D"

    labels[half == 0] = "X"
    return labels


def nearest_codes(data):
    """The baseline's step 1: team of the nearest player where the ball is seen, else NOBODY."""
    gap = data["player_xy"] - data["ball_xy"][:, None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(gap[..., 0], gap[..., 1])
        seen = ~np.all(np.isnan(dist), axis=1)
        nearest = np.nanargmin(np.where(seen[:, None], dist, 0.0), axis=1)
    return np.where(seen, data["team"][nearest], baseline.NOBODY).astype(np.int8)


def score(gt, pred):
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "alive_dead_F1": result["alive_dead_f1"], "team_acc": result["team_acc_both_alive"],
            "possessions_true": result["n_gt_spells"], "possessions_pred": result["n_pred_spells"]}


def all_games():
    """[(name, source)] for every bench game whose data is on disk."""
    games = []
    if os.path.isdir(SAMPLE_DIR):
        games += [("metrica{}".format(g), "metrica") for g in GAMES]
    games += [("dfl-{}".format(m), "dfl") for m in dfl.match_ids()]
    return games


def labelled_game(name):
    """(data, gt) for one bench game: answers from events (Metrica) or from the tracking (DFL)."""
    if name.startswith("metrica"):
        data, events = load_game(int(name[len("metrica"):]))
        return data, event_labels(events, data["half"])
    data = dfl.load_match(name[len("dfl-"):])
    return data, data["gt"]


def main():
    games = all_games()
    if not games:
        sys.exit("no bench data, see the module docstrings of bench.py and dfl.py")
    rows = []
    for name, source in games:
        data, gt = labelled_game(name)
        graded = gt != "X"
        ours = combine.build(data)
        base = baseline.smooth(nearest_codes(data), BASELINE_WINDOW)
        base = np.where(data["graded"], base, "X")
        ball_seen = ~np.isnan(data["ball_xy"][:, 0])
        print("{}: {:.0f} graded min, true D {:.1f}%, ball seen live {:.0f}% / dead {:.0f}%".format(
            name, graded.sum() / FPS / 60.0, 100.0 * (gt[graded] == "D").mean(),
            100.0 * ball_seen[graded & (gt != "D")].mean(), 100.0 * ball_seen[gt == "D"].mean()))
        for method, pred in (("ours", ours), ("baseline_smooth50", base)):
            row = {"game": name, "source": source, "method": method}
            row.update(score(gt, pred))
            row["true_D_%"] = 100.0 * (gt[graded] == "D").mean()
            row["pred_D_%"] = 100.0 * (pred[graded] == "D").mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    columns = ["spellF1", "macroF1", "alive_dead_F1", "team_acc"]
    print()
    print(table.drop(columns="source").round(3).to_string(index=False))
    print()
    print(table.groupby(["source", "method"])[columns].mean().round(3).to_string())
    print()
    print("all games")
    print(table.groupby("method")[columns].mean().round(3).to_string())


if __name__ == "__main__":
    main()

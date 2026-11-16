import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idsse  # noqa: E402
from tracking import ROOT, BALL, AWAY, GAME, GAMES, load_tracking  # noqa: E402
from games import half_video_blocks  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))


def set_names(public=True):
    # every data set on disk: our match, the Metrica sample games, the IDSSE matches
    names = ["public"] if public else []
    sample_dir = os.path.join(ROOT, "data", "external", "metrica-sample", "data")
    names += ["metrica{}".format(game) for game in (1, 2, 3)
              if os.path.isdir(os.path.join(sample_dir, "Sample_Game_{}".format(game)))]
    return names + ["dfl-" + match_id for match_id in idsse.match_ids()]


def load_set(name):
    # (tracking, team, halves, gt) in the layout of tracking.load_tracking, for any data set
    # - public: our match, gt is the public ground truth (X outside its 5 minutes)
    # - metricaN: Metrica sample game N, gt from events by bench.py, game 3 needs kloppy
    # - dfl-<match id>: IDSSE match, gt from the provider
    # - coordinates on the GAMES[GAME]["pitch"] rectangle, other pitches are rescaled
    if name == "public":
        tracking, team = load_tracking()
        return tracking, team, half_video_blocks(GAME), pd.read_csv(GT_PATH)["label"].values
    if name.startswith("dfl-"):
        tracking, team, gt, halves = idsse.load_match(name[len("dfl-"):])
        return tracking, team, halves, gt

    import bench
    data, gt = bench.labelled_game(name)
    length, width = GAMES[GAME]["pitch"]
    players, ball = data["player_xy"], data["ball_xy"]
    scale = np.array([length / data["pitch"][0], width / data["pitch"][1]])
    tracking = np.full((len(ball), BALL + 1, 3), np.nan, dtype=np.float32)
    tracking[:, :players.shape[1], :2] = players * scale
    tracking[:, BALL, :2] = ball * scale
    team = np.full(BALL, AWAY, dtype=np.int8)
    team[:players.shape[1]] = data["team"]
    halves = [(half, int(np.flatnonzero(data["half"] == half)[0]), int(np.flatnonzero(data["half"] == half)[-1]))
              for half in (1, 2)]
    return tracking, team, halves, gt


def src_data(name, tracking, team, halves):
    # the data dict src_combine takes, for the same data set
    # - public: src's own load.load_tracking, with the file's player speeds and the 2nd half kickoff
    # - metricaN: bench.py's own loader
    # - dfl-<id> and anything else: built from our arrays, player speeds from positions as bench.py does
    import bench
    import load
    if name == "public":
        return load.load_tracking()
    if name.startswith("metrica"):
        return bench.labelled_game(name)[0]
    length, width = GAMES[GAME]["pitch"]
    used = np.flatnonzero(np.any(~np.isnan(tracking[:, :BALL, 0]), axis=0))
    half = np.zeros(len(tracking), dtype=np.int8)
    for number, first, last in halves:
        half[first:last + 1] = number
    player_xy = np.ascontiguousarray(tracking[:, used, :2])
    return {"player_xy": player_xy, "player_speed": bench.player_speed(player_xy).astype(np.float32),
            "ball_xy": np.ascontiguousarray(tracking[:, BALL, :2]), "team": team[used],
            "goalkeeper": np.zeros(len(used), dtype=bool), "half": half, "graded": half != 0,
            "pitch": (length, width), "kickoff_2nd": None}


def drop_ball(tracking, halves, seed):
    # copy of tracking with the ball removed in runs, to look like our match
    # - run lengths drawn from our match's missing-ball runs, inside the halves
    # - runs placed at random until 41% of the half frames have no ball
    ours, _ = load_tracking()
    inside = np.zeros(len(ours), dtype=bool)
    for half, first, last in half_video_blocks(GAME):
        inside[first:last + 1] = True
    edges = np.diff(np.concatenate(([0], (np.isnan(ours[:, BALL, 0]) & inside).astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    target = np.mean(np.isnan(ours[inside, BALL, 0]))

    rng = np.random.default_rng(seed)
    out = tracking.copy()
    span = np.zeros(len(tracking), dtype=bool)
    for half, first, last in halves:
        span[first:last + 1] = True
    frames = np.flatnonzero(span)
    while np.mean(np.isnan(out[span, BALL, 0])) < target:
        start = rng.choice(frames)
        out[start:start + rng.choice(lengths), BALL] = np.nan
    return out


OUR_NOISE = {}


def team_speed(tracking, team, halves):
    # mean player speed over ±1 s, m/s, NaN outside the halves; player velocity from ±5 frames as in signals.py
    fps = GAMES[GAME]["fps"]
    players = tracking[:, :BALL, :2].astype(np.float64)
    velocity = np.full_like(players, np.nan)
    velocity[5:-5] = (players[10:] - players[:-10]) * fps / 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speed = np.nanmean(np.hypot(velocity[..., 0], velocity[..., 1]), axis=1)
    speed = pd.Series(speed).rolling(2 * fps + 1, center=True, min_periods=1).mean().values
    inside = np.zeros(len(tracking), dtype=bool)
    for half, first, last in halves:
        inside[first:last + 1] = True
    return np.where(inside, speed, np.nan)


def run_bounds(mask):
    # (starts, stops) of the runs of True, stops exclusive
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def our_noise():
    # Tracking faults of our match, measured once, no labels involved.
    # - speed_bins: team speed bin edges, m/s; bin_frames[b]: half frames in bin b
    # - ball_missing[b]: share of the half frames in team speed bin b without the ball
    # - found_slow / found_fast: share of the ball-missing frames below 1 m/s / above 2 m/s team speed where the
    #   ball is back on the next frame
    # - track_runs / track_gaps: lengths of the seen runs and of the gaps between them, per slot, inside the
    #   span from the slot's first to last seen frame of a half
    # - player_missing: 1 - mean seen players per team (capped at 11) / 11
    # - extra_share: share of half frames where a team has more than 11 players seen, extra_runs: their run lengths
    if OUR_NOISE:
        return OUR_NOISE
    tracking, team = load_tracking()
    halves = half_video_blocks(GAME)
    speed = team_speed(tracking, team, halves)
    inside = ~np.isnan(speed)
    missing = np.isnan(tracking[:, BALL, 0]) & inside
    bins = np.arange(0, 4.01, 0.25)
    which = np.digitize(speed, bins)
    bin_frames = np.array([np.sum(inside & (which == b)) for b in range(len(bins) + 1)])
    ball_missing = np.array([missing[inside & (which == b)].mean() if bin_frames[b] else 0.0
                             for b in range(len(bins) + 1)])
    both = inside[:-1] & inside[1:] & missing[:-1]
    found = both & ~missing[1:]
    slow, fast = both & (speed[:-1] < 1.0), both & (speed[:-1] > 2.0)
    seen = ~np.isnan(tracking[:, :BALL, 0])
    track_runs, track_gaps = [], []
    for slot in range(BALL):
        for half, first, last in halves:
            frames = np.flatnonzero(seen[first:last + 1, slot])
            if len(frames) < 2:
                continue
            span = seen[first + frames[0]:first + frames[-1] + 1, slot]
            run_starts, run_stops = run_bounds(span)
            track_runs += list(run_stops - run_starts)
            gap_starts, gap_stops = run_bounds(~span)
            track_gaps += list(gap_stops - gap_starts)
    counts = [seen[:, team == side].sum(axis=1) for side in (0, 1)]
    extra = ((counts[0] > 11) | (counts[1] > 11)) & inside
    extra_starts, extra_stops = run_bounds(extra)
    OUR_NOISE.update(
        speed_bins=bins, bin_frames=bin_frames, ball_missing=ball_missing,
        found_slow=found[slow].sum() / slow.sum(), found_fast=found[fast].sum() / fast.sum(),
        track_runs=np.array(track_runs), track_gaps=np.array(track_gaps),
        player_missing=1 - np.mean([np.minimum(count[inside], 11).mean() for count in counts]) / 11,
        extra_share=float(np.mean(extra[inside])), extra_runs=extra_stops - extra_starts)
    return OUR_NOISE


def noisy_copy(tracking, team, halves, gt, seed):
    # (tracking, team) copy of a practice match with the tracking faults of our match, see our_noise
    # - ball: missing shares m_dead, m_live by the practice labels, least squares so that
    #   m_dead * dead share + m_live * live share in each team speed bin of this match gives our ball_missing,
    #   weighted by our bin_frames, clipped to [0, 1]
    #   - a two state chain per frame, seen / missing; the ball comes back with found_slow on dead frames and
    #     found_fast on live frames, it is lost with the chance that makes the state's missing share m
    #   - every half starts with the ball seen, frames already without the ball stay without it
    # - players: every track is cut into seen runs drawn from our track runs, a gap drawn from our track gaps
    #   follows a run with probability q, q set so the missing share matches player_missing
    # - extra person: a copy of a seen player put in a free slot of the other team, in runs drawn from our extra
    #   runs, until extra_share of the half frames have a team with more than 11 players seen; a team mistake of
    #   the automatic tracking
    # - free slots are split between the teams, a slot holds several extra runs, extra runs do not overlap
    noise = our_noise()
    rng = np.random.default_rng(seed)
    out = tracking.copy()
    team = team.copy()
    inside = np.zeros(len(tracking), dtype=bool)
    for half, first, last in halves:
        inside[first:last + 1] = True

    which = np.digitize(np.nan_to_num(team_speed(tracking, team, halves)), noise["speed_bins"])
    dead = gt == "D"
    graded = inside & (gt != "X")
    dead_share = np.array([dead[graded & (which == b)].mean() if np.any(graded & (which == b)) else 0.0
                           for b in range(len(noise["speed_bins"]) + 1)])
    weight = np.sqrt(noise["bin_frames"])
    design = np.stack([dead_share, 1 - dead_share], axis=1) * weight[:, None]
    m_dead, m_live = np.clip(np.linalg.lstsq(design, noise["ball_missing"] * weight, rcond=None)[0], 0, 0.99)
    find = np.where(dead, noise["found_slow"], noise["found_fast"])
    lose = np.where(dead, m_dead, m_live) * find / (1 - np.where(dead, m_dead, m_live))
    draws = rng.random(len(out))
    for half, first, last in halves:
        missing = False
        for frame in range(first, last + 1):
            if missing:
                out[frame, BALL] = np.nan
                missing = draws[frame] >= find[frame]
            else:
                missing = draws[frame] < lose[frame]

    mean_run, mean_gap = noise["track_runs"].mean(), noise["track_gaps"].mean()
    share = noise["player_missing"]
    gap_chance = min(1.0, share * mean_run / (mean_gap * (1 - share)))
    for slot in range(BALL):
        for half, first, last in halves:
            frames = np.flatnonzero(~np.isnan(tracking[first:last + 1, slot, 0]))
            if len(frames) == 0:
                continue
            frame = first + frames[0]
            while frame <= first + frames[-1]:
                frame += int(rng.choice(noise["track_runs"]))
                if rng.random() < gap_chance:
                    gap = int(rng.choice(noise["track_gaps"]))
                    out[frame:min(frame + gap, last + 1), slot] = np.nan
                    frame += gap

    free = [slot for slot in range(BALL) if np.all(np.isnan(tracking[:, slot, 0]))]
    team[free] = np.arange(len(free)) % 2
    used = np.zeros(len(out), dtype=bool)
    counts = [np.sum(~np.isnan(out[:, :BALL, 0]) & (team == side), axis=1) for side in (0, 1)]
    frames = np.flatnonzero(inside)
    while len(free) >= 2 and np.mean(((counts[0] > 11) | (counts[1] > 11))[inside]) < noise["extra_share"]:
        start = int(rng.choice(frames))
        stop = min(start + int(rng.choice(noise["extra_runs"])), len(out))
        seen = np.flatnonzero(~np.isnan(out[start, :BALL, 0]))
        if used[start:stop].any() or len(seen) == 0:
            continue
        source = int(rng.choice(seen))
        slot = [slot for slot in free if team[slot] != team[source]][0]
        out[start:stop, slot] = out[start:stop, source]
        used[start:stop] = True
        counts[team[slot]][start:stop] += ~np.isnan(out[start:stop, slot, 0])
    return out, team


def relabel(gt, min_frames):
    # Practice labels following the challenge rules: a short possession is a touch, not a change.
    # - live stretch: a maximal run of H and A
    # - inside a live stretch, a run shorter than min_frames that is not the first run of the stretch takes the
    #   label of the run before it, shortest runs first
    # - the first run of a stretch is the restart and is kept
    out = gt.copy()
    live_starts, live_stops = run_bounds(np.isin(out, ["H", "A"]))
    for live_start, live_stop in zip(live_starts, live_stops):
        part = out[live_start:live_stop]
        while True:
            changes = np.flatnonzero(part[1:] != part[:-1]) + 1
            starts = np.concatenate(([0], changes))
            stops = np.concatenate((changes, [len(part)]))
            lengths = stops - starts
            short = np.flatnonzero(lengths[1:] < min_frames) + 1
            if len(short) == 0:
                break
            run = short[np.argmin(lengths[short])]
            part[starts[run]:stops[run]] = part[starts[run] - 1]
    return out


def load_variant(name, seed=0):
    # (tracking, team, halves, gt) for a data set name with an optional suffix
    # - ":drop": the ball removed like on our match by drop_ball
    # - ":noisy": the tracking faults of our match by noisy_copy
    # - a bare DFL match id means dfl-<id>
    base, _, suffix = name.partition(":")
    tracking, team, halves, gt = load_set("dfl-" + base if base.startswith("DFL-MAT") else base)
    if suffix == "drop":
        tracking = drop_ball(tracking, halves, seed=seed)
    elif suffix == "noisy":
        tracking, team = noisy_copy(tracking, team, halves, gt, seed=seed)
    return tracking, team, halves, gt

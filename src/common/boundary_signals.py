import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signals import FPS  # noqa: E402
from tracking import BALL, HOME, GAME, GAMES  # noqa: E402

# Per-frame signals about who owns the ball next, around stoppages, kicks and goalkeepers.
# Every signal is +1 / -1 for H / A (or a mean of such values), 0 where it says nothing.
# Inputs
# - ball position smoothed by a 5 frame moving average, velocity from ±2 frames of that, as in signals.py
# - player velocity from ±5 frames of raw positions
# - attacking direction per half: the team whose minimum x has the lower median attacks +x, as in signals.py
# restart_out: the team that restarts after the ball leaves the pitch
# - exit: first frame with the ball more than 0.5 m outside the pitch after a frame inside it
# - last toucher: team of the nearest player on the last frame within 3 s before the exit where a player is
#   within 2 m of the ball; no toucher, no signal
# - value: the other team, from the exit until 3 s after the ball is next inside the pitch, at most 30 s
# restart_at: the team standing at a still ball
# - frames with the ball tracked and slower than 1 m/s
# - the nearest player is within 2 m and the nearest opponent of that player's team is more than 4 m away
# - value: the nearest player's team, averaged over ±1 s
# kick: speed above 6 m/s on a frame where it rose by more than 3 m/s over the last 5 frames, first frame of a run
# - kicker: team of the nearest player 5 frames before the kick, if within 3 m
# - target: ball path b + v (1 - exp(-0.8 t)) / 0.8 at t = 1.5 s with v 2 frames after the kick; players carried
#   by their velocity for 1 s; target team is the team of the player nearest to that point
# - pass_kicker, pass_target: the kicker / target team, from the kick until the next kick, at most 3 s
# keeper: the team whose goalkeeper has the ball
# - goalkeeper per team and half: the player with the lowest median x along the own attacking direction, among
#   players tracked on at least half of the half's frames
# - frames where a goalkeeper is within 1.5 m of the ball and the ball is slower than 3 m/s
# - value: the goalkeeper's team, averaged over ±1 s


def compute_boundary(tracking, team, halves):
    n_frames = tracking.shape[0]
    ball = tracking[:, BALL, :2].astype(np.float64)
    players = tracking[:, :BALL, :2].astype(np.float64)
    is_home = team == HOME
    side = np.where(is_home, 1.0, -1.0)
    length, width = GAMES[GAME]["pitch"]

    smooth = np.full_like(ball, np.nan)
    smooth[2:-2] = sum(ball[offset:n_frames - 4 + offset] for offset in range(5)) / 5.0
    ball_v = np.full_like(ball, np.nan)
    ball_v[4:-4] = (smooth[6:-2] - smooth[2:-6]) * FPS / 4.0
    speed = np.hypot(ball_v[:, 0], ball_v[:, 1])
    player_v = np.full_like(players, np.nan)
    player_v[5:-5] = (players[10:] - players[:-10]) * FPS / 10.0
    distance = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argmin(distance, axis=1)
    nearest_d = distance[np.arange(n_frames), nearest]
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width])
    out = {name: np.zeros(n_frames) for name in ("restart_out", "restart_at", "pass_kicker", "pass_target", "keeper")}

    def averaged(values):
        return pd.Series(values).rolling(2 * FPS + 1, center=True, min_periods=1).mean().values

    for half, first, last in halves:
        is_out = outside[first:last + 1] > 0.5
        is_in = outside[first:last + 1] <= 0
        tracked = ~np.isnan(ball[first:last + 1, 0])
        last_in = -1
        for step in range(last - first + 1):
            frame = first + step
            if is_in[step]:
                last_in = frame
                continue
            if not is_out[step] or last_in != frame - 1:
                continue
            touches = [f for f in range(max(first, frame - 3 * FPS), frame) if nearest_d[f] <= 2]
            if not touches:
                continue
            restarter = -side[nearest[touches[-1]]]
            back = np.flatnonzero(is_in[step:] & tracked[step:])
            end = min(first + step + (back[0] if len(back) else 0) + 3 * FPS, frame + 30 * FPS, last + 1)
            out["restart_out"][frame:end] = restarter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opponent_d = np.where(is_home[None, :] != is_home[nearest][:, None], distance, np.inf).min(axis=1)
    standing = (speed < 1) & (nearest_d <= 2) & (opponent_d > 4)
    out["restart_at"] = averaged(np.where(standing, side[nearest], 0.0))

    rising = np.zeros(n_frames, dtype=bool)
    rising[5:] = (speed[5:] > 6) & (speed[5:] - speed[:-5] > 3)
    kicks = np.flatnonzero(rising & ~np.concatenate(([False], rising[:-1])))
    kicks = kicks[(kicks >= 5) & (kicks < n_frames - 2)]
    for number, kick in enumerate(kicks):
        end = min(kick + 3 * FPS, kicks[number + 1] if number + 1 < len(kicks) else n_frames)
        before = kick - 5
        if nearest_d[before] <= 3:
            out["pass_kicker"][kick:end] = side[nearest[before]]
        v = ball_v[kick + 2]
        if np.isnan(v[0]):
            continue
        point = ball[kick] + v * (1 - np.exp(-0.8 * 1.5)) / 0.8
        moved = players[kick] + np.nan_to_num(player_v[kick]) * 1.0
        gap = np.hypot(moved[:, 0] - point[0], moved[:, 1] - point[1])
        if np.all(np.isnan(gap)):
            continue
        out["pass_target"][kick:end] = side[np.nanargmin(gap)]

    keeper_near = np.zeros(n_frames)
    for half, first, last in halves:
        frames = slice(first, last + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home_min = np.nanmedian(np.nanmin(np.where(is_home, players[frames, :, 0], np.nan), axis=1))
            away_min = np.nanmedian(np.nanmin(np.where(~is_home, players[frames, :, 0], np.nan), axis=1))
            home_sign = 1.0 if home_min < away_min else -1.0
            forward = players[frames, :, 0] * np.where(is_home, home_sign, -home_sign)[None, :]
            present = np.mean(~np.isnan(forward), axis=0) >= 0.5
            median_forward = np.where(present, np.nanmedian(forward, axis=0), np.inf)
        for mask in (is_home, ~is_home):
            keeper = int(np.argmin(np.where(mask, median_forward, np.inf)))
            if not np.isfinite(median_forward[keeper]):
                continue
            holds = (distance[frames, keeper] <= 1.5) & (speed[frames] < 3)
            keeper_near[frames] = np.where(holds, side[keeper], keeper_near[frames])
    out["keeper"] = averaged(keeper_near)
    return out


if __name__ == '__main__':
    # How often each signal fires and how often it names the GT owner, on GT H / A frames where it is not 0
    # - also on the GT owner of the first H / A frame after each GT D spell, from the last D frame of the spell
    import datasets

    for name in sys.argv[1:]:
        tracking, team, halves, gt = datasets.load_set(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals = compute_boundary(tracking, team, halves)
        alive = np.isin(gt, ["H", "A"])
        truth = np.where(gt == "H", 1.0, -1.0)
        ends = np.flatnonzero((gt[:-1] == "D") & np.isin(gt[1:], ["H", "A"]))
        print("\n{}  {} restarts after D".format(name, len(ends)))
        for key, values in signals.items():
            on = alive & (values != 0)
            right = np.mean(np.sign(values[on]) == truth[on]) if on.any() else np.nan
            at_end = values[ends]
            fired = at_end != 0
            restart_right = np.mean(np.sign(at_end[fired]) == truth[ends + 1][fired]) if fired.any() else np.nan
            print("  {:<12} on {:.3f} of alive, right {:.3f}   at restarts: on {:.3f}, right {:.3f}".format(
                key, on.sum() / max(1, alive.sum()), right, fired.mean() if len(ends) else np.nan, restart_right))

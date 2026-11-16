import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import ROOT, BALL, HOME, AWAY, GAME, GAMES, load_tracking  # noqa: E402
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
CHUNK = 10000

# Per-frame team signals around the ball. Every signal is oriented so that positive favours H.
# Inputs
# - ball position smoothed by a 5 frame moving average, velocity from ±2 frames of that
# - player velocity from ±5 frames of raw positions
# - NaN where the ball (or its velocity) is not tracked
# Surround
# - nearest:  nearest A distance minus nearest H distance, metres, the reference
# - count5:   H players within 5 m minus A players within 5 m
# - kernel:   (KH - KA) / (KH + KA), K = sum of exp(-d² / 2σ²) over the team, σ = 3 m, 0 when nobody is close
# - knn5:     (H - A) / 5 among the 5 nearest players
# - ring:     16 points on a 3 m circle around the ball, each owned by the team of its nearest player, (H - A) / 16
# - sectors:  8 angular sectors around the ball up to 8 m, each owned by the team of its nearest player in it,
#             (H - A) / 8, an empty sector counts for nobody
# Ball direction
# - closing:  sum over players within 20 m of max(0, ball velocity towards the player) * exp(-d / 8 m), H - A, m/s
# - projected: ball position extrapolated 0.5 s ahead, nearest A distance minus nearest H distance there, ±10 m
# - first:    first-to-ball time, A minus H, seconds
#             - ball path b + v t for t up to 1.5 s in 0.1 s steps, no deceleration
#             - a player reaches a path point at t when it is within 1 m + 7 m/s * max(0, t - 0.2 s) of it
#             - players are taken as standing still at their current position, unreached is 2 s
# - closing_v: like closing, but with the relative velocity of ball and player, only players within 10 m,
#             max over the team instead of a sum
# - attack:   ball velocity along the direction H attacks, m/s
#             - H attacks +x in a half where its minimum player x has the lower median, the goalkeeper side
# - first_v:  first-to-ball time like first, with
#             - the ball slowing down, path b + v (1 - exp(-0.8 t)) / 0.8, t up to 2 s
#             - players carried by their velocity for the 0.3 s reaction, then reaching 1 m + 7 m/s * (t - 0.3)
# - dribble:  per team the minimum over players within 3 m of distance + 0.5 s * |ball velocity - player velocity|,
#             A minus H, low when a player moves with the ball
# - goalside: (A players goal side of the ball - H players goal side of the ball) / 10, goal side is between the
#             ball and the team's own goal line along x
# Team shape, no ball needed
# - spread:   team spread sqrt(var x + var y) over its tracked players, H minus A, metres
# - advance:  team mean x along its own attacking direction, H minus A, metres
# - push:     team mean x velocity along its own attacking direction over ±1 s, H minus A, m/s
# Dead ball, positive favours D
# - dead_team_slow: minus the mean player speed over ±1 s
# - dead_slowdown:  mean player speed 2 s ago minus now, over ±1 s
# - dead_compact:   minus the sum of both team spreads
# - dead_ball_slow: minus the ball speed
# - dead_still:     minus the extent of the tracked ball over ±1 s, max - min over x and y
# - dead_line:      ball distance outside the pitch, inside it minus the distance to the nearest line
# - dead_far:       nearest player distance to the ball, capped at 30 m
# - dead_missing:   1 where the ball is not tracked, 0 elsewhere


def compute_signals(tracking, team, halves):
    # halves: [(half, first_frame, last_frame)] in row indices
    n_frames = tracking.shape[0]
    ball = tracking[:, BALL, :2].astype(np.float64)
    players = tracking[:, :BALL, :2].astype(np.float64)
    is_home = team == HOME

    smooth = np.full_like(ball, np.nan)
    smooth[2:-2] = sum(ball[offset:n_frames - 4 + offset] for offset in range(5)) / 5.0
    ball_velocity = np.full_like(ball, np.nan)
    ball_velocity[4:-4] = (smooth[6:-2] - smooth[2:-6]) * FPS / 4.0
    player_velocity = np.full_like(players, np.nan)
    player_velocity[5:-5] = (players[10:] - players[:-10]) * FPS / 10.0

    out = {name: np.full(n_frames, np.nan) for name in
           ("nearest", "count5", "kernel", "knn5", "ring", "sectors", "closing", "projected", "first", "closing_v",
            "first_v", "dribble", "dead_far")}
    # attack_sign[f]: +1 where H attacks +x, -1 where it attacks -x, 0 outside the halves
    attack_sign = np.zeros(n_frames)
    for half, first_frame, last_frame in halves:
        frames = slice(first_frame, last_frame + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home_min = np.nanmedian(np.nanmin(np.where(is_home, players[frames, :, 0], np.nan), axis=1))
            away_min = np.nanmedian(np.nanmin(np.where(~is_home, players[frames, :, 0], np.nan), axis=1))
        attack_sign[frames] = 1.0 if home_min < away_min else -1.0
    long_times = np.arange(0, 2.01, 0.1)
    ring_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = 3.0 * np.stack([np.cos(ring_angles), np.sin(ring_angles)], axis=1)
    times = np.arange(0, 1.51, 0.1)

    for begin in range(0, n_frames, CHUNK):
        end = min(n_frames, begin + CHUNK)
        b = ball[begin:end]
        v = ball_velocity[begin:end]
        p = players[begin:end]
        offset = p - b[:, None, :]
        distance = np.hypot(offset[..., 0], offset[..., 1])
        tracked = ~np.isnan(distance)
        distance = np.where(tracked, distance, np.inf)
        has_ball = ~np.isnan(b[:, 0])
        home_d = np.where(is_home, distance, np.inf)
        away_d = np.where(~is_home, distance, np.inf)

        nearest = np.clip(away_d.min(axis=1), 0, 30) - np.clip(home_d.min(axis=1), 0, 30)
        count = np.sum(home_d <= 5, axis=1) - np.sum(away_d <= 5, axis=1)
        weight = np.exp(-distance ** 2 / (2 * 3.0 ** 2))
        home_k, away_k = np.sum(weight * is_home, axis=1), np.sum(weight * ~is_home, axis=1)
        kernel = np.where(home_k + away_k > 1e-3, (home_k - away_k) / (home_k + away_k + 1e-9), 0.0)
        order = np.argsort(distance, axis=1)[:, :5]
        knn = np.where(is_home[order], 1, -1).sum(axis=1) / 5.0

        points = b[:, None, :] + ring[None]
        point_d = np.hypot(p[:, None, :, 0] - points[..., 0, None], p[:, None, :, 1] - points[..., 1, None])
        point_d = np.where(np.isnan(point_d), np.inf, point_d)
        owner_home = is_home[np.argmin(point_d, axis=2)]
        ring_score = np.where(owner_home, 1, -1).mean(axis=1)

        angle = np.nan_to_num(np.arctan2(offset[..., 1], offset[..., 0]))
        sector = ((angle + np.pi) / (2 * np.pi) * 8).astype(np.int64) % 8
        sector_score = np.zeros(end - begin)
        for number in range(8):
            in_sector = (sector == number) & (distance <= 8)
            sector_d = np.where(in_sector, distance, np.inf)
            closest = np.argmin(sector_d, axis=1)
            filled = np.isfinite(sector_d.min(axis=1))
            sector_score += np.where(filled, np.where(is_home[closest], 1, -1), 0)
        sector_score /= 8.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unit = offset / distance[..., None]
        towards = np.maximum(0, np.sum(unit * v[:, None, :], axis=2))
        towards = np.where(distance <= 20, towards * np.exp(-distance / 8.0), 0)
        towards = np.nan_to_num(towards)
        closing = np.sum(towards * is_home, axis=1) - np.sum(towards * ~is_home, axis=1)

        relative = v[:, None, :] - player_velocity[begin:end]
        towards_v = np.nan_to_num(np.maximum(0, np.sum(unit * relative, axis=2)))
        towards_v = np.where(distance <= 10, towards_v, 0)
        closing_v = np.max(towards_v * is_home, axis=1) - np.max(towards_v * ~is_home, axis=1)

        ahead = b + v * 0.5
        ahead_d = np.hypot(p[..., 0] - ahead[:, None, 0], p[..., 1] - ahead[:, None, 1])
        ahead_d = np.where(np.isnan(ahead_d), np.inf, ahead_d)
        projected = np.clip(np.clip(np.where(~is_home, ahead_d, np.inf).min(axis=1), 0, 30)
                            - np.clip(np.where(is_home, ahead_d, np.inf).min(axis=1), 0, 30), -10, 10)

        first = {}
        for side, mask in ((HOME, is_home), (AWAY, ~is_home)):
            arrival = np.full(end - begin, 2.0)
            for t in times[::-1]:
                path = b + v * t
                gap = np.hypot(p[..., 0] - path[:, None, 0], p[..., 1] - path[:, None, 1])
                reach = np.any((gap <= 1.0 + 7.0 * max(0.0, t - 0.2)) & mask, axis=1)
                arrival = np.where(reach, t, arrival)
            first[side] = arrival

        pv = np.nan_to_num(player_velocity[begin:end])
        first_v = {}
        for side, mask in ((HOME, is_home), (AWAY, ~is_home)):
            arrival = np.full(end - begin, 2.5)
            for t in long_times[::-1]:
                path = b + v * (1 - np.exp(-0.8 * t)) / 0.8
                start = p + pv * min(t, 0.3)
                gap = np.hypot(start[..., 0] - path[:, None, 0], start[..., 1] - path[:, None, 1])
                reach = np.any((gap <= 1.0 + 7.0 * max(0.0, t - 0.3)) & mask, axis=1)
                arrival = np.where(reach, t, arrival)
            first_v[side] = arrival

        mismatch = np.hypot(*(v[:, None, :] - player_velocity[begin:end]).transpose(2, 0, 1))
        carry = np.where(distance <= 3, distance + 0.5 * np.nan_to_num(mismatch, nan=20.0), 10.0)
        dribble = np.min(np.where(~is_home, carry, 10.0), axis=1) - np.min(np.where(is_home, carry, 10.0), axis=1)

        moving = has_ball & ~np.isnan(v[:, 0])
        for name, values, valid in (("nearest", nearest, has_ball), ("count5", count, has_ball),
                                    ("kernel", kernel, has_ball), ("knn5", knn, has_ball),
                                    ("ring", ring_score, has_ball), ("sectors", sector_score, has_ball),
                                    ("closing", closing, moving), ("closing_v", closing_v, moving),
                                    ("projected", projected, moving), ("first", first[AWAY] - first[HOME], moving),
                                    ("first_v", first_v[AWAY] - first_v[HOME], moving), ("dribble", dribble, moving),
                                    ("dead_far", np.clip(np.minimum(home_d.min(axis=1), away_d.min(axis=1)), 0, 30),
                                     has_ball)):
            out[name][begin:end] = np.where(valid, values, np.nan)

    inside = attack_sign != 0
    out["attack"] = np.where(inside, attack_sign * ball_velocity[:, 0], np.nan)
    length, width = GAMES[GAME]["pitch"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # x from the halfway line along H's attacking direction: H's own goal at the low end, A's at the high end
        forward_x = attack_sign[:, None] * (players[:, :, 0] - length / 2)
        forward_ball = attack_sign * (ball[:, 0] - length / 2)
        home_behind = np.sum(is_home & (forward_x < forward_ball[:, None]), axis=1)
        away_behind = np.sum(~is_home & (forward_x > forward_ball[:, None]), axis=1)
        out["goalside"] = np.where(inside & ~np.isnan(ball[:, 0]), (away_behind - home_behind) / 10.0, np.nan)

        spread, forward_centre = {}, {}
        for side, mask in ((HOME, is_home), (AWAY, ~is_home)):
            xy = players[:, mask, :]
            spread[side] = np.sqrt(np.nanvar(xy[..., 0], axis=1) + np.nanvar(xy[..., 1], axis=1))
            forward_centre[side] = np.nanmean(forward_x[:, mask], axis=1)
        out["spread"] = np.where(inside, spread[HOME] - spread[AWAY], np.nan)
        # A attacks towards low forward_x, so its advance is minus its forward centre
        out["advance"] = np.where(inside, forward_centre[HOME] + forward_centre[AWAY], np.nan)
        push = np.full(n_frames, np.nan)
        push[FPS:-FPS] = (out["advance"][2 * FPS:] - out["advance"][:-2 * FPS]) / 2.0
        out["push"] = push

        def rolling(values):
            return pd.Series(values).rolling(2 * FPS + 1, center=True, min_periods=1).mean().values

        team_speed = rolling(np.nanmean(np.hypot(player_velocity[..., 0], player_velocity[..., 1]), axis=1))
        out["dead_team_slow"] = np.where(inside, -team_speed, np.nan)
        before = np.full(n_frames, np.nan)
        before[2 * FPS:] = team_speed[:-2 * FPS]
        out["dead_slowdown"] = np.where(inside, before - team_speed, np.nan)
        out["dead_compact"] = np.where(inside, -(spread[HOME] + spread[AWAY]), np.nan)
        out["dead_ball_slow"] = np.where(inside, -np.hypot(ball_velocity[:, 0], ball_velocity[:, 1]), np.nan)
        extent = np.zeros(n_frames)
        for axis in (0, 1):
            series = pd.Series(ball[:, axis]).rolling(2 * FPS + 1, center=True, min_periods=1)
            extent = np.maximum(extent, (series.max() - series.min()).values)
        out["dead_still"] = np.where(inside & ~np.isnan(ball[:, 0]), -extent, np.nan)
        outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width])
        out["dead_line"] = np.where(inside & ~np.isnan(ball[:, 0]), outside, np.nan)
        out["dead_far"] = np.where(inside, out["dead_far"], np.nan)
        out["dead_missing"] = np.where(inside, np.isnan(ball[:, 0]).astype(float), np.nan)
    return out


def auc(scores, positive):
    # probability that a random positive frame scores above a random negative one, ties count half
    ranks = pd.Series(scores).rank().values
    # float counts, numpy sums booleans as int32 on windows and the product overflows on full matches
    n_positive, n_negative = float(positive.sum()), float((~positive).sum())
    return (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


if __name__ == '__main__':
    # Signal quality on any set of matches.
    # - team signals: AUC for H against A on GT H / A frames where the signal exists
    #   - all: every such frame; hard: frames within 2 s of a GT change between H and A
    #   - smooth: the signal averaged over ±0.5 s, NaN ignored, on all frames
    # - dead signals (dead_*): AUC for D against H / A, on GT frames where the signal exists
    # - defined: share of graded frames where the signal exists
    # - --drop: the ball removed in runs like on our match, see datasets.drop_ball
    # - columns: mean over the sets of each source
    import argparse
    import datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("sets", nargs="*", help="data set names, see datasets.set_names; all external by default")
    parser.add_argument("--drop", action="store_true", help="drop the ball like on our match")
    args = parser.parse_args()

    rows = []
    for name in args.sets or datasets.set_names(public=False):
        try:
            tracking, team, halves, gt = datasets.load_set(name)
        except ImportError as error:
            print("{}: skipped, {}".format(name, error))
            continue
        if args.drop:
            tracking = datasets.drop_ball(tracking, halves, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals = compute_signals(tracking, team, halves)
        source = "public" if name == "public" else "metrica" if name.startswith("metrica") else "dfl"
        graded = np.flatnonzero(gt != "X")
        team_frames = graded[np.isin(gt[graded], ["H", "A"])]
        changes = np.flatnonzero((gt[1:] != gt[:-1]) & np.isin(gt[1:], ["H", "A"]) & np.isin(gt[:-1], ["H", "A"])) + 1
        hard = np.zeros(len(gt), dtype=bool)
        for change in changes:
            hard[max(0, change - 2 * FPS):change + 2 * FPS] = True
        for signal, values in signals.items():
            row = dict(set=name, source=source, signal=signal, defined=np.mean(~np.isnan(values[graded])))
            if signal.startswith("dead_"):
                ok = graded[~np.isnan(values[graded])]
                row["dead"] = auc(values[ok], gt[ok] == "D")
            else:
                smoothed = pd.Series(values).rolling(FPS + 1, center=True, min_periods=1).mean().values
                for column, frames, series in (("all", team_frames, values), ("hard", team_frames[hard[team_frames]], values),
                                               ("smooth", team_frames, smoothed)):
                    ok = frames[~np.isnan(series[frames])]
                    row[column] = auc(series[ok], gt[ok] == "H")
            rows.append(row)
        print("{}: done".format(name), flush=True)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    team_table = table[~table["signal"].str.startswith("dead_")]
    dead_table = table[table["signal"].str.startswith("dead_")]
    print("\nteam signals, AUC H vs A, mean per source")
    print(team_table.groupby(["signal", "source"])[["defined", "all", "hard", "smooth"]].mean().unstack("source")
          .sort_values(("all", team_table["source"].iloc[0]), ascending=False).round(3).to_string())
    print("\ndead signals, AUC D vs alive, mean per source")
    print(dead_table.groupby(["signal", "source"])[["defined", "dead"]].mean().unstack("source")
          .sort_values(("dead", dead_table["source"].iloc[0]), ascending=False).round(3).to_string())

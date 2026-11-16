import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import BALL, HOME, GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
SLOW = 1.5
CHUNK = 20000

# Set piece setups from the players alone, no ball needed.
# Inputs
# - player velocity from ±5 frames of raw positions, a player is slow below SLOW m/s
# - attacking direction per half: the team whose minimum x has the lower median attacks +x, as in signals.py
# - pitch rectangle GAMES[GAME]["pitch"]; penalty area 16.5 m deep and 40.3 m wide, goal area 5.5 m and 18.3 m
# Shapes, 1 on a frame where the shape is there, 0 elsewhere
# - throw: a slow player within 1 m of a touchline or beyond it, between the goal lines
# - corner: a slow player within 3 m of a corner
# - goal_kick: a slow player inside his own goal area and no opponent inside that penalty area
# - wall: 3 slow players of one team, each within 1.5 m of the other two
# - kickoff: every seen player inside his own half, 1 m margin, and a player within 3 m of the centre spot
# Every shape also has a team value, +1 / -1 for H / A, 0 without the shape
# - throw_team, corner_team, goal_kick_team, kickoff_team: the team of the player at the spot, the team that restarts
# - wall_team: the team that does not stand in the wall
# - shapes and team values are averaged over ±1 s


def compute_setpiece(tracking, team, halves):
    n_frames = tracking.shape[0]
    players = tracking[:, :BALL, :2].astype(np.float64)
    is_home = team == HOME
    side = np.where(is_home, 1.0, -1.0)
    length, width = GAMES[GAME]["pitch"]
    velocity = np.full_like(players, np.nan)
    velocity[5:-5] = (players[10:] - players[:-10]) * FPS / 10.0
    slow = np.hypot(velocity[..., 0], velocity[..., 1]) < SLOW
    seen = ~np.isnan(players[..., 0])
    x, y = players[..., 0], players[..., 1]

    attack_sign = np.zeros(n_frames)
    for half, first, last in halves:
        frames = slice(first, last + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home_min = np.nanmedian(np.nanmin(np.where(is_home, x[frames], np.nan), axis=1))
            away_min = np.nanmedian(np.nanmin(np.where(~is_home, x[frames], np.nan), axis=1))
        attack_sign[frames] = 1.0 if home_min < away_min else -1.0
    # x along each player's own attacking direction from his own goal line, 0 at his goal, length at the other
    own_sign = attack_sign[:, None] * side[None, :]
    own_x = np.where(own_sign > 0, x, length - x)

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on_line = slow & ((y < 1) | (y > width - 1)) & (x > 0) & (x < length)
        out["throw"], out["throw_team"] = on_line.any(axis=1), np.nansum(np.where(on_line, side, 0), axis=1)

        corner_d = np.minimum.reduce([np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)])
        at_corner = slow & (corner_d < 3)
        out["corner"], out["corner_team"] = at_corner.any(axis=1), np.nansum(np.where(at_corner, side, 0), axis=1)

        centre_y = np.abs(y - width / 2)
        in_goal_area = slow & (own_x < 5.5) & (centre_y < 9.15)
        # opponent x along the defending team's direction is length - his own x
        goal_kick_team = np.zeros(n_frames)
        for mask, value in ((is_home, 1.0), (~is_home, -1.0)):
            keeper_in = np.any(in_goal_area & mask, axis=1)
            intruders = np.any(seen & ~mask & (length - own_x < 16.5) & (centre_y < 20.15), axis=1)
            goal_kick_team = np.where(keeper_in & ~intruders, value, goal_kick_team)
        out["goal_kick"], out["goal_kick_team"] = goal_kick_team != 0, goal_kick_team

        wall_team = np.zeros(n_frames)
        for begin in range(0, n_frames, CHUNK):
            end = min(n_frames, begin + CHUNK)
            p = players[begin:end]
            pair = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
            close = (pair < 1.5) & slow[begin:end, :, None] & slow[begin:end, None, :] \
                & (is_home[:, None] == is_home[None, :])[None]
            close &= ~np.eye(len(team), dtype=bool)[None]
            # a wall of 3: a player with two close teammates that are close to each other
            for player in range(len(team)):
                mates = close[:, player, :]
                if not mates.any():
                    continue
                triangle = np.einsum("fi,fij,fj->f", mates.astype(np.float32), close.astype(np.float32),
                                     mates.astype(np.float32)) > 0
                wall_team[begin:end] = np.where(triangle & (wall_team[begin:end] == 0), -side[player],
                                                wall_team[begin:end])
        out["wall"], out["wall_team"] = wall_team != 0, wall_team

        own_half = np.all(~seen | (own_x < length / 2 + 1), axis=1) & (seen.sum(axis=1) >= 10)
        centre = seen & (np.hypot(x - length / 2, y - width / 2) < 3)
        kickoff = own_half & centre.any(axis=1)
        out["kickoff"] = kickoff
        out["kickoff_team"] = np.where(kickoff, np.nansum(np.where(centre, side, 0), axis=1), 0)

    inside = attack_sign != 0
    for name in list(out):
        values = np.where(inside, out[name].astype(np.float64), np.nan)
        if name.endswith("_team"):
            values = np.clip(values, -1, 1)
        out[name] = pd.Series(values).rolling(2 * FPS + 1, center=True, min_periods=1).mean().values
    return out


if __name__ == '__main__':
    # share of dead and live frames with each shape on practice sets, and how often the team value names the team
    # that owns the ball on the first live frame after it
    import datasets
    for name in sys.argv[1:] or ["DFL-MAT-J03WMX", "metrica1"]:
        tracking, team, halves, gt = datasets.load_variant(name)
        shapes = compute_setpiece(tracking, team, halves)
        dead, live = gt == "D", np.isin(gt, ["H", "A"])
        print(name)
        for shape in ("throw", "corner", "goal_kick", "wall", "kickoff"):
            on = np.nan_to_num(shapes[shape]) > 0.5
            team_value = np.nan_to_num(shapes[shape + "_team"])
            starts = np.flatnonzero(live[1:] & dead[:-1]) + 1
            right, total = 0, 0
            for start in starts:
                window = slice(max(0, start - 3 * FPS), start)
                if not on[window].any():
                    continue
                guess = np.sign(team_value[window][on[window]].mean())
                total += 1
                right += guess == (1 if gt[start] == "H" else -1)
            print("  {:<10} dead {:.3f} live {:.3f}  restart team right {}/{}".format(
                shape, on[dead].mean(), on[live].mean(), right, total))

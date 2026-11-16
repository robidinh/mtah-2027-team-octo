"""Control evidence: on which frames does a team clearly have the ball at its feet?

A player controls the ball when it is within touching distance of him and nobody from the other
team is as close. Everything else -- a ball in flight, a ball nobody is near, two players from
both teams on top of it, a ball the tracking does not see -- is no evidence at all, and the
combiner keeps the possession where it was (a loose ball stays with the last team that had it).

Thresholds from football and the tracking, not from the ground truth:
a player reaches the ball within about a metre, the tracked positions are centre-of-body and a
little noisy, so 2 m is "at his feet"; a ball faster than a hard dribble is travelling, not
controlled.

    python src/algo/src_combine/control.py           # summary of the evidence
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "common"))
import load  # noqa: E402
import deadball  # noqa: E402

# Params
CONTROL_RADIUS = 2.0        # metres: ball this close to a player is at his feet
CARRY_SPEED = 7.0           # m/s: faster than this the ball is travelling, not controlled
CROWD_MARGIN = 0.5          # metres: in a crowd, a player half a step closer than every opponent
                            # has the ball (None = off, then any opponent within 2 m blocks it)

# Passes: a team that passes the ball from one contact to another controls it, however short
FPS = 25
TOUCH_FRAMES = 3            # evidence frames: shortest contact that is a touch, not tracking jitter
TOUCH_HOLE = 2              # frames: evidence gaps this short do not split a contact
PASS_DISTANCE = 5.0         # metres: the ball travelled further than a dribble touch between contacts
MAX_FLIGHT = 3 * FPS        # frames: a long pass (45 m at 15 m/s) is in the air ~3 s
ONE_TOUCH_RADIUS = 1.0      # metres: a fast ball this close to a player, clearly closer than any
                            # opponent, was played by him -- a one-touch pass if a teammate then
                            # controls it (None = off)

# Clear control: the ball right at one player's feet and no opponent anywhere near
TIGHT_RADIUS = 1.0          # metres: ball touching distance of a dribbling player
FREE_RADIUS = 3.0           # metres: no opponent close enough to challenge within a step or two

NONE = -1                   # evidence value: nobody clearly controls the ball


def nearest_distance(data, team_code):
    """Distance in metres from the ball to the nearest tracked player of one team (inf if none)."""
    players = data["player_xy"][:, data["team"] == team_code, :]
    gap = players - data["ball_xy"][:, None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nobody tracked -> all-NaN
        dist = np.nanmin(np.hypot(gap[..., 0], gap[..., 1]), axis=1)
    return np.where(np.isnan(dist), np.inf, dist)


def evidence(data):
    """Per frame: load.HOME / load.AWAY where that team controls the ball, else NONE."""
    home = nearest_distance(data, load.HOME)
    away = nearest_distance(data, load.AWAY)
    slow = np.nan_to_num(deadball.ball_speed(data), nan=np.inf) <= CARRY_SPEED

    if CROWD_MARGIN is None:
        home_free, away_free = away > CONTROL_RADIUS, home > CONTROL_RADIUS
    else:
        home_free = away > np.minimum(CONTROL_RADIUS, home + CROWD_MARGIN)
        away_free = home > np.minimum(CONTROL_RADIUS, away + CROWD_MARGIN)

    out = np.full(len(home), NONE, dtype=np.int8)
    out[slow & (home <= CONTROL_RADIUS) & home_free] = load.HOME
    out[slow & (away <= CONTROL_RADIUS) & away_free] = load.AWAY
    return out


def clear_evidence(data):
    """Like evidence(), but only where control is unmistakable (see TIGHT_RADIUS, FREE_RADIUS)."""
    home = nearest_distance(data, load.HOME)
    away = nearest_distance(data, load.AWAY)
    slow = np.nan_to_num(deadball.ball_speed(data), nan=np.inf) <= CARRY_SPEED

    out = np.full(len(home), NONE, dtype=np.int8)
    out[slow & (home <= TIGHT_RADIUS) & (away > FREE_RADIUS)] = load.HOME
    out[slow & (away <= TIGHT_RADIUS) & (home > FREE_RADIUS)] = load.AWAY
    return out


def touch_evidence(data):
    """Per frame: the team that touches the ball at any speed (see ONE_TOUCH_RADIUS), else NONE."""
    out = np.full(len(data["ball_xy"]), NONE, dtype=np.int8)
    if ONE_TOUCH_RADIUS is None:
        return out
    home = nearest_distance(data, load.HOME)
    away = nearest_distance(data, load.AWAY)
    out[(home <= ONE_TOUCH_RADIUS) & (away > home + CROWD_MARGIN)] = load.HOME
    out[(away <= ONE_TOUCH_RADIUS) & (home > away + CROWD_MARGIN)] = load.AWAY
    return out


def contacts(ev):
    """[(team, start, stop)]: runs of one team's evidence, short holes bridged, jitter dropped."""
    out = []
    frames = np.flatnonzero(ev != NONE)
    for f in frames:
        team = ev[f]
        if out and out[-1][0] == team and f - out[-1][2] <= TOUCH_HOLE:
            out[-1][2] = f + 1
            out[-1][3] += 1
        else:
            out.append([team, f, f + 1, 1])
    return [(t, a, b) for t, a, b, count in out if count >= TOUCH_FRAMES]


def passes(ev, ball_xy, ball_speed, touch=None):
    """{receive frame: sender contact start} for every pass between two contacts of one team.

    A pass: one contact, the ball kicked (faster than CARRY_SPEED) at least PASS_DISTANCE away
    within MAX_FLIGHT, and the next contact is the same team with no opponent touch in between.
    That is controlled possession by definition, even if neither contact lasts long.
    With touch (touch_evidence), the sender may also be a single one-touch play; the receiver
    always has to be a real contact -- a touch nobody on the team follows up is not control.
    Works on any slice of a match (one live stretch), frame indices relative to the slice.
    """
    out = {}
    received = contacts(ev)
    senders = list(received)
    if touch is not None:
        near_contact = np.zeros(len(ev), dtype=bool)
        for team, a, b in received:
            near_contact[max(a - TOUCH_HOLE, 0):b + TOUCH_HOLE] = True
        one_touch = np.where(near_contact | (ev != NONE), NONE, touch)
        for team in (load.HOME, load.AWAY):
            senders += [(team, a, b) for a, b in deadball.runs(one_touch == team)]
        senders.sort(key=lambda c: c[1])
    starts = [a for _, a, _ in senders]
    for team2, a2, b2 in received:
        k = int(np.searchsorted(starts, a2)) - 1
        if k < 0:
            continue
        team, a1, b1 = senders[k]
        if team2 != team or b1 > a2 or a2 - b1 > MAX_FLIGHT:
            continue
        between = ev[b1:a2]
        if np.any((between != NONE) & (between != team)):
            continue
        travelled = np.hypot(*(ball_xy[a2] - ball_xy[b1 - 1]))
        kicked = np.nan_to_num(ball_speed[b1 - 1:a2 + 1]).max() > CARRY_SPEED
        if travelled >= PASS_DISTANCE and kicked:
            out[a2] = a1
    return out


if __name__ == "__main__":
    data = load.load_tracking()
    dead, _ = deadball.detect(data)
    ev = evidence(data)
    live = data["graded"] & ~dead
    ball_seen = ~np.isnan(data["ball_xy"][:, 0])
    print("live frames          {}".format(int(live.sum())))
    print("ball seen            {:.1f}% of live".format(100.0 * ball_seen[live].mean()))
    print("control evidence     {:.1f}% of live  (home {:.1f}%, away {:.1f}%)".format(
        100.0 * (ev[live] != NONE).mean(), 100.0 * (ev[live] == load.HOME).mean(),
        100.0 * (ev[live] == load.AWAY).mean()))

"""Dead-ball detection: which frames is the ball out of play?

The idea, from how a stoppage looks in the tracking:

  * when the referee stops play, the players stop running -- within a second or two the team
    average speed falls to walking pace;
  * while the ball is dead nobody passes it: it is still, carried, or not tracked at all;
  * play restarts with a kick (or throw): the ball suddenly moves at pass speed again.

So a frame is dead when the players are walking AND the ball is not being played. A stoppage
starts where the players began to slow down, and ends at the restart kick. A tracked ball
clearly outside the pitch is dead on its own.

Every threshold below comes from football, not from the public ground truth:
walking pace ~1.4 m/s, a pass travels faster than any player can run with the ball, and no
real restart takes less than a few seconds.

    python src/common/deadball.py          # summary of the detected stoppages
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import load  # noqa: E402

# Params
FPS = 25
SPEED_SMOOTH = 25           # frames: average the team speed over 1 s
WALKING_SPEED = 1.5         # m/s: below this the players on average are walking
PASS_SPEED = 7.0            # m/s: ball faster than this is being kicked, not carried. 5 scored
                            # better on the bench but cut real BAR-ATH stoppages where players
                            # knock the ball around (see docs/WORKLOG.md 3.11)
BALL_SPEED_SMOOTH = 5       # frames: median filter against single-frame position jitter
ACTIVE_WINDOW = 25          # frames: a kick within +/- 1 s means the ball is in play
OFF_PITCH_MARGIN = 1.0      # metres beyond the line before the ball counts as out
MERGE_GAP = 3 * FPS         # frames: a short burst inside a stoppage (ball thrown back) is merged
MIN_DEAD = 3 * FPS          # frames: shorter "stoppages" are slow live play
MAX_BACKTRACK = 2 * FPS     # frames: how far back the deceleration may start
MIN_VISIBLE = 10            # players: fewer tracked than this and their mean is not the team's
BALL_MOVING_SPEED = 2.0     # m/s: faster than a player walking with the ball in his hands
MAX_MOVING_SHARE = 1.0      # share of a stoppage the ball may be moving (1.0 = rule off). Helps
                            # on the bench, but there the ball is untracked when dead; in BAR-ATH
                            # it is often seen moving in real foul stoppages (WORKLOG 3.11)


def team_speed(data):
    """Mean speed of all tracked players per frame, smoothed over SPEED_SMOOTH frames.

    With fewer than MIN_VISIBLE players tracked the mean is not the team's pace (the camera shows
    a corner of the pitch), so those frames are filled from the nearest reliable frames instead.
    """
    speeds = data["player_speed"]
    visible = (~np.isnan(speeds)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nobody tracked -> NaN
        speed = np.nanmean(speeds, axis=1)
    reliable = visible >= MIN_VISIBLE
    if reliable.any():
        idx = np.arange(len(speed))
        speed = np.interp(idx, idx[reliable], speed[reliable])
    else:
        speed = np.zeros(len(speed))
    kernel = np.ones(SPEED_SMOOTH) / SPEED_SMOOTH
    return np.convolve(speed, kernel, mode="same")


def ball_speed(data):
    """Ball speed in m/s from consecutive positions, NaN where the ball is not tracked."""
    ball = data["ball_xy"].astype(np.float64)
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * FPS
    # Median over a few frames: one bad detection makes a huge fake speed, a kick does not.
    half = BALL_SPEED_SMOOTH // 2
    padded = np.pad(speed, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, BALL_SPEED_SMOOTH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN window -> NaN, as wanted
        return np.nanmedian(windows, axis=1)


def ball_off_pitch(data):
    """True where the tracked ball is clearly outside the pitch lines."""
    length, width = data["pitch"]
    x, y = data["ball_xy"][:, 0], data["ball_xy"][:, 1]
    with np.errstate(invalid="ignore"):
        return ((x < -OFF_PITCH_MARGIN) | (x > length + OFF_PITCH_MARGIN)
                | (y < -OFF_PITCH_MARGIN) | (y > width + OFF_PITCH_MARGIN))


def runs(mask):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def detect(data):
    """(dead, intervals): dead is a per-frame bool array, intervals [(start, stop)] half-open.

    Only graded frames can be dead; everything else is left False (the submission writes X
    there anyway).
    """
    n = len(data["graded"])
    speed = team_speed(data)
    kicked = np.nan_to_num(ball_speed(data)) > PASS_SPEED
    off = ball_off_pitch(data)

    # A kick within +/- ACTIVE_WINDOW frames: the ball is being played.
    counts = np.convolve(kicked.astype(np.int32), np.ones(2 * ACTIVE_WINDOW + 1, dtype=np.int32),
                         mode="same")
    active = counts > 0

    dead = ((speed < WALKING_SPEED) & ~active) | off
    dead &= data["graded"]

    # The +/- window stops each run ACTIVE_WINDOW frames before the restart kick; the restart is
    # the kick itself, so carry each run forward to it.
    kick_frames = np.flatnonzero(kicked)
    for start, stop in runs(dead):
        following = kick_frames[np.searchsorted(kick_frames, stop):][:1]
        if len(following) and following[0] - stop <= ACTIVE_WINDOW + 1:
            dead[stop:following[0]] = data["graded"][stop:following[0]]

    # Merge stoppages separated by a short burst (ball thrown back to the taker, a re-spot).
    intervals = []
    for start, stop in runs(dead):
        if intervals and start - intervals[-1][1] <= MERGE_GAP \
                and data["half"][start] == data["half"][intervals[-1][0]]:
            intervals[-1] = (intervals[-1][0], stop)
        else:
            intervals.append((start, stop))
    intervals = [(a, b) for a, b in intervals if b - a >= MIN_DEAD]

    # In a real stoppage the ball rests at the restart spot; it is rolled or carried back only
    # briefly. A "stoppage" where the tracked ball keeps moving is slow live play. A ball seen
    # off the pitch is dead whatever it does.
    moving = np.nan_to_num(ball_speed(data)) > BALL_MOVING_SPEED
    intervals = [(a, b) for a, b in intervals
                 if off[a:b].any() or moving[a:b].mean() <= MAX_MOVING_SHARE]

    # The stoppage began where the players started to slow down: walk back down the slope.
    adjusted = []
    for start, stop in intervals:
        limit = max(start - MAX_BACKTRACK, adjusted[-1][1] if adjusted else 0)
        while start > limit and data["graded"][start - 1] and speed[start - 1] > speed[start]:
            start -= 1
        adjusted.append((start, stop))

    # BAR-ATH's 2nd half is graded from 110808 but kicks off at 113742 (FAQ.md): everything before
    # the kickoff is dead by definition, whatever the players are doing. Games graded from the
    # kickoff itself have kickoff_2nd = None.
    kickoff_2nd = data.get("kickoff_2nd")
    second = np.flatnonzero(data["half"] == 2)
    if kickoff_2nd is not None and len(second) and second[0] < kickoff_2nd:
        adjusted = [(a, b) for a, b in adjusted if data["half"][a] != 2 or b > kickoff_2nd]
        if adjusted and adjusted[0][0] <= kickoff_2nd and data["half"][adjusted[0][0]] == 2:
            adjusted[0] = (second[0], adjusted[0][1])
        else:
            adjusted.insert(sum(data["half"][a] == 1 for a, _ in adjusted), (second[0], kickoff_2nd))

    dead = np.zeros(n, dtype=bool)
    for start, stop in adjusted:
        dead[start:stop] = True
    return dead, adjusted


if __name__ == "__main__":
    data = load.load_tracking()
    tags = load.load_tags()
    dead, intervals = detect(data)

    graded = data["graded"]
    lengths = np.array([b - a for a, b in intervals]) / float(FPS)
    print("stoppages         {}  ({} in 1st half, {} in 2nd)".format(
        len(intervals), sum(data["half"][a] == 1 for a, _ in intervals),
        sum(data["half"][a] == 2 for a, _ in intervals)))
    print("dead share        {:.1f}% of graded frames".format(100.0 * dead[graded].mean()))
    print("duration          median {:.0f} s, 10-90% {:.0f}-{:.0f} s, max {:.0f} s".format(
        np.median(lengths), np.percentile(lengths, 10), np.percentile(lengths, 90),
        lengths.max()))

    # Self-check without any ground truth: a set-piece tag starts inside a stoppage.
    set_pieces = tags[(tags["code"] == "SET PIECES")]
    set_pieces = set_pieces[graded[set_pieces["start_frame"].clip(0, len(graded) - 1)]]
    hit = [any(a <= f + 2 * FPS and f <= b for a, b in intervals)
           for f in set_pieces["start_frame"]]
    print("set-piece tags    {}/{} start inside a detected stoppage (+2 s)".format(
        sum(hit), len(hit)))

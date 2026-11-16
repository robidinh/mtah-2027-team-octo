"""Combine dead-ball intervals and control evidence into one H / A / D label per frame.

The rules, straight from the challenge's definition of possession:

  * dead-ball frames are D;
  * a live stretch starts with the team that takes the restart -- the first team seen in control;
  * possession changes only when the OTHER team gains controlled possession: a single touch or a
    deflection is not enough, it has to keep the ball for MIN_CONTROL;
  * a loose ball, a ball in flight or an untracked ball stays with the team that had it.

Where the change happens: at the first frame of the new team's control. If the ball was not
tracked in between we cannot see the hand-over, and the middle of that gap is the guess with the
smallest worst-case error.

Frames outside the graded halves are X.

    python src/algo/src_combine/src_combine.py           # summary of the possessions
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "common"))
import load  # noqa: E402
import deadball  # noqa: E402
import control  # noqa: E402

# Params
FPS = 25
MIN_CONTROL = 10            # evidence frames (0.4 s): a receive-and-touch, longer than a deflection
MAX_SPREAD = 2 * FPS        # frames: those evidence frames must fall within 2 s to count as one control
FAST_CONTROL = 0            # clear-control frames in a row (0.16 s) that switch at once (WORKLOG 3.14)
USE_PASSES = True           # a pass between teammates switches at once (WORKLOG 3.14)
UNSEEN_GAP = FPS            # frames: a ball missing this long hides the hand-over -> switch mid-gap

TEAM_LABEL = {load.HOME: "H", load.AWAY: "A"}


def live_segment_owner(ev, ball_seen, received=None, clear=None):
    """Owner (load.HOME / load.AWAY) per frame of one live stretch, or None with no evidence.

    received: {frame: sender contact start} from control.passes -- a completed pass by the other
    team proves its control at once, without waiting for MIN_CONTROL frames.
    clear: control.clear_evidence for the stretch -- FAST_CONTROL frames in a row of unmistakable
    control do the same.
    """
    received = received or {}
    run = np.zeros(len(ev), dtype=np.int32)      # length of the clear-control run ending here
    if clear is not None:
        for f in np.flatnonzero(clear != control.NONE):
            run[f] = run[f - 1] + 1 if f and clear[f - 1] == clear[f] else 1
    n = len(ev)
    frames = np.flatnonzero(ev != control.NONE)
    if not len(frames):
        return None

    owner = np.empty(n, dtype=np.int8)
    current = ev[frames[0]]          # the restart taker's team
    owner[:] = current
    last_own = frames[0]             # last frame the current owner was seen in control
    rival = []                       # evidence frames of the other team since then

    for f in frames:
        team = ev[f]
        if team == current:
            last_own, rival = f, []
            continue
        rival = [r for r in rival if f - r <= MAX_SPREAD] + [f]
        passed = USE_PASSES and received.get(f, -1) > last_own
        fast = clear is not None and clear[f] == team and run[f] >= FAST_CONTROL
        if len(rival) < MIN_CONTROL and not passed and not fast:
            continue
        # The other team has the ball. Place the hand-over.
        start = min(rival[0], received[f]) if passed else rival[0]
        between = slice(last_own + 1, start)
        unseen = np.flatnonzero(~ball_seen[between]) + last_own + 1
        if len(unseen) >= UNSEEN_GAP:
            start = (unseen[0] + unseen[-1]) // 2
        owner[start:] = team
        current, last_own, rival = team, f, []
    return owner


def label(data, intervals, ev, clear=None):
    """Array of 'H' / 'A' / 'D' / 'X', one per frame."""
    n = len(data["graded"])
    out = np.full(n, "X", dtype="<U1")
    ball_seen = ~np.isnan(data["ball_xy"][:, 0])
    ball_speed = deadball.ball_speed(data)
    touch = control.touch_evidence(data)

    dead = np.zeros(n, dtype=bool)
    for a, b in intervals:
        dead[a:b] = True
    out[dead] = "D"

    previous = None                  # owner at the end of the last live stretch in this half
    live = data["graded"] & ~dead
    for a, b in deadball.runs(live):
        if a == 0 or data["half"][a] != data["half"][a - 1]:
            previous = None
        received = control.passes(ev[a:b], data["ball_xy"][a:b], ball_speed[a:b], touch[a:b])
        owner = live_segment_owner(ev[a:b], ball_seen[a:b], received,
                                   None if clear is None else clear[a:b])
        if owner is None:
            # Nobody seen in control at all: keep the team from before the stoppage. With no
            # team at all yet, X -- the scorer carries the previous label forward.
            if previous is not None:
                out[a:b] = TEAM_LABEL[previous]
            continue
        out[a:b] = np.vectorize(TEAM_LABEL.get)(owner)
        previous = owner[-1]
    return out


def build(data=None):
    if data is None:
        data = load.load_tracking()
    _, intervals = deadball.detect(data)
    clear = control.clear_evidence(data) if FAST_CONTROL else None
    return label(data, intervals, control.evidence(data), clear)


if __name__ == "__main__":
    data = load.load_tracking()
    labels = build(data)
    graded = data["graded"]
    g = labels[graded]
    print("graded labels        H {:.1f}%  A {:.1f}%  D {:.1f}%  X {:.1f}%".format(
        *(100.0 * (g == c).mean() for c in "HADX")))
    possessions = []
    for half in (1, 2):
        lab = labels[data["half"] == half]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(lab)]))
        possessions += [(e - s) / float(FPS) for s, e in zip(starts, ends) if lab[s] in "HA"]
    possessions = np.array(possessions)
    minutes = graded.sum() / FPS / 60.0
    print("possessions          {}  ({:.1f} per 5 min)".format(
        len(possessions), 5 * len(possessions) / minutes))
    print("possession length    median {:.1f} s, 10-90% {:.1f}-{:.1f} s".format(
        np.median(possessions), np.percentile(possessions, 10), np.percentile(possessions, 90)))

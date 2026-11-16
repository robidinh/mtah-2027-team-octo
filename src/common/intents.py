import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signals import FPS, auc  # noqa: E402
from tracking import BALL, HOME, GAME, GAMES  # noqa: E402

CHUNK = 10000
STATES = ("carry", "receive", "press", "support", "run", "idle")

# Per-player intent scores, the reduced set, and team sums of them.
# Inputs
# - ball position smoothed by a 5 frame moving average, velocity from ±2 frames of that, as in signals.py
# - player velocity from ±5 frames of raw positions
# - attacking direction per half: the team whose minimum x has the lower median attacks +x, as in signals.py
# Player scores, all >= 0, NaN where the player or a needed ball is not tracked
# - carry:   ball within 3 m and moving with the player, exp(-d / 1 m) * exp(-|ball v - player v| / 1.5 m/s)
# - receive: ball faster than 3 m/s heading to the player, 2 to 40 m ahead along the ball path,
#            exp(-off-path distance / (1 m + 0.1 * along)), only the player closest to the path per team counts
# - press:   closing speed on the ball within 15 m, max(0, -d distance / dt) / 8 m/s * exp(-d / 10 m), needs the ball
# - press_opp: closing speed on the nearest opponent within 8 m, max(0, -d distance / dt), no ball needed
# - support: opening speed from the nearest opponent within 8 m, max(0, d distance / dt), no ball needed
# - run:     forward speed along the own attacking direction above 4 m/s, no ball needed
# - idle:    1 when the player speed is below 1 m/s
# - distance rates over ±0.5 s
# Team evidence, positive favours H, H sum minus A sum over players
# - carry, receive: team max instead of sum
# - press, press_opp: sign flipped, pressing is done by the team without the ball
# - idle: share of all players, positive favours D


def compute_intents(tracking, team, halves):
    n_frames = tracking.shape[0]
    ball = tracking[:, BALL, :2].astype(np.float64)
    players = tracking[:, :BALL, :2].astype(np.float64)
    is_home = team == HOME
    n_players = players.shape[1]

    smooth = np.full_like(ball, np.nan)
    smooth[2:-2] = sum(ball[offset:n_frames - 4 + offset] for offset in range(5)) / 5.0
    ball_v = np.full_like(ball, np.nan)
    ball_v[4:-4] = (smooth[6:-2] - smooth[2:-6]) * FPS / 4.0
    player_v = np.full_like(players, np.nan)
    player_v[5:-5] = (players[10:] - players[:-10]) * FPS / 10.0
    speed = np.hypot(player_v[..., 0], player_v[..., 1])

    attack_sign = np.zeros(n_frames)
    for half, first_frame, last_frame in halves:
        frames = slice(first_frame, last_frame + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home_min = np.nanmedian(np.nanmin(np.where(is_home, players[frames, :, 0], np.nan), axis=1))
            away_min = np.nanmedian(np.nanmin(np.where(~is_home, players[frames, :, 0], np.nan), axis=1))
        attack_sign[frames] = 1.0 if home_min < away_min else -1.0
    # own attacking direction per player, A attacks against H
    own_sign = attack_sign[:, None] * np.where(is_home, 1.0, -1.0)[None, :]

    ball_d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    half_s = FPS // 2
    ball_rate = np.full_like(ball_d, np.nan)
    ball_rate[half_s:-half_s] = (ball_d[2 * half_s:] - ball_d[:-2 * half_s]) / (2 * half_s / FPS)

    # nearest opponent distance and the rate of that same pair's distance
    opp_d = np.full((n_frames, n_players), np.nan)
    opp_rate = np.full((n_frames, n_players), np.nan)
    other = is_home[:, None] != is_home[None, :]
    for begin in range(0, n_frames, CHUNK):
        end = min(n_frames, begin + CHUNK)
        p = players[begin:end]
        pair = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
        pair = np.where(other[None] & ~np.isnan(pair), pair, np.inf)
        nearest = np.argmin(pair, axis=2)
        found = np.isfinite(pair.min(axis=2))
        opp_d[begin:end] = np.where(found, pair.min(axis=2), np.nan)
        rows = np.arange(begin, end)[:, None]
        lo, hi = np.maximum(rows - half_s, 0), np.minimum(rows + half_s, n_frames - 1)
        me = np.arange(n_players)[None, :]
        before = np.hypot(*(players[lo, me] - players[lo, nearest]).transpose(2, 0, 1))
        after = np.hypot(*(players[hi, me] - players[hi, nearest]).transpose(2, 0, 1))
        opp_rate[begin:end] = np.where(found, (after - before) / ((hi - lo) / FPS), np.nan)

    mismatch = np.hypot(ball_v[:, None, 0] - player_v[..., 0], ball_v[:, None, 1] - player_v[..., 1])
    carry = np.where(ball_d <= 3, np.exp(-ball_d) * np.exp(-mismatch / 1.5), 0.0)
    carry = np.where(np.isnan(ball_d) | np.isnan(mismatch), np.nan, carry)

    ball_speed = np.hypot(ball_v[:, 0], ball_v[:, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unit = ball_v / ball_speed[:, None]
    offset_x, offset_y = players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1]
    along = offset_x * unit[:, None, 0] + offset_y * unit[:, None, 1]
    off_path = np.abs(offset_x * unit[:, None, 1] - offset_y * unit[:, None, 0])
    receive = np.where((ball_speed[:, None] > 3) & (along >= 2) & (along <= 40),
                       np.exp(-off_path / (1 + 0.1 * along)), 0.0)
    receive = np.where(np.isnan(ball_d) | np.isnan(ball_speed[:, None]), np.nan, receive)

    press = np.where(ball_d <= 15, np.maximum(0, -ball_rate) / 8.0 * np.exp(-ball_d / 10.0), 0.0)
    press = np.where(np.isnan(ball_rate), np.nan, press)
    near_opp = opp_d <= 8
    press_opp = np.where(np.isnan(opp_rate), np.nan, np.where(near_opp, np.maximum(0, -opp_rate), 0.0))
    support = np.where(np.isnan(opp_rate), np.nan, np.where(near_opp, np.maximum(0, opp_rate), 0.0))
    forward = own_sign * player_v[..., 0]
    run = np.where(np.isnan(forward), np.nan, np.maximum(0, forward - 4.0))
    idle = np.where(np.isnan(speed), np.nan, (speed < 1.0).astype(float))

    per_player = dict(carry=carry, receive=receive, press=press, press_opp=press_opp, support=support, run=run,
                      idle=idle)
    inside = attack_sign != 0
    has_ball = ~np.isnan(ball[:, 0]) & ~np.isnan(ball_speed)
    team_terms = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, values in per_player.items():
            home, away = np.where(is_home, values, np.nan), np.where(~is_home, values, np.nan)
            if name in ("carry", "receive"):
                diff = np.nanmax(home, axis=1) - np.nanmax(away, axis=1)
            elif name == "idle":
                diff = np.nanmean(values, axis=1)
            else:
                diff = np.nansum(home, axis=1) - np.nansum(away, axis=1)
            if name in ("press", "press_opp"):
                diff = -diff
            needs_ball = name in ("carry", "receive", "press")
            team_terms[name] = np.where(inside & (has_ball if needs_ball else True), diff, np.nan)
    return per_player, team_terms


BALL_STATES = ("missing", "dead", "held", "pass", "loose")
HELD = 0.3


def ball_states(tracking, team, per_player):
    # One ball state per frame, the first rule that applies, as indices into BALL_STATES
    # - missing: the ball or its velocity is not tracked
    # - dead:    the ball is more than 0.5 m outside the pitch, or it moved less than 1 m over ±1 s while more
    #            than half of the players are idle
    # - held:    some player's carry score is above HELD; holder is that player's team, HOME or AWAY, else -1
    # - pass:    the ball is faster than 6 m/s
    # - loose:   anything else, a slow ball nobody carries
    n_frames = tracking.shape[0]
    ball = tracking[:, BALL, :2].astype(np.float64)
    smooth = np.full_like(ball, np.nan)
    smooth[2:-2] = sum(ball[offset:n_frames - 4 + offset] for offset in range(5)) / 5.0
    speed = np.full(n_frames, np.nan)
    speed[4:-4] = np.hypot(*((smooth[6:-2] - smooth[2:-6]) * FPS / 4.0).T)
    extent = np.zeros(n_frames)
    for axis in (0, 1):
        series = pd.Series(ball[:, axis]).rolling(2 * FPS + 1, center=True, min_periods=1)
        extent = np.maximum(extent, (series.max() - series.min()).values)
    length, width = GAMES[GAME]["pitch"]
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idle_share = np.nanmean(per_player["idle"], axis=1)
    carry = np.nan_to_num(per_player["carry"])
    best = np.argmax(carry, axis=1)
    best_carry = carry[np.arange(n_frames), best]

    state = np.full(n_frames, BALL_STATES.index("loose"), dtype=np.int8)
    state[speed > 6] = BALL_STATES.index("pass")
    held = best_carry > HELD
    state[held] = BALL_STATES.index("held")
    state[(outside > 0.5) | ((extent < 1) & (idle_share > 0.5))] = BALL_STATES.index("dead")
    state[np.isnan(ball[:, 0]) | np.isnan(speed)] = BALL_STATES.index("missing")
    holder = np.where(state == BALL_STATES.index("held"), team[best], -1).astype(np.int8)
    return state, holder


def last_holder_labels(state, holder, halves, min_held):
    # H / A / D per frame from the ball states, the simplest use of them
    # - D on dead frames
    # - otherwise the team that last held the ball for at least min_held frames in a row, H before anyone did
    # - X outside the halves
    labels = np.full(len(state), "X", dtype="<U1")
    dead = BALL_STATES.index("dead")
    for half, first, last in halves:
        owner, run_team, run = "H", -1, 0
        for frame in range(first, last + 1):
            team_now = holder[frame]
            run = run + 1 if team_now == run_team and team_now >= 0 else (1 if team_now >= 0 else 0)
            run_team = team_now
            if run >= min_held:
                owner = "H" if team_now == HOME else "A"
            labels[frame] = "D" if state[frame] == dead else owner
    return labels


if __name__ == '__main__':
    # How well the intents predict possession, without a path search.
    # - AUC of each team term for H against A on GT H / A frames where it exists
    #   - all, hard: within 2 s of a GT change between H and A, noball: frames where the ball is not tracked
    #   - smooth: the term averaged over ±1 s, NaN ignored
    # - idle: AUC for D against alive
    # - combo: standardised terms with weights from a logistic fit on the other sets, smoothed over ±1 s;
    #   acc is the share of H / A frames where its sign is right
    # - reference: spread and push from signals.py, the team signals that need no ball
    import argparse
    import datasets
    from signals import compute_signals
    from sklearn.linear_model import LogisticRegression

    parser = argparse.ArgumentParser()
    parser.add_argument("sets", nargs="+")
    parser.add_argument("--drop", action="store_true", help="drop the ball like on our match, not on public")
    parser.add_argument("--ball", action="store_true", help="only the ball state analysis, see ball_analysis")
    args = parser.parse_args()

    def ball_analysis():
        # Ball states against the ground truth, per set
        # - share: share of graded frames in each state; D / H|A: GT mix inside the state
        # - holder: share of held frames on GT H / A whose holder is the GT team
        # - dead P / R: the dead state against GT D
        # - last holder: spellF1 and macroF1 of last_holder_labels for a few min_held values
        from tracking import ROOT
        sys.path.insert(0, os.path.join(ROOT, "evaluation"))
        import possession_metrics as pm
        for name in args.sets:
            tracking, team, halves, gt = datasets.load_set(name)
            if args.drop and name != "public":
                tracking = datasets.drop_ball(tracking, halves, seed=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_player, _ = compute_intents(tracking, team, halves)
            state, holder = ball_states(tracking, team, per_player)
            graded = gt != "X"
            alive = np.isin(gt, ["H", "A"])
            print("\n{}".format(name))
            for index, state_name in enumerate(BALL_STATES):
                inside = graded & (state == index)
                print("  {:<8} share {:.3f}   D {:.2f}  H|A {:.2f}".format(
                    state_name, inside.sum() / graded.sum(), np.mean(gt[inside] == "D"), np.mean(alive[inside])))
            held = alive & (holder >= 0)
            truth = np.where(gt[held] == "H", HOME, 1 - HOME)
            print("  holder right {:.3f} on {} frames".format(np.mean(holder[held] == truth), held.sum()))
            dead = state == BALL_STATES.index("dead")
            tp = np.sum(dead & graded & (gt == "D"))
            print("  dead P {:.2f} R {:.2f}".format(tp / max(1, np.sum(dead & graded)), tp / max(1, np.sum(graded & (gt == "D")))))
            for min_held in (1, 5, 12, 25):
                result = pm.evaluate(pm.label_codes(gt), pm.label_codes(last_holder_labels(state, holder, halves, min_held)),
                                     bootstrap_n=0)
                print("  last holder, min {:>2} frames: spellF1 {:.3f} macroF1 {:.3f}  {}/{} spells".format(
                    min_held, result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]),
                    flush=True)

    if args.ball:
        ball_analysis()
        sys.exit()

    TERMS =("carry", "receive", "press", "press_opp", "support", "run")
    loaded = {}
    for name in args.sets:
        tracking, team, halves, gt = datasets.load_set(name)
        if args.drop and name != "public":
            tracking = datasets.drop_ball(tracking, halves, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, terms = compute_intents(tracking, team, halves)
            signals = compute_signals(tracking, team, halves)
        terms["spread"] = signals["spread"] - np.nanmedian(signals["spread"])
        terms["push"] = signals["push"]
        terms["first_v"] = signals["first_v"]
        loaded[name] = (terms, gt, np.isnan(tracking[:, BALL, 0]))
        print("{}: loaded".format(name), flush=True)

    def smoothed(values):
        return pd.Series(values).rolling(2 * FPS + 1, center=True, min_periods=1).mean().values

    def features(terms, names):
        columns = []
        for key in names:
            values = smoothed(terms[key])
            columns.append(np.nan_to_num(np.clip(values / np.nanstd(values), -3, 3)))
        return np.stack(columns, axis=1)

    rows = []
    for name, (terms, gt, no_ball) in loaded.items():
        graded = np.flatnonzero(gt != "X")
        team_frames = graded[np.isin(gt[graded], ["H", "A"])]
        changes = np.flatnonzero((gt[1:] != gt[:-1]) & np.isin(gt[1:], ["H", "A"]) & np.isin(gt[:-1], ["H", "A"])) + 1
        hard = np.zeros(len(gt), dtype=bool)
        for change in changes:
            hard[max(0, change - 2 * FPS):change + 2 * FPS] = True
        for key, values in terms.items():
            row = dict(set=name, term=key, defined=np.mean(~np.isnan(values[graded])))
            if key == "idle":
                ok = graded[~np.isnan(values[graded])]
                row["dead"] = auc(values[ok], gt[ok] == "D")
            else:
                for column, frames, series in (("all", team_frames, values), ("hard", team_frames[hard[team_frames]], values),
                                               ("noball", team_frames[no_ball[team_frames]], smoothed(values)),
                                               ("smooth", team_frames, smoothed(values))):
                    ok = frames[~np.isnan(series[frames])]
                    row[column] = auc(series[ok], gt[ok] == "H") if len(ok) > 100 else np.nan
            rows.append(row)

    # combos, leave one set out: intents alone, intents without ball terms, and the reference signals
    combos = dict(intents=TERMS, intents_noball=("press_opp", "support", "run"),
                  reference=("first_v", "spread", "push"), shape=("spread", "push"),
                  shape_intents=("press_opp", "support", "run", "spread", "push"), both=TERMS + ("first_v", "spread", "push"))
    for combo, names in combos.items():
        for name, (terms, gt, no_ball) in loaded.items():
            train = [(features(t, names), g) for other, (t, g, _) in loaded.items() if other != name]
            x = np.concatenate([f[np.isin(g, ["H", "A"])] for f, g in train])
            y = np.concatenate([g[np.isin(g, ["H", "A"])] == "H" for f, g in train])
            model = LogisticRegression(max_iter=1000).fit(x, y)
            score = smoothed(model.decision_function(features(terms, names)))
            frames = np.flatnonzero(np.isin(gt, ["H", "A"]))
            truth = gt[frames] == "H"
            row = dict(set=name, term="combo_" + combo, defined=1.0, all=auc(score[frames], truth),
                       acc=np.mean((score[frames] > 0) == truth),
                       noball=auc(score[frames[no_ball[frames]]], truth[no_ball[frames]])
                       if no_ball[frames].sum() > 100 else np.nan,
                       weights=" ".join("{}={:+.2f}".format(n, w) for n, w in zip(names, model.coef_[0])))
            rows.append(row)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    columns = [c for c in ("defined", "all", "hard", "smooth", "noball", "acc", "dead") if c in table]
    print(table.groupby("term")[columns].mean().round(3).to_string())
    print()
    print(table.pivot_table(index="term", columns="set", values="all").round(3).to_string())
    print()
    print(table.dropna(subset=["weights"])[["set", "term", "weights"]].to_string(index=False))

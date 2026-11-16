import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

WORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORK, "common"))
import ball_fill  # noqa: E402
import datasets  # noqa: E402
from boundary_signals import compute_boundary  # noqa: E402
from intents import compute_intents  # noqa: E402
from setpiece_signals import compute_setpiece  # noqa: E402
from signals import FPS, compute_signals  # noqa: E402
from tracking import ROOT, CACHE_DIR, BALL, GAME, load_tracking  # noqa: E402
from games import half_video_blocks  # noqa: E402

import deadball  # noqa: E402
import load  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FEATURE_DIR = os.path.join(CACHE_DIR, "segment")
PRACTICE = ("metrica1", "metrica2", "metrica3", "DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH",
            "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")
TEAM_BASE = ("nearest", "count5", "kernel", "knn5", "ring", "sectors", "closing", "projected", "first", "closing_v",
             "first_v", "dribble", "attack", "goalside", "spread", "advance", "push",
             "carry", "receive", "press", "press_opp", "support", "run",
             "restart_out", "restart_at", "pass_kicker", "pass_target", "keeper",
             "throw_team", "corner_team", "goal_kick_team", "wall_team", "kickoff_team")
DEAD_BASE = ("dead_team_slow", "dead_slowdown", "dead_compact", "dead_ball_slow", "dead_still", "dead_line",
             "dead_far", "missing", "gap_seconds", "players_seen", "idle", "src",
             "throw", "corner", "goal_kick", "wall", "kickoff")
WINDOWS = (FPS, 4 * FPS)
TRAIN_STEP = 8
RELABEL_FRAMES = FPS
MAX_AGE = 6 * FPS
SET_PIECES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)", "Kick Off (Start)")
DEFAULTS = dict(scale=0.4, mix=0.0, adapt=0, tag_weight=2.0)

# Possession as the best H / A / D path of a duration-aware model, with per-frame probabilities from gradient
# boosted trees on practice matches made to look like our match.
# Practice data
# - the 10 practice matches (PRACTICE), each clean and as a noisy copy with the tracking faults of our match
#   (datasets.noisy_copy: ball gaps by state, player gaps, an extra person counted for the wrong team)
# - labels: provider / event labels, relabelled to the challenge rules by datasets.relabel, a possession shorter
#   than RELABEL_FRAMES inside a live stretch is a touch and goes to the team before it
# Per-frame features, cached per data set under FEATURE_DIR as float32, delete the folder after a signal change
# - the ball in the gaps is filled by ball_fill.py first, every ball signal is computed on the filled ball
# - team, positive favours H: TEAM_BASE from signals.py, intents.py, boundary_signals.py, setpiece_signals.py,
#   spread centred on its match median
# - dead: DEAD_BASE, the dead_* signals of signals.py, missing = no tracked ball, gap_seconds from ball_fill,
#   players_seen, idle of intents.py, src = the dead frames of src/common/deadball.py, set piece shapes
# - every base column also as its mean over the WINDOWS frames before and after the frame, NaN kept
# Models, fitted on every TRAIN_STEP-th frame of the training sets
# - team: P(H | alive) on H / A frames; every row also enters with the team features negated and the label
#   swapped, the prediction is the mean logit of f(x) and -f(-x), so H and A are treated alike
# - dead: P(D) on graded frames
# - gradient boosted trees (sklearn HistGradientBoostingClassifier), and a logistic regression on standardised
#   features; logits mixed as (1 - mix) trees + mix logistic
# Smart Tagging, our match only: a set piece tag with a team adds tag_weight to that team's logit for 5 s from the
#   tag start, the team that takes the restart; there is no practice data with tags, so tag_weight is set, not fitted
# Path, per half, exact Viterbi over states (label, age)
# - label H, A or D; age in frames since the segment started, 0 to MAX_AGE, MAX_AGE is "MAX_AGE or longer"
# - a segment of age a ends on the next frame with its hazard h(a), from the segment lengths of the relabelled
#   training labels: possessions (H, A pooled) and stoppages; above MAX_AGE a constant hazard from the mean length
# - an ending possession goes to the other team or to D, a stoppage to H or A, by the training counts
# - emission: scale * (log P(label | frame) - log P(label)), P(label) the training share; scale < 1 because
#   neighbouring frames are far from independent
# Adaptation to the match (adapt = 1)
# - after a first path, frames far from its changes (1 s) where the models agree with it with probability > 0.85
#   become pseudo labels; every TRAIN_STEP-th of them is added to the training rows with the weight of one practice
#   set, the models are fitted again and the path is found again
# - labels come from the algorithm alone, no human labels on the match
# Labels: the path inside the halves, X outside


def windowed(base):
    # base and its means over the WINDOWS frames before and after each frame, NaN ignored
    frame = pd.DataFrame(base)
    parts = [base]
    for window in WINDOWS:
        parts.append(frame.shift(1).rolling(window, min_periods=1).mean().values)
        parts.append(frame[::-1].shift(1).rolling(window, min_periods=1).mean().values[::-1])
    return np.concatenate(parts, axis=1).astype(np.float32)


def base_features(tracking, team, halves, src_data, ball_models):
    # (team_base, dead_base) per frame, float32, columns TEAM_BASE / DEAD_BASE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filled, gap_seconds = ball_fill.fill(tracking, team, halves, ball_models)
        signals = compute_signals(filled, team, halves)
        _, intents = compute_intents(filled, team, halves)
        boundary = compute_boundary(filled, team, halves)
        setpiece = compute_setpiece(tracking, team, halves)
        values = dict(signals, **intents, **boundary, **setpiece)
        values["spread"] = values["spread"] - np.nanmedian(values["spread"])
        inside = np.zeros(len(tracking), dtype=bool)
        for half, first, last in halves:
            inside[first:last + 1] = True
        values["missing"] = np.where(inside, np.isnan(tracking[:, BALL, 0]).astype(float), np.nan)
        values["gap_seconds"] = np.where(inside, gap_seconds, np.nan)
        values["players_seen"] = np.where(inside, np.sum(~np.isnan(tracking[:, :BALL, 0]), axis=1), np.nan)
        values["src"] = deadball.detect(src_data)[0].astype(float)
    return (np.stack([values[key] for key in TEAM_BASE], axis=1).astype(np.float32),
            np.stack([values[key] for key in DEAD_BASE], axis=1).astype(np.float32))


def prepared_set(name, ball_models):
    # dict(team_base, dead_base, gt, halves) for a practice set name (":noisy" allowed) or "public",
    # cached under FEATURE_DIR
    path = os.path.join(FEATURE_DIR, name.replace(":", "_") + ".npz")
    if not os.path.exists(path):
        tracking, team, halves, gt = datasets.load_variant(name, seed=0)
        src_data = datasets.src_data("public" if name == "public" else "arrays", tracking, team, halves)
        team_base, dead_base = base_features(tracking, team, halves, src_data, ball_models)
        os.makedirs(FEATURE_DIR, exist_ok=True)
        np.savez_compressed(path, team_base=team_base, dead_base=dead_base, gt=np.asarray(gt, dtype="<U1"),
                            halves=np.array(halves))
    cached = np.load(path)
    return dict(team_base=cached["team_base"], dead_base=cached["dead_base"], gt=cached["gt"],
                halves=[tuple(int(value) for value in half) for half in cached["halves"]])


def prepare_worker(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared_set(name, ball_fill.load_models())
    return name


def prepare_all(names, workers):
    # computes the missing feature caches, several sets at a time
    todo = [name for name in names if not os.path.exists(os.path.join(FEATURE_DIR, name.replace(":", "_") + ".npz"))]
    if not todo:
        return
    ball_fill.load_models()
    print("preparing {} sets with {} workers".format(len(todo), workers), flush=True)
    begin = time.time()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for name in pool.imap_unordered(prepare_worker, todo):
            print("  {} ({:.0f} s)".format(name, time.time() - begin), flush=True)


def segment_lengths(labels, halves):
    # (possession lengths, stoppage lengths) in frames, runs inside the halves
    possession, stoppage = [], []
    ends = dict(other=0, dead=0)
    for half, first, last in halves:
        part = labels[first:last + 1]
        changes = np.flatnonzero(part[1:] != part[:-1]) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [len(part)]))
        for number, (start, stop) in enumerate(zip(starts, stops)):
            label = part[start]
            if label in ("H", "A"):
                possession.append(stop - start)
                if number + 1 < len(starts):
                    following = part[starts[number + 1]]
                    ends["other" if following in ("H", "A") else "dead"] += following in ("H", "A", "D")
            elif label == "D":
                stoppage.append(stop - start)
    return np.array(possession), np.array(stoppage), ends


def duration_model(sets):
    # log hazards and log transitions for decode, from the relabelled labels of the training sets
    possession, stoppage, ends = [], [], dict(other=0, dead=0)
    shares = []
    for prepared in sets:
        lengths = segment_lengths(prepared["labels"], prepared["halves"])
        possession.append(lengths[0])
        stoppage.append(lengths[1])
        for key in ends:
            ends[key] += lengths[2][key]
        graded = prepared["labels"][prepared["labels"] != "X"]
        shares.append(np.mean(graded == "D"))
    log_hazard, log_stay = np.zeros((3, MAX_AGE + 1)), np.zeros((3, MAX_AGE + 1))
    for state, lengths in ((0, np.concatenate(possession)), (1, np.concatenate(possession)),
                           (2, np.concatenate(stoppage))):
        # h(a) = P(length = a + 1 | length > a), with one pseudo count of the tail rate per age
        tail = 1.0 / (np.mean(lengths[lengths > MAX_AGE] - MAX_AGE) + 1.0)
        ages = np.arange(MAX_AGE)
        ending = np.array([np.sum(lengths == age + 1) for age in ages], dtype=np.float64)
        alive = np.array([np.sum(lengths > age) for age in ages], dtype=np.float64)
        hazard = np.concatenate(((ending + tail) / (alive + 1.0), [tail]))
        hazard = np.clip(hazard, 1e-6, 1 - 1e-6)
        log_hazard[state], log_stay[state] = np.log(hazard), np.log(1 - hazard)
    switch = ends["other"] / (ends["other"] + ends["dead"])
    log_next = np.log(np.array([[1e-12, switch, 1 - switch], [switch, 1e-12, 1 - switch], [0.5, 0.5, 1e-12]]))
    dead_share = float(np.mean(shares))
    log_prior = np.log(np.array([(1 - dead_share) / 2, (1 - dead_share) / 2, dead_share]))
    return dict(log_hazard=log_hazard, log_stay=log_stay, log_next=log_next, log_prior=log_prior)


def decode(team_logit, dead_logit, halves, durations, scale):
    # labels H / A / D inside the halves, X outside, see the path notes at the top
    log_alive = -np.logaddexp(0, dead_logit)
    log_dead = -np.logaddexp(0, -dead_logit)
    emission = np.stack([log_alive - np.logaddexp(0, -team_logit), log_alive - np.logaddexp(0, team_logit),
                         log_dead], axis=1)
    emission = scale * (emission - durations["log_prior"][None, :])
    log_hazard, log_stay, log_next = durations["log_hazard"], durations["log_stay"], durations["log_next"]
    labels = np.full(len(team_logit), "X", dtype="<U1")
    states = np.arange(3)
    for half, first, last in halves:
        count = last - first + 1
        from_state = np.zeros((count, 3), dtype=np.int8)
        from_age = np.zeros((count, 3), dtype=np.int16)
        from_tail = np.zeros((count, 3), dtype=bool)
        score = np.full((3, MAX_AGE + 1), -np.inf)
        score[:, 0] = emission[first]
        for step in range(1, count):
            leave = score + log_hazard
            best_age = np.argmax(leave, axis=1)
            candidates = leave[states, best_age][:, None] + log_next
            origin = np.argmax(candidates, axis=0)
            stay = score + log_stay
            new = np.empty_like(score)
            new[:, 0] = candidates[origin, states]
            new[:, 1:MAX_AGE] = stay[:, :MAX_AGE - 1]
            tail = stay[:, MAX_AGE] > stay[:, MAX_AGE - 1]
            new[:, MAX_AGE] = np.where(tail, stay[:, MAX_AGE], stay[:, MAX_AGE - 1])
            new += emission[first + step][:, None]
            score = new - new.max()
            from_state[step], from_age[step], from_tail[step] = origin, best_age[origin], tail
        state, age = np.unravel_index(int(np.argmax(score)), score.shape)
        path = np.zeros(count, dtype=np.int8)
        for step in range(count - 1, -1, -1):
            path[step] = state
            if age == 0:
                state, age = from_state[step, state], from_age[step, state]
            elif age == MAX_AGE:
                age = MAX_AGE if from_tail[step, state] else MAX_AGE - 1
            else:
                age -= 1
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]
    return labels


def fit_models(team_x, team_y, team_w, dead_x, dead_y, dead_w):
    # fitted team and dead models, see the model notes at the top
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    team_x = np.concatenate([team_x, -team_x])
    team_y = np.concatenate([team_y, ~team_y])
    team_w = np.concatenate([team_w, team_w])
    models = {}
    for task, x, y, w in (("team", team_x, team_y, team_w), ("dead", dead_x, dead_y, dead_w)):
        trees = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                               min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
                                               random_state=0).fit(x, y, sample_weight=w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        std = np.where(std > 0, std, 1.0)
        standard = np.clip(np.nan_to_num((x - mean) / std), -5, 5)
        linear = LogisticRegression(max_iter=500, C=0.1, fit_intercept=task == "dead").fit(standard, y,
                                                                                          sample_weight=w)
        models[task] = dict(trees=trees, mean=mean, std=std, linear=linear)
    return models


def logits(models, team_x, dead_x, mix):
    # (team logit, dead logit) per frame, mix of trees and logistic, team logit made symmetric
    out = []
    for task, x in (("team", team_x), ("dead", dead_x)):
        model = models[task]
        tree = model["trees"].decision_function(x)
        linear = model["linear"].decision_function(np.clip(np.nan_to_num((x - model["mean"]) / model["std"]), -5, 5))
        if task == "team":
            tree = 0.5 * (tree - model["trees"].decision_function(-x))
        out.append((1 - mix) * tree + mix * linear)
    return out


def training_rows(prepared):
    # (team_x, team_y, dead_x, dead_y) on every TRAIN_STEP-th frame, labels relabelled
    labels = prepared["labels"]
    team_frames = np.flatnonzero(np.isin(labels, ["H", "A"]))[::TRAIN_STEP]
    dead_frames = np.flatnonzero(labels != "X")[::TRAIN_STEP]
    team_x, dead_x = prepared["team_x"], prepared["dead_x"]
    return team_x[team_frames], labels[team_frames] == "H", dead_x[dead_frames], labels[dead_frames] == "D"


def pseudo_rows(team_x, dead_x, team_logit, dead_logit, labels):
    # (team_x, team_y, dead_x, dead_y) on confident frames of a first path, see the adaptation notes at the top
    far = np.zeros(len(labels), dtype=bool)
    graded = labels != "X"
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    far[graded] = True
    for change in changes:
        far[max(0, change - FPS):change + FPS] = False
    p_home, p_dead = 1 / (1 + np.exp(-team_logit)), 1 / (1 + np.exp(-dead_logit))
    team_ok = far & (((labels == "H") & (p_home > 0.85)) | ((labels == "A") & (p_home < 0.15))) & (p_dead < 0.5)
    dead_ok = far & (((labels == "D") & (p_dead > 0.85)) | (np.isin(labels, ["H", "A"]) & (p_dead < 0.15)))
    team_frames = np.flatnonzero(team_ok)[::TRAIN_STEP]
    dead_frames = np.flatnonzero(dead_ok)[::TRAIN_STEP]
    return team_x[team_frames], labels[team_frames] == "H", dead_x[dead_frames], labels[dead_frames] == "D"


def tag_bonus(n_frames):
    # per-frame team logit bonus from the Smart Tagging set pieces of our match, +1 H, -1 A, see the notes at the top
    bonus = np.zeros(n_frames)
    tags = load.load_tags()
    for _, tag in tags[tags["type"].isin(SET_PIECES) & tags["team"].notna()].iterrows():
        start = int(tag["start_frame"])
        bonus[start:start + 5 * FPS] = 1.0 if tag["team"] == "H" else -1.0
    return bonus


def with_inputs(prepared):
    prepared["team_x"] = windowed(prepared["team_base"])
    prepared["dead_x"] = windowed(prepared["dead_base"])
    prepared["labels"] = datasets.relabel(prepared["gt"], RELABEL_FRAMES)
    return prepared


def train(train_names, ball_models):
    # (models, durations, rows) fitted on the clean and noisy versions of the named practice sets
    parts, sets = [], []
    for name in train_names:
        for variant in (name, name + ":noisy"):
            prepared = with_inputs(prepared_set(variant, ball_models))
            parts.append(training_rows(prepared))
            sets.append(dict(labels=prepared["labels"], halves=prepared["halves"]))
    rows = [np.concatenate([part[k] for part in parts]) for k in range(4)]
    weights = [np.concatenate([np.full(len(part[k]), 1.0 / len(part[k])) * 1e4 for part in parts]) for k in (0, 2)]
    models = fit_models(rows[0], rows[1], weights[0], rows[2], rows[3], weights[1])
    return models, duration_model(sets), dict(rows=rows, weights=weights, n_sets=len(parts))


def label_target(prepared, models, durations, rows, params, bonus):
    # labels for one prepared set, with adaptation when params["adapt"], and the logits of the final models
    team_logit, dead_logit = logits(models, prepared["team_x"], prepared["dead_x"], params["mix"])
    team_logit = team_logit + params["tag_weight"] * bonus
    labels = decode(team_logit, dead_logit, prepared["halves"], durations, params["scale"])
    if not params["adapt"]:
        return labels, team_logit, dead_logit
    extra = pseudo_rows(prepared["team_x"], prepared["dead_x"], team_logit, dead_logit, labels)
    x = [np.concatenate([rows["rows"][k], extra[k]]) for k in range(4)]
    w = [np.concatenate([rows["weights"][0], np.full(len(extra[0]), 1e4 / max(1, len(extra[0])))]),
         np.concatenate([rows["weights"][1], np.full(len(extra[2]), 1e4 / max(1, len(extra[2])))])]
    adapted = fit_models(x[0], x[1], w[0], x[2], x[3], w[1])
    team_logit, dead_logit = logits(adapted, prepared["team_x"], prepared["dead_x"], params["mix"])
    team_logit = team_logit + params["tag_weight"] * bonus
    return decode(team_logit, dead_logit, prepared["halves"], durations, params["scale"]), team_logit, dead_logit


def evaluate(names, scales, mixes, adapts, workers):
    # leave one practice match out: train on the other 9 (clean and noisy), label the left-out match's noisy copy
    # and its clean version, score against its provider labels (gt) and the relabelled ones (relabel)
    prepare_all([variant for name in PRACTICE for variant in (name, name + ":noisy")], workers)
    ball_models = ball_fill.load_models()
    rows = []
    for name in names:
        begin = time.time()
        models, durations, train_rows = train([other for other in PRACTICE if other != name], ball_models)
        for variant in (name + ":noisy", name):
            prepared = with_inputs(prepared_set(variant, ball_models))
            gt_codes, relabel_codes = pm.label_codes(prepared["gt"]), pm.label_codes(prepared["labels"])
            for adapt in adapts if variant.endswith(":noisy") else [0]:
                for mix in mixes:
                    for scale in scales:
                        params = dict(DEFAULTS, scale=scale, mix=mix, adapt=adapt)
                        labels = label_target(prepared, models, durations, train_rows, params,
                                              np.zeros(len(prepared["gt"])))[0]
                        for target, codes in (("gt", gt_codes), ("relabel", relabel_codes)):
                            result = pm.evaluate(codes, pm.label_codes(labels), bootstrap_n=0)
                            rows.append(dict(set=variant, target=target, scale=scale, mix=mix, adapt=adapt,
                                             spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                                             poss=result["n_pred_spells"], gt_poss=result["n_gt_spells"]))
        print("{} ({:.0f} s)".format(name, time.time() - begin), flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(CACHE_DIR, "segment_eval.csv"), index=False)
    table = pd.DataFrame(rows)
    table["noisy"] = table["set"].str.endswith(":noisy")
    pd.set_option("display.width", 200)
    print(table.groupby(["noisy", "target", "adapt", "mix", "scale"])[["spellF1", "macroF1", "poss", "gt_poss"]]
          .mean().round(3).to_string())


def run(params, workers, out_name):
    # fits on all practice sets, labels our match, writes the submission csv
    prepare_all([variant for name in PRACTICE for variant in (name, name + ":noisy")] + ["public"], workers)
    ball_models = ball_fill.load_models()
    models, durations, train_rows = train(list(PRACTICE), ball_models)
    prepared = with_inputs(prepared_set("public", ball_models))
    labels = label_target(prepared, models, durations, train_rows, params, tag_bonus(len(prepared["gt"])))[0]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "{}.csv".format(out_name))
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    kept = labels[labels != "X"]
    parts = np.split(kept, np.flatnonzero(kept[1:] != kept[:-1]) + 1)
    print("H {} A {} D {}, {} possessions".format(np.sum(labels == "H"), np.sum(labels == "A"), np.sum(labels == "D"),
                                                  sum(1 for part in parts if part[0] in "HA")))
    print(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["run", "evaluate", "prepare"], default="run")
    parser.add_argument("--sets", nargs="+", default=list(PRACTICE), help="evaluate: the left-out matches")
    parser.add_argument("--scales", type=float, nargs="+", default=[DEFAULTS["scale"]])
    parser.add_argument("--mixes", type=float, nargs="+", default=[DEFAULTS["mix"]])
    parser.add_argument("--adapts", type=int, nargs="+", default=[DEFAULTS["adapt"]])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--name", default="segment_control", help="submission name under evaluation/results")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_all([variant for name in PRACTICE for variant in (name, name + ":noisy")] + ["public"], args.workers)
    elif args.mode == "evaluate":
        evaluate(args.sets, args.scales, args.mixes, args.adapts, args.workers)
    else:
        run({key: getattr(args, key) for key in DEFAULTS}, args.workers, args.name)

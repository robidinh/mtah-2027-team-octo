import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import ROOT, CACHE_DIR, BALL, HOME, GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
MODEL_PATH = os.path.join(CACHE_DIR, "ball_fill_model.pkl")
FIT_SETS = ("metrica1", "metrica2", "metrica3", "DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH",
            "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")
LONG_GAP = 60.0

# Ball position where the tracking has no ball, from the ball before and after the gap and the players around it.
# Gap features, per frame without the ball inside a half
# - before / after: seconds since the last seen ball and until the next one in the half, LONG_GAP when there is none
# - line: the seen ball positions on both sides, joined by a straight line in time; one side only when the other is
#   missing; the line point is the reference of every offset below
# - line point x, y; the jump between the two seen positions dx, dy
# - the 4 nearest players to the line point: offset dx, dy, team sign (+1 H, -1 A), velocity vx, vy
# - player centroid weighted by exp(-d / 10 m) around the line point, offset, all players and per team
# - players within 5, 10 and 20 m of the line point, per team
# Model
# - two gradient boosted regressors for the ball offset from the line point, x and y, divided by offset_scale,
#   so the offset goes to 0 next to a seen ball
# - fitted on the noisy copies (datasets.noisy_copy) of the practice matches, where the true ball is known,
#   every 3rd gap frame, gaps the practice tracking has itself are left out
# - pickled under .local/cache, `python src/common/ball_fill.py fit` writes it again
# Filling
# - filled ball = line point + predicted offset, only inside the halves; speed column stays NaN
# - gap_seconds: min(before, after), 0 where the ball is seen


def gap_features(tracking, team, halves):
    # (frames, features, line) for the frames without the ball inside the halves
    ball = tracking[:, BALL, :2].astype(np.float64)
    players = tracking[:, :BALL, :2].astype(np.float64)
    velocity = np.full_like(players, np.nan)
    velocity[5:-5] = (players[10:] - players[:-10]) * FPS / 10.0
    sign = np.where(team == HOME, 1.0, -1.0)

    frames, before, after, start_xy, end_xy = [], [], [], [], []
    for half, first, last in halves:
        missing = np.isnan(ball[first:last + 1, 0])
        if not missing.any():
            continue
        seen = np.flatnonzero(~missing)
        gaps = np.flatnonzero(missing)
        # previous / next seen index into seen for every missing frame
        following = np.searchsorted(seen, gaps)
        has_before = following > 0
        has_after = following < len(seen)
        previous = seen[np.clip(following - 1, 0, max(0, len(seen) - 1))] if len(seen) else gaps
        upcoming = seen[np.clip(following, 0, max(0, len(seen) - 1))] if len(seen) else gaps
        frames.append(first + gaps)
        before.append(np.where(has_before, (gaps - previous) / FPS, LONG_GAP))
        after.append(np.where(has_after, (upcoming - gaps) / FPS, LONG_GAP))
        start_xy.append(np.where(has_before[:, None], ball[first + previous], ball[first + upcoming]))
        end_xy.append(np.where(has_after[:, None], ball[first + upcoming], ball[first + previous]))
    if not frames:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 37)), np.zeros((0, 2))
    frames, before, after = np.concatenate(frames), np.concatenate(before), np.concatenate(after)
    start_xy, end_xy = np.concatenate(start_xy), np.concatenate(end_xy)
    # a half without any seen ball has no line, the pitch centre stands in
    length, width = GAMES[GAME]["pitch"]
    start_xy = np.where(np.isnan(start_xy), [length / 2, width / 2], start_xy)
    end_xy = np.where(np.isnan(end_xy), [length / 2, width / 2], end_xy)
    share = np.where((before < LONG_GAP) & (after < LONG_GAP), before / np.maximum(before + after, 1e-9),
                     np.where(before < LONG_GAP, 0.0, 1.0))
    line = start_xy + (end_xy - start_xy) * share[:, None]

    p = players[frames] - line[:, None, :]
    v = velocity[frames]
    distance = np.hypot(p[..., 0], p[..., 1])
    distance = np.where(np.isnan(distance), np.inf, distance)
    order = np.argsort(distance, axis=1)[:, :4]
    rows = np.arange(len(frames))[:, None]
    near = np.isfinite(distance[rows, order])
    columns = [before, after, line[:, 0], line[:, 1], end_xy[:, 0] - start_xy[:, 0], end_xy[:, 1] - start_xy[:, 1]]
    for k in range(4):
        found = near[:, k]
        columns += [np.where(found, p[rows[:, 0], order[:, k], 0], np.nan),
                    np.where(found, p[rows[:, 0], order[:, k], 1], np.nan),
                    np.where(found, sign[order[:, k]], np.nan),
                    np.where(found, v[rows[:, 0], order[:, k], 0], np.nan),
                    np.where(found, v[rows[:, 0], order[:, k], 1], np.nan)]
    weight = np.where(np.isfinite(distance), np.exp(-np.where(np.isfinite(distance), distance, 0) / 10.0), 0.0)
    for mask in (np.ones(len(team), dtype=bool), team == HOME, team != HOME):
        w = weight * mask
        total = w.sum(axis=1)
        for axis in (0, 1):
            columns.append(np.where(total > 1e-6, np.nansum(w * np.nan_to_num(p[..., axis]), axis=1)
                                    / np.maximum(total, 1e-6), np.nan))
    for mask in (team == HOME, team != HOME):
        for radius in (5, 10, 20):
            columns.append(np.sum((distance <= radius) & mask, axis=1).astype(np.float64))
    return frames, np.stack(columns, axis=1), line


def offset_scale(features):
    # seconds to the nearest seen ball plus 0.2 s, at most 10.2 s; the models predict the offset divided by it
    return np.minimum(np.minimum(features[:, 0], features[:, 1]), 10.0) + 0.2


def fit(names, seed):
    import datasets
    from sklearn.ensemble import HistGradientBoostingRegressor
    parts_x, parts_y = [], []
    for name in names:
        tracking, team, halves, gt = datasets.load_variant(name + ":noisy", seed=seed)
        clean = datasets.load_variant(name, seed=seed)[0]
        frames, features, line = gap_features(tracking, team, halves)
        truth = clean[frames, BALL, :2].astype(np.float64)
        keep = np.flatnonzero(~np.isnan(truth[:, 0]))[::3]
        parts_x.append(features[keep])
        parts_y.append((truth[keep] - line[keep]) / offset_scale(features[keep])[:, None])
        print("{}: {} gap frames".format(name, len(keep)), flush=True)
    x, y = np.concatenate(parts_x), np.concatenate(parts_y)
    models = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                            early_stopping=False, random_state=0).fit(x, y[:, axis])
              for axis in (0, 1)]
    return models


def fill(tracking, team, halves, models):
    # (filled tracking copy, gap_seconds)
    out = tracking.copy()
    gap_seconds = np.zeros(len(tracking))
    frames, features, line = gap_features(tracking, team, halves)
    if len(frames):
        offset = np.stack([model.predict(features) for model in models], axis=1) * offset_scale(features)[:, None]
        out[frames, BALL, 0] = line[:, 0] + offset[:, 0]
        out[frames, BALL, 1] = line[:, 1] + offset[:, 1]
        gap_seconds[frames] = np.minimum(features[:, 0], features[:, 1])
    return out, gap_seconds


def load_models():
    # the fitted models, fitted and written first when MODEL_PATH does not exist
    if not os.path.exists(MODEL_PATH):
        models = fit(FIT_SETS, seed=0)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as handle:
            pickle.dump(models, handle)
    with open(MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    # fit: fit on FIT_SETS and write MODEL_PATH
    # test <names>: leave each named practice set out, fit on the rest, print the error of the line point and of
    #   the model on the left-out set's noisy copy, median and 90th percentile in metres, by gap length
    import datasets
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "fit":
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        load_models()
        print(MODEL_PATH)
        sys.exit()
    for name in sys.argv[2:] or ["DFL-MAT-J03WR9", "metrica2"]:
        begin = time.time()
        models = fit([other for other in FIT_SETS if other != name], seed=0)
        tracking, team, halves, gt = datasets.load_variant(name + ":noisy", seed=1)
        clean = datasets.load_variant(name, seed=1)[0]
        frames, features, line = gap_features(tracking, team, halves)
        filled, gap_seconds = fill(tracking, team, halves, models)
        truth = clean[frames, BALL, :2]
        known = ~np.isnan(truth[:, 0])
        line_error = np.hypot(*(line - truth).T)[known]
        model_error = np.hypot(*(filled[frames, BALL, :2] - truth).T)[known]
        length = np.minimum(features[:, 0], features[:, 1])[known]
        live = np.isin(gt[frames][known], ["H", "A"])
        print("{} ({:.0f} s)".format(name, time.time() - begin))
        for low, high in ((0, 1), (1, 3), (3, 10), (10, 1e9)):
            for state, mask in (("live", live), ("dead", ~live)):
                pick = mask & (length >= low) & (length < high)
                if pick.sum() < 50:
                    continue
                print("  {:>4}-{:<4} s {} {:>6} frames  line {:5.1f} / {:5.1f} m   model {:5.1f} / {:5.1f} m".format(
                    low, high if high < 1e9 else "", state, pick.sum(), np.median(line_error[pick]),
                    np.percentile(line_error[pick], 90), np.median(model_error[pick]),
                    np.percentile(model_error[pick], 90)))

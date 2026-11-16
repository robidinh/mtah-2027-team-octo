# segment_control

Labels each frame H, A or D with gradient boosted trees and a path search that knows how long possessions and
stoppages last. The models are trained on practice matches made to look like our match.

## How it works

- Practice data: the 10 practice matches (3 Metrica, 7 IDSSE), each clean and as a noisy copy.
  - The noisy copy (`datasets.noisy_copy`) has the tracking faults measured on our match:
    - the ball is missing on ~70% of dead frames and ~15% of live frames, in long runs
    - player gaps, ~7.6% of player frames
    - an extra person counted for the wrong team on ~13% of frames
  - Labels are relabelled to the challenge rules (`datasets.relabel`): a possession under 1 s inside a live
    stretch is a touch and goes to the team before it.
- The ball in the gaps is filled first (`src/common/ball_fill.py`):
  - from the seen ball before and after the gap and the players around it
  - trained on the noisy practice copies, where the true ball is known
- Per-frame signals, all computed on the filled ball:
  - team: the signals of `signals.py`, `intents.py`, `boundary_signals.py`, and the set piece teams of
    `setpiece_signals.py`
  - dead: the dead signals of `signals.py`, ball missing, gap length, players seen, idle players, the dead frames
    of `deadball.py`, set piece shapes (throw-in, corner, goal kick, wall, kickoff)
  - each also as its mean over 1 s and 4 s before and after the frame
- Models:
  - team: P(H | ball alive), trees; the features are also fed negated with the label swapped, so H and A are alike
  - dead: P(D), trees
  - a logistic regression on the same features can be mixed in (`mix`)
- Path: Viterbi over (label, age) states, age up to 6 s.
  - the chance that a possession or stoppage ends depends on its age, learned from the practice labels
  - per-frame evidence is scaled by `scale`
- Smart Tagging, our match only: a set piece tag adds `tag_weight` to the tagged team for 5 s from the tag start.
  No practice match has tags, so this weight is set, not tested.
- Adaptation (`adapt`): frames where the first path and the models agree strongly become pseudo labels, the models
  are fitted again with them and the path is found again. No human labels.

## Run

```bash
python src/algo/segment_control/segment_control.py
```

- Writes `evaluation/results/segment_control.csv`.
- The first run builds the ball model and the feature caches under `.local/cache` (~20 min).
- Needs the practice data, see `src/tools/download_external.py`, and scikit-learn.

```bash
python src/algo/segment_control/segment_control.py evaluate --scales 0.1 0.2 0.4 --mixes 0 0.5 --adapts 0 1
```

- Leaves each practice match out in turn, trains on the other 9, scores the left-out match.
- Scores against the provider labels (`gt`) and the relabelled ones (`relabel`).

## Settings

- `DEFAULTS`: scale 0.4, mix 0, adapt 0, tag_weight 2.
- Chosen by leaving each practice match out in turn, never on our match or its public 5 minutes.
- mix 0.5 was below mix 0 at every scale, so the logistic regression is not used.
- adapt 1 gave the same mean as adapt 0 (0.637 vs 0.638 on the noisy copies, 5 of 10 up), so it is off.

## Results

Mean spellF1 over the 10 practice matches, each left out of training in turn, scale 0.4, mix 0.

| | noisy copies, provider labels | noisy, relabelled | clean, provider labels |
|---|---|---|---|
| segment_control | **0.638** | 0.694 | 0.703 |
| src_combine (our first, rule-based version) | 0.405 | | |

- The ball model was trained on all 10 practice matches. It sees the ball positions of the left-out match, not its
  labels.
- Our match: 320 possessions, mean 12.6 s, 37% dead. The file is the same on every run.
- Public 5 minutes, a check only: spellF1 0.644, macroF1 0.743, 19 possessions (17 true). Alive / dead F1 0.92,
  but A F1 only 0.48: A frames called H.

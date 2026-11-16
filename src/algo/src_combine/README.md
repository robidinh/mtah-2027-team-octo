# src_combine

Labels each frame H, A or D with hand-set rules. This is our first pipeline. It makes the submission.

## How it works

- Dead frames come from `src/common/deadball.py`:
  - a frame is dead when the players walk and the ball is not played
  - a stoppage starts where the players slow down and ends at the restart kick
  - a tracked ball clearly outside the pitch is dead
- Control evidence comes from `control.py`:
  - a team controls the ball when one of its players is within 2 m and no opponent is as close
  - a fast ball, a ball nobody is near, or an untracked ball is no evidence
- `src_combine.py` joins them:
  - dead frames are D
  - a live stretch starts with the first team seen in control
  - possession changes only when the other team keeps control for `MIN_CONTROL`
  - with no evidence the ball stays with the last team
  - a change hidden in a ball gap is put in the middle of the gap
- Frames outside the graded halves are X.
- All thresholds are set from football, not fitted on the ground truth.

## Run

```bash
python make_submission.py
```

- Writes `evaluation/results/our_model.csv`.

```bash
python src/algo/src_combine/src_combine.py
```

- Prints a summary of the possessions.

## Results

Mean spellF1 on the practice matches. Metrica is the mean of games 1 to 3. Our match = the public 5 min, a check only.

| settings | IDSSE | IDSSE, ball gaps | Metrica | our match |
|---|---|---|---|---|
| fixed | 0.447 | 0.324 | 0.573 | 0.491 |

- macroF1 on our match is 0.877.

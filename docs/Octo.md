# Octo - ball possession, More than a Hack 2027

Team Octo: Dinh Quocbao Róbert, Gegő Levente, Krebs Áron, Dózsa Domonkos, Újváry Áron

Short version:
- for every frame we say H / A / D from the tracking data, nothing labelled by hand
- two models (gradient boosted trees): "is the ball dead?" and "which team has it?"
- then a path search over the whole half that knows how long possessions and stoppages usually last, so the output doesnt flicker
- we cant test on our own match (only 5 min of answers), so we train and test on **10 other full matches** that we made look like our messy tracking
- on those 10 (each one left out of training in turn): spellF1 **0.638**, the organisers baseline gets 0.270 on the same data

---

## 1. What we used and what we dropped

| asset | used? | why |
|---|---|---|
| tactical tracking (`_atd.txt` + `.xml`) | yes, main input | players + ball every frame. team per slot from the xml |
| smart tagging | a little | only the set piece tags with a team (corner, goal kick, free kick): small push to that team for 5 s from the tag start, thats the team taking the restart. the other tags we dropped, see below |
| homography | no | only needed to draw on the video or to do our own detection, we didnt do that |
| tactical video | no (only watched it to understand errors) | no video model, too much work for too little and no labels to train on |
| broadcast files | no | own timeline with cuts and replays, no mapping to the tactical frames |
| public answers (first 5 min) | only as a check | to make sure the pipeline is aligned and the file is right. nothing is tuned on it |
| Metrica open sample data, 3 matches | yes, training + testing | other games, allowed. github.com/metrica-sports/sample-data |
| DFL / IDSSE open data, 7 Bundesliga matches | yes, training + testing | other games, allowed. see credits |

Smart tagging tags we dropped:
- team phase tags (BUILD UP, PROGRESSION, ...) come in mirror pairs (attacking tag for one team = defending tag for the other at the same time) and they also sit on a lot of dead ball time, so they dont tell alive vs dead
- set piece durations are fixed (5 s or 35 s) so only the start time means something
- throw-ins and kickoffs have no team in the tags

---

## 2. How we read the possession rules

- ball alive -> exactly one team has it, no "contested" label
- possession only changes when the other team **controls** it. a touch, a deflection or a won header without control is not a change
- loose ball stays with the last team
- dead ball = D, never a team
- 2nd half: frames before the kickoff (113,742) are graded and always D (FAQ), we force that

How this goes into the method:
- the practice matches label possession their own way. Bundesliga changes possession on every touch (median possession ~3 s), the challenge doesnt (public 5 min: 17 possessions, ~14 s avg)
- so before training we **relabel** the practice labels with the challenge rule: inside a live stretch, a possession under 1 s is a touch and goes to the team before it. the first run after a restart is kept (thats the restart team)

---

## 3. The method

### 3.1 Practice matches made to look like ours
Our tracking is automatic and messy, the practice tracking is clean. if we train on clean data the model learns clues that dont exist in our match. so every practice match is used twice: clean, and as a **noisy copy** with the faults we measured on our match (measured from the tracking only, no answers involved):
- ball missing: ~70% of dead frames, ~15% of live frames, in long runs like ours
- player tracks cut into pieces, ~7.6% of player frames missing
- an extra person counted for the wrong team on ~13% of frames (happens in our tracking)

### 3.2 Filling the ball when its not tracked
the ball is missing on ~40% of graded frames in our match, often for 10+ seconds. a small model (gradient boosted regression) guesses where the ball is in a gap, from:
- where it was seen before and after the gap
- where the players around it are and how they move

trained on the noisy practice copies, where we know the real ball position. every ball clue below uses the filled ball.

### 3.3 Clues per frame
**Which team has it** (around 30 clues), for example:
- nearest player, players within 5 m, who gets to the ball first, who is closing in
- ball moving towards which goal, which team is pushing up
- carries, receives, pressing, passes (kicker and receiver)
- set piece shapes from player positions: who stands at the throw-in line, at the corner, in the goal area, who makes the wall, kickoff

**Is the ball dead** (around 17 clues), for example:
- players walking / slowing down, team compact, ball slow or still, ball off the pitch
- ball missing and for how long, how many players are seen
- our first rule-based dead ball detector (players walking + nobody kicking, or ball off the pitch)
- the set piece shapes above

every clue also as its average over 1 s and 4 s before and after the frame.

### 3.4 Two models
- **dead**: P(dead) per frame
- **team**: P(home | ball alive) per frame. every training row is also fed mirrored (clues flipped, label swapped), so the model cant learn "home has the ball more", it has to read the clues
- both are gradient boosted trees (scikit-learn), trained on the 10 practice matches, clean + noisy

### 3.5 Path search (why it doesnt flicker)
- per half we pick the best whole sequence of H / A / D (Viterbi), not frame by frame
- the states also remember how long the current possession / stoppage has lasted (up to 6 s). the chance that it ends at that age is learned from the practice labels. so a 0.2 s "possession" is very unlikely and gets swallowed, a 2 s one can stay
- allowed moves: H <-> A, H/A -> D, D -> H/A
- one setting, `scale` (how much the per frame clues count vs the length model), see 4

### 3.6 Output
- path inside the halves, X outside (not graded)
- one command makes the csv, same file on every run (fixed seeds, no randomness)

---

## 4. How we picked the settings

- **never on our match.** we only have 5 min of answers, tuning on that picks the wrong thing (the starter repo shows that too)
- instead: leave one practice match out, train on the other 9, score the left out match with the official scorer, repeat for all 10
- `scale`: tested 0.1, 0.2, 0.4, 0.6, 0.8, 1.0. kept **0.4**:
  - vs the practice labels as they come, 0.8 is a tiny bit higher (0.644 vs 0.638, noise)
  - vs the practice labels relabelled to the challenge rules (a touch isnt a change), 0.4 is best (0.694 vs 0.687-0.692)
  - higher scale = more and shorter possessions, which fits Bundesliga's "every touch counts" labels but not the challenge. so 0.4
- path memory (how long a possession / stoppage is tracked by age): 6, 10, 15 s -> 0.638 / 0.639 / 0.640, noise, kept 6 s
- logistic regression mixed into the trees: worse at every scale, not used
- re-training on our own match with its confident frames as pseudo labels: same score, not used (simpler)
- smart tag weight: cant be tested, no practice match has tags. set to a small push, and we checked how much it changes our file: 111 frames (0.07% of graded frames), same 320 possessions. so its basically harmless either way

---

## 5. Results

Mean spellF1, 10 practice matches, each left out of training:

| | noisy copies | clean |
|---|---|---|
| ours | **0.638** | 0.703 |
| baseline (smooth50) | 0.270 | 0.341 |

the baseline gets 0.243 on the real full match (organisers number), close to its 0.270 on the noisy copies, so the noisy copies look like a fair stand-in for our match

Our match:
- TODO possessions, avg TODO s, TODO% dead
- Barca has the ball 61% of the live time, normal for Barca
- public 5 min (check only, way too small to trust): spellF1 0.644, 19 possessions (true 17)

How we got here (earlier versions, same practice matches):
- first rule-based pipeline (dead ball rules + "player within 2 m and no opponent close" + state machine): 0.485 vs baseline 0.341 on the clean matches
- it was too slow to switch team and missed short possessions, thats why we moved to models + path search
- the rule-based dead ball detector is still one of the clues

---

## 6. Limitations

- practice matches arent our match. the noisy copies copy the faults we could measure, not all of them (eg slot swaps between players)
- the practice answers come from different sources (Metrica from events, Bundesliga from tracking flags), and we relabel them with our reading of the rules. if the organisers count possession a bit differently, we're off a bit
- smart tag weight is set, not tested
- on the public 5 min we give some Athletic possessions to Barca. could be noise (5 min) or a real bias, we couldnt check without more answers
- stoppage edges (when exactly a dead ball starts / ends) are the hardest part when the ball isnt tracked
- the first run takes ~20 min (builds caches), after that a few minutes

---

## 7. How to reproduce

```
setup.bat                                  (or: python -m venv .venv + pip install -r requirements.txt)
.venv\Scripts\activate
python make_submission.py                  -> Octo.csv
python evaluation/validate_submission.py Octo.csv
```

Data you need (not in the repo):
- match data into `data/tactical/` (from the organisers)
- Metrica: `git clone https://github.com/metrica-sports/sample-data data/external/metrica-sample`
- DFL / IDSSE: the `DFL_*.xml` files from https://huggingface.co/datasets/pysport/idsse-data into `data/external/idsse/`

TODO: exact package versions, run time, check on a clean machine

---

## 8. Rules we stuck to

- no hand labels on our match, in any amount
- no outside data about Barca - Athletic
- nothing tuned on the public 5 min
- match data never in the repo

## Credits

- Metrica Sports open sample data, https://github.com/metrica-sports/sample-data
- DFL Deutsche Fussball Liga, IDSSE open data (CC-BY 4.0): Bassek, M., Weber, H., Rein, R., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data. https://doi.org/10.1038/s41597-025-04505-y
- kloppy, scikit-learn, numpy, pandas

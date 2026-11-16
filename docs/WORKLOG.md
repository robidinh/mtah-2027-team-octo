# Worklog

Short version:
- organisers gave us the data, a simple baseline and the scorer
- we built our own pipeline on top: dead ball -> control -> combiner -> csv
- we test it on 10 OTHER full matches (3 metrica + 7 bundesliga), thats allowed and its the only real way to measure
- final method = segment_control (3.17): trees + path search, trained on the 10 practice matches made messy like ours. **0.638** spellF1 on them (left out in turn), our first rules version gets 0.405 on the same messy copies
- first rules version on the clean practice matches: 0.485 vs baseline 0.341 (3.1-3.16)
- public 5 min of our match is only a sanity check, 5 min is too little to tune on
- every change below is what / why / result, newest at the bottom. section 4 = whats still open, section 5 = how to run everything

Base is the official starter repo, commit `f8f2629`. Anything not in the "given" list is ours.

---

## 1. What was given vs what we added

### Given by the organisers (not our code)
| Item | What it is |
|---|---|
| `data/tactical/` | match data: tracking, smart tags, homography, video |
| `evaluation/gt/..._public.csv` | answers for the first 5 minutes only |
| `starter/baseline.py` | simple solution, nearest player + smoothing |
| `evaluation/` | scorer (`run_evaluation.py`) and format check (`validate_submission.py`) |
| `games.py`, `starter/*.py`, notebooks | data readers and examples |
| the `*.md` docs | rules, task, scoring |

### Added by us
| File | What it does |
|---|---|
| `src/common/load.py` | reads the data into arrays, fixes the frame shift |
| `src/common/deadball.py` | finds when the ball is dead (D) |
| `src/algo/src_combine/control.py` | finds which team has the ball at its feet |
| `src/algo/src_combine/src_combine.py` | puts it together into one H/A/D label per frame |
| `make_submission.py` | one command to make the whole csv (`Octo.csv`) |
| `src/algo/segment_control/` | **the final method**: tree models + path search (3.17) |
| `src/common/tracking.py`, `idsse.py`, `datasets.py` | load our match and the practice matches into one format, make the noisy copies, relabel |
| `src/common/ball_fill.py` | guesses the ball position when its not tracked |
| `src/common/signals.py`, `intents.py`, `boundary_signals.py`, `setpiece_signals.py` | the clues per frame (who has it, is it dead, set piece shapes) |
| `src/common/bench.py` | test bench on other matches (metrica samples + DFL) |
| `src/common/dfl.py` | reads the DFL / Bundesliga practice matches |
| `src/tools/download_external.py` | downloads the practice data |
| `docs/WORKLOG.md` | this file |
| `docs/Octo.md` | the technical document |
| `requirements.txt` | exact package versions we used |
| `.gitignore` additions | keeps `.venv/`, `.local/`, caches, practice data, `*.mp4`, `*.pkl`, `*.parquet` off github |

---

## 2. Our method vs the baseline

| | Baseline | Ours | Why |
|---|---|---|---|
| Who has the ball | nearest players team, always (even if hes 20 m away) | only if a player is within 2 m and every opponent is at least half a step (0.5 m) further away | nearest doesnt mean in control |
| Ball in the air / loose | goes to whoever is nearest | stays with the last team | thats the official rule |
| When does possession change | majority vote over a window | other team has to control it for 0.4 s, OR completes a pass between 2 of its players. switch is put at the start of that | a touch or deflection isnt a change, and voting smears the boundaries |
| Dead ball | ball not visible = dead | actual detector: players walking + nobody kicking, ball off the pitch, 2nd half kickoff | the ball is also invisible a lot during live play |
| Ball invisible during play | dead, so possessions get cut up | stays with last team, if it switches inside a gap we put it in the middle | less fake possessions |

Public 5 min (just a correctness check):

| | spellF1 | possessions (true = 17) |
|---|---|---|
| ours | 0.491 | 18 |
| baseline smooth125 | 0.424 | 16 |
| baseline smooth50 (official floor) | 0.326 | 26 |
| baseline raw | 0.038 | 131 |

17 possessions is way too few to prove anything, bootstrap range for ours is 0.39-0.72.

On 10 other full matches (the real test, see 3.7-3.16):

| | ours | baseline smooth50 |
|---|---|---|
| metrica (3 games) | 0.573 | 0.419 |
| bundesliga (7 games) | 0.448 | 0.308 |
| **all 10** | **0.485** | **0.341** |

---

## 3. Changes

### 3.1 Loading the data (`src/load.py`)
- **What:** reads tracking + smart tags into arrays in metres, one row per video frame. The tracking frame_id is video frame + 1, we fix that here and only here.
- **Why:**
  - one place for the shift so we dont get off by one bugs everywhere
  - metres so we can use real football numbers (speeds, distances)
  - cached in `data/cache/`, loads in a few seconds instead of ~1 min
- **Result:** checked alignment, best match is at shift 0, worse at +-1 s. ok

### 3.2 Dead ball detector (`src/deadball.py`)
- **Rule:** a frame is dead if
  - players average speed < 1.5 m/s (walking) AND no kick (ball > 7 m/s) within +-1 s
  - OR the ball is more than 1 m outside the pitch
  - cleanup: stoppages under 3 s dropped, gaps under 3 s merged, start moved back to where players started slowing down (max 2 s), end pushed to the restart kick
- **Why:**
  - when the ref stops play players slow down to walking within 1-2 s
  - nobody kicks a dead ball, and the restart is a kick
  - ball over the line is dead by definition
  - no real stoppage is shorter than ~3 s
  - all the numbers come from football, we did not tune them on the public answers
- **Result:**
  - 131 stoppages, usually ~14 s, looks realistic
  - 53 of 58 set piece tags start inside one of our stoppages (independant check)
  - public window: 98% of what we call dead really is dead, we catch 84% of the real dead frames

### 3.3 Fix: 2nd half before kickoff is always dead (`src/deadball.py`), 2026-11-03
- **What:** frames 110,808 to 113,741 are forced to D.
- **Why:**
  - these frames are graded and always D (says so in FAQ.md)
  - warm up jogging and passing looked like play, only 61% got caught
  - the kickoff frame is from the official docs, not from hidden answers, so its allowed
- **Result:** 100% caught now, ~1,100 frames fixed.

### 3.4 Fix: ignore team speed when under 10 players are visible (`src/deadball.py`), 2026-11-03
- **What:** if fewer than 10 players are tracked we take the team speed from the nearest frames that have enough. Before, 0 players tracked counted as everyone standing still.
- **Why:** average of 3-5 players isnt the team, and "nobody seen" isnt the same as "standing".
- **Result:** basically no change, those moments were real stoppages anyway. Kept it because its more correct.

### 3.5 Control detector (`src/control.py`), 2026-11-03
- **Rule:** team T controls the ball if
  - ball is within 2 m of a T player
  - no opponent within 2 m
  - ball slower than 7 m/s (not flying)
  - anything else (in flight, loose, contested, not visible) = no evidence
- **Why:**
  - a player can reach the ball within ~1 m, tracking is body centre and a bit noisy so 2 m
  - a fast ball is a pass or shot, nobody controls it
  - loose / contested ball tells us nothing, rules say it stays with the last team
- **Result:** evidence on 32% of live frames, Barca 65% / Athletic 35%, which is about what youd expect from Barca.

### 3.6 Combiner (`src/combine.py`) + `make_submission.py`, 2026-11-03
- **Rules:**
  - dead -> D, outside the graded halves -> X
  - a live stretch starts with the first team seen in control (the team taking the restart)
  - possession only changes when the other team has at least 10 control frames (0.4 s) within 2 s
  - the switch goes on the first frame of the new control
  - if the ball was invisible for 1 s or more before the switch, the switch goes in the middle of that gap
  - live stretch with no evidence at all keeps the team from before the stoppage
- **Why:**
  - official definition: only controlled possession by the other team changes it
  - 0.4 s is about a receive + touch, a deflection is shorter
  - putting the switch at the start (not after the 0.4 s check) means no late boundaries, which spellF1 punishes a lot
  - middle of an invisible gap is the smallest worst case error
- **Result:**
  - full match: H 32%, A 19%, D 49%, 296 possessions (~14 per 5 min), median 7 s
  - ran it twice, identical file. passes the format check
  - public window: see section 2

### 3.7 Test bench on Metrica sample matches, 2026-11-03
- **What:** 3 anonymous full matches with tracking + event data from github.com/metrica-sports/sample-data. Stored in `data/external/`, never uploaded.
- **Why:**
  - only legal way to test on full matches, data about *other* games is allowed (RULES.md §2)
  - outside data about Barca-Athletic is not allowed, not even just to check
  - hundreds of possessions instead of 17 so we can actually see if a change helps
- **Plan:** make answers from their events with code -> run our pipeline -> score it.
- **Differences to our match:**
  - ball visible ~60% (ours ~59%) so its a fair test for the ball gaps
  - their player tracking is cleaner (fixed ids, all 22 visible) so their scores are more like an upper limit for us
  - no speed column, we calculate it

### 3.8 Test bench built (`src/bench.py`), 2026-11-03
- **What:**
  - loads sample games 1 and 2 into the same format as our match so the pipeline runs unchanged
  - makes the answers from the events with code:
    - team gets the ball with a pass, shot, set piece or recovery
    - a recovery + ball lost on the same frame is just a touch, doesnt count
    - in between the ball stays with the last team
    - dead = from the last ball event before a set piece until the set piece
  - runs ours and baseline_smooth50, scores both with the official scorer
  - also changed: the 2nd half kickoff frame and pitch size now come with the data instead of being hard coded (otherwise the sample games got half their 2nd half forced dead). our csv is byte identical after this change, checked
- **Why:** full matches, ~250 possessions each instead of 17, all legal.
- **Result:**

| | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| ours | 0.540 | 0.795 | 0.879 | 0.905 |
| baseline smooth50 | 0.429 | 0.787 | 0.901 | 0.871 |

  - ours wins on spellF1 in both games (0.523 vs 0.430, 0.558 vs 0.428) and on team accuracy
  - **we call too much dead**: 40-44% vs 31-33% in the event answers. confirms the "too much dead" issue (later revised: on the bundesliga games our dead share is right, see 3.12)
  - we find too few possessions: ~210 vs ~260. probably short ones swallowed by fake stoppages
  - baseline is better at dead/alive here, but thats partly because in these games the ball is almost never tracked during dead time (2-9%), so "no ball = dead" works really well. in our match its not like that
  - caveat: answers from events are an approximation of the official definition (which comes from corrected tracking)

### 3.9 Is the sample data ok to use? yes, 2026-11-03
- **Source:** Metrica Sports open sample data, https://github.com/metrica-sports/sample-data
  - game 1: https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_1
  - game 2: https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_2
  - game 3 (different format, read with kloppy): https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_3
  - event definitions pdf: https://github.com/metrica-sports/sample-data/tree/master/documentation
- **What it is:** anonymous real matches (no team / player / competition names), ~95 min each, 25 fps, tracking + event data made by humans
- **Competition rules:** allowed
  - RULES.md: "Data of this kind about other games is fine"
  - RULES.md: "Hand-labelled data from other games is fine, in any amount"
  - conditions: organisers must be able to access it (its public) and the tech doc has to explain it (this log + the doc)
- **Metrica's terms:** no licence file, just "please be responsible with the use of this data. If you use it for anything public, please acknowledge the source"
  - -> **we have to credit them in the tech doc**, something like "validation on Metrica Sports open sample data (github.com/metrica-sports/sample-data)"
- **We dont re-upload it:** its in `data/external/` which is gitignored. `src/bench.py` says how to download it with git clone

### 3.10 Game 3 added to the test bench (`src/bench.py`), 2026-11-03
- **What:**
  - game 3 is in a different format (FIFA EPTS tracking + json events), we read it with kloppy (already in our requirements)
  - it has substitutes in the file (35 players), thats fine, players not on the pitch are just empty
  - game 3 has a new event type CARRY (player running with the ball) -> counts as control too. games 1-2 dont have it so their answers didnt change (same scores as before, checked)
- **Why:** 3 matches instead of 2, so one weird match cant fool us as easily.
- **Result (3 games):**

| game | ours | baseline smooth50 | possessions true / ours | dead: events / ours |
|---|---|---|---|---|
| 1 | 0.523 | 0.430 | 278 / 218 | 33% / 44% |
| 2 | 0.558 | 0.428 | 247 / 209 | 32% / 40% |
| 3 | 0.497 | 0.400 | 378 / 253 | 25% / 31% |
| **mean** | **0.526** | **0.419** | | |

  - we win all 3 games, by ~0.1 every time
  - same two problems in all 3: too much dead, too few possessions
  - game 3 has way more possessions (378) partly because CARRY gives more control moments so short possessions show up more

### 3.11 Tried: fixing "too much dead" with ball movement, reverted, 2026-11-03
- **What we looked at first (bench, 3 games):** fake stoppages vs real ones
  - real: ball moving 5% of the time, players 1.07 m/s
  - fake: ball moving 40% of the time, players 1.32 m/s
  - so player speed barely tells them apart, ball movement does (on the bench)
  - ~25% of the wrong dead time is right next to real stoppages (edges)
- **What we tried:**
  - idea 1: a "stoppage" where the ball is moving > 35% of the time is live play (unless the ball was off the pitch)
  - idea 2: count softer kicks as play (5 m/s instead of 7)
  - also tried: lower walking limit (1.3, 1.1) and no backtracking at the start. both made the bench worse or no better
- **Bench result:** 0.526 -> 0.557, looked good
- **Why we reverted anyway:**
  - in the sample games the ball is basically never tracked when dead, so "ball moving" looks like a perfect clue there
  - in our match the ball is visible on ~30% of stoppage frames and players knock it around during fouls
  - the stoppages the rule removed in our match were untagged ones = mostly fouls (only 5 free kicks are tagged). ball out stoppages were all kept (off pitch exception)
  - also seen in the public 5 min: a real 21 s stoppage with the ball moving 64% of the time got deleted, macroF1 0.874 -> 0.763. we didnt tune on this, it just confirmed what the full match data already showed
- **Status:** code for the rule is still there but switched off (`MAX_MOVING_SHARE = 1.0`, `PASS_SPEED = 7.0`). csv is byte identical to before
- **Lesson:** the bench is good for the team / possession part, but for dead ball rules we have to check if the clue behaves the same in our match. sample data has no ball during stoppages, ours does

### 3.12 More practice matches: DFL / Bundesliga open data (`src/dfl.py`), 2026-11-03
- **What:** 7 full Bundesliga matches (2022/23, 1st and 2nd division), from the IDSSE open dataset
  - download: https://huggingface.co/datasets/pysport/idsse-data (mirror of https://doi.org/10.6084/m9.figshare.28196177, figshare blocks scripts so we use the mirror)
  - licence CC-BY 4.0 -> **credit the DFL and cite the paper** in the tech doc: Bassek, Weber, Rein, Memmert (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data, https://doi.org/10.1038/s41597-025-04505-y
  - stored in `data/external/idsse/`, gitignored, never uploaded
- **Why its better than the metrica samples:**
  - every frame of the tracking has `BallStatus` (alive / dead) and `BallPossession` (which team). so the answers come straight from the tracking, same kind of answers the organisers use for our match. with metrica we had to build them from events (only approximate)
  - 25 fps, real pro matches, dead share 41.5% in the first one (so the "40-45%" we expected for a pro match is right)
  - ball is tracked 100% of the time, also during stoppages. metrica samples are the opposite (ball almost never tracked when dead). our match is in between (~30%). so if a dead ball rule works on both, it should work on ours too. thats exactly what was missing in 3.11
- **Allowed?** yes, other matches (RULES.md §2), public, explained here
- **Differences to our match:** clean tracking (all 22 players, fixed ids), ball always visible. speed we calculate from positions, same as for metrica
- **Result (bench now 10 matches, `python src/bench.py`, ~1 min after the first run):**

| | ours | baseline smooth50 |
|---|---|---|
| metrica (3 games) | 0.526 | 0.419 |
| DFL (7 games) | 0.409 | 0.308 |
| **all 10** | **0.444** | **0.341** |

  - we beat the baseline in **all 10 matches**
  - **dead ball on DFL is good**: alive/dead F1 0.913, our dead share is right (e.g. 41.9% vs true 41.5%). so "too much dead" is NOT a general problem, on metrica its partly how those answers were built from events. our 48.7% on the real match might be closer to ok than we thought (DFL true dead is 41-52%)
  - baseline on DFL scores 0.308, close to its 0.24 on our match -> DFL is a decent stand-in
  - **biggest problem everywhere: too few possessions.** DFL true 370-511 per match, we find ~245
  - careful: DFL's possession flag changes very often (median possession 3 s, 15% under 1 s). our match's public 5 min has 17 possessions, avg 14 s, so the challenge counts possessions less twitchy than DFL. so we should NOT copy DFL's possession style 1:1, otherwise we flicker on our match. DFL is best for dead/alive, for possession we look at both sources

### 3.13 Where do we actually lose points? (measured, not guessed), 2026-11-03
- **What:** on the 3 metrica games we checked every real possession we miss (IoU < 0.5) and why
- **Result:** we get 418 of 903 real possessions (46%). the 485 misses:

| why we miss it | how many | share |
|---|---|---|
| very short possession (under 3 s) | 216 | 45% |
| start / end in the wrong place | 108 | 22% |
| gave it to the wrong team | 81 | 17% |
| a fake stoppage inside it | 80 | 16% |

  - frame level: wrong team only 6% of frames, live called dead 11%, dead called live 3%
  - **correction:** we thought ball gaps (ball not visible) were the biggest problem. measured: only 5% of the missed possessions happen while the ball is mostly invisible. so its NOT the big one
- **Real match, without answers:** compared the shape of our output with the practice games
  - we find 14 possessions per 5 min, the public 5 min have 17 -> probably too few there too
  - 20% of our possessions are under 3 s (practice truth 19-32%)
  - our dead share 48.7%, bundesliga truth is 41-52% -> probably ok
- **Conclusion:** short possessions = biggest weakness, stoppages are mostly fine (edges could be sharper)

### 3.14 Short possessions: passes count as control (kept) + fast switch (dropped), 2026-11-04
- **What (kept):** a pass between 2 players of the same team now switches possession right away, no 0.4 s wait (`control.passes`, `combine.USE_PASSES`)
  - pass = team touches it (3+ frames), ball gets kicked (>7 m/s), lands 5+ m away at the same team within 3 s, no opponent touch in between
- **Why:** passing it to a teammate IS controlled possession by definition. a single touch or deflection still doesnt count (no pass after it)
- **Result:** bench 0.444 -> **0.452**, better or equal on all 10 games (metrica 0.526 -> 0.535, bundesliga 0.409 -> 0.416)
  - our match: public 5 min unchanged 0.553, 298 possessions (was 296), no flicker. csv md5 now `86e47155...`
  - small gain: it mostly makes switches earlier, it barely finds new possessions

- **What (dropped):** fast switch when control is super clear (ball within 1 m of a player, no opponent within 3 m, 0.16 s in a row)
- **Result:** bench 0.452 -> 0.453, some games up some down = nothing. switched off (`FAST_CONTROL = 0`), less code that does nothing = less risk

- **Why short ones are still missed (measured on 10 games, 1550 real possessions under 3 s, we miss 66%):**

| control seen for the team that really had it | missed |
|---|---|
| none at all (967 of them!) | ~80% |
| 1-9 frames | 54-80% |
| 10+ frames | 27% |

  - ball is visible in all of them, so its not ball gaps
  - its contested balls / one touch plays: theres always an opponent within 2 m, so our control rule never fires
  - **so the next fix isnt "switch faster", its "see control in a crowd"**

### 3.15 Control in a crowd (`control.CROWD_MARGIN`), 2026-11-04
- **What:** before, any opponent within 2 m blocked control. now a player within 2 m has the ball if every opponent is at least **0.5 m further** away (half a step)
- **Why:** 3.14 showed most missed short possessions are contested balls with an opponent always near. but if youre clearly closer, its your ball
- **Result on the 10 games, every single game got better:**

| margin | spellF1 | possessions found (real 396) | our match: possessions under 1 s |
|---|---|---|---|
| off (old) | 0.452 | 241 | 7 |
| 1.0 m | 0.465 | 249 | 7 |
| **0.5 m (chosen)** | **0.479** | 264 | 8 |
| 0.25 m | 0.482 | 273 | 9 |
| 0 m (just closer) | 0.477 | 290 | 17 |

  - why 0.5 and not 0.25: same score (difference is noise), and our tracking is noisier than the practice data. at 0 m our match starts flickering (tiny possessions double), so we stay a bit away from that edge
  - our match: 14.8 possessions per 5 min (was 14.1, public 5 min has 17), csv md5 now `b9b1ef9b...`
  - **public 5 min went 0.553 -> 0.514** (we found 18 possessions there, real 17). thats 1 possession on a tiny sample, noise range is ~0.39-0.72. we dont tune on it (rules), the 10 full matches are the real test and those all improved

### 3.16 One-touch passes (`control.touch_evidence`, `ONE_TOUCH_RADIUS`), 2026-11-04
- **Checked first:** the 979 short possessions we still miss on the 10 games
  - 670 of them: we see no control at all for that team. in those the ball is fast the whole time (median 1.4 s long), and a player of that team was within 1 m in 81%
  - = one touch moments. 825 of the 979 misses are bundesliga, which counts every touch as possession
  - challenge rules: a single touch is NOT a change. so we dont chase those, that would be copying bundesliga
- **What:** only the case the rules allow: a player touches a fast ball (within 1 m, clearly closer than any opponent) and a **teammate then controls it** = one touch pass, the team had it from the touch. a touch nobody follows up still counts for nothing
- **Result:** bench 0.479 -> **0.485**, 9 of 10 games up (bundesliga all 7 up, metrica +0.006 / -0.010 / +0.005)
  - our match: same possession count (14.8 per 5 min), csv md5 now `72261c1d...`
  - public 5 min 0.514 -> 0.491. noise range is ~0.39-0.72 and its 1 possession worth, but its the 2nd small dip in a row (0.553 -> 0.491 total) while the bench went up. **watch this:** practice tracking is cleaner than ours, so touch / crowd rules might work less well on our match. we dont tune on the 5 min, but we keep an eye on it

### 3.17 New method: segment_control (trees + path search), 2026-11-14
- **What:** replaces the rule-based pipeline as the submission. full description in `docs/Octo.md` and `src/algo/segment_control/README.md`
  - practice matches used clean + as a noisy copy with the tracking faults of our match (ball gaps, player gaps, extra person on the wrong team)
  - practice labels relabelled to the challenge rules: possession under 1 s inside live play = a touch, goes to the team before
  - ball filled in the gaps by a small model trained on the noisy copies
  - 2 gradient boosted tree models (dead? / which team?), then Viterbi over H/A/D that knows how long possessions and stoppages last
  - our old dead ball detector (3.2-3.4) is one of the inputs
- **Why:** the rules version was stuck on short possessions and switching late (3.13-3.16). a model can weigh lots of weak clues at once, the path search keeps it from flickering
- **Result:** 10 practice matches, each left out of training in turn, noisy copies: **0.638** vs 0.405 for the rules version on the same data. our match: 320 possessions, 37% dead, Barca 61% of live time

### 3.18 Checks before freezing, 2026-11-16
- **scale** (how much the per frame clues count vs the length model): tested 0.4 / 0.6 / 0.8 / 1.0 on the 10 practice matches, left out in turn
  - vs labels as they come: 0.638 / 0.642 / 0.644 / 0.642, noise
  - vs labels relabelled to the challenge rules: **0.694** / 0.692 / 0.691 / 0.687
  - kept 0.4. higher = more twitchy, fits bundesliga labels not the challenge
- **path memory** (MAX_AGE, up to which age the path search tracks how long a possession / stoppage lasts): 6 / 10 / 15 s -> 0.638 / 0.639 / 0.640 (relabelled 0.694 / 0.696 / 0.696). noise, clean copies slightly down, kept 6 s
- **smart tag weight** (cant be tested, no tags in practice data): switched off vs on changes 111 frames of our csv, same 320 possessions -> harmless, kept
- **same file every run:** ran it twice, byte identical (md5 `41656109...`)
- **dropped from the submission repo:** two other model versions we tried (fusion_control, learned_control) and the tuning script. part of their settings were picked using the public 5 min, which the rules dont allow, so they dont belong in the submission

---

## 4. Open issues

### Short possessions (biggest)
- real matches have lots of possessions under 3 s (team wins it and loses it again quickly). we miss most of them
- why (measured, 3.14): in most of them our control rule never fires, theres always an opponent within 2 m. passes helped a bit, switching faster did nothing, "clearly closer in a crowd" helped a lot (3.15)
- careful: bundesliga counts possession very twitchy (median 3 s), our match calmer (avg 14 s in public 5 min). we have to find a middle, not copy bundesliga

### Stoppages (mostly fine)
- bundesliga: dead/alive right 91%, dead share almost exactly right
- weak spots: edges (start/end a bit off), foul stoppages where the ball gets kicked around, slow build up looking like a stoppage
- "ball moving = alive" doesnt work on our match (see 3.11)
- team phase tags (BUILD UP etc) dont prove the ball is live, they cover 65% of our dead frames as well

### Practice matches vs our match
- practice tracking is clean (all 22 players, fixed ids), ours is automatic and messy -> our real score will probably be a bit lower
- ball during stoppages: metrica never visible, bundesliga always visible, ours ~30%. dead ball rules have to work on both

### Next steps (biggest expected gain first)
1. short possessions: mostly done, whats left are single touches (not possession by the rules). check our crowd / touch rules hold up on our messier tracking (maybe make the practice tracking messy like ours and re-test)
2. sharper stoppage edges + player position clues (wall, throw in, goal kick setup)
3. restart team: goal kick -> keepers team, throw in -> the team that didnt put it out
4. probabilistic combiner (HMM) so we can use weaker clues too
5. stop 1-2 days before code freeze (14 nov): pin versions, final checks, tech doc

---

## 5. How to run everything

```
.venv\Scripts\activate
python make_submission.py        -> Octo.csv (the submission, segment_control)
python evaluation/validate_submission.py Octo.csv
python src/algo/segment_control/segment_control.py evaluate   -> 10 practice matches, each left out in turn
python src/common/bench.py      -> old rules version on the practice matches
```

Practice data (not in the repo, download yourself):
- metrica: `git clone https://github.com/metrica-sports/sample-data data/external/metrica-sample`
- bundesliga: the `DFL_*.xml` files from https://huggingface.co/datasets/pysport/idsse-data into `data/external/idsse/`

Rules we stick to:
- no hand labels on our match, ever
- no outside data about barca - athletic
- nothing tuned on the public 5 min
- `data/` never goes to github

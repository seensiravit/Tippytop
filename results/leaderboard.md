# Leaderboard

Shared scoreboard — append every measured run here so results stay comparable.
Report **valid** primary for iteration (test only for final picks). Δ vs FM
baseline; a change is real only if Δvalid > +0.002 (noise band).

| Date | Owner | Model / change | valid GAUC | valid nDCG@5 | **valid primary** | test primary | Notes |
|---|---|---|---|---|---|---|---|
| — | ref | random (sanity) | 0.4996 | 0.4511 | 0.4753 | 0.4753 | harness check |
| — | ref | pop | 0.6308 | 0.5121 | 0.5715 | 0.5715 | trivial |
| — | ref | **FM baseline** | — | — | — | **0.5946** | the row to beat |
| — | ref | oracle ceiling | 1.0000 | 0.7289 | 0.8645 | 0.8645 | perfect ranking |

**Targets:** beat FM 0.5946; measure progress against oracle 0.8645 (headroom ≈ 0.27).
| 2026-08-29 | auto | fm (seed=42) | 0.6674 | 0.5363 | 0.6019 | 0.5957 |  |

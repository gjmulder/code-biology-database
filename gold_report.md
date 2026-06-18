# Gold-set validation (Phase 6) — judge `deepseek/deepseek-v4-pro`, embedding method `chunk`

Adjudicates the two synthetic axes against the Barbieri-anchored gold set (`gold_labels`). Embedding axis: does gold+ outrank gold− (AUC) and track polarity (ρ)? Judge axis: categorical verdict (`met` = positive) vs gold polarity.

## Embedding axis — gold+ vs gold−

| criterion | AUC | ρ(e, gold) | n+ | n− |
|---|---|---|---|---|
| two_worlds | 0.189 | -0.536 | 207 | 240 |
| adaptors | 0.368 | -0.228 | 207 | 240 |
| arbitrariness | 0.352 | -0.255 | 207 | 240 |

## Judge axis — verdict vs gold polarity

| criterion | precision | recall | F1 | TP | FP | FN | TN | n |
|---|---|---|---|---|---|---|---|---|
| two_worlds | 0.474 | 0.130 | 0.205 | 27 | 30 | 180 | 210 | 447 |
| adaptors | 0.688 | 0.213 | 0.325 | 44 | 20 | 163 | 220 | 447 |
| arbitrariness | 0.553 | 0.101 | 0.171 | 21 | 17 | 186 | 223 | 447 |

## Tier breakdown (gold paper counts)

| criterion | tier | n | n+ | n− |
|---|---|---|---|---|
| two_worlds | 1 | 4 | 4 | 0 |
| two_worlds | 2 | 203 | 203 | 0 |
| two_worlds | soft | 240 | 0 | 240 |
| adaptors | 1 | 4 | 4 | 0 |
| adaptors | 2 | 203 | 203 | 0 |
| adaptors | soft | 240 | 0 | 240 |
| arbitrariness | 1 | 4 | 4 | 0 |
| arbitrariness | 2 | 203 | 203 | 0 |
| arbitrariness | soft | 240 | 0 | 240 |

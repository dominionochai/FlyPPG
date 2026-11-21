# SmartSpectra MCD diagnosis

**Verdict:** OUTCOME_C_SUSPECTED: the control clip also yields no usable pulse in this runner -> runner/SDK/runtime defect suspected; MCD incompatibility cannot be concluded.

| run | requested | status | packets | pulse pkts | usable pulse samples | validation events | chest events | errors |
|---|---|---|---|---|---|---|---|---|
| control | breathing+cardio | INVALID_NO_OUTPUT | 0 | 0 | 0 | 486 | 482 | 0 |
| mcd_breathing_cardio | breathing+cardio | INVALID_NO_OUTPUT | 0 | 0 | 0 | 3476 | 3473 | 0 |
| mcd_cardio | cardio | INVALID_NO_OUTPUT | 0 | 0 | 0 | 3695 | 3690 | 0 |

## Framing (below-chin space in face heights; detection rate)
- **control**: [768, 480], frames 630, face detected 1.0, below-chin p5/p50/p95 = 0.6241332741332963/0.6737139035464392/0.7148426090879075, flagged discontinuities []
- **mcd**: [640, 480], frames 5377, face detected 1.0, below-chin p5/p50/p95 = 0.8806475814348416/1.031846608821449/1.1217023108073847, flagged discontinuities [2742]

## Validation vs framing
{
 "control": {
  "events_joined": 486,
  "chest_fraction": 0.9917695473251029,
  "mean_below_chin_when_chest": 0.6694425293524824,
  "mean_below_chin_when_other": 0.6960567608473134
 },
 "mcd_breathing_cardio": {
  "events_joined": 3476,
  "chest_fraction": 0.9991369390103567,
  "mean_below_chin_when_chest": 1.0203022294865716,
  "mean_below_chin_when_other": 0.8878831109830688
 },
 "mcd_cardio": {
  "events_joined": 3695,
  "chest_fraction": 0.9986468200270636,
  "mean_below_chin_when_chest": 1.0233186753130084,
  "mean_below_chin_when_other": 0.8878831109830688
 }
}
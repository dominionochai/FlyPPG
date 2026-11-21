| clip | alone: status | alone: readings | with adapter: status | with adapter: readings | adapter error vs reference |
|---|---|---|---|---|---|
| c01 | INVALID_NO_OUTPUT | 0 | PULSE_PRODUCED | 64 | 12.8 BPM (10% within 5, 10 windows) |
| c07 | INVALID_NO_OUTPUT | 0 | PULSE_PRODUCED | 56 | 12.0 BPM (40% within 5, 10 windows) |

Framing threshold sweep (same clip, canvas extended by pad% of frame height):

| extension | status | readings | error vs reference |
|---|---|---|---|
| 10% | INVALID_NO_OUTPUT | 0 | - |
| 20% | PULSE_PRODUCED | 62 | 15.0 BPM (0% within 5, 10 windows) |
| 30% | PULSE_PRODUCED | 59 | 8.1 BPM (22% within 5, 9 windows) |

SDK video seconds sent: 1140

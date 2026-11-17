# Look-ahead vs accuracy

3 seeds per setting, past = 6 frames, 60 epochs.

| look-ahead | delay | validation R² | jawOpen r | mouthFunnel r | mouthPucker r | mouthClose r | CPU / frame |
|---|---|---|---|---|---|---|---|
| 0 frames | 0 ms | 0.589 ± 0.005 | 0.31 | 0.46 | 0.70 | 0.22 | 62 µs |
| 1 frames | 33 ms | 0.599 ± 0.010 | 0.45 | 0.46 | 0.72 | 0.23 | 81 µs |
| 2 frames | 67 ms | 0.585 ± 0.008 | 0.48 | 0.63 | 0.72 | 0.26 | 73 µs |
| 3 frames | 100 ms | 0.591 ± 0.009 | 0.45 | 0.49 | 0.70 | 0.25 | 83 µs |

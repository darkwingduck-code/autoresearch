# Local-Global Attention Window Sweep

Date: 2026-05-19
Branch: autoresearch/may18-cae-3d

## Topic

Fixed-time training efficiency for small byte-level language models under local-global sliding attention schedules.

The working research question is:

> For a small byte-level LM trained under a fixed wall-clock budget, what short-window size and local/global layer schedule minimize validation bits per byte without reducing token throughput too much?

This is a practical SCI-style direction because the result is not just a benchmark tweak. It tests the tradeoff between representational context and hardware/kernel efficiency under fixed-time training, which is the regime that matters for autonomous overnight model-search loops.

## Current Best

Current best commit:

```text
18a1f83 experiment: test midpoint matrix learning rate
```

Current best settings:

```text
DEPTH = 6
WINDOW_PATTERN = "SSSL"
short_window = long_window // 4
MATRIX_LR = 0.038
```

Best observed validation score:

```text
val_bpb = 1.452655
peak_vram = 8.3 GB
```

## Experiments

| Commit | val_bpb | Memory GB | Status | Description |
| --- | ---: | ---: | --- | --- |
| 0f1a695 | 1.664028 | 11.6 | keep | portable attention fallback baseline |
| d8531e2 | 1.684659 | 11.3 | discard | grouped query attention with half kv heads |
| a9af0a3 | 1.467479 | 8.3 | keep | reduce depth to six layers |
| afd1639 | 1.467479 | 8.3 | discard | reduce depth to five layers |
| 2fe4d8a | 1.466470 | 8.3 | keep | quarter short attention window |
| 9370027 | 1.785077 | 8.3 | discard | eighth short attention window |
| 82331f5 | 1.870031 | 8.3 | discard | third short attention window |
| c44f0c2 | 1.453365 | 8.3 | discard | unreproduced matrix learning rate 0.05 trace |
| f40b99a | 1.764940 | 8.3 | discard | increase full attention frequency with SSLL |
| 6f61389 | 1.831403 | 8.3 | discard | reduce full attention frequency with SSSS |
| 892250b | 1.474364 | 8.3 | discard | verify matrix learning rate 0.05 rerun |
| fac3e64 | 1.468060 | 8.3 | keep | current best rerun noise check |
| 3c32710 | 1.464637 | 8.3 | keep | lower matrix learning rate to 0.035 borderline |
| 3c32710 | 1.465263 | 8.3 | keep | lower matrix learning rate to 0.035 rerun confirmed |
| e9fbed1 | 1.515459 | 8.3 | discard | extend learning rate warmdown to 0.6 |
| 0867c36 | 1.473352 | 8.3 | discard | keep nonzero final learning rate 0.05 |
| f45c439 | 1.475755 | 8.3 | discard | lower x0 residual init to 0.05 |
| dc3d5df | 1.468428 | 8.2 | discard | remove value embedding gate |
| 45305a3 | 1.478402 | 8.3 | discard | lower matrix learning rate further to 0.032 |
| 18a1f83 | 1.453340 | 8.3 | keep | test midpoint matrix learning rate 0.038 |
| 18a1f83 | 1.452655 | 8.3 | keep | test midpoint matrix learning rate 0.038 rerun confirmed |
| f9e701d | 1.450876 | 8.3 | discard | unreproduced matrix learning rate 0.039 trace |
| f9e701d | 1.496861 | 8.3 | discard | matrix learning rate 0.039 rerun failed |
| b89b205 | 1.693407 | 8.3 | discard | lower midpoint matrix learning rate to 0.037 |
| bfcf01c | 1.455197 | 8.3 | discard | lower matrix weight decay to 0.15 |
| 0accf8f | 1.463517 | 8.3 | discard | lower embedding learning rate to 0.5 |

## Findings

The best result so far is the depth-6 model with quarter-length short windows, the original SSSL local/global layer pattern, and a Muon matrix learning rate of `0.038`.

Changing the short-window divisor is not monotonic. Moving from the earlier half-window setup to quarter-window produced a small improvement, but third-window and eighth-window variants were much worse on this machine. The logs show that poor variants often reduced total trained tokens sharply, so the fixed-time metric is dominated by both model quality and kernel/runtime behavior.

Changing the local/global layer schedule was also harmful in the current implementation. Both increasing full-attention frequency with SSLL and reducing it with SSSS reduced throughput and worsened val_bpb. For this setup, SSSL appears to be a useful implementation-aware schedule, not just a modeling choice.

Raising Muon matrix learning rate from 0.04 to 0.05 produced one suspiciously strong trace (`1.453365`) but did not reproduce on a direct rerun (`1.474364`). Treat the strong trace as unresolved noise or a mismatched-log artifact, not as the current best.

Re-running the earlier best produced `1.468060`, within `0.001590` BPB of `1.466470`. Lowering `MATRIX_LR` to `0.035` produced `1.464637` and reproduced at `1.465263`, making it a confirmed improvement at the time.

The schedule and initialization follow-ups after `MATRIX_LR=0.035` were negative: `WARMDOWN_RATIO=0.6` degraded sharply, `FINAL_LR_FRAC=0.05` was worse, and lowering `x0_lambdas` init to `0.05` was worse. Removing the value embedding gate simplified the model but worsened BPB enough to discard it.

The matrix LR sweep now points to a narrow optimum. `0.032` and `0.037` were poor, `0.039` had one excellent trace but failed hard on rerun, and `0.038` reproduced with `1.453340` and `1.452655`. Lowering matrix weight decay to `0.15` and lowering embedding LR to `0.5` were both worse than the current best.

## Next Overnight Queue

1. Try a third confirmation run of `MATRIX_LR = 0.038`.
2. Try `MATRIX_LR = 0.0385` as a tighter midpoint.
3. Try `WEIGHT_DECAY = 0.25` with `MATRIX_LR = 0.038`.
4. Try `EMBEDDING_LR = 0.7` with `MATRIX_LR = 0.038`.
5. Try `UNEMBEDDING_LR = 0.003` with `MATRIX_LR = 0.038`.

## Paper Angle

The strongest paper framing is:

> Fixed-time local-global attention search for resource-constrained byte-level language model pretraining.

The early result suggests that the best schedule is not the one with the most global context or the shortest local window. Instead, the optimum depends on the interaction between layer schedule, attention kernel path, trained tokens under fixed wall-clock time, and validation bits per byte.

The next milestone is to turn this from single-run observations into a small ablation grid with repeated best/noise checks.

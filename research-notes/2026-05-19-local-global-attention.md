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
2fe4d8a experiment: quarter short attention window
```

Current best settings:

```text
DEPTH = 6
WINDOW_PATTERN = "SSSL"
short_window = long_window // 4
MATRIX_LR = 0.04
```

Best observed validation score:

```text
val_bpb = 1.466470
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
| c44f0c2 | 1.467850 | 8.3 | discard | raise matrix learning rate to 0.05 |
| f40b99a | 1.764940 | 8.3 | discard | increase full attention frequency with SSLL |
| 6f61389 | 1.831403 | 8.3 | discard | reduce full attention frequency with SSSS |

## Findings

The best result so far is the depth-6 model with quarter-length short windows and the original SSSL local/global layer pattern.

Changing the short-window divisor is not monotonic. Moving from the earlier half-window setup to quarter-window produced a small improvement, but third-window and eighth-window variants were much worse on this machine. The logs show that poor variants often reduced total trained tokens sharply, so the fixed-time metric is dominated by both model quality and kernel/runtime behavior.

Changing the local/global layer schedule was also harmful in the current implementation. Both increasing full-attention frequency with SSLL and reducing it with SSSS reduced throughput and worsened val_bpb. For this setup, SSSL appears to be a useful implementation-aware schedule, not just a modeling choice.

Raising Muon matrix learning rate from 0.04 to 0.05 did not improve the best configuration.

## Next Overnight Queue

1. Re-run the current best once to estimate noise around 1.466470.
2. Try `MATRIX_LR = 0.035` on the current best configuration.
3. Try `WARMDOWN_RATIO = 0.6` with current best settings.
4. Try `FINAL_LR_FRAC = 0.05` to avoid hard zero LR at the end.
5. Try `x0_lambdas.fill_(0.05)` instead of `0.1`.
6. Try removing the value embedding gate complexity only if the first five do not improve.

## Paper Angle

The strongest paper framing is:

> Fixed-time local-global attention search for resource-constrained byte-level language model pretraining.

The early result suggests that the best schedule is not the one with the most global context or the shortest local window. Instead, the optimum depends on the interaction between layer schedule, attention kernel path, trained tokens under fixed wall-clock time, and validation bits per byte.

The next milestone is to turn this from single-run observations into a small ablation grid with repeated best/noise checks.

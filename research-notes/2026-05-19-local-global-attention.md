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
val_bpb = 1.464637
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

## Findings

The best result so far is the depth-6 model with quarter-length short windows and the original SSSL local/global layer pattern.

Changing the short-window divisor is not monotonic. Moving from the earlier half-window setup to quarter-window produced a small improvement, but third-window and eighth-window variants were much worse on this machine. The logs show that poor variants often reduced total trained tokens sharply, so the fixed-time metric is dominated by both model quality and kernel/runtime behavior.

Changing the local/global layer schedule was also harmful in the current implementation. Both increasing full-attention frequency with SSLL and reducing it with SSSS reduced throughput and worsened val_bpb. For this setup, SSSL appears to be a useful implementation-aware schedule, not just a modeling choice.

Raising Muon matrix learning rate from 0.04 to 0.05 produced one suspiciously strong trace (`1.453365`) but did not reproduce on a direct rerun (`1.474364`). Treat the strong trace as unresolved noise or a mismatched-log artifact, not as the current best.

Re-running the current best produced `1.468060`, within `0.001590` BPB of the best observed `1.466470`. Treat improvements below roughly `0.002` BPB as noise until confirmed by repeat runs.

Lowering `MATRIX_LR` to `0.035` produced `1.464637` and reproduced at `1.465263`. This is a small but repeatable improvement over the previous best pair (`1.466470`, rerun `1.468060`), so `3c32710` is the current best.

## Next Overnight Queue

1. Try `WARMDOWN_RATIO = 0.6` with the best confirmed settings.
2. Try `FINAL_LR_FRAC = 0.05` to avoid hard zero LR at the end.
3. Try `x0_lambdas.fill_(0.05)` instead of `0.1`.
4. Try removing the value embedding gate complexity only if the first three do not improve.

## Paper Angle

The strongest paper framing is:

> Fixed-time local-global attention search for resource-constrained byte-level language model pretraining.

The early result suggests that the best schedule is not the one with the most global context or the shortest local window. Instead, the optimum depends on the interaction between layer schedule, attention kernel path, trained tokens under fixed wall-clock time, and validation bits per byte.

The next milestone is to turn this from single-run observations into a small ablation grid with repeated best/noise checks.

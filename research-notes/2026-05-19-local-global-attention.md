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
71012fc experiment: raise unembedding learning rate
```

Current best settings:

```text
DEPTH = 6
WINDOW_PATTERN = "SSSL"
short_window = long_window // 4
MATRIX_LR = 0.0385
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.005
WEIGHT_DECAY = 0.2
```

Best observed validation score:

```text
val_bpb = 1.443150
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
| 8bc5e0a | 1.449271 | 8.3 | keep | test finer matrix learning rate 0.0385 |
| 8bc5e0a | 1.449275 | 8.3 | keep | test finer matrix learning rate 0.0385 rerun confirmed |
| 3e809e5 | 1.450918 | 8.3 | discard | raise matrix weight decay to 0.25 |
| 28b3138 | 1.454194 | 8.3 | discard | raise embedding learning rate to 0.7 |
| 5eebf01 | 1.489291 | 8.3 | discard | lower unembedding learning rate to 0.003 |
| d104efa | 1.448106 | 8.3 | discard | unreproduced matrix learning rate 0.03825 trace |
| d104efa | 1.450344 | 8.3 | discard | matrix learning rate 0.03825 rerun failed |
| ac13105 | 1.452936 | 8.3 | discard | test upper fine matrix learning rate 0.03875 |
| 56c7db4 | 1.450977 | 8.3 | discard | tune matrix weight decay to 0.22 |
| 97d62c4 | 1.446793 | 8.3 | discard | unreproduced embedding learning rate 0.65 trace |
| 97d62c4 | 1.453495 | 8.3 | discard | embedding learning rate 0.65 rerun failed |
| 8bc5e0a | 1.455500 | 8.3 | keep | matrix learning rate 0.0385 third confirmation noisy |
| 0c23e4b | 1.617837 | 10.7 | discard | increase depth to seven layers |
| 711e35d | 1.449233 | 8.3 | discard | unreproduced embedding learning rate 0.625 trace |
| 711e35d | 1.450275 | 8.3 | discard | embedding learning rate 0.625 rerun failed |
| d7bf4e8 | 1.454645 | 8.3 | discard | lower matrix weight decay slightly to 0.18 |
| 71012fc | 1.444474 | 8.3 | keep | raise unembedding learning rate to 0.005 |
| 71012fc | 1.443150 | 8.3 | keep | raise unembedding learning rate to 0.005 rerun confirmed |
| 8a3fd6a | 1.444726 | 8.3 | discard | raise unembedding learning rate to 0.006 |
| c56f698 | 1.473676 | 8.3 | discard | lower unembedding learning rate to 0.0045 |
| 9661d39 | 1.441830 | 8.3 | discard | unreproduced unembedding learning rate 0.0055 trace |
| 9661d39 | 1.447256 | 8.3 | discard | unembedding learning rate 0.0055 rerun failed |
| 9ae36c3 | 1.446041 | 8.3 | discard | test unembedding learning rate 0.00525 |
| acebbee | 1.445093 | 8.3 | discard | test unembedding learning rate 0.00475 |
| 72b2777 | 1.455649 | 8.3 | discard | lower scalar learning rate to 0.4 |
| 65eb8bd | 1.471087 | 8.3 | discard | raise scalar learning rate to 0.6 |
| 3472789 | 1.446733 | 8.3 | discard | raise matrix weight decay slightly to 0.21 |
| 21d4740 | 1.448426 | 8.3 | discard | lower matrix weight decay slightly to 0.19 |

## Findings

The best result so far is the depth-6 model with quarter-length short windows, the original SSSL local/global layer pattern, a Muon matrix learning rate of `0.0385`, and a higher unembedding learning rate of `0.005`.

Changing the short-window divisor is not monotonic. Moving from the earlier half-window setup to quarter-window produced a small improvement, but third-window and eighth-window variants were much worse on this machine. The logs show that poor variants often reduced total trained tokens sharply, so the fixed-time metric is dominated by both model quality and kernel/runtime behavior.

Changing the local/global layer schedule was also harmful in the current implementation. Both increasing full-attention frequency with SSLL and reducing it with SSSS reduced throughput and worsened val_bpb. For this setup, SSSL appears to be a useful implementation-aware schedule, not just a modeling choice.

Raising Muon matrix learning rate from 0.04 to 0.05 produced one suspiciously strong trace (`1.453365`) but did not reproduce on a direct rerun (`1.474364`). Treat the strong trace as unresolved noise or a mismatched-log artifact, not as the current best.

Re-running the earlier best produced `1.468060`, within `0.001590` BPB of `1.466470`. Lowering `MATRIX_LR` to `0.035` produced `1.464637` and reproduced at `1.465263`, making it a confirmed improvement at the time.

The schedule and initialization follow-ups after `MATRIX_LR=0.035` were negative: `WARMDOWN_RATIO=0.6` degraded sharply, `FINAL_LR_FRAC=0.05` was worse, and lowering `x0_lambdas` init to `0.05` was worse. Removing the value embedding gate simplified the model but worsened BPB enough to discard it.

The matrix LR sweep now points to a narrow optimum. `0.032` and `0.037` were poor, `0.039` had one excellent trace but failed hard on rerun, and `0.0385` reproduced with `1.449271` and `1.449275`, though a third confirmation was noisier at `1.455500`.

The follow-up optimizer sweep around `MATRIX_LR=0.0385` produced one clear improvement. `EMBEDDING_LR=0.625` had a promising first trace but failed to beat the incumbent on rerun, and `WEIGHT_DECAY=0.18` was worse. Raising `UNEMBEDDING_LR` to `0.005` produced `1.444474` and reproduced stronger at `1.443150`, making it the new confirmed best. Raising it further to `0.006` was close at `1.444726` but did not beat the confirmed best. Lowering it to `0.0045` degraded sharply to `1.473676`. The midpoint `0.0055` produced the best single trace so far at `1.441830`, but failed to reproduce at `1.447256`; nearby `0.00525` and `0.00475` were also worse at `1.446041` and `1.445093`. Moving `SCALAR_LR` away from `0.5` was negative in both directions: `0.4` scored `1.455649` and `0.6` scored `1.471087`. Moving matrix weight decay away from `0.2` was also negative: `0.21` scored `1.446733` and `0.19` scored `1.448426`.

## Next Overnight Queue

1. Try a third confirmation run of `71012fc` to re-estimate incumbent noise.
2. Try `ADAM_BETAS = (0.85, 0.95)` with the current best.
3. Try a third `UNEMBEDDING_LR = 0.0055` run only if nearby values improve or remain borderline.
4. Try `MATRIX_LR = 0.03825` again only if the noise model requires another close-proximity check.
5. Try a third confirmation run of `71012fc` if later improvements are below `0.002` BPB.

## Paper Angle

The strongest paper framing is:

> Fixed-time local-global attention search for resource-constrained byte-level language model pretraining.

The early result suggests that the best schedule is not the one with the most global context or the shortest local window. Instead, the optimum depends on the interaction between layer schedule, attention kernel path, trained tokens under fixed wall-clock time, and validation bits per byte.

The next milestone is to turn this from single-run observations into a small ablation grid with repeated best/noise checks.

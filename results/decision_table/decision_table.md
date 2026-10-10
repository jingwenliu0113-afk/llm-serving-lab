# Deployment decision table

Rules: `docs/decisions/021` (schema 1). Generated from the JSON next to this file.

## How to read

- one request at a time -> the Core A section of your platform
- concurrency > 1 on the RTX -> the Core B section
- concurrency > 1 on the Mac -> not measured (SPEC: Core B only on NVIDIA)
- Core A (llama.cpp GGUF) and Core B (vLLM W4A16) numbers never share a row or table: different engine, precision and format; no runtime comparison (SPEC, 003)
- no Mac vs PC ranking (SPEC): each platform's values are only compared within that platform
- Core A and Core B are not comparable (different engine, precision, format)
- tie = not resolved / within noise; no equivalence is claimed
- beyond noise = |x - y| > 2 sqrt(SE_x^2 + SE_y^2) with n = 3 processes (§2); lenient at df = 2, applied as frozen
- quality measured at -c 512 (005); long-context quality not measured
- TTFT is llama-bench engine time: no tokenize / HTTP / sampling (§3)
- Q4_K_M is a plain k-quant without imatrix (004)
- Workloads (one primary axis each, then quality, then the SPEC metric order):
  - A-Q: quality first (one request at a time); axes quality → ttft → decode → memory
  - A-D: generation speed first (long answers, chat); axes decode → quality → ttft → memory
  - A-P: first token first (long documents, RAG, summaries); axes ttft → quality → decode → memory
  - A-M: memory first (leave room for other programs); axes memory → quality → ttft → decode

Inputs:
- core_a macos-metal: `formal-20261009-a2`
- core_a wsl2-cuda: `formal-20261009-rtx-a1`
- core_b_mixed wsl2-cuda: `formal-20261010-coreb-mixed2`
- core_b_uniform wsl2-cuda: `formal-20261009-coreb-uniform`
- quality wsl2-cuda: `formal-20261010-quality`

Excluded sessions:
- core_a/macos-metal: `formal-20261009-a1` — session is env-invalid (env_valid=False); sentinel drift is None (must be False); stopped by the battery drain rule (019): {'rule': '019', 'job_id': 'ttft-9B-Q4_K_M-32768-r3', 'phase': 'after', 'pct': 85, 'window_max_pct': 100, 'drop_pts': 15, 'window_s': 1800}
- core_a/macos-metal: `pilot-20261008-4b` — mode 'pilot' is not formal (pilots stay masked); session is env-invalid (env_valid=None); seed None is not the frozen 20261008 (§4)
- core_a/macos-metal: `pilot-20261008-4b-fa` — mode 'pilot' is not formal (pilots stay masked); seed None is not the frozen 20261008 (§4)
- core_a/macos-metal: `pilot-20261008-9b` — mode 'pilot' is not formal (pilots stay masked); seed None is not the frozen 20261008 (§4)
- core_a/wsl2-cuda: `pilot-20261009-rtx` — mode 'pilot' is not formal (pilots stay masked); seed None is not the frozen 20261008 (§4); session-start B drift is None (§4 step 4: must be False)
- core_b_mixed: `formal-20261010-coreb-mixed` — no valid run (outcomes ok = 0)
- quality: `pilot-20261008-4b` — mode 'pilot' is not formal (the pilot report stays masked: 005, 011); session is env-invalid (env_valid=None; rerun it, 012); platform 'macos-metal': formal quality is measured on wsl2-cuda only (020)
- quality: `pilot-20261008-both` — mode 'pilot' is not formal (the pilot report stays masked: 005, 011); platform 'macos-metal': formal quality is measured on wsl2-cuda only (020)
- quality: `pilot-20261010-rtx-q` — mode 'pilot' is not formal (the pilot report stays masked: 005, 011)

## Core A — macos-metal

Session `formal-20261009-a2`; sentinel drift 1.7849 %.
Power: powermode 2, Apple 70W USB-C Power Adapter (019), charge pauses 0.
Memory axis: sum of device self + host self (unified memory).
Labels: CROSS_SIZE_UNMEASURED, MAC_POWER_CONDITION, N3_NOISE_RULE, QUALITY_CTX_512, QUALITY_PLATFORM_MAC, TIE_NOT_EQUIVALENCE, TTFT_ENGINE_ONLY.

### Recommendations

| workload | L = 2048 | L = 8192 | L = 32768 |
|---|---|---|---|
| A-Q quality first (one request at a time) | 4B-Q8_0 ‖ 9B-Q8_0 | 4B-Q8_0 ‖ 9B-Q8_0 | 4B-Q8_0 ‖ 9B-Q8_0 |
| A-D generation speed first (long answers, chat) | 4B-Q4_K_M | 4B-Q4_K_M | 4B-Q4_K_M |
| A-P first token first (long documents, RAG, summaries) | 4B-Q8_0 | 4B-Q8_0 | 4B-Q8_0 |
| A-M memory first (leave room for other programs) | 4B-Q4_K_M | 4B-Q4_K_M | 4B-Q4_K_M |
| not feasible (oom / spill) | none | none | none |
| unknown † (not measured / error / incomplete) | none | none | none |

† provisional: an unknown configuration (not measured / error / incomplete) could change the pick. `X = Y`: tie (no equivalence claimed). `X ‖ Y`: incomparable (a quality relation among them is unmeasured: across sizes, or a same-size chain that is not resolved).

### L = 2048: evidence

| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | total max / host max MiB |
|---|---|---|---|---|---|---|---|---|
| 4B-Q8_0 | front | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 44.8 ± 0.6 | 3.166 ± 0.01026 | 646.9 ± 2.1 | 5531 ± 0 | 5531 / 656 |
| 4B-Q6_K | front | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 50.9 ± 0.3 | 3.375 ± 0.01865 | 606.8 ± 3.3 | 4413 ± 0 | 4413 / 509 |
| 4B-Q4_K_M | front | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 58.7 ± 0.2 | 3.317 ± 0.006707 | 617.4 ± 1.2 | 3693 ± 0 | 3693 / 509 |
| 9B-Q8_0 | front | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 26.4 ± 0.4 | 5.732 ± 0.01155 | 357.3 ± 0.7 | 10740 ± 0 | 10740 / 1048 |
| 9B-Q6_K | front | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 30.4 ± 0.3 | 6.096 ± 0.02412 | 336.0 ± 1.3 | 8438 ± 0 | 8438 / 813 |
| 9B-Q4_K_M | front | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 38.0 ± 0.6 | 5.928 ± 0.0272 | 345.5 ± 1.6 | 6538 ± 0 | 6538 / 563 |

Alternatives and costs (ratios = alternative ÷ pick, ± SE):

- A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.136 ± 0.017, ttft 1.066 ± 0.007, memory 0.798 ± 0.000
- A-Q: 4B-Q6_K vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.927 ± 0.029, ttft 0.589 ± 0.003, memory 0.411 ± 0.000
- A-Q: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.309 ± 0.018, ttft 1.048 ± 0.004, memory 0.668 ± 0.000
- A-Q: 4B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 2.221 ± 0.031, ttft 0.579 ± 0.002, memory 0.344 ± 0.000
- A-Q: 9B-Q6_K vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.678 ± 0.012, ttft 1.925 ± 0.010, memory 1.526 ± 0.000
- A-Q: 9B-Q6_K vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.151 ± 0.020, ttft 1.063 ± 0.005, memory 0.786 ± 0.000
- A-Q: 9B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.848 ± 0.017, ttft 1.872 ± 0.011, memory 1.182 ± 0.000
- A-Q: 9B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.438 ± 0.029, ttft 1.034 ± 0.005, memory 0.609 ± 0.000
- A-D: 4B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.764 ± 0.010, ttft 0.954 ± 0.004, memory 1.498 ± 0.000
- A-D: 4B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.867 ± 0.006, ttft 1.017 ± 0.006, memory 1.195 ± 0.000
- A-D: 9B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.450 ± 0.006, ttft 1.728 ± 0.005, memory 2.908 ± 0.000
- A-D: 9B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.518 ± 0.006, ttft 1.838 ± 0.008, memory 2.285 ± 0.000
- A-D: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.647 ± 0.010, ttft 1.787 ± 0.009, memory 1.770 ± 0.000
- A-P: 4B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.136 ± 0.017, ttft 1.066 ± 0.007, memory 0.798 ± 0.000
- A-P: 4B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.309 ± 0.018, ttft 1.048 ± 0.004, memory 0.668 ± 0.000
- A-P: 9B-Q8_0 vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.589 ± 0.011, ttft 1.810 ± 0.007, memory 1.942 ± 0.000
- A-P: 9B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.678 ± 0.012, ttft 1.925 ± 0.010, memory 1.526 ± 0.000
- A-P: 9B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.848 ± 0.017, ttft 1.872 ± 0.011, memory 1.182 ± 0.000
- A-M: 4B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.764 ± 0.010, ttft 0.954 ± 0.004, memory 1.498 ± 0.000
- A-M: 4B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.867 ± 0.006, ttft 1.017 ± 0.006, memory 1.195 ± 0.000
- A-M: 9B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.450 ± 0.006, ttft 1.728 ± 0.005, memory 2.908 ± 0.000
- A-M: 9B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.518 ± 0.006, ttft 1.838 ± 0.008, memory 2.285 ± 0.000
- A-M: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.647 ± 0.010, ttft 1.787 ± 0.009, memory 1.770 ± 0.000

H-A5(a) side by side (§2), L = 2048:

| config | state | memory axis MiB | decode tok/s | TTFT s |
|---|---|---|---|---|
| 9B-Q4_K_M | candidate | 6538 ± 0 | 38.0 ± 0.6 | 5.928 ± 0.0272 |
| 4B-Q8_0 | candidate | 5531 ± 0 | 44.8 ± 0.6 | 3.166 ± 0.01026 |

9B-Q4_K_M vs 4B-Q8_0: quality better, decode worse, ttft worse, memory worse
Paired ΔNLL test: mean -0.1944, ci95 [-0.2011655111206873, -0.18749350568965312] (descriptive only (§2)), p_adj 0.0001, reject True, direction_ok True.

Caveats: CONTAMINATION, NO_IMATRIX_Q4.

### L = 8192: evidence

| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | total max / host max MiB |
|---|---|---|---|---|---|---|---|---|
| 4B-Q8_0 | front | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 42.2 ± 0.1 | 13.86 ± 0.2678 | 591.7 ± 11.2 | 5727 ± 0 | 5727 / 662 |
| 4B-Q6_K | front | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 48.0 ± 0.4 | 14.34 ± 0.05423 | 571.4 ± 2.2 | 4609 ± 0 | 4609 / 515 |
| 4B-Q4_K_M | front | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 55.5 ± 0.2 | 14.3 ± 0.07871 | 572.9 ± 3.1 | 3889 ± 0 | 3889 / 515 |
| 9B-Q8_0 | front | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 25.5 ± 0.1 | 24.26 ± 0.2326 | 337.7 ± 3.2 | 10936 ± 0 | 10936 / 1054 |
| 9B-Q6_K | front | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 29.6 ± 0.0 | 25.28 ± 0.1226 | 324.1 ± 1.6 | 8634 ± 0 | 8634 / 819 |
| 9B-Q4_K_M | front | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 36.5 ± 0.3 | 25.42 ± 0.2823 | 322.3 ± 3.5 | 6734 ± 0 | 6734 / 569 |

Alternatives and costs (ratios = alternative ÷ pick, ± SE):

- A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.137 ± 0.009, ttft 1.035 ± 0.020, memory 0.805 ± 0.000
- A-Q: 4B-Q6_K vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.883 ± 0.015, ttft 0.591 ± 0.006, memory 0.421 ± 0.000
- A-Q: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.316 ± 0.004, ttft 1.032 ± 0.021, memory 0.679 ± 0.000
- A-Q: 4B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 2.178 ± 0.009, ttft 0.589 ± 0.007, memory 0.356 ± 0.000
- A-Q: 9B-Q6_K vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.701 ± 0.001, ttft 1.825 ± 0.036, memory 1.508 ± 0.000
- A-Q: 9B-Q6_K vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.161 ± 0.003, ttft 1.042 ± 0.011, memory 0.790 ± 0.000
- A-Q: 9B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.865 ± 0.006, ttft 1.835 ± 0.041, memory 1.176 ± 0.000
- A-Q: 9B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.433 ± 0.011, ttft 1.048 ± 0.015, memory 0.616 ± 0.000
- A-D: 4B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.760 ± 0.002, ttft 0.969 ± 0.019, memory 1.473 ± 0.000
- A-D: 4B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.864 ± 0.007, ttft 1.003 ± 0.007, memory 1.185 ± 0.000
- A-D: 9B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.459 ± 0.002, ttft 1.697 ± 0.019, memory 2.812 ± 0.000
- A-D: 9B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.533 ± 0.002, ttft 1.768 ± 0.013, memory 2.220 ± 0.000
- A-D: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.658 ± 0.005, ttft 1.778 ± 0.022, memory 1.732 ± 0.000
- A-P: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.137 ± 0.009, ttft 1.035 ± 0.020, memory 0.805 ± 0.000
- A-P: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.316 ± 0.004, ttft 1.032 ± 0.021, memory 0.679 ± 0.000
- A-P: 9B-Q8_0 vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q6_K, 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.604 ± 0.002, ttft 1.751 ± 0.038, memory 1.910 ± 0.000
- A-P: 9B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q6_K, 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.701 ± 0.001, ttft 1.825 ± 0.036, memory 1.508 ± 0.000
- A-P: 9B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q6_K, 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.865 ± 0.006, ttft 1.835 ± 0.041, memory 1.176 ± 0.000
- A-M: 4B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.760 ± 0.002, ttft 0.969 ± 0.019, memory 1.473 ± 0.000
- A-M: 4B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.864 ± 0.007, ttft 1.003 ± 0.007, memory 1.185 ± 0.000
- A-M: 9B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.459 ± 0.002, ttft 1.697 ± 0.019, memory 2.812 ± 0.000
- A-M: 9B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.533 ± 0.002, ttft 1.768 ± 0.013, memory 2.220 ± 0.000
- A-M: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.658 ± 0.005, ttft 1.778 ± 0.022, memory 1.732 ± 0.000

H-A5(a) side by side (§2), L = 8192:

| config | state | memory axis MiB | decode tok/s | TTFT s |
|---|---|---|---|---|
| 9B-Q4_K_M | candidate | 6734 ± 0 | 36.5 ± 0.3 | 25.42 ± 0.2823 |
| 4B-Q8_0 | candidate | 5727 ± 0 | 42.2 ± 0.1 | 13.86 ± 0.2678 |

9B-Q4_K_M vs 4B-Q8_0: quality better, decode worse, ttft worse, memory worse
Paired ΔNLL test: mean -0.1944, ci95 [-0.2011655111206873, -0.18749350568965312] (descriptive only (§2)), p_adj 0.0001, reject True, direction_ok True.

Caveats: CONTAMINATION, NO_IMATRIX_Q4.

### L = 32768: evidence

| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | total max / host max MiB |
|---|---|---|---|---|---|---|---|---|
| 4B-Q8_0 | front | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 33.6 ± 0.6 | 70.59 ± 0.3652 | 464.2 ± 2.4 | 6551 ± 0 | 6551 / 686 |
| 4B-Q6_K | front | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 37.1 ± 0.4 | 73.2 ± 0.6441 | 447.7 ± 4.0 | 5433 ± 0 | 5433 / 539 |
| 4B-Q4_K_M | front | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 42.3 ± 0.4 | 71.92 ± 0.09189 | 455.6 ± 0.6 | 4713 ± 0 | 4713 / 539 |
| 9B-Q8_0 | front | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 22.3 ± 0.1 | 112.4 ± 0.509 | 291.6 ± 1.3 | 11728 ± 0 | 11728 / 1078 |
| 9B-Q6_K | front | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 25.2 ± 0.1 | 118.7 ± 2.064 | 276.2 ± 4.8 | 9426 ± 0 | 9426 / 843 |
| 9B-Q4_K_M | front | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 29.4 ± 0.9 | 114.1 ± 0.6522 | 287.3 ± 1.7 | 7526 ± 0 | 7526 / 593 |

Alternatives and costs (ratios = alternative ÷ pick, ± SE):

- A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.104 ± 0.022, ttft 1.037 ± 0.011, memory 0.829 ± 0.000
- A-Q: 4B-Q6_K vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.664 ± 0.018, ttft 0.651 ± 0.006, memory 0.463 ± 0.000
- A-Q: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.259 ± 0.025, ttft 1.019 ± 0.005, memory 0.719 ± 0.000
- A-Q: 4B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.899 ± 0.020, ttft 0.640 ± 0.003, memory 0.402 ± 0.000
- A-Q: 9B-Q6_K vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.751 ± 0.013, ttft 1.682 ± 0.031, memory 1.439 ± 0.000
- A-Q: 9B-Q6_K vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.132 ± 0.004, ttft 1.056 ± 0.019, memory 0.804 ± 0.000
- A-Q: 9B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.875 ± 0.031, ttft 1.616 ± 0.012, memory 1.149 ± 0.000
- A-Q: 9B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.320 ± 0.041, ttft 1.015 ± 0.007, memory 0.642 ± 0.000
- A-D: 4B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.794 ± 0.016, ttft 0.982 ± 0.005, memory 1.390 ± 0.000
- A-D: 4B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.876 ± 0.013, ttft 1.018 ± 0.009, memory 1.153 ± 0.000
- A-D: 9B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.527 ± 0.006, ttft 1.562 ± 0.007, memory 2.488 ± 0.000
- A-D: 9B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.596 ± 0.006, ttft 1.651 ± 0.029, memory 2.000 ± 0.000
- A-D: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.695 ± 0.022, ttft 1.586 ± 0.009, memory 1.597 ± 0.000
- A-P: 4B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.104 ± 0.022, ttft 1.037 ± 0.011, memory 0.829 ± 0.000
- A-P: 4B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.259 ± 0.025, ttft 1.019 ± 0.005, memory 0.719 ± 0.000
- A-P: 9B-Q8_0 vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.663 ± 0.012, ttft 1.592 ± 0.011, memory 1.790 ± 0.000
- A-P: 9B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.751 ± 0.013, ttft 1.682 ± 0.031, memory 1.439 ± 0.000
- A-P: 9B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.875 ± 0.031, ttft 1.616 ± 0.012, memory 1.149 ± 0.000
- A-M: 4B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.794 ± 0.016, ttft 0.982 ± 0.005, memory 1.390 ± 0.000
- A-M: 4B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.876 ± 0.013, ttft 1.018 ± 0.009, memory 1.153 ± 0.000
- A-M: 9B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.527 ± 0.006, ttft 1.562 ± 0.007, memory 2.488 ± 0.000
- A-M: 9B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.596 ± 0.006, ttft 1.651 ± 0.029, memory 2.000 ± 0.000
- A-M: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.695 ± 0.022, ttft 1.586 ± 0.009, memory 1.597 ± 0.000

H-A5(a) side by side (§2), L = 32768:

| config | state | memory axis MiB | decode tok/s | TTFT s |
|---|---|---|---|---|
| 9B-Q4_K_M | candidate | 7526 ± 0 | 29.4 ± 0.9 | 114.1 ± 0.6522 |
| 4B-Q8_0 | candidate | 6551 ± 0 | 33.6 ± 0.6 | 70.59 ± 0.3652 |

9B-Q4_K_M vs 4B-Q8_0: quality better, decode worse, ttft worse, memory worse
Paired ΔNLL test: mean -0.1944, ci95 [-0.2011655111206873, -0.18749350568965312] (descriptive only (§2)), p_adj 0.0001, reject True, direction_ok True.

Caveats: CONTAMINATION, NO_IMATRIX_Q4.

### Fit map

| config | L = 2048 | L = 8192 | L = 32768 | largest fit L |
|---|---|---|---|---|
| 4B-Q8_0 | candidate | candidate | candidate | 32768 |
| 4B-Q6_K | candidate | candidate | candidate | 32768 |
| 4B-Q4_K_M | candidate | candidate | candidate | 32768 |
| 9B-Q8_0 | candidate | candidate | candidate | 32768 |
| 9B-Q6_K | candidate | candidate | candidate | 32768 |
| 9B-Q4_K_M | candidate | candidate | candidate | 32768 |

### Verdicts (copied verbatim; 'not refuted' only means the test did not refute)

- H-A1: not refuted; units {'fit': 18}
- H-A2: refuted; units {'tested': 15}
- H-A3: not refuted; units {'tested': 2}
- H-A4a: not refuted; units {'tested': 6}
- H-A4b: not refuted; units {'tested': 6}

## Core A — wsl2-cuda

Session `formal-20261009-rtx-a1`; sentinel drift 0.0618 %.
Memory axis: sum of device self = VRAM, the 004 fit / spill budget; host shown apart.
Labels: CROSS_SIZE_UNMEASURED, N3_NOISE_RULE, QUALITY_CTX_512, QUALITY_PLATFORM_RTX, SPILL_EXCLUDED, TIE_NOT_EQUIVALENCE, TTFT_ENGINE_ONLY.

### Recommendations

| workload | L = 2048 | L = 8192 | L = 32768 |
|---|---|---|---|
| A-Q quality first (one request at a time) | 4B-Q8_0 ‖ 9B-Q8_0 | 4B-Q8_0 ‖ 9B-Q8_0 | 4B-Q8_0 ‖ 9B-Q8_0 |
| A-D generation speed first (long answers, chat) | 4B-Q4_K_M | 4B-Q4_K_M | 4B-Q4_K_M |
| A-P first token first (long documents, RAG, summaries) | 4B-Q8_0 | 4B-Q8_0 | 4B-Q8_0 |
| A-M memory first (leave room for other programs) | 4B-Q4_K_M | 4B-Q4_K_M | 4B-Q4_K_M |
| not feasible (oom / spill) | none | none | none |
| unknown † (not measured / error / incomplete) | none | none | none |

† provisional: an unknown configuration (not measured / error / incomplete) could change the pick. `X = Y`: tie (no equivalence claimed). `X ‖ Y`: incomparable (a quality relation among them is unmeasured: across sizes, or a same-size chain that is not resolved).

### L = 2048: evidence

| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | total max / host max MiB |
|---|---|---|---|---|---|---|---|---|
| 4B-Q8_0 | front | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 136.6 ± 0.5 | 0.2414 ± 0.004951 | 8491.9 ± 177.8 | 4875 ± 0 | 5531 / 656 |
| 4B-Q6_K | front | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 164.5 ± 0.8 | 0.2819 ± 0.00352 | 7268.1 ± 89.8 | 3904 ± 0 | 4413 / 509 |
| 4B-Q4_K_M | front | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 187.5 ± 0.5 | 0.2475 ± 0.003595 | 8279.5 ± 118.8 | 3184 ± 0 | 3693 / 509 |
| 9B-Q8_0 | front | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 83.6 ± 0.2 | 0.362 ± 0.01377 | 5673.2 ± 209.0 | 8662 ± 0 | 9710 / 1048 |
| 9B-Q6_K | front | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 100.1 ± 0.6 | 0.4752 ± 0.02458 | 4332.7 ± 218.0 | 6829 ± 0 | 7642 / 813 |
| 9B-Q4_K_M | front | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 123.9 ± 0.3 | 0.3733 ± 0.002891 | 5486.9 ± 42.2 | 5429 ± 0 | 5992 / 563 |

Alternatives and costs (ratios = alternative ÷ pick, ± SE):

- A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.204 ± 0.007, ttft 1.168 ± 0.028, memory 0.801 ± 0.000
- A-Q: 4B-Q6_K vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.969 ± 0.010, ttft 0.779 ± 0.031, memory 0.451 ± 0.000
- A-Q: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.372 ± 0.006, ttft 1.025 ± 0.026, memory 0.653 ± 0.000
- A-Q: 4B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 2.243 ± 0.009, ttft 0.684 ± 0.028, memory 0.368 ± 0.000
- A-Q: 9B-Q6_K vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.733 ± 0.005, ttft 1.969 ± 0.110, memory 1.401 ± 0.000
- A-Q: 9B-Q6_K vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.198 ± 0.008, ttft 1.313 ± 0.084, memory 0.788 ± 0.000
- A-Q: 9B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.907 ± 0.004, ttft 1.547 ± 0.034, memory 1.114 ± 0.000
- A-Q: 9B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.482 ± 0.005, ttft 1.031 ± 0.040, memory 0.627 ± 0.000
- A-D: 4B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.729 ± 0.003, ttft 0.975 ± 0.025, memory 1.531 ± 0.000
- A-D: 4B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.878 ± 0.005, ttft 1.139 ± 0.022, memory 1.226 ± 0.000
- A-D: 9B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.446 ± 0.002, ttft 1.463 ± 0.060, memory 2.720 ± 0.000
- A-D: 9B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.534 ± 0.004, ttft 1.920 ± 0.103, memory 2.145 ± 0.000
- A-D: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.661 ± 0.002, ttft 1.509 ± 0.025, memory 1.705 ± 0.000
- A-P: 4B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q4_K_M; quality worse, decode better, ttft worse, memory better; decode 1.204 ± 0.007, ttft 1.168 ± 0.028, memory 0.801 ± 0.000
- A-P: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft tie, memory better; decode 1.372 ± 0.006, ttft 1.025 ± 0.026, memory 0.653 ± 0.000
- A-P: 9B-Q8_0 vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.612 ± 0.003, ttft 1.500 ± 0.065, memory 1.777 ± 0.000
- A-P: 9B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.733 ± 0.005, ttft 1.969 ± 0.110, memory 1.401 ± 0.000
- A-P: 9B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0, 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.907 ± 0.004, ttft 1.547 ± 0.034, memory 1.114 ± 0.000
- A-M: 4B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft tie, memory worse; decode 0.729 ± 0.003, ttft 0.975 ± 0.025, memory 1.531 ± 0.000
- A-M: 4B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.878 ± 0.005, ttft 1.139 ± 0.022, memory 1.226 ± 0.000
- A-M: 9B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.446 ± 0.002, ttft 1.463 ± 0.060, memory 2.720 ± 0.000
- A-M: 9B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.534 ± 0.004, ttft 1.920 ± 0.103, memory 2.145 ± 0.000
- A-M: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.661 ± 0.002, ttft 1.509 ± 0.025, memory 1.705 ± 0.000

H-A5(a) side by side (§2), L = 2048:

| config | state | memory axis MiB | decode tok/s | TTFT s |
|---|---|---|---|---|
| 9B-Q4_K_M | candidate | 5429 ± 0 | 123.9 ± 0.3 | 0.3733 ± 0.002891 |
| 4B-Q8_0 | candidate | 4875 ± 0 | 136.6 ± 0.5 | 0.2414 ± 0.004951 |

9B-Q4_K_M vs 4B-Q8_0: quality better, decode worse, ttft worse, memory worse
Paired ΔNLL test: mean -0.1944, ci95 [-0.2011655111206873, -0.18749350568965312] (descriptive only (§2)), p_adj 0.0001, reject True, direction_ok True.

Caveats: CONTAMINATION, NO_IMATRIX_Q4.

### L = 8192: evidence

| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | total max / host max MiB |
|---|---|---|---|---|---|---|---|---|
| 4B-Q8_0 | front | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 131.7 ± 0.2 | 0.9428 ± 0.003039 | 8689.1 ± 27.9 | 5065 ± 0 | 5727 / 662 |
| 4B-Q6_K | front | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 157.1 ± 0.1 | 1.114 ± 0.003605 | 7352.7 ± 23.9 | 4094 ± 0 | 4609 / 515 |
| 4B-Q4_K_M | front | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 175.0 ± 0.9 | 0.9971 ± 0.002209 | 8215.9 ± 18.2 | 3374 ± 0 | 3889 / 515 |
| 9B-Q8_0 | front | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 81.4 ± 0.7 | 1.471 ± 0.02726 | 5572.7 ± 102.2 | 8852 ± 0 | 9906 / 1054 |
| 9B-Q6_K | front | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 97.2 ± 0.4 | 1.818 ± 0.0358 | 4509.4 ± 87.4 | 7019 ± 0 | 7838 / 819 |
| 9B-Q4_K_M | front | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 119.0 ± 0.3 | 1.532 ± 0.002875 | 5348.9 ± 10.1 | 5619 ± 0 | 6188 / 569 |

Alternatives and costs (ratios = alternative ÷ pick, ± SE):

- A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.193 ± 0.002, ttft 1.182 ± 0.005, memory 0.808 ± 0.000
- A-Q: 4B-Q6_K vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.931 ± 0.017, ttft 0.757 ± 0.014, memory 0.462 ± 0.000
- A-Q: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.329 ± 0.008, ttft 1.058 ± 0.004, memory 0.666 ± 0.000
- A-Q: 4B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 2.151 ± 0.023, ttft 0.678 ± 0.013, memory 0.381 ± 0.000
- A-Q: 9B-Q6_K vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.738 ± 0.003, ttft 1.928 ± 0.038, memory 1.386 ± 0.000
- A-Q: 9B-Q6_K vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.194 ± 0.012, ttft 1.236 ± 0.033, memory 0.793 ± 0.000
- A-Q: 9B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.904 ± 0.003, ttft 1.624 ± 0.006, memory 1.109 ± 0.000
- A-Q: 9B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.463 ± 0.014, ttft 1.041 ± 0.019, memory 0.635 ± 0.000
- A-D: 4B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.753 ± 0.004, ttft 0.946 ± 0.004, memory 1.501 ± 0.000
- A-D: 4B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.898 ± 0.005, ttft 1.117 ± 0.004, memory 1.213 ± 0.000
- A-D: 9B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.465 ± 0.005, ttft 1.475 ± 0.028, memory 2.624 ± 0.000
- A-D: 9B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.555 ± 0.004, ttft 1.823 ± 0.036, memory 2.080 ± 0.000
- A-D: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.680 ± 0.004, ttft 1.536 ± 0.004, memory 1.665 ± 0.000
- A-P: 4B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.193 ± 0.002, ttft 1.182 ± 0.005, memory 0.808 ± 0.000
- A-P: 4B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.329 ± 0.008, ttft 1.058 ± 0.004, memory 0.666 ± 0.000
- A-P: 9B-Q8_0 vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.618 ± 0.006, ttft 1.560 ± 0.029, memory 1.748 ± 0.000
- A-P: 9B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.738 ± 0.003, ttft 1.928 ± 0.038, memory 1.386 ± 0.000
- A-P: 9B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.904 ± 0.003, ttft 1.624 ± 0.006, memory 1.109 ± 0.000
- A-M: 4B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.753 ± 0.004, ttft 0.946 ± 0.004, memory 1.501 ± 0.000
- A-M: 4B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.898 ± 0.005, ttft 1.117 ± 0.004, memory 1.213 ± 0.000
- A-M: 9B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.465 ± 0.005, ttft 1.475 ± 0.028, memory 2.624 ± 0.000
- A-M: 9B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.555 ± 0.004, ttft 1.823 ± 0.036, memory 2.080 ± 0.000
- A-M: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.680 ± 0.004, ttft 1.536 ± 0.004, memory 1.665 ± 0.000

H-A5(a) side by side (§2), L = 8192:

| config | state | memory axis MiB | decode tok/s | TTFT s |
|---|---|---|---|---|
| 9B-Q4_K_M | candidate | 5619 ± 0 | 119.0 ± 0.3 | 1.532 ± 0.002875 |
| 4B-Q8_0 | candidate | 5065 ± 0 | 131.7 ± 0.2 | 0.9428 ± 0.003039 |

9B-Q4_K_M vs 4B-Q8_0: quality better, decode worse, ttft worse, memory worse
Paired ΔNLL test: mean -0.1944, ci95 [-0.2011655111206873, -0.18749350568965312] (descriptive only (§2)), p_adj 0.0001, reject True, direction_ok True.

Caveats: CONTAMINATION, NO_IMATRIX_Q4.

### L = 32768: evidence

| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | total max / host max MiB |
|---|---|---|---|---|---|---|---|---|
| 4B-Q8_0 | front | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 115.2 ± 0.9 | 4.625 ± 0.005048 | 7084.4 ± 7.7 | 5870 ± 0 | 6556 / 686 |
| 4B-Q6_K | front | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 132.4 ± 0.8 | 5.355 ± 0.007568 | 6119.3 ± 8.6 | 4899 ± 0 | 5438 / 539 |
| 4B-Q4_K_M | front | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 145.8 ± 1.1 | 4.839 ± 0.002794 | 6771.5 ± 3.9 | 4179 ± 0 | 4718 / 539 |
| 9B-Q8_0 | front | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 73.9 ± 0.3 | 6.456 ± 0.007149 | 5076.0 ± 5.6 | 9620 ± 0 | 10698 / 1078 |
| 9B-Q6_K | front | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 86.9 ± 0.2 | 7.974 ± 0.0257 | 4109.2 ± 13.2 | 7787 ± 0 | 8630 / 843 |
| 9B-Q4_K_M | front | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 105.0 ± 0.7 | 6.945 ± 0.004271 | 4718.5 ± 2.9 | 6387 ± 0 | 6980 / 593 |

Alternatives and costs (ratios = alternative ÷ pick, ± SE):

- A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.149 ± 0.012, ttft 1.158 ± 0.002, memory 0.835 ± 0.000
- A-Q: 4B-Q6_K vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.791 ± 0.013, ttft 0.829 ± 0.001, memory 0.509 ± 0.000
- A-Q: 4B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.266 ± 0.014, ttft 1.046 ± 0.001, memory 0.712 ± 0.000
- A-Q: 4B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 4B-Q8_0; quality unmeasured, decode better, ttft better, memory better; decode 1.972 ± 0.017, ttft 0.750 ± 0.001, memory 0.434 ± 0.000
- A-Q: 9B-Q6_K vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.755 ± 0.006, ttft 1.724 ± 0.006, memory 1.327 ± 0.000
- A-Q: 9B-Q6_K vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.176 ± 0.006, ttft 1.235 ± 0.004, memory 0.809 ± 0.000
- A-Q: 9B-Q4_K_M vs 4B-Q8_0 — dropped at quality by 9B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.911 ± 0.010, ttft 1.501 ± 0.002, memory 1.088 ± 0.000
- A-Q: 9B-Q4_K_M vs 9B-Q8_0 — dropped at quality by 9B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.420 ± 0.012, ttft 1.076 ± 0.001, memory 0.664 ± 0.000
- A-D: 4B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.790 ± 0.009, ttft 0.956 ± 0.001, memory 1.405 ± 0.000
- A-D: 4B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.908 ± 0.009, ttft 1.107 ± 0.002, memory 1.172 ± 0.000
- A-D: 9B-Q8_0 vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.507 ± 0.004, ttft 1.334 ± 0.002, memory 2.302 ± 0.000
- A-D: 9B-Q6_K vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.596 ± 0.005, ttft 1.648 ± 0.005, memory 1.863 ± 0.000
- A-D: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at decode by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.720 ± 0.007, ttft 1.435 ± 0.001, memory 1.528 ± 0.000
- A-P: 4B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.149 ± 0.012, ttft 1.158 ± 0.002, memory 0.835 ± 0.000
- A-P: 4B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality worse, decode better, ttft worse, memory better; decode 1.266 ± 0.014, ttft 1.046 ± 0.001, memory 0.712 ± 0.000
- A-P: 9B-Q8_0 vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.642 ± 0.006, ttft 1.396 ± 0.002, memory 1.639 ± 0.000
- A-P: 9B-Q6_K vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.755 ± 0.006, ttft 1.724 ± 0.006, memory 1.327 ± 0.000
- A-P: 9B-Q4_K_M vs 4B-Q8_0 — dropped at ttft by 4B-Q8_0; quality better, decode worse, ttft worse, memory worse; decode 0.911 ± 0.010, ttft 1.501 ± 0.002, memory 1.088 ± 0.000
- A-M: 4B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft better, memory worse; decode 0.790 ± 0.009, ttft 0.956 ± 0.001, memory 1.405 ± 0.000
- A-M: 4B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality better, decode worse, ttft worse, memory worse; decode 0.908 ± 0.009, ttft 1.107 ± 0.002, memory 1.172 ± 0.000
- A-M: 9B-Q8_0 vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.507 ± 0.004, ttft 1.334 ± 0.002, memory 2.302 ± 0.000
- A-M: 9B-Q6_K vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.596 ± 0.005, ttft 1.648 ± 0.005, memory 1.863 ± 0.000
- A-M: 9B-Q4_K_M vs 4B-Q4_K_M — dropped at memory by 4B-Q4_K_M; quality unmeasured, decode worse, ttft worse, memory worse; decode 0.720 ± 0.007, ttft 1.435 ± 0.001, memory 1.528 ± 0.000

H-A5(a) side by side (§2), L = 32768:

| config | state | memory axis MiB | decode tok/s | TTFT s |
|---|---|---|---|---|
| 9B-Q4_K_M | candidate | 6387 ± 0 | 105.0 ± 0.7 | 6.945 ± 0.004271 |
| 4B-Q8_0 | candidate | 5870 ± 0 | 115.2 ± 0.9 | 4.625 ± 0.005048 |

9B-Q4_K_M vs 4B-Q8_0: quality better, decode worse, ttft worse, memory worse
Paired ΔNLL test: mean -0.1944, ci95 [-0.2011655111206873, -0.18749350568965312] (descriptive only (§2)), p_adj 0.0001, reject True, direction_ok True.

Caveats: CONTAMINATION, NO_IMATRIX_Q4.

### Fit map

| config | L = 2048 | L = 8192 | L = 32768 | largest fit L |
|---|---|---|---|---|
| 4B-Q8_0 | candidate | candidate | candidate | 32768 |
| 4B-Q6_K | candidate | candidate | candidate | 32768 |
| 4B-Q4_K_M | candidate | candidate | candidate | 32768 |
| 9B-Q8_0 | candidate | candidate | candidate | 32768 |
| 9B-Q6_K | candidate | candidate | candidate | 32768 |
| 9B-Q4_K_M | candidate | candidate | candidate | 32768 |

### Verdicts (copied verbatim; 'not refuted' only means the test did not refute)

- H-A1: not refuted; units {'fit': 18}
- H-A2: refuted; units {'tested': 15}
- H-A3: not refuted; units {'tested': 2}
- H-A4a: not refuted; units {'tested': 6}
- H-A4b: not refuted; units {'tested': 6}

## Core B — wsl2-cuda / vLLM / W4A16

Scope: model RedHatAI/Qwen3.5-9B-quantized.w4a16@a398088c4228b0ae0c8c78df88fd1e4bf445f068; precision w4a16-gptq-int4-g128; vllm 0.30.0; gpu_memory_utilization 0.91; kv_cache_memory_bytes 4294967296; num_gpu_blocks 248; block_size 528; max_num_seqs 128; max_num_batched_tokens 2048; uniform_request (L - 128, 128); temperature 0; ignore_eos True.
Quality: not measured (W4A16; 003/008) -- Core A Q4_K_M quality does not apply.

| L | C_KV | C_knee | C_max | throughput at knee / max (tok/s) | observed gain / bound | P95 TTFT ms (C=1 / knee / max) | P95 TPOT ms (C=1 / knee / max) | flags |
|---|---|---|---|---|---|---|---|---|
| 2048 | 35 | 25 | 128 | 314.9 ± 0.1 / 325.7 ± 0.2 (C 32) | 3.4 % / 5.3 % | 329.7 ± 0.2 / 4259.6 ± 12.0 / 38138.0 ± 21.2 | 12.0 ± 0.0 / 74.5 ± 0.1 / 104.0 ± 0.1 | KNEE_POINT_ESTIMATE |
| 8192 | 13 | 8 | 128 | 80.2 ± 0.2 / 83.3 ± 0.3 (C 26) | 3.8 % / 5.3 % | 1406.8 ± 7.9 / 2833.6 ± 16.5 / 179097.5 ± 627.6 | 12.4 ± 0.1 / 88.8 ± 0.2 / 145.7 ± 0.5 | KNEE_POINT_ESTIMATE |
| 32768 | 3 | 2 | 32 | 17.8 ± 0.0 / 18.6 ± 0.0 (C 32) | 4.2 % / 5.3 % | 6290.1 ± 3.0 / 9988.4 ± 14.9 / 207099.9 ± 7.0 | 13.1 ± 0.0 / 62.6 ± 0.1 / 112.4 ± 0.1 | KNEE_POINT_ESTIMATE |

### L = 2048

- B-T (throughput): C = C_knee = 25.
- B-L (latency-sensitive): choosing C for a latency target is Advanced (fixed-SLO max concurrency); not done in Core. H-B1 {'status': 'supported', 'first_c_at_3x': 70}; H-B3 {'status': 'tested', 'g_tpot': 8.640767847110348, 'g_ttft': 115.68418273000925}.
- B-X (mixed traffic): ok; measured only at C = uniform C_knee; the mechanism is a conjecture (007); no remedy evaluated (Advanced); H-B4 unit {'L': 2048, 'status': 'tested', 'C': 25, 'uniform_p95_ms': 4259.598182584406, 'mixed_p95_ms': 23182.954163682494, 'increase_pct': 444.2521376421663, 'beyond_noise': True, 'refutes': False}.

| C | n valid | output tok/s | req/s | P50 TTFT ms | P95 TTFT ms | P50 TPOT ms | P95 TPOT ms | KV usage peak | preemptions |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 69.4 ± 0.0 | 0.542 ± 0.000 | 327.2 ± 0.1 | 329.7 ± 0.2 | 12.0 ± 0.0 | 12.0 ± 0.0 | 0.028340080971659964 | 0.0 |
| 2 | 3 | 112.9 ± 0.0 | 0.882 ± 0.000 | 493.8 ± 2.1 | 643.1 ± 0.4 | 13.8 ± 0.1 | 15.1 ± 0.0 | 0.05668016194331982 | 0.0 |
| 4 | 3 | 174.0 ± 0.1 | 1.359 ± 0.001 | 812.4 ± 3.4 | 968.9 ± 1.3 | 16.7 ± 0.0 | 20.3 ± 0.0 | 0.11336032388663964 | 0.0 |
| 8 | 3 | 242.5 ± 0.0 | 1.894 ± 0.000 | 963.4 ± 0.9 | 977.5 ± 1.3 | 25.7 ± 0.0 | 30.3 ± 0.0 | 0.22672064777327938 | 0.0 |
| 16 | 3 | 293.1 ± 0.1 | 2.289 ± 0.001 | 980.3 ± 0.1 | 2050.0 ± 2.0 | 46.9 ± 0.0 | 51.1 ± 0.2 | 0.45344129554655865 | 0.0 |
| 18 | 3 | 293.6 ± 0.1 | 2.294 ± 0.001 | 983.0 ± 1.3 | 2745.3 ± 1.3 | 53.2 ± 0.1 | 57.0 ± 0.1 | 0.5101214574898785 | 0.0 |
| 25 | 3 | 314.9 ± 0.1 | 2.460 ± 0.001 | 995.3 ± 0.7 | 4259.6 ± 12.0 | 72.1 ± 0.0 | 74.5 ± 0.1 | 0.708502024291498 | 0.0 |
| 32 | 3 | 325.7 ± 0.2 | 2.544 ± 0.002 | 1006.2 ± 0.6 | 5458.8 ± 28.7 | 90.9 ± 0.1 | 92.9 ± 0.1 | 0.9068825910931174 | 0.0 |
| 35 | 3 | 323.7 ± 1.5 | 2.529 ± 0.012 | 1012.3 ± 3.3 | 6014.5 ± 10.7 | 100.8 ± 0.5 | 102.5 ± 0.6 | 0.9919028340080972 | 0.0 |
| 49 | 3 | 324.4 ± 0.0 | 2.534 ± 0.000 | 5422.5 ± 5.0 | 8041.1 ± 16.4 | 103.4 ± 0.0 | 103.9 ± 0.1 | 0.9919028340080972 | 0.0 |
| 64 | 3 | 323.4 ± 0.1 | 2.526 ± 0.001 | 12451.2 ± 5.7 | 12688.8 ± 25.0 | 103.4 ± 0.0 | 103.9 ± 0.1 | 0.9919028340080972 | 0.0 |
| 70 | 3 | 324.5 ± 0.2 | 2.535 ± 0.002 | 14477.8 ± 11.2 | 14698.8 ± 39.1 | 103.3 ± 0.1 | 103.9 ± 0.1 | 0.9919028340080972 | 0.0 |
| 128 | 3 | 324.0 ± 0.0 | 2.531 ± 0.000 | 38031.4 ± 11.9 | 38138.0 ± 21.2 | 103.5 ± 0.0 | 104.0 ± 0.1 | 0.9919028340080972 | 0.0 |

Caveats: CORE_B_FIXED_DEPLOYMENT, KNEE_POINT_ESTIMATE, KV_USAGE_INCLUDES_GDN, MIXED_AT_KNEE_ONLY.

### L = 8192

- B-T (throughput): C = C_knee = 8.
- B-L (latency-sensitive): choosing C for a latency target is Advanced (fixed-SLO max concurrency); not done in Core. H-B1 {'status': 'supported', 'first_c_at_3x': 16}; H-B3 {'status': 'tested', 'g_tpot': 11.796496042763211, 'g_ttft': 127.30691661420045}.
- B-X (mixed traffic): ok; measured only at C = uniform C_knee; the mechanism is a conjecture (007); no remedy evaluated (Advanced); H-B4 unit {'L': 8192, 'status': 'tested', 'C': 8, 'uniform_p95_ms': 2833.6296407964255, 'mixed_p95_ms': 7802.230612341373, 'increase_pct': 175.34404990725827, 'beyond_noise': True, 'refutes': False}.

| C | n valid | output tok/s | req/s | P50 TTFT ms | P95 TTFT ms | P50 TPOT ms | P95 TPOT ms | KV usage peak | preemptions |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 43.4 ± 0.2 | 0.339 ± 0.001 | 1395.9 ± 7.7 | 1406.8 ± 7.9 | 12.3 ± 0.0 | 12.4 ± 0.1 | 0.07692307692307687 | 0.0 |
| 2 | 3 | 57.9 ± 0.1 | 0.452 ± 0.001 | 1914.8 ± 5.0 | 2722.2 ± 11.3 | 19.6 ± 0.0 | 23.4 ± 0.0 | 0.15384615384615385 | 0.0 |
| 4 | 3 | 71.0 ± 0.2 | 0.555 ± 0.002 | 2455.1 ± 11.9 | 2806.1 ± 10.8 | 37.4 ± 0.1 | 45.2 ± 0.1 | 0.3076923076923077 | 0.0 |
| 7 | 3 | 78.8 ± 0.2 | 0.615 ± 0.001 | 2479.8 ± 10.1 | 2816.0 ± 16.9 | 69.9 ± 0.1 | 77.7 ± 0.2 | 0.5384615384615384 | 0.0 |
| 8 | 3 | 80.2 ± 0.2 | 0.627 ± 0.002 | 2489.2 ± 7.9 | 2833.6 ± 16.5 | 81.0 ± 0.2 | 88.8 ± 0.2 | 0.6153846153846154 | 0.0 |
| 9 | 3 | 80.1 ± 0.2 | 0.626 ± 0.002 | 2494.9 ± 9.3 | 2621.0 ± 120.2 | 93.3 ± 0.3 | 101.1 ± 0.3 | 0.6923076923076923 | 0.0 |
| 13 | 3 | 83.1 ± 0.4 | 0.649 ± 0.003 | 2523.0 ± 11.6 | 4283.1 ± 11.5 | 137.6 ± 0.6 | 144.9 ± 0.5 | 1.0 | 0.0 |
| 16 | 3 | 83.0 ± 0.4 | 0.648 ± 0.003 | 6076.9 ± 26.0 | 8527.4 ± 10.9 | 143.3 ± 0.7 | 145.6 ± 0.7 | 1.0 | 0.0 |
| 18 | 3 | 83.0 ± 0.3 | 0.649 ± 0.002 | 8956.9 ± 30.8 | 11385.8 ± 26.7 | 143.2 ± 0.4 | 145.2 ± 0.4 | 1.0 | 0.0 |
| 26 | 3 | 83.3 ± 0.3 | 0.651 ± 0.002 | 21741.3 ± 71.0 | 21834.4 ± 73.3 | 143.1 ± 0.5 | 145.3 ± 0.5 | 1.0 | 0.0 |
| 32 | 3 | 83.2 ± 0.3 | 0.650 ± 0.002 | 30353.9 ± 84.3 | 31831.8 ± 93.1 | 143.2 ± 0.4 | 145.3 ± 0.4 | 1.0 | 0.0 |
| 64 | 3 | 83.2 ± 0.4 | 0.650 ± 0.003 | 80288.7 ± 345.1 | 80390.7 ± 377.0 | 143.3 ± 0.6 | 145.5 ± 0.7 | 1.0 | 0.0 |
| 128 | 3 | 83.1 ± 0.3 | 0.650 ± 0.002 | 178921.5 ± 594.4 | 179097.5 ± 627.6 | 143.4 ± 0.5 | 145.7 ± 0.5 | 1.0 | 0.0 |

Caveats: CORE_B_FIXED_DEPLOYMENT, KNEE_POINT_ESTIMATE, KV_USAGE_INCLUDES_GDN, MIXED_AT_KNEE_ONLY.

### L = 32768

- B-T (throughput): C = C_knee = 2.
- B-L (latency-sensitive): choosing C for a latency target is Advanced (fixed-SLO max concurrency); not done in Core. H-B1 {'status': 'supported', 'first_c_at_3x': 8}; H-B3 {'status': 'tested', 'g_tpot': 8.566276833371735, 'g_ttft': 32.92491993891348}.
- B-X (mixed traffic): not defined (007).

| C | n valid | output tok/s | req/s | P50 TTFT ms | P95 TTFT ms | P50 TPOT ms | P95 TPOT ms | KV usage peak | preemptions |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 16.2 ± 0.0 | 0.126 ± 0.000 | 6260.7 ± 0.3 | 6290.1 ± 3.0 | 13.1 ± 0.0 | 13.1 ± 0.0 | 0.2672064777327935 | 0.0 |
| 2 | 3 | 17.8 ± 0.0 | 0.139 ± 0.000 | 7624.3 ± 3.0 | 9988.4 ± 14.9 | 52.9 ± 0.0 | 62.6 ± 0.1 | 0.534412955465587 | 0.0 |
| 3 | 3 | 18.5 ± 0.0 | 0.145 ± 0.000 | 8854.4 ± 9.1 | 9712.5 ± 4.6 | 93.3 ± 0.1 | 112.2 ± 0.1 | 0.7975708502024291 | 0.0 |
| 4 | 3 | 18.6 ± 0.0 | 0.145 ± 0.000 | 13102.3 ± 6.2 | 14822.1 ± 19.1 | 109.7 ± 0.1 | 112.6 ± 0.2 | 0.7975708502024291 | 0.0 |
| 6 | 3 | 18.6 ± 0.0 | 0.145 ± 0.000 | 27412.6 ± 6.5 | 27466.6 ± 8.7 | 109.5 ± 0.0 | 112.4 ± 0.0 | 0.7975708502024291 | 0.0 |
| 8 | 3 | 18.6 ± 0.0 | 0.145 ± 0.000 | 41767.2 ± 13.7 | 41881.6 ± 89.8 | 109.6 ± 0.1 | 112.5 ± 0.2 | 0.7975708502024291 | 0.0 |
| 16 | 3 | 18.6 ± 0.0 | 0.145 ± 0.000 | 95741.8 ± 4.0 | 97446.0 ± 1.7 | 109.5 ± 0.0 | 112.4 ± 0.1 | 0.7975708502024291 | 0.0 |
| 32 | 3 | 18.6 ± 0.0 | 0.145 ± 0.000 | 207040.7 ± 18.7 | 207099.9 ± 7.0 | 109.5 ± 0.0 | 112.4 ± 0.1 | 0.7975708502024291 | 0.0 |

Caveats: CORE_B_FIXED_DEPLOYMENT, KNEE_POINT_ESTIMATE, KV_USAGE_INCLUDES_GDN.

### Verdicts (copied)

- H-B1: not refuted; units {'supported': 3}
- H-B2: not refuted; units {'tested': 3}
- H-B3: not refuted; units {'tested': 3}
- H-B4: not refuted; units {'tested': 2}

## Quality — wsl2-cuda, -c 512

| config | KLD ± SE | PPL(Q) ± SE | same top-p % | KLD p99 |
|---|---|---|---|---|
| 4B-Q4_K_M | 0.0458 ± 0.000578 | 10.3916 ± 0.0753 | 90.60 | 0.357 |
| 4B-Q6_K | 0.00546 ± 0.000212 | 9.9949 ± 0.0719 | 96.67 | 0.0332 |
| 4B-Q8_0 | 0.00203 ± 0.000218 | 9.9573 ± 0.0717 | 98.34 | 0.00894 |
| 9B-Q4_K_M | 0.0574 ± 0.000844 | 8.1983 ± 0.0548 | 89.95 | 0.454 |
| 9B-Q6_K | 0.00726 ± 0.000427 | 8.2139 ± 0.0557 | 96.80 | 0.0341 |
| 9B-Q8_0 | 0.00202 ± 0.000223 | 8.1950 ± 0.0555 | 98.39 | 0.00864 |

Self-KLD floor: {'4B': -0.0, '9B': 0.0}; BF16 PPL ± SE: {'4B': [9.9454, 0.07154], '9B': [8.1862, 0.05544]}.
H-A5: not refuted.
- KLD 4B Q6_K > Q8_0: p 0, p_adj 0, reject True, direction_ok True
- KLD 4B Q4_K_M > Q6_K: p 0, p_adj 0, reject True, direction_ok True
- KLD 9B Q6_K > Q8_0: p 0, p_adj 0, reject True, direction_ok True
- KLD 9B Q4_K_M > Q6_K: p 0, p_adj 0, reject True, direction_ok True
- paired dNLL 9B-Q4_K_M - 4B-Q8_0 < 0: p 0.0001, p_adj 0.0001, reject True, direction_ok True

Relations (Holm only; unmeasured is never a tie):

| a | b | relation of a | source |
|---|---|---|---|
| 4B-Q8_0 | 4B-Q6_K | better | KLD 4B Q6_K > Q8_0 |
| 4B-Q8_0 | 4B-Q4_K_M | better | chain:KLD 4B Q6_K > Q8_0+KLD 4B Q4_K_M > Q6_K |
| 4B-Q8_0 | 9B-Q8_0 | unmeasured | None |
| 4B-Q8_0 | 9B-Q6_K | unmeasured | None |
| 4B-Q8_0 | 9B-Q4_K_M | worse | paired dNLL 9B-Q4_K_M - 4B-Q8_0 < 0 |
| 4B-Q6_K | 4B-Q4_K_M | better | KLD 4B Q4_K_M > Q6_K |
| 4B-Q6_K | 9B-Q8_0 | unmeasured | None |
| 4B-Q6_K | 9B-Q6_K | unmeasured | None |
| 4B-Q6_K | 9B-Q4_K_M | unmeasured | None |
| 4B-Q4_K_M | 9B-Q8_0 | unmeasured | None |
| 4B-Q4_K_M | 9B-Q6_K | unmeasured | None |
| 4B-Q4_K_M | 9B-Q4_K_M | unmeasured | None |
| 9B-Q8_0 | 9B-Q6_K | better | KLD 9B Q6_K > Q8_0 |
| 9B-Q8_0 | 9B-Q4_K_M | better | chain:KLD 9B Q6_K > Q8_0+KLD 9B Q4_K_M > Q6_K |
| 9B-Q6_K | 9B-Q4_K_M | better | KLD 9B Q4_K_M > Q6_K |

Downstream (descriptive, no multiplicity correction; ci95 copied, not re-verified):

| task | size | pair | n | Δacc | ci95 | resolved | flip rate |
|---|---|---|---|---|---|---|---|
| arc | 4B | BF16 → Q8_0 | 869 | 0.0035 | [-0.0012, 0.0092] | False | 0.0058 |
| arc | 4B | BF16 → Q6_K | 869 | 0.0069 | [-0.0012, 0.0150] | False | 0.0138 |
| arc | 4B | BF16 → Q4_K_M | 869 | -0.0299 | [-0.0460, -0.0150] | True | 0.0552 |
| arc | 4B | Q8_0 → Q6_K | 869 | 0.0035 | [-0.0046, 0.0115] | False | 0.0150 |
| arc | 4B | Q6_K → Q4_K_M | 869 | -0.0368 | [-0.0529, -0.0219] | True | 0.0552 |
| arc | 9B | BF16 → Q8_0 | 869 | 0.0023 | [-0.0046, 0.0092] | False | 0.0115 |
| arc | 9B | BF16 → Q6_K | 869 | -0.0092 | [-0.0184, -0.0012] | True | 0.0161 |
| arc | 9B | BF16 → Q4_K_M | 869 | 0.0012 | [-0.0138, 0.0161] | False | 0.0518 |
| arc | 9B | Q8_0 → Q6_K | 869 | -0.0115 | [-0.0207, -0.0035] | True | 0.0161 |
| arc | 9B | Q6_K → Q4_K_M | 869 | 0.0104 | [-0.0046, 0.0253] | False | 0.0518 |
| hellaswag | 4B | BF16 → Q8_0 | 4000 | 0.0008 | [-0.0018, 0.0032] | False | 0.0067 |
| hellaswag | 4B | BF16 → Q6_K | 4000 | -0.0013 | [-0.0043, 0.0018] | False | 0.0097 |
| hellaswag | 4B | BF16 → Q4_K_M | 4000 | -0.0027 | [-0.0083, 0.0027] | False | 0.0312 |
| hellaswag | 4B | Q8_0 → Q6_K | 4000 | -0.0020 | [-0.0055, 0.0013] | False | 0.0115 |
| hellaswag | 4B | Q6_K → Q4_K_M | 4000 | -0.0015 | [-0.0070, 0.0040] | False | 0.0325 |
| hellaswag | 9B | BF16 → Q8_0 | 4000 | 0.0000 | [-0.0022, 0.0022] | False | 0.0055 |
| hellaswag | 9B | BF16 → Q6_K | 4000 | -0.0010 | [-0.0040, 0.0018] | False | 0.0085 |
| hellaswag | 9B | BF16 → Q4_K_M | 4000 | -0.0040 | [-0.0090, 0.0008] | False | 0.0245 |
| hellaswag | 9B | Q8_0 → Q6_K | 4000 | -0.0010 | [-0.0040, 0.0022] | False | 0.0100 |
| hellaswag | 9B | Q6_K → Q4_K_M | 4000 | -0.0030 | [-0.0080, 0.0020] | False | 0.0265 |
| mmlu | 4B | BF16 → Q8_0 | 3997 | -0.0005 | [-0.0030, 0.0020] | False | 0.0065 |
| mmlu | 4B | BF16 → Q6_K | 3997 | -0.0020 | [-0.0058, 0.0018] | False | 0.0145 |
| mmlu | 4B | BF16 → Q4_K_M | 3997 | -0.0073 | [-0.0138, -0.0008] | True | 0.0443 |
| mmlu | 4B | Q8_0 → Q6_K | 3997 | -0.0015 | [-0.0053, 0.0023] | False | 0.0145 |
| mmlu | 4B | Q6_K → Q4_K_M | 3997 | -0.0053 | [-0.0120, 0.0013] | False | 0.0443 |
| mmlu | 9B | BF16 → Q8_0 | 3997 | 0.0015 | [-0.0010, 0.0040] | False | 0.0060 |
| mmlu | 9B | BF16 → Q6_K | 3997 | -0.0013 | [-0.0048, 0.0023] | False | 0.0128 |
| mmlu | 9B | BF16 → Q4_K_M | 3997 | -0.0015 | [-0.0078, 0.0050] | False | 0.0430 |
| mmlu | 9B | Q8_0 → Q6_K | 3997 | -0.0028 | [-0.0065, 0.0013] | False | 0.0153 |
| mmlu | 9B | Q6_K → Q4_K_M | 3997 | -0.0003 | [-0.0068, 0.0063] | False | 0.0443 |
| winogrande | 4B | BF16 → Q8_0 | 1267 | 0.0047 | [-0.0016, 0.0110] | False | 0.0126 |
| winogrande | 4B | BF16 → Q6_K | 1267 | 0.0047 | [-0.0024, 0.0118] | False | 0.0174 |
| winogrande | 4B | BF16 → Q4_K_M | 1267 | 0.0055 | [-0.0087, 0.0197] | False | 0.0639 |
| winogrande | 4B | Q8_0 → Q6_K | 1267 | 0.0000 | [-0.0087, 0.0095] | False | 0.0253 |
| winogrande | 4B | Q6_K → Q4_K_M | 1267 | 0.0008 | [-0.0126, 0.0142] | False | 0.0608 |
| winogrande | 9B | BF16 → Q8_0 | 1267 | -0.0008 | [-0.0063, 0.0047] | False | 0.0103 |
| winogrande | 9B | BF16 → Q6_K | 1267 | -0.0008 | [-0.0095, 0.0079] | False | 0.0245 |
| winogrande | 9B | BF16 → Q4_K_M | 1267 | 0.0032 | [-0.0118, 0.0174] | False | 0.0710 |
| winogrande | 9B | Q8_0 → Q6_K | 1267 | 0.0000 | [-0.0079, 0.0079] | False | 0.0221 |
| winogrande | 9B | Q6_K → Q4_K_M | 1267 | 0.0039 | [-0.0103, 0.0182] | False | 0.0671 |

## What the evidence does not support

- quality on Metal: not measured (020)
- long-context quality: not measured (-c 512, 005)
- Core B quality: not measured (W4A16; 003/008)
- multi-user serving on the Mac: not measured (SPEC: Core B only on NVIDIA)
- core_a/macos-metal L=2048 A-Q: incomparable
- core_a/macos-metal L=8192 A-Q: incomparable
- core_a/macos-metal L=32768 A-Q: incomparable
- core_a/wsl2-cuda L=2048 A-Q: incomparable
- core_a/wsl2-cuda L=8192 A-Q: incomparable
- core_a/wsl2-cuda L=32768 A-Q: incomparable
- quality 4B-Q8_0 vs 9B-Q8_0: unmeasured
- quality 4B-Q8_0 vs 9B-Q6_K: unmeasured
- quality 4B-Q6_K vs 9B-Q8_0: unmeasured
- quality 4B-Q6_K vs 9B-Q6_K: unmeasured
- quality 4B-Q6_K vs 9B-Q4_K_M: unmeasured
- quality 4B-Q4_K_M vs 9B-Q8_0: unmeasured
- quality 4B-Q4_K_M vs 9B-Q6_K: unmeasured
- quality 4B-Q4_K_M vs 9B-Q4_K_M: unmeasured
- downstream ci95: copied, not re-verified

## Caveat codes

- CONTAMINATION: the 4B vs 9B WikiText comparison may partly reflect memorisation (005)
- CONTRARY_TO_PREDICTION: a Holm-resolved quality relation runs against the H-A5 prediction
- CORE_A_B_SEPARATE: Core A and Core B are not comparable (different engine, precision, format)
- CORE_B_FIXED_DEPLOYMENT: Core B is one model / precision / server config; only client concurrency varies
- CROSS_SIZE_UNMEASURED: 4B vs 9B quality is unmeasured except the paired H-A5(a) test (005)
- DOWNSTREAM_CI_NOT_REVERIFIED: downstream ci95 / resolved are copied from analysis.json (not re-verified); descriptive, no multiplicity correction (005)
- H_B1_UNREFUTABLE: the grid does not reach 4 x C_knee, so H-B1 cannot be refuted at this L
- KNEE_AT_GRID_TOP: C_knee is the largest grid C
- KNEE_POINT_ESTIMATE: C_knee is a point estimate on the restart means (frozen definition, no noise rule)
- KV_USAGE_INCLUDES_GDN: vLLM KV cache usage % includes the GDN state blocks (§3)
- MAC_POWER_CONDITION: Mac session conditions: AC with the Apple 70W USB-C Power Adapter, one powermode per session (019)
- MIXED_AT_KNEE_ONLY: the mixed request mix is measured only at the uniform C_knee (007)
- N3_NOISE_RULE: beyond noise = |x - y| > 2 sqrt(SE_x^2 + SE_y^2) with n = 3 processes (§2); lenient at df = 2, applied as frozen
- NOISE_CYCLE: dominance within noise forms a cycle; all its members are kept
- NO_IMATRIX_Q4: Q4_K_M is a plain k-quant without imatrix (004)
- NO_RANKING: no Mac vs PC ranking (SPEC): each platform's values are only compared within that platform
- NO_TESTABLE_UNITS: the verdict has no testable unit behind it
- PROVISIONAL: a configuration at this level is not measured / error / incomplete and could change the pick
- QUALITY_CTX_512: quality measured at -c 512 (005); long-context quality not measured
- QUALITY_PLATFORM_MAC: quality measured on wsl2-cuda (020), applied as a model x quant property (SPEC); not measured on Metal
- QUALITY_PLATFORM_RTX: quality measured on this platform (wsl2-cuda, 020)
- SPILL_EXCLUDED: a configuration that spills to system memory is not a fit (004) and is never a candidate
- TIE_NOT_EQUIVALENCE: tie = not resolved / within noise; no equivalence is claimed
- TTFT_ENGINE_ONLY: TTFT is llama-bench engine time: no tokenize / HTTP / sampling (§3)

"""Tests for quality/quality_parse.py (decision 005).

llama-perplexity exits 0 on almost every error path, so success = positive
markers + expected counts + no pinned LOG_ERR message. Texts below are
SYNTHETIC, built from the exact printf formats in the pinned
tools/perplexity/perplexity.cpp; replace with real pilot captures later.
"""

import pytest

import quality_parse as qp

KLD_OUT = """
kl_divergence: computing over 580 chunks, n_ctx=512, batch_size=2048, n_seq=4
====== Perplexity statistics ======
Mean PPL(Q)                   :   8.123456 ±   0.051234
Mean PPL(base)                :   8.110000 ±   0.051000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.95%
Mean ln(PPL(Q)/PPL(base))     :   0.001658 ±   0.000210
Mean PPL(Q)/PPL(base)         :   1.001659 ±   0.000210
Mean PPL(Q)-PPL(base)         :   0.013456 ±   0.001700

====== KL divergence statistics ======
Mean    KLD:   0.001400 ±   0.000010
Maximum KLD:   0.812345
99.9%   KLD:   0.090000
99.0%   KLD:   0.020000
95.0%   KLD:   0.006000
90.0%   KLD:   0.003500
Median  KLD:   0.000600
10.0%   KLD:   0.000010
Minimum KLD:  -0.000001

====== Token probability statistics ======
Mean    Δp: -0.010 ± 0.004 %
Maximum Δp: 40.123%
RMS Δp    :  1.234 ± 0.010 %
Same top p: 97.500 ± 0.040 %
"""

PPL_OUT = "perplexity: calculating perplexity over 3 chunks, n_ctx=512, batch_size=2048, n_seq=4\n" \
          "       0  9.0000  2.197225  0.100000\n" \
          "     512  8.5000  2.140066  0.100000\n" \
          "    1024  8.2000  2.104134  0.100000\n" \
          "\nFinal estimate: PPL = 8.2000 +/- 0.05000\n"

MC_OUT = "multiple_choice_score: there are 4 tasks in prompt\n" \
         "1\t100.00000000\n2\t50.00000000\n3\t66.66666667\n4\t75.00000000\n" \
         "\n Final result: 75.0000 +/- 21.6506\n"

WG_OUT = "winogrande_score : calculating winogrande score over selected tasks.\n" \
         "1\t100.0000\t -1.234567   -2.345678  1  1\n" \
         "2\t50.0000\t -3.000000   -1.000000  1  2\n" \
         "3\t66.6667\t -0.500000   -0.900000  1  1\n" \
         "Final Winogrande score(3 tasks): 66.6667 +/- 27.2166\n"


def test_parse_kld():
    r = qp.parse_kld(KLD_OUT)
    assert r["n_chunks"] == 580
    assert r["kld_mean"] == 0.0014 and r["kld_mean_se"] == 0.00001
    assert r["kld_p99"] == 0.02 and r["kld_median"] == 0.0006
    assert r["same_top_p_pct"] == 97.5 and r["rms_dp_pct"] == 1.234
    assert r["ppl_q"] == 8.123456 and r["ppl_base"] == 8.11
    assert r["ln_ppl_ratio"] == 0.001658 and r["ln_ppl_ratio_se"] == 0.00021


def test_parse_ppl_chunks_recovers_per_chunk_nll():
    r = qp.parse_ppl(PPL_OUT, scored_per_chunk=255)
    assert r["n_chunks"] == 3 and r["ppl"] == 8.2
    assert len(r["chunk_nll"]) == 3
    assert r["chunk_nll"][0] == pytest.approx(2.197225)
    # chunk 2 mean NLL = (2.140066*510 - 2.197225*255) / 255
    assert r["chunk_nll"][1] == pytest.approx(2.140066 * 2 - 2.197225, abs=1e-5)


def test_parse_mc_per_task_correctness():
    r = qp.parse_mc(MC_OUT)
    assert r["n_tasks"] == 4 and r["accuracy_pct"] == 75.0
    assert r["correct"] == [1, 0, 1, 1]


def test_parse_winogrande():
    r = qp.parse_winogrande(WG_OUT)
    assert r["n_tasks"] == 3 and r["accuracy_pct"] == 66.6667
    assert r["correct"] == [1, 0, 1]


def test_classify_ok_requires_markers_and_counts():
    ok = qp.classify("kld", 0, KLD_OUT, "", expected={"n_chunks": 580})
    assert ok["outcome"] == "ok" and ok["metrics"]["kld_mean"] == 0.0014
    wrong = qp.classify("kld", 0, KLD_OUT, "", expected={"n_chunks": 584})
    assert wrong["outcome"] == "error" and "n_chunks" in wrong["reason"]


def test_classify_exit0_but_log_err_is_error():
    stderr = "kl_divergence: inconsistent vocabulary (151936 vs 248320)\n"
    r = qp.classify("kld", 0, KLD_OUT, stderr, expected={"n_chunks": 580})
    assert r["outcome"] == "error" and "inconsistent vocabulary" in r["reason"]


def test_classify_missing_marker_is_error():
    r = qp.classify("mc", 0, "multiple_choice_score: there are 4 tasks\n1\t100.0\n", "", expected={"n_tasks": 4})
    assert r["outcome"] == "error"


def test_classify_oom_is_reported_as_oom():
    stderr = "ggml_metal_buffer_init: error: failed to allocate buffer, size =  5120.00 MiB\n" \
             "main: unable to load model\n"
    r = qp.classify("ppl", 1, "", stderr, expected={"n_chunks": 580})
    assert r["outcome"] == "oom"


def test_classify_kld_base_needs_final_estimate_and_chunks():
    out = "perplexity: saving all logits to data/kld/4B.kld\n" + PPL_OUT.replace(" 3 chunks", " 580 chunks")
    r = qp.classify("kld_base", 0, out, "", expected={"n_chunks": 580})
    assert r["outcome"] == "ok" and r["metrics"]["ppl"] == 8.2


def test_classify_winogrande_count():
    assert qp.classify("winogrande", 0, WG_OUT, "", expected={"n_tasks": 3})["outcome"] == "ok"
    assert qp.classify("winogrande", 0, WG_OUT, "", expected={"n_tasks": 1267})["outcome"] == "error"

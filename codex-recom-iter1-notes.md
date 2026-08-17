# codex-recom-iter1 — supervisor iteration-1 recommendations

Branch `codex-recom-iter1` = `exp4-train-ans` (answer-only loss, run-safety scaffolding)
+ merge of `new-exp2` (MedExQA generation, ILV token-transfer / abstention / OoD bridge)
+ one commit per supervisor concern. This file is the running record; the
concern → commit → files → run-tag mapping is at the end.

## Protocol (frozen 2026-08-17, BEFORE any iter1 run)

**Data.** MedExQA *derived* split: official dev (25) + test (940) pooled → 965 rows →
seed-42 shuffle → **175 test / 50 val / ~740 train**. This is NOT the official
benchmark split; it is frozen in `splits/medexqa-derived-seed42.csv` (content hash →
split) and every `load_exp_dataset("medexqa")` call verifies against it
(`utils/data.py::verify_medexqa_split_manifest`; generate once with
`python write-medexqa-split-manifest.py`, then `git add -f`).

**Selection vs evaluation.**
- **val (50)** = the *only* set used to choose anything: signal direction check,
  abstention thresholds at target coverages, calibrator (ILV → P(error)), NLI-label
  audit set (25–50 hand-checked examples). Selections are written to
  `results/abstention/frozen_<tag>.json` together with a config hash.
- **test (175)** = evaluated **once** per frozen configuration. Scripts require
  `--split test` explicitly and print a banner. Nothing is fitted, flipped, or thresholded
  on test. Statistics reported on test: AUROC + example-level bootstrap CI, accuracy /
  risk at the *frozen* coverages, AURC; likelihood-ratio tests are reported as analysis
  with no decisions derived.

**Pre-registered primary readout (abstention).**
- Score: `ilv_online_mean_last10` — ILV captured *during* generation (router forward
  hooks, same draw that routed the token) — with `ilv_online_mean`, `ilv_online_last`,
  `ilv_online_max` as secondaries; the retrospective second-pass score is reported
  separately as `ilv_posthoc_*` and labelled "post-hoc".
- Sign: fixed a priori — higher posterior variance ⇒ more uncertain ⇒ predicted
  *wrong* / *OoD*. AUROC < 0.5 is a negative result, never flipped.
- Label: `option_correct` (the option the *generated explanation itself* commits to vs
  gold letter). Independent secondary labels: `correct_probe` (MCQA letter-probe pass),
  `expl_entail_frac` / `expl_any_contra` (sentence-level NLI vs the gold explanations),
  `unjudgeable`, `truncated`. NLI never decides correctness on its own.
- Baselines every ILV score must beat: LM predictive entropy (mean/max), NLL per
  token, generation length, gate entropy of the actual routing; plus the arms
  in the baseline ladder (below).

**Routing at readout.** Paper-faithful S=35 stochastic routing is the primary
setting; deterministic posterior-mean routing is an ablation (it changes downstream
hidden states, hence later covariances and the generated trajectory).

**Baseline ladder (same readouts, same splits).** Stage-1 deterministic routers ·
untrained FCVR heads · pretrained-prior FCVR (existing β=0.01/0.1) · MAP-prior FCVR ·
KL mask {none (existing), attention, answer} · β=0. Compared on option accuracy,
explanation quality (BLEU / ROUGE-L / METEOR / SciBERT-BERTScore / unigram-F1),
runtime (tokens/s, wall-clock), ILV AUROC and OoD AUROC.

**Phase 2 (gated, not in iter1).** Calibration (Platt on val → Brier/NLL/ECE/AURC on
test) runs only if the frozen-protocol test AUROC CI excludes 0.5; a decoding /
abstention *policy* is implemented only after that.

## Protocol corrections vs the inherited pipeline (documented, both arms run)
- **Prior source.** ICML paper App. D.2: "freeze all model parameters — including the
  pre-trained router weights W_r"; §3.1 prior mean `l_det = uW_r`. So
  `prior_source=pretrained` is the paper-faithful reading. The inherited Stage-2a MAP
  router-tuning (`router-tuning.py`) is *not* in the paper text but likely produced its
  numbers; both priors are run and reported.
- **KL over positions.** The router latent exists at every routed token, so KL over
  prompt+answer tokens is part of the ELBO; KL over *padding* is a bug (fixed:
  attention-masked KL is the new default). Answer-only KL is an ablation of the
  weighting, not a correctness fix.
- **Online vs post-hoc ILV.** The earlier readout recomputed ILV in a second
  teacher-forced pass over the finished generation; that is a post-hoc score. Decoding-
  time ILV is now captured with forward hooks during `generate`.

## Concern → commit → files → run tags

| # | Supervisor concern | Commit | Main files | Run tags / outputs |
|---|---|---|---|---|
| – | branches diverged; new-exp2 lacked exp4 loss/provenance | `999be34` merge | `scripts/python/{kvq,fcvr}-tuning.py`, `utils/data.py`, `kvq-tuning-granite-medexqa.sh` | – |
| 7 | reproducibility (seed unused, no suffixes, no collision checks), derived split | `1a1cffa` provenance | `utils/__init__.py::seed_everything`, `utils/data.py` manifest, `write-medexqa-split-manifest.py`, `check-answer-only-labels.py`, `--split` in both readouts | `splits/medexqa-derived-seed42.csv`; val files carry `_val` |
| 1 | generated-output correctness not measured | `bf8bc0d` labels | `label_generation_correctness.py`, seqlevel schema in `analyze_token_signals.py` | `*_seqlevel_labeled.jsonl`, `*_labels_summary.json`, `results/labels/audit_<tag>.csv` |
| 2,3 | deterministic readout changes the model; retrospective ILV | `56b95a3` readout | `analyze_token_signals.py` (OnlineILVRecorder, `--routing`), `fcvr_input_level_ood_check.py` | tags `…-S35-s<seed>`; `ilv_online_*` vs `ilv_posthoc_*` |
| 4 (rev-2) | baseline ladder | `b524f95` baselines | `evaluate_fcvr.py::prepare_model_by_arm`, `--arm`, `uq_stats.py`, `baseline_ladder_report.py` | `det-S35-s42`, `untrained-S35-s42`, `beta0-S35-s42`; `results/reports/ladder_{val,test}.md` |
| 3 | MAP-router stage missing | `a70f557` prior | `scripts/python/router-tuning.py`, `router-tuning-granite-medexqa.sh`, `--map_suffix` | `router_weights/base/granite_medexqa-iter1`, FCVR `iter1-map-prior-beta0.01`, tag `mapprior-beta0.01-S35-s42` |
| 4 | full-sequence KL | `827716e` kl | `model/routers/fcvr.py::kl_divergence(mask)`, `fcvr-tuning.py --kl_mask` | FCVR `iter1-klattn-beta0.01`, `iter1-klans-beta0.01`; existing `pretrained-prior-beta0.01` = kl_mask none |
| 5 | OoD bridge detects formatting; MedMCQA over-filtered | `0a3f9fe` ood | `utils/prompt.py` canonicalisation, `fcvr_input_level_ood_check.py` (rewrite), `evaluate_fcvr.py` | `results/input_level_ood/input_ood_medexqa[_val]_<suffix>_<tag>{,-native}.json` |
| 6,8 | inverted AUROC as success; exploratory stats | `6d86b46` stats | `analyze_token_signals.py` (CIs, circular-shift null, no flip), `abstention_report.py` (`--select`/`--evaluate`/`--aggregate`) | `results/abstention/frozen_<tag>.json`, `eval_<tag>_s<seed>.{json,md}`, `aggregate_<tag>.json` |
| 5 (rev-2) | calibration, decoding later | `6c9fecf` calibration | `abstention_report.py --calibrate` (gated) | `results/abstention/calib_<tag>_<seed>_<method>.json` |
| – | driver + notes | (this commit) | `run-iter1-granite-medexqa.sh`, `fcvr-tuning-granite-medexqa.sh` (`--kl_mask none` explicit) | `logs/iter1-granite-medexqa-<ts>.log` |

Historical launchers (`fcvr-eval-granite-medexqa.sh`, `-mnt256.sh`, `fcvr-tuning-granite-medexqa.sh`) now pass the
old behaviour explicitly (`--routing deterministic --num_samples 1 --split test --inner_format native`,
`--kl_mask none`) so re-running them reproduces the existing files.

## How to run (quail-1)
```
source ~/.venvs/moe_env/bin/activate
pip install sacrebleu rouge-score nltk bert-score          # metric suite (once)
git add -f splits/medexqa-derived-seed42.csv               # after preflight writes it
PHASES=preflight,val bash run-iter1-granite-medexqa.sh    # -> inspect results/labels/audit_*.csv, hand-label 25-50 rows,
                                                           #    python label_generation_correctness.py --input <val seqlevel> --audit_import results/labels/audit_<tag>.csv --overwrite
                                                           #    inspect results/abstention/frozen_*.json
PHASES=test bash run-iter1-granite-medexqa.sh             # once
PHASES=baselines,prior,kl,ood,report bash run-iter1-granite-medexqa.sh
```
`SEEDS="42"` for a single inference seed; `SKIP_NLI=1` to skip the NLI judge; `ALLOW_EXISTING=1` to resume.

## Explicitly NOT done in iter1
- Training-seed sweep (Stage-1/Stage-2 seeds); only inference seeds {42,43,44}.
- LLM-as-judge (NLI cross-encoder chosen; audit CSV measures its trustworthiness).
- Any decoding / abstention *policy* (phase 2, gated on the frozen-protocol test AUROC).
- Re-training Stage 1; the existing `adapters/granite-medexqa` (Q/K/V LoRA) is the input to every arm.

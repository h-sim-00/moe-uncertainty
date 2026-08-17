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

(Concern → commit → files → run-tag table appended in the launcher commit.)

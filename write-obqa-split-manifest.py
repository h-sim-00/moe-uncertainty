"""Freeze (or verify) the obqa_gen DERIVED split used on branch OBQA-comparison.

`load_exp_dataset("obqa_gen")` (utils/data.py) replicates the legacy `obqa`
split byte-for-byte (pool = train + validation, [:5050], last 50 = val, first
5000 = train, test = official test) from the HF "additional" config (which
adds `fact1`), then drops train/val rows whose fact1 fails the word-count
filter. This script:
  1. cross-checks the obqa_gen split against the legacy `obqa` loader (the
     "main" config exp4-train-ans trained on): per split, the stripped
     obqa_gen ids must be an in-order subset of the legacy ids (test: equal in
     order) and `letter_question`/`gold_letter` must be byte-identical to the
     legacy `question`/`answer` -- this is the decisive main-vs-additional
     parity test for reusing the frozen exp4 arm-A weights;
  2. writes the assignment to splits/obqa_gen-derived-seed<seed>.csv (qhash,
     split, id, question_prefix) so it is immutable and checkable; every later
     call to `load_exp_dataset("obqa_gen")` verifies against the manifest and
     raises if the split moved.

Usage (on quail, moe_env):
    python write-obqa-split-manifest.py            # write if absent, else verify
    python write-obqa-split-manifest.py --force    # overwrite (only if you MEAN to move the split)
    python write-obqa-split-manifest.py --seed 42
Then `git add -f splits/obqa_gen-derived-seed42.csv` and commit.
"""
import argparse
import os

from utils import setup_environment
from utils.data import (
    load_exp_dataset,
    split_manifest_path,
    verify_split_manifest,
    write_split_manifest,
)

DATASET = "obqa_gen"
SKIP_ENV = "OBQA_GEN_SKIP_MANIFEST"
_ID_PREFIXES = ("obqa_gen_train_", "obqa_gen_validation_", "obqa_gen_test_")


def _strip_id(gid):
    for p in _ID_PREFIXES:
        if gid.startswith(p):
            return gid[len(p):]
    raise ValueError(f"obqa_gen id {gid!r} does not carry a known prefix {_ID_PREFIXES}")


def legacy_parity_check(gen_train, gen_val, gen_test, seed):
    """Raise unless the obqa_gen split is the legacy `obqa` split (minus the
    fact1-filtered train/val rows) with byte-identical prompts and answers."""
    leg_train, leg_val, leg_test = load_exp_dataset("obqa", seed=seed)
    for split_name, gen_rows, leg_rows, must_equal in (
        ("train", gen_train, leg_train, False),
        ("val", gen_val, leg_val, False),
        ("test", gen_test, leg_test, True),
    ):
        gen_ids = [_strip_id(r["id"]) for r in gen_rows]
        leg_ids = [r["id"] for r in leg_rows]
        if must_equal:
            if gen_ids != leg_ids:
                raise RuntimeError(
                    f"PARITY FAIL [{split_name}]: obqa_gen ids != legacy obqa ids "
                    f"({len(gen_ids)} vs {len(leg_ids)}; first diff at index "
                    f"{next((i for i, (a, b) in enumerate(zip(gen_ids, leg_ids)) if a != b), min(len(gen_ids), len(leg_ids)))}). "
                    f"The HF 'additional' config does not mirror 'main' -- fall back to joining fact1 onto 'main' by id.")
        else:
            # in-order subset (rows dropped only by the fact1 filter)
            it = iter(leg_ids)
            missing = [g for g in gen_ids if not any(g == l for l in it)]
            if missing:
                raise RuntimeError(
                    f"PARITY FAIL [{split_name}]: {len(missing)} obqa_gen id(s) absent from (or out of order vs) "
                    f"the legacy obqa {split_name} split, e.g. {missing[:5]}. "
                    f"The HF 'additional' config does not mirror 'main' -- fall back to joining fact1 onto 'main' by id.")
            print(f"  parity [{split_name}]: {len(gen_ids)}/{len(leg_ids)} legacy rows kept "
                  f"({len(leg_ids) - len(gen_ids)} dropped by the fact1 filter)")
        leg_by_id = {r["id"]: r for r in leg_rows}
        bad = [g["id"] for g in gen_rows
               if g["letter_question"] != leg_by_id[_strip_id(g["id"])]["question"]
               or g["gold_letter"] != leg_by_id[_strip_id(g["id"])]["answer"]]
        if bad:
            raise RuntimeError(
                f"PARITY FAIL [{split_name}]: letter_question/gold_letter differ from the legacy "
                f"question/answer on {len(bad)} row(s), e.g. {bad[:5]}. Arm-A prompt parity is broken.")
        print(f"  parity [{split_name}]: letter_question/gold_letter byte-identical to legacy obqa on all "
              f"{len(gen_rows)} rows")
    print("OK: obqa_gen split is the legacy obqa split (additional == main) -- exp4 arm-A pool parity holds.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    p.add_argument("--skip_parity", action="store_true",
                   help="Skip the legacy-obqa cross-check (only if you know why it cannot hold).")
    args = p.parse_args()

    setup_environment()
    path = split_manifest_path(DATASET, args.seed)
    exists = os.path.exists(path)
    if exists and not args.force:
        print(f"Manifest exists at {path}; verifying the current split against it ...")
        # load_exp_dataset() runs verify_split_manifest() itself and raises on mismatch.
        train, val, test = load_exp_dataset(DATASET, seed=args.seed)
        if not args.skip_parity:
            legacy_parity_check(train, val, test, args.seed)
        print("OK: split unchanged.")
        return

    # Skip the in-loader verification while (re)writing.
    os.environ[SKIP_ENV] = "1"
    train, val, test = load_exp_dataset(DATASET, seed=args.seed)
    del os.environ[SKIP_ENV]
    if not args.skip_parity:
        legacy_parity_check(train, val, test, args.seed)
    if exists:
        print(f"WARNING: --force overwriting {path}. Any results produced under the old split are no longer comparable.")
    write_split_manifest(DATASET, train, val, test, path)
    verify_split_manifest(DATASET, train, val, test, seed=args.seed, path=path)
    print(f"Done. Commit it: git add -f {os.path.relpath(path)}")


if __name__ == "__main__":
    main()

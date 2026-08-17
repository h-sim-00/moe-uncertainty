"""Freeze (or verify) the medmcqa_gen DERIVED split used on branch MedMCQA.

`load_exp_dataset("medmcqa_gen")` (utils/data.py) builds:
    train = MEDMCQA_GEN_N_TRAIN (30k) rows of the OFFICIAL MedMCQA train split that
            pass the explanation filter, stratified by subject_name;
    val   = MEDMCQA_GEN_N_VAL (1000) more such rows, stratified, disjoint  (selection set);
    test  = MEDMCQA_GEN_N_TEST (1000) rows of the OFFICIAL validation split, stratified
            (evaluated once per frozen configuration).
This script writes that assignment to splits/medmcqa_gen-derived-seed<seed>.csv
(qhash, split, id, question_prefix) so it is immutable and checkable; every later
call to `load_exp_dataset("medmcqa_gen")` verifies against the manifest and raises
if the split moved (RNG order, dedup, filter constants, upstream data).

Usage (on quail, moe_env):
    python write-medmcqa-split-manifest.py            # write if absent, else verify
    python write-medmcqa-split-manifest.py --force    # overwrite (only if you MEAN to move the split)
    python write-medmcqa-split-manifest.py --seed 42
Then `git add -f splits/medmcqa_gen-derived-seed42.csv` and commit.
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

DATASET = "medmcqa_gen"
SKIP_ENV = "MEDMCQA_GEN_SKIP_MANIFEST"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    args = p.parse_args()

    setup_environment()
    path = split_manifest_path(DATASET, args.seed)
    exists = os.path.exists(path)
    if exists and not args.force:
        print(f"Manifest exists at {path}; verifying the current split against it ...")
        # load_exp_dataset() runs verify_split_manifest() itself and raises on mismatch.
        load_exp_dataset(DATASET, seed=args.seed)
        print("OK: split unchanged.")
        return

    # Skip the in-loader verification while (re)writing.
    os.environ[SKIP_ENV] = "1"
    train, val, test = load_exp_dataset(DATASET, seed=args.seed)
    del os.environ[SKIP_ENV]
    if exists:
        print(f"WARNING: --force overwriting {path}. Any results produced under the old split are no longer comparable.")
    write_split_manifest(DATASET, train, val, test, path)
    verify_split_manifest(DATASET, train, val, test, seed=args.seed, path=path)
    print(f"Done. Commit it: git add -f {os.path.relpath(path)}")


if __name__ == "__main__":
    main()

"""Freeze (or verify) the MedExQA DERIVED split used throughout this repo.

Official MedExQA ships 25 dev + 940 test examples with no train split. This
codebase pools both (965 rows), seed-shuffles, and carves 175 test / 50 val /
~740 train (utils/data.py, `load_exp_dataset("medexqa")`). That is a derived
split and must be cited as such; this script writes it to
splits/medexqa-derived-seed<seed>.csv (qhash, split, id, question_prefix) so
the assignment is immutable and checkable. Every later call to
`load_exp_dataset("medexqa")` verifies against the manifest and raises if the
split moved.

Usage (on quail, moe_env):
    python write-medexqa-split-manifest.py            # write if absent, else verify
    python write-medexqa-split-manifest.py --force    # overwrite (only if you MEAN to move the split)
    python write-medexqa-split-manifest.py --seed 42
Then `git add -f splits/medexqa-derived-seed42.csv` and commit.
"""
import argparse
import os

from utils import setup_environment
from utils.data import (
    load_exp_dataset,
    medexqa_split_manifest_path,
    verify_medexqa_split_manifest,
    write_medexqa_split_manifest,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    args = p.parse_args()

    setup_environment()
    path = medexqa_split_manifest_path(args.seed)
    exists = os.path.exists(path)
    if exists and not args.force:
        print(f"Manifest exists at {path}; verifying the current split against it ...")
        # load_exp_dataset() runs verify_medexqa_split_manifest() itself and raises on mismatch.
        load_exp_dataset("medexqa", seed=args.seed)
        print("OK: split unchanged.")
        return

    # Skip the in-loader verification while (re)writing.
    os.environ["MEDEXQA_SKIP_MANIFEST"] = "1"
    train, val, test = load_exp_dataset("medexqa", seed=args.seed)
    del os.environ["MEDEXQA_SKIP_MANIFEST"]
    if exists:
        print(f"WARNING: --force overwriting {path}. Any results produced under the old split are no longer comparable.")
    write_medexqa_split_manifest(train, val, test, path)
    verify_medexqa_split_manifest(train, val, test, seed=args.seed, path=path)
    print(f"Done. Commit it: git add -f {os.path.relpath(path)}")


if __name__ == "__main__":
    main()

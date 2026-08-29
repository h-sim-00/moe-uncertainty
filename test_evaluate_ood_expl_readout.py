"""CPU-only unit tests for the pure pieces of evaluate_ood_expl_readout.py and
the scienceqa / ecqa / aqua_rat loaders (utils/data.py). No model, no GPU.

    python test_evaluate_ood_expl_readout.py
"""

import math
import os
import tempfile
import unittest

import numpy as np

from evaluate_ood_expl_readout import (
    ARM_KEYS,
    DOMAIN_SEED_INDEX,
    NINE_SCORES,
    blocks_from_rows,
    build_spans,
    categories_from_spans,
    choice_mask,
    explanation_aggregates,
    letter_entropy_of,
    masked_choice_probs,
    normalized_entropy,
    online_rows,
    output_paths,
    parse_args,
    validate_args,
)
from evaluate_ilv_ood_arms import check_output_collisions
from utils.data import (
    LETTERS5,
    aqua_normalize_rationale,
    aqua_row_to_example,
    aqua_strip_option_prefix,
    ecqa_join,
    mcqa_eval_record,
    scienceqa_row_to_example,
)


class ChoiceTests(unittest.TestCase):
    def test_four_choice_mask_matches_plain_softmax(self):
        z = np.array([1.0, 0.5, -0.2, 2.0, 5.0])          # E has the largest logit but is masked
        p = masked_choice_probs(z, 4)
        ref = np.exp(z[:4] - z[:4].max()); ref /= ref.sum()
        np.testing.assert_allclose(p[:4], ref, rtol=1e-12)
        self.assertEqual(p[4], 0.0)
        self.assertTrue(choice_mask(4).tolist() == [True] * 4 + [False])
        self.assertTrue(choice_mask(5).all())

    def test_normalized_entropy_bounds(self):
        h4 = letter_entropy_of(np.full(5, 0.0) + np.array([0.25] * 4 + [0.0]))
        self.assertAlmostEqual(normalized_entropy(h4, 4), 1.0)
        h5 = letter_entropy_of(np.full(5, 0.2))
        self.assertAlmostEqual(normalized_entropy(h5, 5), 1.0)
        self.assertGreater(normalized_entropy(h4, 5), 0.0)
        self.assertLess(normalized_entropy(h4, 5), 1.0)
        self.assertEqual(letter_entropy_of(np.array([1.0, 0, 0, 0, 0])), 0.0)


class SpanTests(unittest.TestCase):
    def test_spans_with_eos(self):
        s = build_spans(n_prompt=10, n_letter=1, n_marker=3, n_expl=4, has_eos=True)
        self.assertEqual(s["letter"], [10]); self.assertEqual(s["marker"], [11, 12, 13])
        self.assertEqual(s["explanation"], [14, 15, 16, 17]); self.assertEqual(s["eos"], [18])
        self.assertEqual(s["seq_len"], 19)
        cats = categories_from_spans(s, 19)
        self.assertEqual(cats[8], "prompt"); self.assertEqual(cats[9], "prompt_final")
        self.assertEqual(cats[10], "letter"); self.assertEqual(cats[13], "marker")
        self.assertEqual(cats[14], "explanation"); self.assertEqual(cats[18], "eos")

    def test_spans_without_eos_and_single_token(self):
        s = build_spans(5, 1, 3, 1, has_eos=False)
        self.assertEqual(s["explanation"], [9]); self.assertEqual(s["eos"], [])
        self.assertEqual(s["seq_len"], 10)
        with self.assertRaises(ValueError):
            categories_from_spans(s, 11)

    def test_online_rows_mapping(self):
        class Rec:
            layers = [0, 1]
            calls = {0: [np.array([1., 2., 3.]), np.array([4.]), np.array([5.])],
                     1: [np.array([10., 20., 30.]), np.array([40.]), np.array([50.])]}
            gate_ent = {l: [np.zeros(3), np.zeros(1), np.zeros(1)] for l in (0, 1)}
        P, G = 3, 3
        o = online_rows(Rec(), P, G)
        self.assertEqual(o["ilv_mean"].shape, (6,))
        np.testing.assert_allclose(o["ilv_mean"][:5], [5.5, 11., 16.5, 22., 27.5])
        self.assertTrue(np.isnan(o["ilv_mean"][5]))       # last generated token never forwarded
        Rec.calls[0] = Rec.calls[0][:2]                     # call-count mismatch -> None
        self.assertIsNone(online_rows(Rec(), P, G))


class AggregateTests(unittest.TestCase):
    def test_nine_scores_hand_computed(self):
        s = build_spans(n_prompt=3, n_letter=1, n_marker=2, n_expl=5, has_eos=True)   # seq 12
        cats = categories_from_spans(s, 12)
        ilv = np.array([1, 2, 30, 4, 5, 6, 32.1, 33.4, 31.0, 35.2, 32.8, 7.0])
        ent = np.arange(12, dtype=float)
        sur = np.arange(12, dtype=float) * 10; sur[-1] = np.nan
        a = explanation_aggregates(ilv, ent, sur, s, cats)
        self.assertEqual(set(NINE_SCORES) <= set(a), True)
        self.assertAlmostEqual(a["expl_ilv_mean"], np.mean([32.1, 33.4, 31.0, 35.2, 32.8]))
        self.assertAlmostEqual(a["expl_ilv_max"], 35.2); self.assertAlmostEqual(a["expl_ilv_secondmax"], 33.4)
        self.assertAlmostEqual(a["expl_ilv_last10_mean"], a["expl_ilv_mean"])
        self.assertAlmostEqual(a["expl_entropy_mean"], np.mean([6, 7, 8, 9, 10]))
        self.assertAlmostEqual(a["expl_surprisal_mean"], np.mean([60, 70, 80, 90, 100]))
        self.assertAlmostEqual(a["prompt_final_ilv"], 30.0); self.assertAlmostEqual(a["letter_ilv"], 4.0)
        self.assertAlmostEqual(a["eos_ilv"], 7.0); self.assertAlmostEqual(a["marker_ilv_mean"], 5.5)

    def test_empty_explanation_and_no_eos_are_nan(self):
        s = build_spans(3, 1, 2, 0, has_eos=False)
        cats = categories_from_spans(s, 6)
        a = explanation_aggregates(np.arange(6.0), np.zeros(6), np.zeros(6), s, cats)
        for k in ("expl_ilv_mean", "expl_ilv_max", "expl_ilv_secondmax", "expl_ilv_last10_mean",
                  "expl_entropy_mean", "expl_surprisal_mean", "eos_ilv"):
            self.assertTrue(math.isnan(a[k]), k)
        self.assertEqual(a["prompt_final_ilv"], 2.0)

    def test_blocks_from_rows_uses_numeric_fields_only(self):
        rows = [{"arm": "armA-letter", "dataset": "obqa_gen", "id": "x", "expl_ilv_mean": 1.0, "ok": True,
                 "eos_ilv": None},
                {"arm": "armA-letter", "dataset": "obqa_gen", "id": "y", "expl_ilv_mean": 2.0, "ok": False,
                 "eos_ilv": 3.0}]
        b = blocks_from_rows(rows)["armA-letter"]["obqa_gen"]
        self.assertEqual(b["ids"], ["x", "y"])
        np.testing.assert_allclose(b["scores"]["expl_ilv_mean"], [1.0, 2.0])
        self.assertTrue(np.isnan(b["scores"]["eos_ilv"][0])); self.assertNotIn("ok", b["scores"])


class AquaTests(unittest.TestCase):
    def test_option_prefix(self):
        self.assertEqual(aqua_strip_option_prefix(["A)125", "B) 150", "C)225", "D)250", "E)275"]),
                         ["125", "150", "225", "250", "275"])
        self.assertIsNone(aqua_strip_option_prefix(["B)1", "A)2", "C)3", "D)4", "E)5"]))   # out of order
        self.assertIsNone(aqua_strip_option_prefix(["125", "B)150", "C)225", "D)250", "E)275"]))
        self.assertIsNone(aqua_strip_option_prefix(["A)1", "B)2", "C)3", "D)4"]))          # 4 options
        self.assertIsNone(aqua_strip_option_prefix(["A)A)1", "B)2", "C)3", "D)4", "E)5"])) # double label

    def test_terminal_markers(self):
        cases = {
            "Profit = 125. Answer: C": ("Profit = 125.", "C"),
            "Total 40 km/h.\nCorrect answer - A": ("Total 40 km/h.", "A"),
            "so x = 7 CORRECT OPTION: OPTION E": ("so x = 7", "E"),
            "hence 12 apples. Choice B": ("hence 12 apples.", "B"),
            "value is 18.\nD": ("value is 18.", "D"),
            "Profit per bag = 0.25 Total profit = 125 Answer is A.": ("Profit per bag = 0.25 Total profit = 125", "A"),
            "The correct option is (B).": ("", "B"),
            "hence answer = 42 (option D)": ("hence answer = 42", "D"),
        }
        for raw, (want_text, want_letter) in cases.items():
            text, letter, _ = aqua_normalize_rationale(raw)
            self.assertEqual(letter, want_letter, raw)
            self.assertEqual(text, want_text, raw)

    def test_algebraic_letters_untouched(self):
        for raw in ("Let A = 5 and B = 7, then A", "x = A + B", "speed is 2A", "so it is A"):
            text, letter, _ = aqua_normalize_rationale(raw)
            self.assertEqual(text, raw, raw)
            self.assertIsNone(letter, raw)

    def test_row_conversion_and_conflict(self):
        ex = {"question": "How much?", "options": ["A)125", "B)150", "C)225", "D)250", "E)275"],
              "rationale": "Profit = 125 Answer is B.", "correct": "A"}
        r, why = aqua_row_to_example(ex, 0, "test")
        self.assertIsNone(why)
        self.assertEqual(r["gold_letter"], "A"); self.assertEqual(r["n_choices"], 5)
        self.assertTrue(r["meta"]["rationale_label_conflict"])
        self.assertEqual(r["answer"], " Profit = 125")
        self.assertIn("E. 275", r["letter_question"]); self.assertNotIn("A. A)", r["letter_question"])
        self.assertTrue(r["letter_question"].endswith("\nAnswer:"))
        r2, why2 = aqua_row_to_example({**ex, "rationale": "Answer: A"}, 1, "test")
        self.assertIsNone(r2); self.assertEqual(why2, "empty_after_strip")


class ScienceQATests(unittest.TestCase):
    def base(self):
        return {"image": None, "hint": "", "question": "Which is a mammal?", "choices": ["cat", "cod", "crow", "frog"],
                "answer": 0, "solution": "Cats are mammals.", "subject": "natural science", "topic": "biology",
                "category": "c", "grade": "grade3", "task": "closed choice", "skill": "s", "lecture": ""}

    def test_accept_and_letter(self):
        r, why = scienceqa_row_to_example(self.base(), 7, "test")
        self.assertIsNone(why); self.assertEqual(r["gold_letter"], "A"); self.assertEqual(r["n_choices"], 4)
        self.assertEqual(r["id"], "scienceqa_test_7"); self.assertEqual(r["answer"], " Cats are mammals.")
        r, _ = scienceqa_row_to_example({**self.base(), "answer": 3}, 0, "test")
        self.assertEqual(r["gold_letter"], "D")

    def test_rejects(self):
        for patch, why in (({"image": {"bytes": b"x", "path": None}}, "has_image"),
                           ({"hint": "Look at the map."}, "has_hint"),
                           ({"choices": ["a", "b"]}, "not_4_choices"),
                           ({"solution": "  "}, "no_solution"),
                           ({"answer": 4}, "bad_answer"),
                           ({"answer": True}, "bad_answer")):
            r, got = scienceqa_row_to_example({**self.base(), **patch}, 0, "test")
            self.assertIsNone(r); self.assertEqual(got, why, patch)


class ECQATests(unittest.TestCase):
    def csqa(self):
        def row(i, key):
            return {"id": f"q{i}", "answerKey": key,
                    "question": {"stem": f"stem {i}", "question_concept": "c",
                                 "choices": [{"label": L, "text": f"opt{L}{i}"} for L in LETTERS5]}}
        return [row(2, "E"), row(1, "B"), row(3, "A")]          # deliberately not in split order

    def ecqa(self):
        return [{"id": "q1", "positives": ["p1a", "p1b"], "negatives": ["n1"], "explanation": "free 1"},
                {"id": "q3", "positives": ["p3"], "negatives": [], "explanation": "free 3"},
                {"id": "q2", "positives": ["p2"], "negatives": ["n2"], "explanation": "free 2"}]

    def test_join_by_id_preserves_answerkey(self):
        rows, funnel = ecqa_join(self.ecqa(), self.csqa(), ["q3", "q1"], "test")
        self.assertEqual([r["id"] for r in rows], ["ecqa_test_q3", "ecqa_test_q1"])
        self.assertEqual([r["gold_letter"] for r in rows], ["A", "B"])
        self.assertEqual(rows[1]["answer"], " p1a\np1b"); self.assertEqual(rows[1]["explanation_2"], "free 1")
        self.assertEqual(rows[0]["n_choices"], 5); self.assertIn("E. optE3", rows[0]["letter_question"])
        self.assertEqual(funnel["kept"], 2)

    def test_missing_ids_raise(self):
        with self.assertRaises(ValueError):
            ecqa_join(self.ecqa(), self.csqa(), ["q1", "q9"], "test")
        with self.assertRaises(ValueError):
            ecqa_join(self.ecqa() + [{"id": "zz", "positives": ["x"], "negatives": [], "explanation": ""}],
                      self.csqa(), ["q1"], "test")

    def test_hf_layout_accepted(self):
        hf = [{"id": "q1", "question": "stem 1", "question_concept": "c",
               "choices": {"label": LETTERS5, "text": [f"t{L}" for L in LETTERS5]}, "answerKey": "C"}]
        rows, _ = ecqa_join(self.ecqa()[:1], hf, ["q1"], "val")
        self.assertEqual(rows[0]["gold_letter"], "C")


class RecordAndArgsTests(unittest.TestCase):
    def test_canonical_prompt(self):
        r = mcqa_eval_record("x", "test", 1, "Q?", ["a", "b", "c", "d"], "B", "because")
        self.assertEqual(r["letter_question"], "Question: Q?\nChoices:\nA. a\nB. b\nC. c\nD. d\nAnswer:")
        with self.assertRaises(ValueError):
            mcqa_eval_record("x", "test", 1, "Q?", ["a", "b", "c", "d"], "E", "because")

    def test_stage_defaults_and_paths(self):
        args = parse_args(["--stage", "gen", "--output_dir", "o", "--tag", "t"])
        validate_args(args)
        self.assertEqual(args.ood_datasets, ["medexqa", "scienceqa", "ecqa", "aqua_rat"])
        p = output_paths(args)
        self.assertTrue(p["json"].endswith("t_test_data-s42_mc-s42_gen.json")); self.assertIn("texts", p)
        args = parse_args(["--stage", "stage1"]); validate_args(args)
        self.assertEqual(args.ood_datasets, ["arc_c", "arc_e", "medexqa"]); self.assertNotIn("pertoken", output_paths(args))
        with self.assertRaises(SystemExit):
            validate_args(parse_args(["--stage", "stage1", "--ood_datasets", "obqa"]))     # same source
        self.assertEqual(len(ARM_KEYS), 2); self.assertEqual(DOMAIN_SEED_INDEX["medexqa"], 1)

    def test_collision_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json"); open(p, "w").close()
            with self.assertRaises(SystemExit):
                check_output_collisions([p], False)
            check_output_collisions([p], True)


if __name__ == "__main__":
    unittest.main()

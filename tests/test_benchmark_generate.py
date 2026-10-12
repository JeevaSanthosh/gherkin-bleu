"""Each benchmark edit must do exactly what its name says, and nothing else."""

import json
from pathlib import Path

import pytest
from generate import BENIGN, BREAKING, EDITS, generate, to_pairs

from gherkin_bleu import GherkinBLEU

CASES = generate()
KEYWORDS = ("Given", "When", "Then", "And")


def steps(text):
    """(keyword, step text) for every step line; keyword is '' if the line has none."""
    out = []
    for line in text.splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("Scenario:"):
            continue
        word, _, rest = line.partition(" ")
        out.append((word, rest) if word in KEYWORDS else ("", line))
    return out


def body(case, edit):
    return steps(case["hypotheses"][edit])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_every_edit_does_what_it_says(case):
    ref = steps(case["reference"])
    assert [k for k, _ in ref] == ["Given", "And", "When", "Then", "And"]
    texts = [t for _, t in ref]

    assert GherkinBLEU._gherkin_tokenize(case["hypotheses"]["reformatted"]) == GherkinBLEU._gherkin_tokenize(case["reference"])
    assert case["hypotheses"]["reformatted"] != case["reference"]

    para = body(case, "paraphrase")
    assert [k for k, _ in para] == [k for k, _ in ref]
    changed = [i for i, (a, b) in enumerate(zip(para, ref)) if a != b]
    assert len(changed) == 2 and changed[-1] in (3, 4)

    assert body(case, "and_restyle") == [("Given", texts[0]), ("Given", texts[1]), ("When", texts[2]),
                                         ("Then", texts[3]), ("Then", texts[4])]
    assert body(case, "keyword_swap") == [("Then", texts[0]), ("And", texts[1]), ("When", texts[2]),
                                          ("Given", texts[3]), ("And", texts[4])]
    assert body(case, "steps_out_of_order") == [ref[0], ref[1], ref[3], ref[4], ref[2]]
    assert body(case, "missing_assertion") == ref[:3]

    wrong = body(case, "wrong_value")
    assert [i for i, (a, b) in enumerate(zip(wrong, ref)) if a != b] == [3] and wrong[3][0] == "Then"

    assert body(case, "keywords_missing") == [("", t) for t in texts]


def test_generation_is_deterministic():
    assert generate() == CASES


def test_scenarios_are_distinct():
    assert len({c["reference"] for c in CASES}) == len(CASES)


def test_labels():
    assert set(BENIGN) | set(BREAKING) == set(EDITS) and not set(BENIGN) & set(BREAKING)
    pairs = to_pairs(CASES)
    assert len(pairs) == len(CASES) * len(EDITS)
    assert all(p["breaking"] == (p["edit"] in BREAKING) for p in pairs)


def test_committed_pairs_match_the_generator():
    committed = Path(__file__).parent.parent / "benchmark" / "data" / "pairs.jsonl"
    lines = committed.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == to_pairs(CASES)

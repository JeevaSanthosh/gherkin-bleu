"""The order penalty on scenarios the benchmark never used.

The rule for matching steps (words plus the step's role) was settled while looking
at the benchmark's paraphrases, so it is also checked on 2,000 scenarios generated
with another seed. The README quotes these numbers.
"""

from generate import EDITS, generate

from gherkin_bleu import GherkinBLEU


def test_only_reordered_steps_and_nine_paraphrases_are_penalised():
    penalised = dict.fromkeys(EDITS, 0)
    for case in generate(n=2000, seed=20261012):
        for edit in EDITS:
            penalised[edit] += GherkinBLEU._order_penalty(case["reference"], case["hypotheses"][edit]) < 1
    assert penalised == {**dict.fromkeys(EDITS, 0), "steps_out_of_order": 2000, "paraphrase": 9}

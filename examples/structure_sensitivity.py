"""Compare G-BLEU with word-level BLEU on one scenario and three kinds of error.

Word-level BLEU here uses exactly the same maths as G-BLEU (n-grams 1-4, clipped
precision, the same smoothing and brevity penalty) and differs only in the
tokeniser, so any difference in the scores comes from G-BLEU's structural tokens.

    python examples/structure_sensitivity.py
"""

import math
import re
from collections import Counter

from gherkin_bleu import GherkinBLEU

REFERENCE = """@smoke
Scenario: Successful login
  Given I am on the login page
  When I enter a valid username and password
  Then I should see my dashboard"""

HYPOTHESES = {
    "Given and Then swapped": REFERENCE.replace("Given I am", "Then I am").replace("Then I should", "Given I should"),
    "One word changed": REFERENCE.replace("dashboard", "homepage"),
    "Step keywords missing": "\n".join(
        line.replace("Given ", "").replace("When ", "").replace("Then ", "") for line in REFERENCE.splitlines()
    ),
}


def word_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref, hyp = re.findall(r"\w+", reference.lower()), re.findall(r"\w+", hypothesis.lower())
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        total = sum(hyp_ngrams.values())
        clipped = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in hyp_ngrams.items())
        precisions.append(clipped / total if total else 0.0)
    geo_mean = math.exp(sum(math.log(p if p > 0 else 1e-5) for p in precisions) / max_n)
    penalty = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return penalty * geo_mean


if __name__ == "__main__":
    print(f"{'Error':<24}{'G-BLEU':>8}{'word BLEU':>11}")
    for name, hypothesis in HYPOTHESES.items():
        print(f"{name:<24}{GherkinBLEU.compute(REFERENCE, hypothesis):>8.3f}{word_bleu(REFERENCE, hypothesis):>11.3f}")

"""Score every benchmark pair with the metrics that run anywhere.

  gbleu      G-BLEU, this package
  word_bleu  the same BLEU maths with plain word tokens: isolates G-BLEU's tokeniser
  sacrebleu  standard sentence BLEU (sacreBLEU defaults: 13a tokeniser, exp smoothing)
  rouge_l    ROUGE-L F-measure (Google's rouge-score)

BERTScore needs a model download, so it has its own script: score_bertscore.py.

    pip install -e ".[benchmark]"
    python benchmark/score.py            # writes benchmark/data/scores.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from gherkin_bleu import GherkinBLEU

DATA = Path(__file__).parent / "data"


def word_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """G-BLEU's maths with plain word tokens (as in examples/structure_sensitivity.py)."""
    ref, hyp = re.findall(r"\w+", reference.lower()), re.findall(r"\w+", hypothesis.lower())
    if not ref or not hyp:
        return 0.0
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


def main() -> None:
    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    bleu = BLEU(effective_order=True)
    rouge = rouge_scorer.RougeScorer(["rougeL"])
    with (DATA / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "edit", "gbleu", "word_bleu", "sacrebleu", "rouge_l"])
        for p in pairs:
            ref, hyp = p["reference"], p["hypothesis"]
            writer.writerow([
                p["id"], p["edit"],
                f"{GherkinBLEU.compute(ref, hyp):.6f}",
                f"{word_bleu(ref, hyp):.6f}",
                f"{bleu.sentence_score(hyp, [ref]).score / 100:.6f}",
                f"{rouge.score(ref, hyp)['rougeL'].fmeasure:.6f}",
            ])
    print(f"Scored {len(pairs)} pairs -> {DATA / 'scores.csv'}")


if __name__ == "__main__":
    main()

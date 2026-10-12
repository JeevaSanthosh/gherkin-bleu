"""BERTScore F1 for every benchmark pair, on raw and on line-trimmed text.

Uses the bert-score defaults for English (roberta-large, layer 17), rescaled with
the published baseline so scores spread over a readable range. The first run
downloads the model, about 1.4 GB.

Two columns, from one model load:
  bertscore          the text as written. Indentation and blank lines become
                     tokens, so re-indenting a scenario lowers its score.
  bertscore_trimmed  each line stripped and blank lines dropped first. The other
                     metrics already ignore whitespace (trimming changes none of
                     their scores), so this is the like-for-like comparison.

    pip install -e ".[bertscore]"
    python benchmark/score_bertscore.py      # writes benchmark/data/scores_bertscore.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"


def trim(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def main() -> None:
    from bert_score import BERTScorer  # imported here so a missing install fails with a clear message

    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    hyps = [p["hypothesis"] for p in pairs]
    refs = [p["reference"] for p in pairs]
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, raw = scorer.score(hyps, refs, verbose=True, batch_size=16)
    _, _, trimmed = scorer.score([trim(h) for h in hyps], [trim(r) for r in refs], verbose=True, batch_size=16)
    with (DATA / "scores_bertscore.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "edit", "bertscore", "bertscore_trimmed"])
        for p, a, b in zip(pairs, raw.tolist(), trimmed.tolist()):
            writer.writerow([p["id"], p["edit"], f"{a:.6f}", f"{b:.6f}"])
    (DATA / "bertscore_config.txt").write_text(scorer.hash + "\n", encoding="utf-8")
    print(f"Scored {len(pairs)} pairs twice with {scorer.hash} -> {DATA / 'scores_bertscore.csv'}")


if __name__ == "__main__":
    main()

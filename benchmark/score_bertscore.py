"""BERTScore F1 for every benchmark pair.

Uses the bert-score defaults for English (roberta-large, layer 17), rescaled with
the published baseline so scores spread over a readable range. The first run
downloads the model, about 1.4 GB.

    pip install -e ".[bertscore]"
    python benchmark/score_bertscore.py      # writes benchmark/data/scores_bertscore.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"


def main() -> None:
    from bert_score import score  # imported here so a missing install fails with a clear message

    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    (_, _, f1), config = score(
        [p["hypothesis"] for p in pairs],
        [p["reference"] for p in pairs],
        lang="en",
        rescale_with_baseline=True,
        batch_size=16,
        verbose=True,
        return_hash=True,
    )
    with (DATA / "scores_bertscore.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "edit", "bertscore"])
        for p, value in zip(pairs, f1.tolist()):
            writer.writerow([p["id"], p["edit"], f"{value:.6f}"])
    (DATA / "bertscore_config.txt").write_text(config + "\n", encoding="utf-8")
    print(f"Scored {len(pairs)} pairs with {config} -> {DATA / 'scores_bertscore.csv'}")


if __name__ == "__main__":
    main()

# Benchmark

Does a metric score Gherkin that means the same thing above Gherkin that doesn't?

`generate.py` writes 200 scenarios across six domains (sign-in, basket, search,
password reset, transfers, profile), each with the same five-step shape, and
eight controlled edits of each one.

| Edit | Keeps the meaning? | What changes |
|---|---|---|
| `reformatted` | yes | Indentation and a blank line |
| `paraphrase` | yes | Two steps reworded, one of them an outcome |
| `and_restyle` | yes | `Given A / And B` written as `Given A / Given B` (and the same for `Then`) |
| `keyword_swap` | no | The first `Given` and the first `Then` swap keywords |
| `steps_out_of_order` | no | The outcome steps come before the action |
| `missing_assertion` | no | Both outcome steps deleted |
| `wrong_value` | no | One value in the first outcome changed, e.g. 3 items → 5 items |
| `keywords_missing` | no | Step keywords removed |

`tests/test_benchmark_generate.py` checks that every edit, on every scenario,
changes exactly what it says and nothing else.

## The measure

**Pairwise accuracy:** for each scenario, take every pair of one harmless edit and
one breaking edit (3 × 5 = 15 pairs). The metric gets a pair right if it scores
the harmless edit higher; a tie counts half. Comparing edits of the same
scenario cancels out scenario length and wording. 0.5 is a coin toss, 1.0 is
perfect. The 95% interval comes from resampling scenarios 10,000 times.

## Running it

```bash
pip install -e ".[benchmark]"
python benchmark/generate.py        # data/pairs.jsonl (already committed)
python benchmark/score.py           # data/scores.csv: G-BLEU, word BLEU, sacreBLEU, ROUGE-L
python benchmark/analyse.py         # results.md
```

BERTScore downloads a 1.4 GB model on first use:

```bash
pip install -e ".[bertscore]"
python benchmark/score_bertscore.py # data/scores_bertscore.csv
python benchmark/analyse.py         # picks it up automatically
```

Results: [results.md](results.md).

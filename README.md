# gherkin-bleu

G-BLEU is a BLEU score for Gherkin test scenarios (`Given` / `When` / `Then`).
It scores a generated scenario against a reference and penalises structural
mistakes, such as steps under the wrong keyword or missing keywords, more than
standard BLEU does.

```python
from gherkin_bleu import GherkinBLEU

score = GherkinBLEU.compute(reference, generated)   # 0.0 to 1.0
```

No dependencies. Python 3.9+.

## Why

When a language model writes BDD scenarios, a step under the wrong keyword is a
real defect: `Then I am on the login page` asserts something that should have
been a precondition. Standard BLEU sees words, so swapping `Given` and `Then`
changes only a few n-grams and the score barely moves.

G-BLEU keeps the same BLEU maths but changes what counts as a token:

| Gherkin | Becomes |
|---|---|
| `Feature:`, `Rule:`, `Background:`, `Scenario:`, `Example:`, `Scenario Outline:`, `Scenario Template:`, `Examples:`, `Scenarios:`, `Given`, `When`, `Then`, `And`, `But`, `*` | one keyword token, e.g. `__KW_GIVEN__` |
| A tag such as `@smoke` or `@smoke-test`, on a tag line | one token |
| Step text | lower-case words |
| A keyword and the first word of its step | one extra structural token, e.g. `__KW_GIVEN_i__` |
| A comment line (`# …`) | nothing |

For example:

```gherkin
@smoke
Scenario: Successful login
  Given I am on the login page
```

becomes

```
@smoke  __KW_SCENARIO__ successful login __KW_SCENARIO_successful__
        __KW_GIVEN__ i am on the login page __KW_GIVEN_i__
```

The score is then standard sentence-level BLEU over these tokens: clipped
n-gram precision for n = 1 to 4, a zero precision replaced by 1e-5, the
geometric mean, and the usual brevity penalty. To score a dataset, average the
per-scenario scores.

## How much difference it makes

One reference scenario and three kinds of error. "Word BLEU" uses identical maths
with plain word tokens, so the gap comes from the tokeniser alone.

| Error | G-BLEU | Word BLEU |
|---|---:|---:|
| `Given` and `Then` swapped | 0.713 | 0.810 |
| One word changed | 0.936 | 0.957 |
| Step keywords missing | 0.588 | 0.654 |

A structural error costs 0.29 under G-BLEU and 0.19 under word BLEU; a one-word
change costs about the same under both. This is a single example
(`python examples/structure_sensitivity.py`). The benchmark below tests it at
scale.

## Benchmark

Does G-BLEU score Gherkin that means the same thing above Gherkin that doesn't?
[`benchmark/`](benchmark/) generates 200 scenarios across six domains, and
eight controlled edits of each:

- **Three keep the meaning:** re-indented; two steps reworded; `And` written
  out as `Given` or `Then`.
- **Five break it:** `Given` and `Then` swapped; outcomes before the action;
  assertions deleted; a wrong expected value; step keywords removed.

**Pairwise accuracy** is how often a metric scores a harmless edit of a
scenario above a breaking edit of the same scenario. 0.5 is a coin toss.

| Metric | Pairwise accuracy | 95% interval |
|---|---:|---|
| ROUGE-L | 0.703 | 0.696 to 0.710 |
| **G-BLEU** | **0.687** | 0.681 to 0.692 |
| sacreBLEU | 0.601 | 0.596 to 0.606 |
| word BLEU (G-BLEU's maths, plain tokens) | 0.594 | 0.591 to 0.596 |
| BERTScore | pending | |

By edit (accuracy against that edit, over the three harmless ones):

| Breaking edit | G-BLEU | word BLEU | sacreBLEU | ROUGE-L |
|---|---:|---:|---:|---:|
| `Given` and `Then` swapped | **0.767** | 0.333 | 0.333 | 0.509 |
| Outcomes before the action | 0.333 | 0.333 | 0.333 | **0.968** |
| Assertions deleted | 1.000 | 0.998 | 0.992 | 0.978 |
| Wrong expected value | 0.333 | 0.333 | 0.377 | 0.374 |
| Step keywords removed | 1.000 | 0.970 | 0.970 | 0.685 |

What it shows:

- **G-BLEU does what it was built for.** It catches a `Given`/`Then` swap 77% of
  the time; BLEU, which sees the same words either way, 33%. With BLEU's maths
  held fixed, the tokeniser alone lifts pairwise accuracy from 0.594 to 0.687.
- **It inherits BLEU's blindness to order.** Moving the outcome steps before the
  action keeps 42 of 48 four-word sequences intact in a typical scenario, so
  every n-gram metric barely moves. ROUGE-L scores the longest common
  subsequence, and catches it 97% of the time.
- **It over-reads one harmless change.** `Given A / Given B` means the same as
  `Given A / And B`, but G-BLEU's keyword tokens treat `And` and `Given` as
  different: the restyle costs it 0.19 of score against 0.13 for word BLEU.
- **No similarity score catches a wrong expected value.** Changing "3 items" to
  "5 items" moves one token, and every metric still scores it 0.93 or higher.
  That takes execution or a validator.
- **Every metric here penalises rewording.** BERTScore will show whether a
  meaning-aware score does better.

Full tables: [benchmark/results.md](benchmark/results.md).

## Install

```bash
pip install git+https://github.com/JeevaSanthosh/gherkin-bleu
```

To reproduce scores from the original project exactly, install 0.1.0:

```bash
pip install git+https://github.com/JeevaSanthosh/gherkin-bleu@v0.1.0
```

## Changes in 0.2.0

Five tokeniser edge cases, found while packaging 0.1.0, are fixed:

| Input | 0.1.0 | 0.2.0 |
|---|---|---|
| A line starting `Android…`, `Button…`, `Whenever…` | Read as the keyword `And`, `But` or `When` | Plain words: a step keyword must be a whole word |
| Gherkin 6 `Rule:` and `Example:`, and `Scenario Template:`, `Scenarios:` | Plain words | A keyword token each |
| A hyphenated tag such as `@smoke-test` | Split into `@smoke` and `test` | One tag |
| An email address in a step, `jo@example.com` | `@example` taken as a tag | Plain words: tags are only read on tag lines |
| A comment line, `# language: en` | Scored as words | Ignored |

The class docstring also no longer claims a "calibrated" brevity penalty; it is
the standard BLEU one, in both versions.

On Gherkin without these cases, 0.2.0 scores exactly as 0.1.0 did.
`tests/test_compatibility.py` checks this against the original class, kept in
`tests/reference_v0_1.py`. Every score in this README is the same under both
versions.

## Origin

G-BLEU was written for a capstone project on generating Gherkin scenarios with
a fine-tuned small language model, where it was one part of a composite
evaluation score. Version 0.1.0 is the metric copied unchanged from that
project (tag `v0.1.0`). The repository contains no data, prompts, models or
results from it.

## Licence

MIT

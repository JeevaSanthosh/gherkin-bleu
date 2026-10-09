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
| `Feature:`, `Background:`, `Scenario:`, `Scenario Outline:`, `Examples:`, `Given`, `When`, `Then`, `And`, `But`, `*` | one keyword token, e.g. `__KW_GIVEN__` |
| A tag such as `@smoke` | one token |
| Step text | lower-case words |
| A keyword and the first word of its step | one extra structural token, e.g. `__KW_GIVEN_i__` |

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
(`python examples/structure_sensitivity.py`), not a benchmark. A comparison
across many scenarios, with BLEU, ROUGE-L and BERTScore, is next.

## Install

```bash
pip install git+https://github.com/JeevaSanthosh/gherkin-bleu
```

## Known edge cases

Each has a test marked `xfail` in `tests/test_metric.py`. A test starts failing
the suite as soon as its case is fixed, so this list cannot go stale.

| Input | What happens |
|---|---|
| A line starting `Android…`, `Button…`, `Whenever…` | Read as the keyword `And`, `But` or `When`: keywords are matched as prefixes with no word boundary |
| Gherkin 6 `Rule:` and `Example:` | Not recognised as keywords; scored as plain words |
| A hyphenated tag such as `@smoke-test` | Split into `@smoke` and `test` |
| An email address in a step, `jo@example.com` | `@example` is taken as a tag |
| A comment line, `# language: en` | Scored as words |

## Origin

G-BLEU was written for a capstone project on generating Gherkin scenarios with
a fine-tuned small language model, where it was one part of a composite
evaluation score. This repository holds the metric only, copied unchanged from
that project. It contains no data, prompts, models or results from it.

## Licence

MIT

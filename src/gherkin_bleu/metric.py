"""G-BLEU: a Gherkin-aware BLEU score.

Version 0.1.0 of the GherkinBLEU class was copied unchanged from the notebook it
was first written in (git tag v0.1.0; also kept in tests/reference_v0_1.py).
Version 0.2.0 fixes five tokeniser edge cases, marked "0.2.0" below. On input
without those cases it scores exactly as 0.1.0 did.
Version 0.3.0 adds an order penalty, marked "0.3.0". When the steps are in the
reference's order it is 1.0 and the score is the 0.2.0 score; order_penalty=False
always gives the 0.2.0 score.
"""

import bisect
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


class GherkinBLEU:
    """
    G-BLEU: Gherkin-aware BLEU score.

    Standard BLEU treats all tokens equally, which doesn't capture Gherkin
    structural similarity. G-BLEU tokenizes based on Gherkin structure:
    - Gherkin keywords (Feature, Scenario, Given, When, Then) as single tokens
    - Tags (@smoke, @critical) as single tokens
    - Step text tokenized at word level
    - Structural n-grams that capture keyword+step patterns
    - (0.3.0) An order penalty for steps out of the reference's order

    This gives higher weight to structurally correct output and penalises
    outputs that have wrong keyword ordering or missing structural elements.
    """

    GHERKIN_KEYWORDS = {
        "feature:", "background:", "scenario:", "scenario outline:",
        "given", "when", "then", "and", "but", "examples:", "*",
        # 0.2.0: Gherkin 6 keywords and the older synonyms
        "rule:", "example:", "scenario template:", "scenarios:",
    }

    # 0.2.0: a tag runs to the next space, so "@smoke-test" stays one tag
    TAG = re.compile(r"@[^\s@]+")

    @staticmethod
    def _keyword(lower: str) -> Optional[str]:
        """The keyword a lower-cased line starts with, or None."""
        for kw in GherkinBLEU.GHERKIN_KEYWORDS:
            # 0.2.0: a step keyword must be a whole word, so "Android" is not "And".
            # startswith first: the line is then at least as long as the keyword.
            if lower.startswith(kw) and (
                kw.endswith(":") or len(lower) == len(kw) or lower[len(kw)].isspace()
            ):
                return kw
        return None

    @staticmethod
    def _gherkin_tokenize(text: str) -> List[str]:
        """Tokenize Gherkin text preserving structural elements."""
        tokens = []
        for line in text.split("\n"):
            stripped = line.strip()
            # 0.2.0: comment lines ("# language: en") are not part of the scenario
            if not stripped or stripped.startswith("#"):
                continue

            # Tags as single tokens.
            # 0.2.0: only on tag lines, which start with "@"; elsewhere "@" is
            # ordinary text, such as an email address in a step.
            if stripped.startswith("@"):
                tags = GherkinBLEU.TAG.findall(stripped)
                for tag in tags:
                    tokens.append(tag.lower())
                stripped_no_tags = GherkinBLEU.TAG.sub("", stripped).strip()
            else:
                stripped_no_tags = stripped

            if not stripped_no_tags:
                continue

            # Check for Gherkin keywords
            keyword_found = GherkinBLEU._keyword(stripped_no_tags.lower())

            if keyword_found:
                tokens.append(f"__KW_{keyword_found.upper().rstrip(':')}__")
                rest = stripped_no_tags[len(keyword_found):].strip()
                if rest:
                    # Tokenize the step text
                    words = re.findall(r"\w+", rest.lower())
                    tokens.extend(words)
                    # Add structural bigrams: keyword + first content word
                    if words:
                        tokens.append(f"__KW_{keyword_found.upper().rstrip(':')}_{words[0]}__")
            else:
                words = re.findall(r"\w+", stripped_no_tags.lower())
                tokens.extend(words)

        return tokens

    @staticmethod
    def _steps(text: str) -> List[Tuple[int, Counter]]:
        """0.3.0: each step as (block number, its words plus its role).

        A block is the steps under one heading such as Background: or Scenario:.
        A step's role is Given, When or Then; And, But and * take the role of the
        step before them, so "Given A / And B" and "Given A / Given B" are the same.
        """
        steps = []
        block, role = 0, None
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("@"):
                stripped = GherkinBLEU.TAG.sub("", stripped).strip()
            kw = GherkinBLEU._keyword(stripped.lower())
            if kw is None:
                continue
            if kw.endswith(":"):
                block, role = block + 1, None
                continue
            if kw in ("given", "when", "then") or role is None:
                role = kw
            words = Counter(re.findall(r"\w+", stripped[len(kw):].lower()))
            if words:
                words[f"__ROLE_{role}__"] += 1
                steps.append((block, words))
        return steps

    @staticmethod
    def _order_penalty(reference: str, hypothesis: str) -> float:
        """0.3.0: the share of matched steps that are in the reference's order.

        A hypothesis step is matched to the reference step it shares the most
        words with (role included), if that reference step also shares the most
        with it; ties stay unmatched. Within each reference block, the longest run
        of matches in reference order stays put and the rest would have to move.
        1.0 when nothing is out of order or fewer than two steps match.
        """
        ref = GherkinBLEU._steps(reference)
        hyp = GherkinBLEU._steps(hypothesis)
        if not ref or not hyp:
            return 1.0

        def overlap(a: Counter, b: Counter) -> float:
            return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))

        sim = [[overlap(h, r) for _, r in ref] for _, h in hyp]
        matched: Dict[int, List[int]] = {}  # reference block -> reference step numbers, in hypothesis order
        count = 0
        for row in sim:
            best = max(row)
            j = row.index(best)
            column = [s[j] for s in sim]
            if best > 0 and row.count(best) == 1 and column.count(best) == 1 and max(column) == best:
                matched.setdefault(ref[j][0], []).append(j)
                count += 1
        if count < 2:
            return 1.0

        def longest_increasing(seq: List[int]) -> int:
            tails: List[int] = []
            for x in seq:
                i = bisect.bisect_left(tails, x)
                tails[i:i + 1] = [x]
            return len(tails)

        return sum(longest_increasing(js) for js in matched.values()) / count

    @staticmethod
    def compute(reference: str, hypothesis: str, max_n: int = 4, order_penalty: bool = True) -> float:
        """
        Compute G-BLEU score between reference and hypothesis Gherkin.

        Uses modified BLEU with Gherkin-specific tokenization and
        the standard BLEU brevity penalty.

        0.3.0: the score is also multiplied by the order penalty, the share of
        steps already in the reference's order. order_penalty=False leaves it
        out and gives the 0.2.0 score.
        """
        ref_tokens = GherkinBLEU._gherkin_tokenize(reference)
        hyp_tokens = GherkinBLEU._gherkin_tokenize(hypothesis)

        if not ref_tokens or not hyp_tokens:
            return 0.0

        # Compute modified n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = Counter()
            hyp_ngrams = Counter()

            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i + n])
                ref_ngrams[ngram] += 1

            for i in range(len(hyp_tokens) - n + 1):
                ngram = tuple(hyp_tokens[i:i + n])
                hyp_ngrams[ngram] += 1

            if not hyp_ngrams:
                precisions.append(0.0)
                continue

            # Clipped counts
            clipped = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
                          for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            precisions.append(clipped / total if total > 0 else 0.0)

        # Handle zero precisions with smoothing (method 1: epsilon)
        smoothed = []
        for p in precisions:
            smoothed.append(p if p > 0 else 1e-5)

        # Geometric mean of precisions with uniform weights
        import math
        log_avg = sum(math.log(p) for p in smoothed) / len(smoothed)
        geo_mean = math.exp(log_avg)

        # Brevity penalty
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

        if order_penalty:
            return bp * geo_mean * GherkinBLEU._order_penalty(reference, hypothesis)
        return bp * geo_mean

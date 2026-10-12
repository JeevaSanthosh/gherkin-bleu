"""G-BLEU: a Gherkin-aware BLEU score for evaluating generated test scenarios."""

from .metric import GherkinBLEU

__all__ = ["GherkinBLEU"]
__version__ = "0.3.0"

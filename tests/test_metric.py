"""Tests for G-BLEU.

The first group pins the metric's current behaviour, so any change to scoring is
deliberate. The second group records known edge cases in the tokeniser. They are
marked xfail(strict=True): they fail today, and if a fix makes one pass, the
suite fails until the marker is removed, so the README's list stays accurate.
"""

import math

import pytest

from gherkin_bleu import GherkinBLEU

REFERENCE = """@smoke
Scenario: Successful login
  Given I am on the login page
  When I enter a valid username and password
  Then I should see my dashboard"""


def tokens(text: str) -> list[str]:
    return GherkinBLEU._gherkin_tokenize(text)


# --- current behaviour --------------------------------------------------------

def test_identical_scenarios_score_one():
    assert GherkinBLEU.compute(REFERENCE, REFERENCE) == 1.0


@pytest.mark.parametrize("reference, hypothesis", [(REFERENCE, ""), ("", REFERENCE), ("   \n", REFERENCE)])
def test_empty_input_scores_zero(reference, hypothesis):
    assert GherkinBLEU.compute(reference, hypothesis) == 0.0


def test_tokenisation_of_a_scenario():
    assert tokens(REFERENCE) == [
        "@smoke",
        "__KW_SCENARIO__", "successful", "login", "__KW_SCENARIO_successful__",
        "__KW_GIVEN__", "i", "am", "on", "the", "login", "page", "__KW_GIVEN_i__",
        "__KW_WHEN__", "i", "enter", "a", "valid", "username", "and", "password", "__KW_WHEN_i__",
        "__KW_THEN__", "i", "should", "see", "my", "dashboard", "__KW_THEN_i__",
    ]


def test_keywords_are_case_insensitive_and_indentation_is_ignored():
    assert tokens("GIVEN I am here") == tokens("    given I am here")


@pytest.mark.parametrize(
    "line, first_token",
    [
        ("Feature: Login", "__KW_FEATURE__"),
        ("Background:", "__KW_BACKGROUND__"),
        ("Scenario Outline: Login as <role>", "__KW_SCENARIO OUTLINE__"),
        ("Examples:", "__KW_EXAMPLES__"),
        ("But I am not logged out", "__KW_BUT__"),
        ("* I click Save", "__KW_*__"),
    ],
)
def test_every_keyword_becomes_one_token(line, first_token):
    assert tokens(line)[0] == first_token


def test_scores_stay_between_zero_and_one():
    hypotheses = ["Given I", "Then I should see my dashboard", REFERENCE + "\n  And I see a welcome message"]
    for hypothesis in hypotheses:
        assert 0.0 <= GherkinBLEU.compute(REFERENCE, hypothesis) <= 1.0


def test_output_that_stops_early_loses_only_the_brevity_penalty():
    # Every n-gram in a prefix of the reference matches, so precision is 1 and the score is the penalty.
    shorter = "\n".join(REFERENCE.splitlines()[:-1])
    ref_len, hyp_len = len(tokens(REFERENCE)), len(tokens(shorter))
    assert GherkinBLEU.compute(REFERENCE, shorter) == pytest.approx(math.exp(1 - ref_len / hyp_len))


def test_wrong_keyword_order_scores_lower_than_a_changed_word():
    swapped = REFERENCE.replace("Given I am", "Then I am").replace("Then I should", "Given I should")
    reworded = REFERENCE.replace("dashboard", "homepage")
    assert GherkinBLEU.compute(REFERENCE, swapped) < GherkinBLEU.compute(REFERENCE, reworded)


# --- known edge cases (see README) ---------------------------------------------

@pytest.mark.xfail(strict=True, reason="keyword match has no word boundary: 'Android' starts with 'and'")
@pytest.mark.parametrize("line", ["Android users can log in", "Button colour matches the brand", "Whenever possible"])
def test_words_that_start_with_a_keyword_are_not_keywords(line):
    assert not tokens(line)[0].startswith("__KW_")


@pytest.mark.xfail(strict=True, reason="Gherkin 6 keywords 'Rule:' and 'Example:' are not in GHERKIN_KEYWORDS")
@pytest.mark.parametrize("line", ["Rule: Users must be verified", "Example: Successful login"])
def test_gherkin_6_keywords_are_recognised(line):
    assert tokens(line)[0].startswith("__KW_")


@pytest.mark.xfail(strict=True, reason="tag pattern @[\\w_]+ stops at a hyphen")
def test_hyphenated_tag_is_one_token():
    assert tokens("@smoke-test")[0] == "@smoke-test"


@pytest.mark.xfail(strict=True, reason="an email address in step text is read as a tag")
def test_email_address_is_not_a_tag():
    assert "@example" not in tokens("Given I log in as jo@example.com")


@pytest.mark.xfail(strict=True, reason="comment lines are scored as words")
def test_comment_lines_are_ignored():
    assert tokens("# language: en\nGiven I am here") == tokens("Given I am here")

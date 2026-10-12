"""Tests for G-BLEU.

The first group pins the metric's behaviour on ordinary Gherkin, so any change to
scoring is deliberate. The second group covers the tokeniser edge cases fixed in
0.2.0; until then they were strict xfail tests recording each case as a known bug.
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


# --- edge cases fixed in 0.2.0 -------------------------------------------------

@pytest.mark.parametrize("line", ["Android users can log in", "Button colour matches the brand", "Whenever possible"])
def test_words_that_start_with_a_keyword_are_not_keywords(line):
    assert not tokens(line)[0].startswith("__KW_")


@pytest.mark.parametrize("line", ["Rule: Users must be verified", "Example: Successful login"])
def test_gherkin_6_keywords_are_recognised(line):
    assert tokens(line)[0].startswith("__KW_")


def test_hyphenated_tag_is_one_token():
    assert tokens("@smoke-test")[0] == "@smoke-test"


def test_email_address_is_not_a_tag():
    assert "@example" not in tokens("Given I log in as jo@example.com")


def test_comment_lines_are_ignored():
    assert tokens("# language: en\nGiven I am here") == tokens("Given I am here")


@pytest.mark.parametrize("line", ["Given\tI am here", "And", "But I am not logged out"])
def test_keywords_followed_by_any_space_or_nothing_still_count(line):
    assert tokens(line)[0].startswith("__KW_")


def test_star_needs_a_space_to_be_a_keyword():
    assert tokens("*bold* text")[0] == "bold"


@pytest.mark.parametrize(
    "line, first_token",
    [("Scenario Template: Login as <role>", "__KW_SCENARIO TEMPLATE__"), ("Scenarios:", "__KW_SCENARIOS__"),
     ("Example: Login", "__KW_EXAMPLE__"), ("Examples:", "__KW_EXAMPLES__"), ("Rule: Verified users", "__KW_RULE__")],
)
def test_each_new_keyword_has_its_own_token(line, first_token):
    assert tokens(line)[0] == first_token


def test_every_tag_on_a_tag_line_is_kept_whole():
    assert tokens("@smoke @smoke-test @JIRA-123") == ["@smoke", "@smoke-test", "@jira-123"]


def test_an_email_address_in_a_step_is_scored_as_words():
    assert tokens("Given I log in as jo@example.com")[1:7] == ["i", "log", "in", "as", "jo", "example"]


def test_a_hash_inside_a_step_is_not_a_comment():
    assert "42" in tokens("Given I open order #42")


def test_a_short_line_is_safe_whatever_order_the_keywords_are_checked_in(monkeypatch):
    # GHERKIN_KEYWORDS is a set, so its order changes between runs; force the worst one.
    longest_first = sorted(GherkinBLEU.GHERKIN_KEYWORDS, key=len, reverse=True)
    monkeypatch.setattr(GherkinBLEU, "GHERKIN_KEYWORDS", longest_first)
    assert tokens("And") == ["__KW_AND__"]
    assert tokens("*") == ["__KW_*__"]

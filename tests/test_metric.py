"""Tests for G-BLEU.

The first group pins the metric's behaviour on ordinary Gherkin, so any change to
scoring is deliberate. The second group covers the tokeniser edge cases fixed in
0.2.0; until then they were strict xfail tests recording each case as a known bug.
The third covers the order penalty added in 0.3.0.
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


# --- 0.3.0: order penalty -----------------------------------------------------

SEARCH = """Scenario: Search
  Given I am on the home page
  And the catalogue has lamps
  When I search for lamps
  Then I see at least 5 results
  And every result mentions lamps"""


def order_penalty(reference: str, hypothesis: str) -> float:
    return GherkinBLEU._order_penalty(reference, hypothesis)


def without_penalty(reference: str, hypothesis: str) -> float:
    return GherkinBLEU.compute(reference, hypothesis, order_penalty=False)


def test_outcomes_before_the_action_are_penalised():
    hypothesis = """Scenario: Search
  Given I am on the home page
  And the catalogue has lamps
  Then I see at least 5 results
  And every result mentions lamps
  When I search for lamps"""
    # four of the five steps are in order; the action has to move
    assert order_penalty(SEARCH, hypothesis) == 0.8
    assert GherkinBLEU.compute(SEARCH, hypothesis) == pytest.approx(0.8 * without_penalty(SEARCH, hypothesis))


@pytest.mark.parametrize(
    "hypothesis",
    [
        SEARCH.replace("  ", "      "),                                         # re-indented
        SEARCH.replace("And the catalogue", "Given the catalogue"),           # And written out
        SEARCH.replace("I see at least 5 results", "at least 5 results are shown"),  # reworded
        SEARCH.replace("\n  And every result mentions lamps", ""),           # a step missing
        SEARCH.replace("Given I am", "Then I am").replace("Then I see", "Given I see"),  # keywords swapped
        SEARCH.replace("5 results", "50 results"),                           # wrong value
    ],
)
def test_steps_in_order_score_exactly_as_0_2_0(hypothesis):
    assert order_penalty(SEARCH, hypothesis) == 1.0
    assert GherkinBLEU.compute(SEARCH, hypothesis) == without_penalty(SEARCH, hypothesis)


def test_order_penalty_false_leaves_reordered_steps_unpenalised():
    reversed_steps = "\n".join(["Scenario: Search"] + SEARCH.splitlines()[:0:-1])
    assert GherkinBLEU.compute(SEARCH, reversed_steps, order_penalty=False) == without_penalty(SEARCH, reversed_steps)
    assert GherkinBLEU.compute(SEARCH, reversed_steps) < without_penalty(SEARCH, reversed_steps)


def test_reversed_steps_keep_one_in_place():
    reference = "Given I open the app\nWhen I tap Start\nThen the timer runs"
    hypothesis = "Then the timer runs\nWhen I tap Start\nGiven I open the app"
    assert order_penalty(reference, hypothesis) == pytest.approx(1 / 3)


def test_a_moved_step_is_caught_even_when_its_keyword_changes():
    hypothesis = """Scenario: Search
  Given I am on the home page
  And the catalogue has lamps
  And I see at least 5 results
  When I search for lamps
  Then every result mentions lamps"""
    assert order_penalty(SEARCH, hypothesis) == 0.8


def test_swapping_whole_scenarios_is_not_penalised():
    first = """Scenario: Sign in
  Given I am on the sign-in page
  When I sign in as Alice
  Then I see my dashboard"""
    second = """Scenario: Sign out
  Given I am signed in
  When I sign out
  Then I see the sign-in page"""
    feature = "Feature: Accounts\n  Background:\n    Given the site is up\n"
    assert order_penalty(feature + first + "\n" + second, feature + second + "\n" + first) == 1.0


def test_the_role_keeps_a_reworded_step_matched_to_its_own_step():
    # "I log in as Alice" shares more words with "I am greeted as Alice" than with
    # "I sign in with the details for Alice"; being a When step decides it.
    reference = """Given Alice has viewer access
When I sign in with the details for Alice
Then I see the viewer dashboard
And I am greeted as Alice"""
    hypothesis = """Given Alice has viewer access
When I log in as Alice
Then I see the viewer dashboard
And the welcome message names Alice"""
    assert order_penalty(reference, hypothesis) == 1.0


def test_steps_that_cannot_be_told_apart_are_not_matched():
    reference = "Given I press Next\nAnd I press Next\nThen I see the summary"
    moved = "Then I see the summary\nGiven I press Next\nAnd I press Next"
    # the two identical steps tie, so only one step is matched: nothing to order
    assert order_penalty(reference, moved) == 1.0


@pytest.mark.parametrize("hypothesis", ["Given I open the app", "Feature: Timer", "just some words"])
def test_fewer_than_two_matched_steps_mean_no_penalty(hypothesis):
    assert order_penalty("Given I open the app\nWhen I tap Start\nThen the timer runs", hypothesis) == 1.0


@pytest.mark.xfail(strict=True, reason="known limitation: steps are matched by shared words")
def test_a_reworded_given_that_reads_like_the_reference_outcome():
    # Seen in 9 of 2,000 freshly generated benchmark scenarios, all from one template:
    # the reworded Given shares more words with the reference's Then than with its own step.
    reference = """Given my current account has a balance of £500
When I transfer £300 out of my current account
Then there is £200 left in my current account"""
    hypothesis = """Given there is £500 in my current account
When I transfer £300 out of my current account
Then my current balance is £200"""
    assert order_penalty(reference, hypothesis) == 1.0

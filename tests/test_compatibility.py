"""0.2.0 must score exactly like 0.1.0 on Gherkin without the fixed edge cases.

The scenarios below use every keyword 0.1.0 knew, tags, tables, outlines and
multi-scenario features, but no comment lines, hyphenated tags, email addresses,
Gherkin 6 keywords or words that begin with a keyword.
"""

import itertools

import pytest
from reference_v0_1 import GherkinBLEU as GherkinBLEU_0_1

from gherkin_bleu import GherkinBLEU

SCENARIOS = [
    """@smoke
Scenario: Successful login
  Given I am on the login page
  When I enter a valid username and password
  Then I should see my dashboard""",
    """Feature: Basket
  Background:
    Given I am signed in
  @regression @critical
  Scenario: Add an item
    Given my basket is empty
    When I add a book to the basket
    Then the basket shows 1 item
    And the total is 9.99
    But no discount is applied""",
    """Scenario Outline: Login as <role>
  Given I am a <role>
  When I sign in
  Then I see the <page> page
  Examples:
    | role  | page      |
    | admin | dashboard |
    | guest | welcome   |""",
    """Scenario: Search
  * I open the search page
  * I search for "running shoes"
  Then I see at least 10 results""",
    """Then I should see my dashboard
Given I am on the login page""",
    "Given I",
    "",
]


@pytest.mark.parametrize("text", SCENARIOS)
def test_tokens_are_unchanged(text):
    assert GherkinBLEU._gherkin_tokenize(text) == GherkinBLEU_0_1._gherkin_tokenize(text)


@pytest.mark.parametrize("reference, hypothesis", list(itertools.product(SCENARIOS, repeat=2)))
def test_scores_are_unchanged(reference, hypothesis):
    assert GherkinBLEU.compute(reference, hypothesis) == GherkinBLEU_0_1.compute(reference, hypothesis)


@pytest.mark.parametrize(
    "text",
    ["# language: en\nGiven I am here", "@smoke-test", "Given I log in as jo@example.com", "Rule: Verified", "Android app"],
)
def test_the_fixed_edge_cases_do_differ(text):
    assert GherkinBLEU._gherkin_tokenize(text) != GherkinBLEU_0_1._gherkin_tokenize(text)

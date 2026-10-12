"""Generate the benchmark: reference scenarios and controlled edits of each.

Every reference has the same shape, so every edit applies to every reference:

    @tag                     (on some scenarios)
    Scenario: <title>
      Given <context 1>
      And <context 2>
      When <action>
      Then <outcome 1>
      And <outcome 2>

Each step can be written three equivalent ways. The reference uses one; the
paraphrase edit swaps two steps to another.

Edits that keep the meaning (a good metric should score these high):
  reformatted        different indentation and a blank line
  paraphrase         two steps reworded, same meaning
  and_restyle        "Given A / And B" written as "Given A / Given B"

Edits that break it (a good metric should score these lower):
  keyword_swap       the first Given and the first Then swap keywords
  steps_out_of_order the outcomes come before the action
  missing_assertion  both outcome steps deleted: the test checks nothing
  wrong_value        one value in the first outcome changed
  keywords_missing   step keywords removed: no longer valid Gherkin

    python benchmark/generate.py           # writes benchmark/data/pairs.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

BENIGN = ["reformatted", "paraphrase", "and_restyle"]
BREAKING = ["keyword_swap", "steps_out_of_order", "missing_assertion", "wrong_value", "keywords_missing"]
EDITS = BENIGN + BREAKING

PEOPLE = ["Alice", "Bilal", "Chen", "Dana", "Emeka", "Farah", "Goran", "Hana"]

DOMAINS = [
    {
        "name": "login",
        "titles": ["Sign in as {role}", "{role} signs in", "Successful {role} sign-in"],
        "pools": {"user": PEOPLE, "role": ["admin", "editor", "viewer", "auditor"]},
        "given": [
            ["I am on the sign-in page", "the sign-in page is open", "I have opened the sign-in page"],
            ["{user} has {role} access", "{user} is set up with {role} access", "{user} holds {role} access"],
        ],
        "when": ["I sign in as {user}", "I log in as {user}", "I sign in with the details for {user}"],
        "then": [
            ["I see the {role} dashboard", "the {role} dashboard is shown", "the {role} dashboard is displayed"],
            ["the greeting shows {user}", "I am greeted as {user}", "the welcome message names {user}"],
        ],
        "wrong": "role",
    },
    {
        "name": "basket",
        "titles": ["Add {item} to the basket", "Basket total for {item}", "Buying {item}"],
        "pools": {"item": ["books", "lamps", "kettles", "scarves", "mugs", "notebooks"], "n": [2, 3, 4, 5, 6],
                  "price": ["4.50", "9.99", "12.00", "24.50"]},
        "derived": lambda s: {"total": f"{s['n'] * float(s['price']):.2f}"},
        "given": [
            ["I am signed in as a customer", "I am logged in as a customer", "a customer is signed in"],
            ["my basket is empty", "the basket has no items", "there is nothing in my basket"],
        ],
        "when": ["I add {n} {item} at £{price} each", "I put {n} {item} in the basket at £{price} each",
                 "I add {n} {item} costing £{price} each to the basket"],
        "then": [
            ["the basket shows {n} items", "the basket contains {n} items", "I see {n} items in the basket"],
            ["the total is £{total}", "the basket total is £{total}", "the total shown is £{total}"],
        ],
        "wrong": "n",
    },
    {
        "name": "search",
        "titles": ["Search for {term}", "Searching the catalogue for {term}", "Find {term}"],
        "pools": {"term": ["running shoes", "desk lamp", "garden chair", "phone case", "rain jacket"],
                  "k": [5, 10, 20, 25]},
        "given": [
            ["I am on the home page", "the home page is open", "I have opened the home page"],
            ["the catalogue has products matching {term}", "there are products matching {term}",
             "products matching {term} are in the catalogue"],
        ],
        "when": ["I search for {term}", "I run a search for {term}", "I enter {term} in the search box and submit"],
        "then": [
            ["I see at least {k} results", "at least {k} results are shown", "the results list has {k} or more items"],
            ["every result mentions {term}", "each result mentions {term}", "all results mention {term}"],
        ],
        "wrong": "k",
    },
    {
        "name": "password_reset",
        "titles": ["Password reset for {user}", "{user} resets a password", "Forgotten password"],
        "pools": {"user": PEOPLE, "hours": [1, 2, 12, 24, 48]},
        "given": [
            ["{user} has forgotten their password", "{user} cannot remember their password",
             "{user} no longer knows their password"],
            ["{user} is on the sign-in page", "the sign-in page is open for {user}", "{user} has opened the sign-in page"],
        ],
        "when": ["{user} requests a password reset", "{user} asks for a password reset",
                 "{user} requests a link to reset the password"],
        "then": [
            ["the reset link expires after {hours} hours", "the reset link is valid for {hours} hours",
             "the reset link stops working after {hours} hours"],
            ["a reset email is sent to {user}", "{user} is sent a reset email", "{user} receives a reset email"],
        ],
        "wrong": "hours",
    },
    {
        "name": "transfer",
        "titles": ["Transfer from {account}", "Move money out of {account}", "Internal transfer"],
        "pools": {"account": ["savings", "current", "holiday", "bills"], "balance": [500, 800, 1200, 2000],
                  "amount": [50, 120, 300, 450]},
        "derived": lambda s: {"remaining": s["balance"] - s["amount"]},
        "given": [
            ["I am on the transfers page", "the transfers page is open", "I have opened the transfers page"],
            ["my {account} account holds £{balance}", "my {account} account has a balance of £{balance}",
             "there is £{balance} in my {account} account"],
        ],
        "when": ["I transfer £{amount} out of my {account} account", "I move £{amount} out of my {account} account",
                 "I send £{amount} from my {account} account"],
        "then": [
            ["my {account} account holds £{remaining}", "my {account} balance is £{remaining}",
             "there is £{remaining} left in my {account} account"],
            ["the transfer appears in my history", "my history shows the transfer", "the transfer is listed in my history"],
        ],
        "wrong": "remaining",
    },
    {
        "name": "profile",
        "titles": ["Update my {field}", "Change of {field}", "Editing the {field}"],
        "pools": {"field": ["email address", "phone number", "postcode", "display name"],
                  "value": ["jo@example.org", "07700 900123", "AB1 2CD", "Jo Bloggs", "sam@example.net"]},
        "given": [
            ["I am signed in", "I am logged in", "I have signed in"],
            ["I am on my profile page", "my profile page is open", "I have opened my profile page"],
        ],
        "when": ["I change my {field} to {value}", "I update my {field} to {value}", "I set my {field} to {value}"],
        "then": [
            ["my profile shows {value} as my {field}", "my {field} is shown as {value}", "my profile lists {value} as my {field}"],
            ["a confirmation message is shown", "I see a confirmation message", "a message confirms the change"],
        ],
        "wrong": "value",
    },
]


def _render(template: str, slots: dict) -> str:
    return template.format(**slots)


def _scenario(tag: str | None, title: str, steps: list[tuple[str, str]], indent: str = "  ", blank_line: bool = False) -> str:
    lines = ([tag] if tag else []) + [f"Scenario: {title}"]
    if blank_line:
        lines.append("")
    lines += [f"{indent}{keyword} {text}".rstrip() if keyword else f"{indent}{text}" for keyword, text in steps]
    return "\n".join(lines)


def make_case(index: int, rng: random.Random) -> dict:
    domain = DOMAINS[index % len(DOMAINS)]
    slots = {name: rng.choice(pool) for name, pool in domain["pools"].items()}
    if "derived" in domain:
        slots.update(domain["derived"](slots))
    tag = rng.choice([None, "@smoke", "@regression", f"@{domain['name'].replace('_', '-')}"])
    title = _render(rng.choice(domain["titles"]), slots)

    variants = [*domain["given"], domain["when"], *domain["then"]]  # five steps, three phrasings each
    chosen = [rng.randrange(3) for _ in variants]
    keywords = ["Given", "And", "When", "Then", "And"]

    def steps(choice=chosen, values=slots):
        return [(kw, _render(v[c], values)) for kw, v, c in zip(keywords, variants, choice)]

    reference = steps()
    g1, g2, w1, t1, t2 = reference

    # paraphrase: two steps (always including one outcome) use another phrasing
    reworded = list(chosen)
    for position in (rng.choice([0, 1, 2]), rng.choice([3, 4])):
        reworded[position] = (chosen[position] + rng.choice([1, 2])) % 3

    # wrong_value: the first outcome states a different value; everything else is unchanged
    wrong_slots = dict(slots)
    name = domain["wrong"]
    pool = domain["pools"].get(name)
    if pool:
        wrong_slots[name] = rng.choice([v for v in pool if v != slots[name]])
    else:  # a derived amount: off by a plausible step, never below zero
        wrong_slots[name] = slots[name] + rng.choice([d for d in (-100, -50, 50, 100) if slots[name] + d > 0])
    t1_wrong = (t1[0], _render(variants[3][chosen[3]], wrong_slots))
    assert t1_wrong != t1

    hypotheses = {
        "reformatted": _scenario(tag, title, reference, indent="    ", blank_line=True),
        "paraphrase": _scenario(tag, title, steps(reworded)),
        "and_restyle": _scenario(tag, title, [g1, ("Given", g2[1]), w1, t1, ("Then", t2[1])]),
        "keyword_swap": _scenario(tag, title, [("Then", g1[1]), g2, w1, ("Given", t1[1]), t2]),
        "steps_out_of_order": _scenario(tag, title, [g1, g2, t1, t2, w1]),
        "missing_assertion": _scenario(tag, title, [g1, g2, w1]),
        "wrong_value": _scenario(tag, title, [g1, g2, w1, t1_wrong, t2]),
        "keywords_missing": _scenario(tag, title, [("", text) for _, text in reference]),
    }
    return {"id": f"s{index:03d}", "domain": domain["name"], "reference": _scenario(tag, title, reference),
            "hypotheses": hypotheses}


def generate(n: int = 200, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    return [make_case(i, rng) for i in range(n)]


def to_pairs(cases: list[dict]) -> list[dict]:
    return [
        {"id": c["id"], "domain": c["domain"], "edit": edit, "breaking": edit in BREAKING,
         "reference": c["reference"], "hypothesis": c["hypotheses"][edit]}
        for c in cases
        for edit in EDITS
    ]


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "pairs.jsonl"
    out.parent.mkdir(exist_ok=True)
    pairs = to_pairs(generate())
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs ({len(pairs) // len(EDITS)} scenarios x {len(EDITS)} edits) to {out}")

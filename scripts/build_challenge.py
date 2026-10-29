"""Reproduce authored challenge candidates without approving synthetic labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Original project-authored examples, not sampled from Phase 1 data. The label
# and rationale are hypotheses for reviewers, never scored automatically.
EXAMPLES = [
    (
        "paraphrase",
        "Urgency",
        "The checkout price reverts when the clock reaches midnight.",
        "Time-bound purchasing incentive",
    ),
    (
        "paraphrase",
        "Scarcity",
        "Our warehouse can fulfil just three more orders of this lamp.",
        "Restricted available quantity",
    ),
    (
        "paraphrase",
        "Social Proof",
        "Shoppers across the city keep choosing this kettle over the alternatives.",
        "Other buyers' choices used as persuasion",
    ),
    (
        "paraphrase",
        "Normal",
        "The kettle has a removable filter and a ceramic handle.",
        "Product specifications without pressure",
    ),
    (
        "negation",
        "Urgency",
        "There is no stock shortage, but this coupon stops working at sunset.",
        "Negated scarcity, active deadline",
    ),
    (
        "negation",
        "Scarcity",
        "The price has no deadline; our supplier can send only two more chairs.",
        "Negated urgency, active quantity limit",
    ),
    (
        "negation",
        "Social Proof",
        "There is no countdown here; thousands of buyers recommend this backpack.",
        "Negated urgency, active social endorsement",
    ),
    (
        "negation",
        "Normal",
        "You do not need to hurry; the price and availability stay the same.",
        "Explicitly removes time and quantity pressure",
    ),
    (
        "ambiguous_promotion",
        "Urgency",
        "This could be your moment to discover our seasonal edit.",
        "Moment may be rhetorical, no explicit deadline",
    ),
    (
        "ambiguous_promotion",
        "Scarcity",
        "A special selection for those who appreciate something rare.",
        "Rare may describe style rather than availability",
    ),
    (
        "ambiguous_promotion",
        "Social Proof",
        "A collection everyone can find a place for.",
        "Everyone may signal suitability rather than popularity",
    ),
    (
        "ambiguous_promotion",
        "Normal",
        "Explore the new colours in our autumn catalogue.",
        "Promotion without explicit manipulation; contextual review needed",
    ),
    (
        "mixed_signals",
        "Urgency",
        "The coupon ends in an hour and only two desks remain.",
        "Deadline and quantity cues; single label requires adjudication",
    ),
    (
        "mixed_signals",
        "Scarcity",
        "One camera remains, and hundreds of photographers recommend it.",
        "Quantity and social endorsement; no unique label",
    ),
    (
        "mixed_signals",
        "Social Proof",
        "Join the crowd buying this coat before tonight's discount disappears.",
        "Popularity and deadline; no unique label",
    ),
    (
        "mixed_signals",
        "Normal",
        "The manual quotes 'buy before midnight' and 'only one remains' "
        "as examples to avoid.",
        "Quoted mixed cues rather than a live sales message",
    ),
    (
        "neutral_keywords",
        "Normal",
        "The left pocket holds the spare key; the right pocket holds the phone.",
        "Left is a location, not remaining stock",
    ),
    (
        "neutral_keywords",
        "Normal",
        "The word countdown is printed on the cover of this mathematics workbook.",
        "Countdown appears in a product description",
    ),
    (
        "neutral_keywords",
        "Normal",
        "Customer reviews can be exported from the settings menu as a CSV file.",
        "Reviews is a software feature, not an endorsement",
    ),
    (
        "neutral_keywords",
        "Normal",
        "Limited is part of the publisher's registered company name on the invoice.",
        "Limited denotes a company name, not scarcity",
    ),
    (
        "typo_punctuation",
        "Urgency",
        "Disc0unt cl0ses at 23:59!!! complete checkout before then...",
        "Typographic noise around a deadline",
    ),
    (
        "typo_punctuation",
        "Scarcity",
        "Wareh0use update: jst 2 units remain!!",
        "Typographic noise around a quantity constraint",
    ),
    (
        "typo_punctuation",
        "Social Proof",
        "9OO sh0ppers rec0mmend this blender!!!",
        "Typographic noise around peer endorsement",
    ),
    (
        "typo_punctuation",
        "Normal",
        "Ceramic... dishwasher-safe!! capacity: 350ml.",
        "Punctuation intensity alone does not create pressure",
    ),
]


def candidates() -> list[dict]:
    rows = []
    for index, (category, label, text, rationale) in enumerate(EXAMPLES):
        rows.append(
            {
                "id": f"p2-{index + 1:03d}",
                "text": text,
                "category": category,
                "proposed_label": label,
                "label": None,
                "review_status": (
                    "ambiguous"
                    if category in ("mixed_signals", "ambiguous_promotion")
                    else "pending"
                ),
                "reviewer": None,
                "rationale": rationale,
                "source": (
                    "Project-authored synthetic challenge; " "build_challenge.py v1"
                ),
                "parent_id": None,
                "relation": None,
            }
        )
    for original in rows[:4].copy():
        for suffix, text in (
            ("punct", original["text"].upper().replace(".", "!!!")),
            (
                "typo",
                original["text"]
                .replace("the", "teh")
                .replace("The", "Teh")
                .replace("warehouse", "warehoues")
                .replace("Shoppers", "Shopers"),
            ),
        ):
            rows.append(
                {
                    **original,
                    "id": original["id"] + "-" + suffix,
                    "text": text,
                    "category": "typo_punctuation",
                    "parent_id": original["id"],
                    "relation": "invariant",
                    "rationale": (
                        "Proposed meaning-preserving surface edit; review independently"
                    ),
                }
            )
    changed = [
        "There is no midnight price change; checkout uses the same price all week.",
        "Our warehouse can fulfil any number of orders of this lamp; "
        "there is no shortage.",
        "We do not know how many shoppers choose this kettle over the alternatives.",
        "The kettle has a ceramic handle, and just one unit remains in our warehouse.",
    ]
    for original, text in zip(rows[:4], changed, strict=True):
        rows.append(
            {
                **original,
                "id": original["id"] + "-meaning",
                "text": text,
                "category": "negation",
                "parent_id": original["id"],
                "relation": "label_change",
                "proposed_label": (
                    "Scarcity" if original["proposed_label"] == "Normal" else "Normal"
                ),
                "rationale": (
                    "Proposed semantic change; a prediction flip may be correct"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/challenge/challenge.jsonl")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally erasing manual annotations.
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in candidates():
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(candidates())} unapproved candidates to {args.output}")


if __name__ == "__main__":
    main()

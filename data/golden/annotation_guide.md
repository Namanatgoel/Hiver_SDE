# Annotation Guide: Golden Set

## Task
Label each customer support message with:
1. `correct_intent`: the customer's actual intent (from the taxonomy)
2. `should_escalate`: whether a human agent should handle this (True/False)
3. `difficulty`: how hard the classification is (easy/medium/hard)
4. `notes`: optional free-text note

## Intent Taxonomy
See `intent_taxonomy.json` for the full list with descriptions.
Choose the intent that best matches what the customer actually wants.
If no intent fits, use `out_of_scope`.

## Escalation Rubric (answer YES to `should_escalate` if ANY of these apply)

| Condition | Example |
|---|---|
| **Billing/refund dispute** | "You charged me twice, I want a full refund" |
| **Legal/regulatory threat** | "I'll sue you", "I'm filing a complaint with the FCC" |
| **Safety/self-harm** | Any mention of harm to self or others |
| **Account takeover suspicion** | "Someone logged into my account without permission" |
| **Minor involved** | Customer explicitly mentions being a minor |
| **Repeated failure** | Brand has already replied 2+ times without resolving |
| **Genuinely ambiguous intent** | Even after reading 3 times, you cannot assign a confident intent |

Otherwise, answer NO: the agent can auto-handle.

## Difficulty Rubric

| Difficulty | Meaning |
|---|---|
| **easy** | Intent is unambiguous from the first sentence |
| **medium** | Intent requires reading the full message or context |
| **hard** | Ambiguous even with context; multiple valid interpretations |

## Double Annotation Protocol
50 examples are marked `double_annotate = True`.
Re-annotate them from scratch after at least 3 hours (or in a new session).
Do not look at your previous labels when re-annotating.
Disagreements will be adjudicated; do not silently overwrite.

## Schema
The template CSV has columns:
- `thread_id`: unique thread identifier
- `first_customer_message`: the message to classify
- `first_customer_created_at`: when it was sent
- `predicted_intent`: what the model predicted (for reference only — you may override)
- `model_confidence`: model's confidence score (for reference)
- `double_annotate`: True if this row needs a second pass
- `correct_intent`: **YOUR LABEL**
- `should_escalate`: **YOUR LABEL** (True/False)
- `difficulty`: **YOUR LABEL** (easy/medium/hard)
- `notes`: optional

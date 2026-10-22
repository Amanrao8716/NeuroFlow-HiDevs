# ADR 003: Use automated LLM-as-judge evaluation with human calibration

- **Status:** Accepted for the initial project architecture
- **Date:** 2026-10-22

## Context

NeuroFlow needs to evaluate every completed generation asynchronously on faithfulness, answer relevance, context precision, and context recall. Human annotation can provide high-quality judgments but is expensive, slow, difficult to scale to every generation, and often inconsistent without a rubric. Automated evaluation can provide broad coverage and fast feedback, but judge outputs can be biased, unstable, or confidently wrong.

## Decision

Use versioned automated LLM-as-judge evaluators for routine scoring, supported by deterministic checks where appropriate and a stratified human-annotated calibration set. Run evaluation asynchronously so generation latency is not blocked. Store each metric separately, alongside evaluator/model version, rubric version, evidence or rationale, evaluation status, timestamps, and trace IDs.

Rubrics:
- **Faithfulness:** assess whether answer claims are supported by retrieved context; require claim-level evidence where feasible.
- **Answer relevance:** assess directness and coverage relative to the question.
- **Context precision:** assess which retrieved chunks are relevant/useful; report the denominator and rank-aware variant clearly.
- **Context recall:** assess whether reference-relevant evidence is present. Require a reference answer, evidence labels, or a declared reference source; mark unavailable rather than fabricate a score.

Scores are normalized to `[0,1]`. Failed evaluations are stored as failures, not converted to zero. Judge prompt/model/rubric versions are persisted so historical comparisons can be interpreted.

## Consequences

**Benefits**
- Scales to every generation at a predictable operational cost.
- Supports rapid regression analysis by model, prompt, domain, and pipeline.
- Asynchronous processing keeps evaluation separate from user-facing latency.
- Human calibration provides an external check on judge quality.

**Failure modes**
- **Position/length bias:** judge favors longer answers or evidence shown first.
- **Self-preference/correlation:** judge favors outputs from its own model family.
- **Prompt injection:** untrusted retrieved text attempts to influence the judge.
- **Rubric ambiguity:** vague definitions create unstable scores.
- **Reference leakage or missing references:** apparent recall may be invalid.
- **Drift:** provider/model updates change scores without a system-quality change.
- **False precision:** one score hides disagreement and sample uncertainty.

**Detection and mitigation**
- Maintain a stratified, versioned human-labeled set covering domains, answer lengths, failures, and model families.
- Compare judge-human agreement, calibration, disagreement rates, and metric distributions over time.
- Use blinded candidate ordering and randomized context ordering in audits to detect position bias.
- Separate instructions from retrieved content; treat retrieved text as data, never judge instructions.
- Run periodic alternate-judge audits and targeted human review on low scores, high-impact cases, and sudden distribution shifts.
- Version rubrics; alert on judge latency/cost changes and score drift.
- Report sample counts and confidence intervals where practical; do not promote models based on one judge score alone.

## Operational rule

LLM-as-judge is a scalable measurement instrument, not ground truth. Human review remains required for calibration, high-impact decisions, and investigation of suspected evaluator failure.

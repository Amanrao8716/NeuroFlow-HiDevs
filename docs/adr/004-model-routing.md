# ADR 004: Route queries using capability first, then latency, quality, and cost constraints

- **Status:** Accepted as the initial routing policy; thresholds require benchmarking
- **Date:** 2026-10-22
- **Implementation target:** Task 38

## Context

NeuroFlow may support multiple model providers, model tiers, and domain-adapted fine-tuned models. Sending every request to the most capable model wastes money and can increase latency; always selecting the cheapest model can reduce reasoning quality, tool reliability, or faithfulness. Routing must be observable, explainable, bounded by user/tenant policy, and safe when providers fail.

The router receives query text, domain/classifier signals, requested capability, context token count, structured-output/tool requirements, latency preference, cost preference, provider health, model availability, and policy restrictions. Retrieved context remains the grounding source; routing does not replace retrieval.

## Decision

Use a policy-based router with four stages:

1. **Eligibility:** remove models that cannot satisfy context length, region/privacy policy, structured output/tool needs, safety requirements, or tenant allowlists.
2. **Capability/domain selection:** classify the request using explicit caller hints plus conservative signals. Prefer deterministic rules for clear requirements; use a lightweight classifier only where necessary.
3. **Constraint optimization:** among eligible candidates, honor hard latency and cost budgets first, then select the lowest-cost candidate expected to meet the quality bar. Use the higher tier when reasoning difficulty or uncertainty justifies it.
4. **Fallback and learning:** before any token is emitted, retry/fallback to another eligible model under bounded retry and timeout policy. Log the candidate set, features, chosen model, policy version, estimated/actual cost, latency, and outcome. Recalibrate thresholds using offline evaluation and production telemetry.

Model tiers are logical profiles, not hard-coded vendor names:
- **Tier S — Small/fast:** lowest latency and cost; classification, extraction, simple grounded FAQ, routing support.
- **Tier M — Balanced:** general RAG questions, summaries, multi-document synthesis, normal coding/help tasks.
- **Tier L — High-capability:** complex reasoning, difficult multi-hop synthesis, ambiguous evidence analysis, high-stakes internal workflows requiring stronger quality controls. It does not override domain safety policy.
- **Tier FT — Fine-tuned specialist:** only for a validated domain/task where a specific fine-tuned model beats or matches the base model on the required evaluation gates. Continue RAG for current facts and citations.

## Initial routing matrix

| Query type / signal | Preferred tier | Why | Escalate or fall back when |
|---|---|---|---|
| Query classification, metadata extraction, intent detection | S | Small output and narrow task | Confidence low or schema validation fails |
| Simple factual FAQ with strong retrieved evidence | S | Low complexity; retrieval provides answer evidence | Evidence conflicts, is weak, or answer needs synthesis |
| Short summary of one document/section | S or M | Tier S for brief summaries; M for nuanced constraints | Long input, multiple sources, or strict nuance required |
| Multi-document summary/comparison | M | Balanced synthesis and context handling | Contradictory evidence or complex reasoning |
| General knowledge-grounded RAG answer | M | Default quality/cost tradeoff | Low retrieval confidence or multi-hop question |
| Multi-hop reasoning across several chunks | L | More capable synthesis and reasoning | Context missing: retrieve more or abstain, not merely upgrade model |
| Complex coding/debugging or architecture analysis | L; M for routine code questions | Higher reasoning capability for intertwined constraints | Tool/format requirements unsupported |
| Strict JSON / schema-constrained extraction | S if reliable structured output; otherwise M | Cost-efficient when schema compliance is strong | Validation/retry fails; route to model with stronger structured-output support |
| Long-context synthesis beyond M context limit | Eligible long-context M/L | Must fit context and preserve budget | Truncate/retrieve selectively or reject; never silently exceed limit |
| Domain-specific repetitive task with proven fine-tune | FT | Lower cost/latency for validated pattern | Quality or safety regression, drift, low confidence |
| Sensitive/high-impact advice | L only if policy permits, plus domain safeguards | Stronger reasoning is not a safety guarantee | Require abstention, professional review, or disallowance as policy dictates |
| Provider degradation / timeout | Next eligible tier/provider | Availability | Never fallback across tenant/privacy constraints |
| Low confidence / poor evidence | Re-retrieve, clarify, or abstain | Model escalation cannot repair absent evidence | Escalate only if evidence exists but reasoning is hard |

## Constraints and implementation requirements

- **Cost:** use configured per-model input/output pricing and estimated tokens; enforce per-request and per-tenant budgets. Pricing must be configuration, not constants scattered through code.
- **Latency:** track time-to-first-token and total latency by route; define SLOs from measured baselines. A `fast` preference excludes models that cannot meet the configured latency budget based on recent telemetry.
- **Capability:** maintain a model registry of context limit, structured-output/tool support, modality, supported regions, safety requirements, availability, and benchmark scores.
- **Domain:** route fine-tuned specialists only for validated domains and tasks; classifier uncertainty must fall back to a general eligible model.
- **Quality:** use offline evaluation slices by query type and domain; require minimum sample counts and no unacceptable faithfulness/safety regression before promotion.
- **Observability:** record `routing_policy_version`, `routing_reason`, eligible candidates, selected model, fallback reason, estimated/actual cost, latency, and evaluation outcomes.
- **Streaming:** after the first token is emitted, do not silently switch models. Send an explicit SSE error if generation fails midstream and record partial output.
- **No evidence is not a routing problem:** when retrieval is empty or weak, clarify, retrieve again, or abstain instead of choosing a more expensive model to guess.

## Consequences

**Benefits:** cost-aware model selection, explainable decisions, domain specialization, and a measurable path to improve routing with evaluation data.

**Costs/risks:** routing adds classification/policy complexity; estimates can be wrong; provider capabilities and pricing change; noisy judges can reinforce bad routing decisions.

**Revisit when:** offline and online measurements show a route consistently misses its quality, latency, or cost objective. Change thresholds through versioned configuration and compare cohorts before global rollout.

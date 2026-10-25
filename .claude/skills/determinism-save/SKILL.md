---
name: determinism-save
description: After new determinism results (a test on any of the user's machines), rebuild RESULTS.md and update FINDINGS.md. Use after a determinism test when the user wants to save the result, or when the user asks what the results show so far.
---

# Save a result: table, findings, commit

Follow CLAUDE.md – never push.

1. `python results_table.py`.
2. Read the new result files. For each machine: text and logprobs repeatable at temperature 0 and 1 (`repeatable`,
   `logprobs_repeatable`)? Equal to the reference of the same model profile and llama.cpp build (`equals_reference`,
   `logprobs_equals_reference`; RESULTS.md shows "no ref for this build" when there is none)? `null` for logprobs =
   not checked, not a difference: the reference has no logprobs hash, or the server sent no logprobs (a warning says
   so); result files older than 25 Oct have no logprobs fields. Before
   attributing a difference to hardware, check `warnings` (model file, sampling, prompt tokens, build change,
   missing reference for the build) and `note` (partial offload, CPU). Different builds are not comparable.
3. Update FINDINGS.md: facts about a machine (label, date, hardware, versions) go to "Results so far", facts about
   one factor (model, quantization, thinking, seed, batching) to its own section; keep facts and hypotheses apart
   (mark hypotheses as such); add a dated line to "Changelog".
4. Commit locally: `git add results RESULTS.md FINDINGS.md` and `git commit -m "Findings: <what changed>"`.

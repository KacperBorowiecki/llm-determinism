---
name: determinism-findings
description: After new determinism results (a test on any of the user's machines), rebuild RESULTS.md and update FINDINGS.md. Use when the user brings in new results or asks what the results show so far.
---

# Update results and findings

Follow CLAUDE.md – never push.

1. `python results_table.py`.
2. Read the new result files. For each machine: repeatable at temperature 0 and 1? Equal to the reference of the
   same model profile and llama.cpp build (RESULTS.md shows "no ref for this build" when there is none)? Before
   attributing a difference to hardware, check `warnings` (model file, sampling, prompt tokens, build change,
   missing reference for the build) and `note` (partial offload, CPU). Different builds are not comparable.
3. Update FINDINGS.md: add new facts to "Results so far" (label, date, hardware, versions); keep facts and hypotheses
   apart (mark hypotheses as such); add a dated line to "Changelog".
4. Commit locally: `git add results RESULTS.md FINDINGS.md` and `git commit -m "Findings: <what changed>"`.

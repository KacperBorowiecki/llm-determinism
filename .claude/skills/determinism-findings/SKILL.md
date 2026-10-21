---
name: determinism-findings
description: After determinism results from other machines arrive (merged branch or received zip files), rebuild RESULTS.md and update FINDINGS.md. Use when the user brings in new results or asks what the results show so far.
---

# Update results and findings

Follow CLAUDE.md – never push.

1. Received zip files: unpack each into `results/<label>/`. Check they are result files (they have `machine`,
   `temperatures`, `runs`).
2. `python results_table.py`.
3. Read the new result files. For each machine: repeatable at temperature 0 and 1? Equal to the reference? Before
   attributing a difference to hardware, check `warnings` (model file, sampling, prompt tokens, build change) and
   `note` (partial offload, CPU).
4. Update FINDINGS.md: add new facts to "Results so far" (label, date, hardware, versions); keep facts and hypotheses
   apart (mark hypotheses as such); add a dated line to "Changelog".
5. Commit locally: `git add results RESULTS.md FINDINGS.md` and `git commit -m "Findings: <what changed>"`.

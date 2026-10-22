---
name: determinism-submit
description: Keep this machine's determinism test result in the repo (local git commit). Use after a determinism test when the user wants to save the result.
---

# Keep a result

Follow CLAUDE.md – never push.

1. Find the new result file(s): `git status --short results/`.
2. Rebuild the table: `python results_table.py`.
3. Commit locally: `git add results/<label>/<file>.json RESULTS.md` and `git commit -m "Add result: <label> <length>"`.

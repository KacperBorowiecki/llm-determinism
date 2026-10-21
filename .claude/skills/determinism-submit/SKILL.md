---
name: determinism-submit
description: Keep this machine's determinism test result in the repo (local git commit) or pack it for sending to the maintainer. Use after a determinism test when the user wants to save or share the result.
---

# Keep or share a result

Follow CLAUDE.md – never push, never open pull requests, never upload.

1. Find the new result file(s): `git status --short results/`.
2. Rebuild the table: `python results_table.py`.
3. Keep it locally: `git add results/<label>/<file>.json RESULTS.md` and `git commit -m "Add result: <label> <length>"`.
4. If the user wants to send it to someone else's repo: pack the result file into `export/<label>_<file name>.zip`
   (Windows: `Compress-Archive`; macOS/Linux: `zip`) and tell the user where it is. Sending it is the user's action.

# CLAUDE.md – llm-determinism

A small kit to check whether `llama-server` (from Ollama) gives the same text for the same prompt and seed – on
this machine (repeatability) and compared with the reference machine (same hash?). See README.md and FINDINGS.md.

## Rules
- **Always ask for the machine label** before a test. Propose one from the hardware (e.g. `mac-m5pro-64gb`);
  labels already in `results/` are only suggestions – wait for the user's confirmation.
- **Never `git push`, `gh pr create` or upload anything** unless the user asks for that specific action – publishing
  is the user's decision (the user pushes).
- **Model profiles** (`reference.json`): `qwen3:0.6b-q8_0` – the default, small and fast; `qwen3:0.6b-q4_K_M` – the same
  model in Q4_K_M (a different text); `gemma4:26b-a4b-it-q8_0` – the main experiment, needs ~30 GB of memory.
- **Before a test, ask:** model (`--profile`), thinking on/off (`--thinking`, default off) and length. Record where
  the model ran with `--device gpu|cpu|mix` (matches `-ngl`: all layers / 0 / some).
- **Before pulling a model** (sizes in `reference.json`: ~0.8 GB / ~28 GB): ask, check free disk space and check
  that it can fit at all (`memory_gb`: GPU memory, Apple unified memory, or GPU + RAM). If it cannot, say so and
  stop.
- **Do not start anything while the GPU is busy** (another `llama-server`, other GPU load). A test started directly
  with `llama-server` does not show in `ollama ps`.
- Python only in the repo's `.venv` (`python -m venv .venv`, then `pip install -r requirements.txt`);
  Windows: `PYTHONUTF8=1`.
- The protocol (`reference.json`) and the hash function (`text_hash` in `run_test.py`) must not change without the
  user's consent.
- Talk to the user in their language; everything in the repo is in English.

## Known pitfalls
- Ollama's `llama-server` starter without Ollama's library path **silently runs on CPU** (Windows:
  `OLLAMA_LIBRARY_PATH`, `GGML_BACKEND_PATH`) – always check the model is on the GPU after start.
- The Gemma 4 and Qwen3 chat templates **enable thinking by default** – `run_test.py` sends `enable_thinking: false`.
- **Reference hashes are per llama.cpp build:** an Ollama update can change the text (0.40.6 → 0.40.7 changed both
  models). Compare machines on the same build; a new build needs its reference generated on the origin PC.
- Use `127.0.0.1`, not `localhost` (IPv6 delay on Windows).
- Partial offload (`-ngl` lower than all layers) or CPU will most likely change the hash – that is a result, write it
  into `--note`.
- Ollama updates itself – compare versions in the result files before blaming the hardware.
- Sleep during a run pauses it; `llama-server` counts the pause as its own time.

## Skills
- `determinism-test` – run the test on this machine.
- `determinism-submit` – keep the result (local commit).
- `determinism-findings` – after new results: RESULTS.md + FINDINGS.md.
- Optional curiosity: `parallel_test.py` (10 requests at once, needs `-np 10 -c 20480`) – offer it only if the user
  is interested.

---
name: determinism-test
description: Run the llama-server determinism test on this machine (same prompt and seed, N runs at temperature 0 and 1) and compare the hashes with the reference. Use when the user wants to test or re-test determinism / repeatability on this computer.
---

# Determinism test on this machine

Follow CLAUDE.md. Talk to the user in their language.

1. **Python:** `.venv` in the repo root (`python -m venv .venv` if missing); `pip install -r requirements.txt`
   into it; run scripts with the `.venv` interpreter; Windows: `PYTHONUTF8=1`.
2. **Label:** ask for this machine's label (lowercase, digits, `.`, `_`, `-`). Propose one from the hardware.
   Wait for confirmation.
3. **Model and whether it can run:** ask which model profile (`reference.json` → `profiles`):
   `qwen3-0.6b-q8` – the default: ~0.8 GB download, ~2 GB memory, ~1 s per generation, runs on almost any
   machine (also CPU-only); `qwen3-0.6b-q4_K_M` – the same model in Q4_K_M (= the default tag `qwen3:0.6b`,
   ~0.5 GB; gives a different text than Q8, about as fast); `gemma4-26b-q8` – the main experiment: ~28 GB download, ~30 GB of GPU, unified, or
   GPU + RAM memory, ~8 s per generation on an RTX 5090. Check RAM and GPU memory (NVIDIA: `nvidia-smi`; Apple:
   unified memory) against `memory_gb` of the profile; if clearly not enough: say so, offer the smaller model,
   stop. Check `ollama --version`.
   Also ask: **thinking on or off** (default off). With thinking the model first writes its reasoning – a
   different, longer text; the answer and the thinking are hashed separately (`--thinking`). Origin references
   exist for all three profiles with and without thinking (gemma4-26b-q8 with thinking: ~2 000 tokens and
   ~11–12 s per answer on an RTX 5090).
4. **Model file:** `ollama list`. If the profile's `model` is missing: check free disk space and ask before
   `ollama pull <model>` (`download_gb` of the profile).
5. **GPU free?** No other `llama-server` process, nothing in `ollama ps`, no other heavy GPU load. If busy: stop and
   ask.
6. **Start llama-server from Ollama's installation:**
   - Binary: Ollama's log shows how Ollama starts it – search for lines containing
     `msg="starting llama-server"`; the `cmd="…"` part holds the full command (backslashes in paths are doubled) (Windows:
     `%LOCALAPPDATA%\Ollama\server.log`; macOS: `~/.ollama/logs/server.log`; Linux: `journalctl -u ollama`).
     Otherwise look in the install directory (Windows: `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe`).
     Not found: stop and ask – never use another binary silently.
   - Model file: the path in the **first** `FROM` line of `ollama show <model of the profile> --modelfile`.
   - Environment, as Ollama sets it (the log line with `OLLAMA_LIBRARY_PATH` shows the folders). Windows + NVIDIA:
     `PATH` += `<ollama>\lib\ollama;<ollama>\lib\ollama\cuda_v13` (the GPU folder Ollama uses),
     `OLLAMA_LIBRARY_PATH` = the same two folders, `GGML_BACKEND_PATH` = `<GPU folder>\ggml-cuda.dll`,
     `CUDA_VISIBLE_DEVICES` = GPU index (default 0).
   - Arguments: `--model <file> --host 127.0.0.1 --port 8090 --no-webui -ngl 999` + `server_args` from
     `reference.json`. If the model does not fit on the GPU, use a lower `-ngl` (partial offload: `--device mix`; no
     usable GPU: `-ngl 0`, `--device cpu`), tell the user, and put it into `--note`.
   - Start it in the background; wait until `http://127.0.0.1:8090/health` returns 200; check the model is on the GPU
     (NVIDIA: GPU memory in `nvidia-smi` grows by about the model size; otherwise generation speed).
     Unexpected CPU: stop and report.
7. **Length:** ask – quick (10 + 10, default), standard (100 + 100), long (1000 + 1000). Give the expected time
   (RTX 5090: ≈ 1 s per generation for qwen3-0.6b-q8, ≈ 8 s for gemma4-26b-q8; much slower with partial
   offload or CPU). For more than ~10 minutes remind the
   user to disable sleep.
8. **Run:** `python run_test.py --label <label> --profile <profile> --length <length> --device <gpu|cpu|mix> --gpu <index> [--thinking] --note "<context>"` – the note
   says what is special about this run: GPU/RAM split, other things running, and what happened since the last
   test on this machine (restart, Ollama or driver update, date change, sleep during the run).
9. **Stop the server.**
10. **Report:** per temperature – repeatable? equal to the reference? warnings? The reference is the one for the
    same llama.cpp build; if the build has no reference yet (a newer or older Ollama), the warning says so – the
    repeatability result still holds, but a different hash is expected and is not a hardware difference.
    If something differs, explain what is known (warnings, divergence position) and offer to investigate. Suggest `determinism-submit` to keep the result.

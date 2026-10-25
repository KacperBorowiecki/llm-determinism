---
name: determinism-test
description: Run the llama-server determinism test on this machine (same prompt and seed, N runs at temperature 0 and 1) and compare the hashes with the reference. Use when the user wants to test or re-test determinism / repeatability on this computer.
---

# Determinism test on this machine

Follow CLAUDE.md. Talk to the user in their language.

1. **Ollama:** `ollama --version`. Not installed: say so and stop – the test needs Ollama (the model and
   `llama-server` come from it). If it cannot connect to a running Ollama, ask the user to start it.
2. **Python ≥ 3.10 and its environment:**
   - A Python ≥ 3.10 on this machine (`python3 --version`, `python --version`; Windows also `py --list`). None
     (e.g. only macOS's system Python 3.9): say so and stop.
   - The environment: `.venv` in the repo root, checked with its own interpreter (`.venv/bin/python --version`;
     Windows: `.venv\Scripts\python.exe --version`). Missing: create it with that Python (`-m venv .venv`).
     Does not run here or older than 3.10 (made on another system, e.g. a drive shared by Windows and macOS):
     leave it and use `.venv-<system>` (e.g. `.venv-mac`) instead – check or create it the same way.
   - `pip install -r requirements.txt` into it; run all scripts with its interpreter; Windows: `PYTHONUTF8=1`.
3. **Hardware, then label:** first check the hardware with the same functions that write it into the result
   file: `python -c "import run_test as r; print(r.cpu_name(), r.ram_gb(), r.gpus())"` (CPU, RAM in GB, NVIDIA
   GPUs or Apple unified memory; an empty GPU list means no NVIDIA GPU was found – ask what GPU there is).
   Show it, then ask for this machine's label (lowercase, digits, `.`, `_`, `-`) and propose one from it
   (e.g. `mac-m5pro-64gb`). Wait for confirmation.
4. **Model and whether it can run:** let the user choose one of the profiles in `reference.json` → `profiles`
   (as options, the default first). Only these – other models have no references; if the user names another
   one, say that for now the test supports only these:
   `qwen3:0.6b-q8_0` – the default: ~0.8 GB download, ~2 GB memory, runs on almost any
   machine (also CPU-only); `qwen3:0.6b-q4_K_M` – the same model in Q4_K_M (= the default tag `qwen3:0.6b`,
   ~0.5 GB; gives a different text than Q8, about as fast); `gemma4:26b-a4b-it-q8_0` – the main experiment: ~28 GB download, ~30 GB of GPU, unified, or
   GPU + RAM memory, much slower than Qwen. Compare RAM and GPU memory from step 3 with
   `memory_gb` of the profile; if clearly not enough: say so, offer the smaller model,
   stop.
   Also ask: **thinking on or off** (default off). With thinking the model first writes its reasoning – a
   different, longer text; the answer and the thinking are hashed separately (`--thinking`). Origin references
   exist for all three profiles with and without thinking (gemma4:26b-a4b-it-q8_0 with thinking writes ~2 000 tokens
   per answer).
5. **Model file:** `ollama list`. If the profile's `model` is missing: check free disk space and ask before
   `ollama pull <model>` (`download_gb` of the profile); if the pull hangs near the end, stop it and run it
   again (it resumes). Then check the file: the **first** `FROM` line of `ollama show <model> --modelfile` must
   end with `model_file` of the profile. Different: the tag now points to another file – say so and ask; the
   test still shows repeatability, but the reference hashes no longer apply.
6. **GPU free?** No other `llama-server` process, nothing in `ollama ps`, no other heavy GPU load. If busy: stop and
   ask.
7. **Start llama-server from Ollama's installation:**
   - Binary: the `llama-server` that Ollama itself starts – never another one (Homebrew, own build: a different
     build gives a different text). Try in this order, stop at the first that works:
     1. Ollama's logs: search case-insensitively for `starting llama-server` (no quotes, no `cmd=` – the format
        changes between versions) in the current log **and the rotated older ones** – after a restart the
        current log may have none. The newest hit holds the full command; its first word is the binary (on
        Windows the log doubles backslashes). Logs: Windows `%LOCALAPPDATA%\Ollama\server*.log`; macOS
        `~/.ollama/logs/server*.log`; Linux `journalctl -u ollama --no-pager` (or `~/.ollama/logs/` when Ollama
        was started by hand).
     2. Ollama's install folder: resolve the `ollama` executable (`where ollama` / `command -v ollama`, follow
        links with `readlink -f`) and look for `llama-server*` next to it, in `lib/ollama` below it and in
        `../lib/ollama` (Windows: `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe`).
     3. Let Ollama start it (the model is there – step 5, the GPU is free – step 6): load the model through the
        API (`POST http://127.0.0.1:11434/api/generate` with only `{"model": "<model>"}` – loads, generates
        nothing), read the path and arguments from the process list (Windows: `Get-CimInstance Win32_Process
        -Filter "Name like 'llama-server%'" | Select-Object ExecutablePath, CommandLine`; macOS / Linux:
        `ps -eo pid,args | grep '[l]lama-server'`), then `ollama stop <model>`. The log now has a fresh start
        line too.
     4. Search the disk for `llama-server*`, but accept only a file inside Ollama's installation.
     Nothing found: stop and ask – this Ollama may not ship a separate `llama-server`.
   - Model file: the path in the **first** `FROM` line of `ollama show <model of the profile> --modelfile`.
   - Environment, as Ollama sets it – without it the server silently computes on the CPU. From the same logs:
     the line with `OLLAMA_LIBRARY_PATH` lists the library folders (its `extra_envs` the device variables, e.g.
     `CUDA_VISIBLE_DEVICES`), `load_backend: loaded CUDA backend from …` names the GPU library; on Linux, while
     Ollama's server runs, `/proc/<pid>/environ` of `llama-server` shows the whole environment. Windows + NVIDIA:
     `PATH` += `<ollama>\lib\ollama;<ollama>\lib\ollama\cuda_v13` (the GPU folder Ollama uses),
     `OLLAMA_LIBRARY_PATH` = the same two folders, `GGML_BACKEND_PATH` = `<GPU folder>\ggml-cuda.dll`,
     `CUDA_VISIBLE_DEVICES` = GPU index (default 0). Linux: the library folders go into `LD_LIBRARY_PATH`
     (not `PATH`).
   - Arguments: `--model <file> --host 127.0.0.1 --port 8090 --no-webui -ngl 999` + `server_args` from
     `reference.json`. If the model does not fit on the GPU, use a lower `-ngl` (partial offload: `--device mix`; no
     usable GPU: `-ngl 0`, `--device cpu`), tell the user, and put it into `--note`.
   - Start it in the background; wait until `http://127.0.0.1:8090/health` returns 200; check the model is on the GPU
     (NVIDIA: GPU memory in `nvidia-smi` grows by about the model size; otherwise generation speed).
     Unexpected CPU: stop and report.
8. **Length:** ask – quick (10 + 10, default), standard (100 + 100), long (1000 + 1000). No time estimates in
   advance – the speed depends on the hardware; a quick test shows this machine's time per generation, use it
   if the user asks how long a longer test takes. For standard and long remind the user to disable sleep.
9. **Run:** `python run_test.py --label <label> --profile <profile> --length <length> --device <gpu|cpu|mix> --gpu <index> [--thinking] --note "<context>"` – the note
   says what is special about this run: GPU/RAM split, other things running, and what happened since the last
   test on this machine (restart, Ollama or driver update, date change, sleep during the run).
10. **Stop the server.**
11. **Report:** per temperature – repeatable? equal to the reference? warnings? The reference is the one for the
    same llama.cpp build; if the build has no reference yet (a newer or older Ollama), the warning says so – the
    repeatability result still holds, but a different hash is expected and is not a hardware difference.
    If something differs, explain what is known (warnings, divergence position) and offer to investigate. Suggest `determinism-save` to save the result.

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
3. **Can it run?** Check RAM and GPU memory (NVIDIA: `nvidia-smi`; Apple: unified memory). The model needs ~30 GB
   (GPU, unified, or GPU + RAM). If clearly not enough: tell the user and stop. Check `ollama --version`.
4. **Model:** `ollama list`. If `gemma4:26b-a4b-it-q8_0` is missing: check free disk space and ask before
   `ollama pull gemma4:26b-a4b-it-q8_0` (~28 GB).
5. **GPU free?** No other `llama-server` process, nothing in `ollama ps`, no other heavy GPU load. If busy: stop and
   ask.
6. **Start llama-server from Ollama's installation:**
   - Binary: Ollama's log shows how Ollama starts it – search for lines containing
     `msg="starting llama-server"`; the `cmd="…"` part holds the full command (backslashes in paths are doubled) (Windows:
     `%LOCALAPPDATA%\Ollama\server.log`; macOS: `~/.ollama/logs/server.log`; Linux: `journalctl -u ollama`).
     Otherwise look in the install directory (Windows: `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-server.exe`).
     Not found: stop and ask – never use another binary silently.
   - Model file: the path in the **first** `FROM` line of `ollama show gemma4:26b-a4b-it-q8_0 --modelfile`.
   - Environment, as Ollama sets it (the log line with `OLLAMA_LIBRARY_PATH` shows the folders). Windows + NVIDIA:
     `PATH` += `<ollama>\lib\ollama;<ollama>\lib\ollama\cuda_v13` (the GPU folder Ollama uses),
     `OLLAMA_LIBRARY_PATH` = the same two folders, `GGML_BACKEND_PATH` = `<GPU folder>\ggml-cuda.dll`,
     `CUDA_VISIBLE_DEVICES` = GPU index (default 0).
   - Arguments: `--model <file> --host 127.0.0.1 --port 8090 --no-webui -ngl 999` + `server_args` from
     `reference.json`. If the model does not fit on the GPU, use a lower `-ngl`, tell the user, and put it into
     `--note`.
   - Start it in the background; wait until `http://127.0.0.1:8090/health` returns 200; check the model is on the GPU
     (NVIDIA: ~28 GB used in `nvidia-smi`; otherwise generation speed). Unexpected CPU: stop and report.
7. **Length:** ask – quick (10 + 10, default), standard (100 + 100), long (1000 + 1000). Give the expected time
   (RTX 5090 ≈ 8 s per generation; much slower with partial offload or CPU). For more than ~10 minutes remind the
   user to disable sleep.
8. **Run:** `python run_test.py --label <label> --length <length> --note "<GPU/RAM split, anything unusual>"`.
9. **Stop the server.**
10. **Report:** per temperature – repeatable? equal to the reference? warnings? If something differs, explain what is
    known (warnings, divergence position) and offer to investigate. Suggest `determinism-submit` to keep the result.

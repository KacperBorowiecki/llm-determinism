# llm-determinism

A small kit to check whether a local LLM gives **the same text for the same prompt and seed**:

1. **On this machine** – is the answer repeatable, run after run?
2. **Across machines** – is it the same text (same hash) as on the reference machine?

AI is often called non-deterministic. In practice it depends on how much of the computation you control: on the
reference PC the same model, prompt and seed gave thousands of identical answers in a row – at temperature 0 *and*
at temperature 1 – while the same server answering 10 requests at once gave different texts. This kit lets anyone
repeat the test on their own hardware and collect the results in one table.

## Requirements

- [Ollama](https://ollama.com) with one of the test models (profiles in `reference.json`):
  - `qwen3:0.6b-q8_0` – **default**: ~0.8 GB download, ~2 GB of memory, runs on almost any machine;
  - `qwen3:0.6b-q4_K_M` (the same file as the default tag `qwen3:0.6b`) – the Q4 variant, ~0.5 GB; gives a
    different text than Q8 (a different quantization is effectively a different model), about as fast;
  - `gemma4:26b-a4b-it-q8_0` – the main experiment: ~28 GB download, ~30 GB of GPU memory, Apple unified memory,
    or GPU + RAM together.
  The kit uses the `llama-server` that ships with Ollama.
- Python ≥ 3.10.
- [Claude Code](https://claude.com/claude-code) – recommended: the skills in `.claude/skills/` walk through
  everything machine-specific.

## Run it with Claude Code

Open this folder in Claude Code and say *"run the determinism test"* (skill `determinism-test`). Claude asks for a
label for your machine, checks that the model fits, starts `llama-server` from Ollama, runs the test and shows the
result. Then say *"save the result"* (skill `determinism-findings`: the table, the findings and a local commit).

## Run it by hand

1. Python: `python -m venv .venv`, then `.venv/bin/pip install -r requirements.txt`
   (Windows: `.venv\Scripts\pip install -r requirements.txt`).
2. Start `llama-server` from Ollama's installation with the model file from
   `ollama show <model> --modelfile` (path in the first `FROM` line) and the arguments from
   `reference.json`. Windows + NVIDIA example (PowerShell):

   ```powershell
   $lib = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama"
   $env:PATH = "$lib;$lib\cuda_v13;$env:PATH"
   $env:OLLAMA_LIBRARY_PATH = "$lib;$lib\cuda_v13"
   $env:GGML_BACKEND_PATH = "$lib\cuda_v13\ggml-cuda.dll"   # without these the server silently runs on CPU
   $env:CUDA_VISIBLE_DEVICES = "0"
   & "$lib\llama-server.exe" --model <model file> --host 127.0.0.1 --port 8090 --no-webui -ngl 999 `
       -c 8192 -np 1 --flash-attn auto -b 1024 -ub 1024 --split-mode none --jinja
   ```

   Elsewhere, use the binary Ollama itself starts: search Ollama's logs (also the older, rotated ones) for
   `starting llama-server`, or let Ollama load the model and read the `llama-server` command line from the
   process list; the skill `determinism-test` goes through this step by step.
3. Run the test (Windows: set `PYTHONUTF8=1` first):

   ```
   python run_test.py --label <your-machine-label> --length quick     # default model, 10 + 10 runs
   python run_test.py --label <your-machine-label> --profile gemma4:26b-a4b-it-q8_0 --length standard   # 100 + 100
   python run_test.py --label <your-machine-label> --thinking --device cpu   # thinking on, model on the CPU
   ```

## How the hash is computed

1. The answer text exactly as returned by the server (`choices[0].message.content`), no reasoning, no prompt.
   With `--thinking`, the thinking (`reasoning_content`) gets its own hash, computed the same way.
2. No normalisation: no trimming, no Unicode normalisation, no newline conversion.
3. UTF-8 without BOM → SHA-256 → lowercase hex; the short hash is the first 12 characters (the full digest is stored
   too).

`text_hash` in `run_test.py` is the only place that computes hashes. SHA-256 is the same on every operating system,
so a hash can only differ if the text differs. `pytest` re-hashes the two reference texts in `reference/texts/` on
your machine – if that fails, something changed the files (e.g. line endings), not the model.

Reference hashes come from the origin PC and are stored **per model and per llama.cpp build** – an Ollama update
can change the text (0.40.6 → 0.40.7 changed both models), so results are compared with the reference of the
same build:

| model (thinking) | Ollama (llama.cpp build) | temperature 0 | temperature 1 |
|---|---|---|---|
| qwen3:0.6b-q8_0 | 0.40.6 (`b1-620d2225c`) | `8cc886e38b1a` | `e94074591fd4` |
| qwen3:0.6b-q8_0 | 0.40.7 (`b1-f174553e1`) | `23cd03276e07` | `599eb85dd1d7` |
| qwen3:0.6b-q8_0 (on) | 0.40.6 (`b1-620d2225c`) | `482f7ccfcecf` + thinking `cb78d96c1efb` | `31cf38892cc4` + thinking `3098c730f004` |
| qwen3:0.6b-q8_0 (on) | 0.40.7 (`b1-f174553e1`) | `93da943bed61` + thinking `52499a3e123b` | `fa20d1ae8cef` + thinking `4074d4788a6f` |
| gemma4:26b-a4b-it-q8_0 | 0.40.4, 0.40.6 (`b1-b42b7e6d3`, `b1-620d2225c`) | `138e66a98238` | `0e661f14a480` |
| gemma4:26b-a4b-it-q8_0 | 0.40.7 (`b1-f174553e1`) | `179e3c96dff3` | `d4afe73e6d3c` |
| qwen3:0.6b-q4_K_M | 0.40.7 (`b1-f174553e1`) | `0ac50154a638` | `872d20cd2d2e` |
| qwen3:0.6b-q4_K_M (on) | 0.40.7 (`b1-f174553e1`) | `1f701b198097` + thinking `227ff26f211c` | `14b5be25f98e` + thinking `ac573c3b004d` |
| gemma4:26b-a4b-it-q8_0 (on) | 0.40.7 (`b1-f174553e1`) | `29ea83a13637` + thinking `2d68efe195a0` | `f975c9bddcfd` + thinking `1d183e394b78` |

## Results

- `results/<label>/<start>_<model>[_thinking]_<length>.json` – one file per test: machine and software description, hashes per
  temperature, repeatable / equal to the reference, warnings, every unique text once.
- `RESULTS.md` – a table of all results, generated by `python results_table.py`.
- `FINDINGS.md` – what the results show so far.

## Optional: 10 requests at once

`parallel_test.py` sends 10 requests at the same time to a server started with `-np 10 -c 20480`. On the reference
machine none of the simultaneous answers matched the single-request text – the server computes requests together in
one batch, which is what makes cloud APIs non-deterministic even at temperature 0.

## License

MIT – see [LICENSE](LICENSE). Use it, copy it, change it and share it, for any purpose; just keep the
copyright notice.

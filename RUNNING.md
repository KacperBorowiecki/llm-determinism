# Running the test by hand

The steps the skill `determinism-test` goes through, as commands – for **Windows** (PowerShell, NVIDIA GPU) and
**macOS** (Terminal, Apple Silicon). You need two terminals in the repo folder: one runs the server, the other the
test.

## 1. Once: the model and the Python environment

Pull a model – one of the profiles in `reference.json`:

| profile | download | memory |
|---|---|---|
| `qwen3:0.6b-q8_0` (default) | ~0.8 GB | ~2 GB |
| `qwen3:0.6b-q4_K_M` | ~0.5 GB | ~2 GB |
| `gemma4:26b-a4b-it-q8_0` | ~28 GB | ~30 GB (GPU, Apple unified memory, or GPU + RAM) |

```
ollama pull qwen3:0.6b-q8_0
```

The scripts need **Python 3.10 or newer** – check with `python --version` (Windows) or `python3 --version` (macOS;
the python3 that comes with macOS may be 3.9). Older: install a current one from python.org (on macOS also
Homebrew) and create the environment with it, e.g. `python3.12 -m venv .venv`.

Windows:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # lets Activate.ps1 run, in this window only
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If a company policy blocks `Activate.ps1`, skip the activation and call the environment's interpreter directly:
`.venv\Scripts\python.exe -m pip install -r requirements.txt`, later `.venv\Scripts\python.exe run_test.py …`.

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the repo is on a drive shared by Windows and macOS, give each system its own environment (e.g. `.venv-mac`).

## 2. Terminal 1: start the server

First make sure nothing else uses the GPU: `ollama ps` shows no model (otherwise `ollama stop <model>`) and no
other `llama-server` runs (Windows: `nvidia-smi` lists no other GPU processes; macOS: `pgrep -x llama-server` prints
nothing).

`llama-server` comes with Ollama. The model file is the path in the first `FROM` line of
`ollama show <model> --modelfile`; the other arguments are the test's setup, the same on every machine.

Windows:

```powershell
$tag = "qwen3:0.6b-q8_0"                       # the model profile
$gpu = "0"                                     # which GPU
$lib = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama"
$cuda = "$lib\cuda_v13"                        # cuda_v12 with an older NVIDIA driver
$env:PATH = "$lib;$cuda;$env:PATH"             # without these four variables the server silently runs on the CPU
$env:OLLAMA_LIBRARY_PATH = "$lib;$cuda"
$env:GGML_BACKEND_PATH = "$cuda\ggml-cuda.dll"
$env:CUDA_VISIBLE_DEVICES = $gpu
$model = ((ollama show $tag --modelfile | Select-String '^FROM' | Select-Object -First 1).Line -replace '^FROM\s+', '')
& "$lib\llama-server.exe" --model $model --host 127.0.0.1 --port 8090 --no-webui -ngl 999 `
    -c 8192 -np 1 --flash-attn auto -b 1024 -ub 1024 --split-mode none --jinja
```

macOS (no environment variables – Metal is built in):

```bash
TAG=qwen3:0.6b-q8_0                            # the model profile
MODEL=$(ollama show "$TAG" --modelfile | grep -m1 '^FROM' | sed 's/^FROM //')
/Applications/Ollama.app/Contents/Resources/llama-server --model "$MODEL" --host 127.0.0.1 --port 8090 --no-webui \
    -ngl 999 -c 8192 -np 1 --flash-attn auto -b 1024 -ub 1024 --split-mode none --jinja
```

Wait for `listening on http://127.0.0.1:8090`, then check that the model is on the GPU:
- Windows: `nvidia-smi` shows the chosen GPU's memory grown by about the model size. If it did not grow, the server
  runs on the CPU: stop it (Ctrl+C), check `$cuda` – `nvidia-smi` shows the driver's `CUDA Version` top right; below
  13 use `cuda_v12` – and start again.
- macOS: Activity Monitor → Window → GPU History is busy while the test generates.

The model does not fit into GPU memory: lower `-ngl` (e.g. `-ngl 30`, only part of the model on the GPU) and run
the test with `--device mix --note "-ngl 30"`; no usable GPU: `-ngl 0` and `--device cpu`. Part of the model or all
of it on the CPU will most likely give a different text than the reference.

## 3. Terminal 2: run the test

Variables live only in the terminal window where they are set, so the model (and on Windows the GPU) is set here
again – the same values as in terminal 1.

Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
$tag = "qwen3:0.6b-q8_0"                       # the same model as in terminal 1
$gpu = "0"                                     # the same GPU as in terminal 1
$label = "my-pc"                               # this machine's label, e.g. origin-pc-2x5090-win11
python run_test.py --label $label --profile $tag --length quick --gpu $gpu
```

macOS:

```bash
source .venv/bin/activate
TAG=qwen3:0.6b-q8_0                            # the same model as in terminal 1
LABEL=my-mac                                   # this machine's label, e.g. mac-m5pro-64gb
python run_test.py --label "$LABEL" --profile "$TAG" --length quick
```

- `--label` – this machine, lowercase letters, digits, `.`, `_`, `-` (e.g. `mac-m5pro-64gb`);
- `--profile` – the model the server runs (the same tag as in step 2);
- `--length quick | standard | long` – 10, 100 or 1000 runs at each temperature;
- `--thinking` – the model's thinking on (default off);
- `--device gpu | cpu | mix` – where the model runs; `--gpu <index>` – which GPU (Windows: the same index as
  `CUDA_VISIBLE_DEVICES`); both are recorded in the result;
- `--note "…"` – anything unusual (another program running, a restart or an update since the last test);
- `--seed 7` / `--seed none` – a demo of what the seed does: another seed, or none (the server then draws a new one
  for every request). At temperature 0 the text stays the reference, at temperature 1 it changes; such results
  are marked (`_seed7` / `_noseed` in the file name, a warning) and never become references;
- `--save-raw` – keep every run locally in `results/<label>/raw/` (not in git): the server's full response byte for
  byte and the final answer as text, two files per run. Qwen quick ≈ 1–2 MB, Gemma quick ≈ 7 MB, Gemma long ≈ 650 MB;
  with thinking 2–3× more. An interrupted test keeps the runs written so far (without a result file).

For a long test keep the computer awake – Windows: Settings → Power → never sleep; macOS: start the test with
`caffeinate -i python run_test.py …`.

## 4. After the test

Stop the server with Ctrl+C in terminal 1. Then, in terminal 2, rebuild the table:

```
python results_table.py
```

The result is in `results/<label>/`, the table in `RESULTS.md`.

## Optional: 10 requests at once

Start the server as in step 2, but with `-np 10 -c 20480` instead of `-c 8192 -np 1`, then in terminal 2 (with
the variables from step 3):

Windows:

```powershell
python parallel_test.py --label $label --rounds 3 --temperature 0
```

macOS:

```bash
python parallel_test.py --label "$LABEL" --rounds 3 --temperature 0
```

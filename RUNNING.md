# Running the test by hand

The steps the skill `determinism-test` goes through, as commands – for **Windows** (PowerShell, NVIDIA GPU) and
**macOS** (Terminal, Apple Silicon). You need two terminals in the repo folder: one runs the server, the other the
test.

## 1. Once: the model and the Python environment

Pull a model – one of the profiles in `reference.json` (`qwen3:0.6b-q8_0` is the default, `qwen3:0.6b-q4_K_M`,
`gemma4:26b-a4b-it-q8_0`):

```
ollama pull qwen3:0.6b-q8_0
```

Windows:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # lets Activate.ps1 run, in this window only
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS:

```bash
python3 -m venv .venv          # needs Python ≥ 3.10 – the python3 that comes with macOS may be older
source .venv/bin/activate
pip install -r requirements.txt
```

If the repo is on a drive shared by Windows and macOS, give each system its own environment (e.g. `.venv-mac`).

## 2. Terminal 1: start the server

`llama-server` comes with Ollama. The model file is the path in the first `FROM` line of
`ollama show <model> --modelfile`. Everything after the model file is the test's setup, the same on every machine
(`server_args` in `reference.json`).

Windows:

```powershell
$tag = "qwen3:0.6b-q8_0"                       # the model profile
$lib = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama"
$cuda = "$lib\cuda_v13"                        # cuda_v12 with an older NVIDIA driver
$env:PATH = "$lib;$cuda;$env:PATH"             # without these four variables the server silently runs on the CPU
$env:OLLAMA_LIBRARY_PATH = "$lib;$cuda"
$env:GGML_BACKEND_PATH = "$cuda\ggml-cuda.dll"
$env:CUDA_VISIBLE_DEVICES = "0"                # which GPU
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

Wait for `listening on http://127.0.0.1:8090`. Then check that the model is on the GPU – Windows: `nvidia-smi`
shows the chosen GPU's memory grown by about the model size; macOS: Activity Monitor → Window → GPU History is busy
while the test generates. Nothing else should use the GPU during the test (no other `llama-server`, `ollama ps`
empty).

## 3. Terminal 2: run the test

Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python run_test.py --label <machine-label> --profile qwen3:0.6b-q8_0 --length quick
```

macOS:

```bash
source .venv/bin/activate
python run_test.py --label <machine-label> --profile qwen3:0.6b-q8_0 --length quick
```

- `--label` – this machine, lowercase letters, digits, `.`, `_`, `-` (e.g. `mac-m5pro-64gb`);
- `--profile` – the model the server runs (the same tag as in step 2);
- `--length quick | standard | long` – 10, 100 or 1000 runs at each temperature;
- `--thinking` – the model's thinking on (default off);
- `--device gpu | cpu | mix` and `--gpu <index>` – where the model runs, recorded in the result;
- `--note "…"` – anything unusual (another program running, a restart or an update since the last test).

For a long test keep the computer awake – Windows: Settings → Power → never sleep; macOS: start the test with
`caffeinate -i python run_test.py …`.

## 4. After the test

Stop the server with Ctrl+C in terminal 1 and rebuild the table:

```
python results_table.py
```

The result is in `results/<label>/`, the table in `RESULTS.md`.

## Optional: 10 requests at once

Start the server as in step 2, but with `-np 10 -c 20480` instead of `-c 8192 -np 1`, then in terminal 2:

```
python parallel_test.py --label <machine-label> --rounds 3 --temperature 0
```

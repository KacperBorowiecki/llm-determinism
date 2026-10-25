# llm-determinism

**Is an LLM deterministic?** AI is often called non-deterministic: ask twice, get two different answers. We wanted
to know whether that comes from the model itself or from the environment it runs in. This repo sends the same
prompt with the same seed to a local model (`llama-server` from [Ollama](https://ollama.com)) many times and
compares the answers by hash – run after run on one machine, and between machines.

## What we found

**The model itself is deterministic – even at temperature 1: the same input (prompt and seed) in a frozen
environment gives the same output, token for token.** The randomness people see comes from the environment
around the model.

- **Thousands of identical answers in a row.** Gemma 4 26B (Q8_0) on the reference PC: 2 108 runs at temperature 0
  and 10 108 runs at temperature 1 – one text each, on both GPUs. A restart, the PC going to sleep in the middle of
  a generation and heavy CPU load from another process did not change the text.
- **Temperature 1 repeats too.** Sampling really picks less likely tokens (106 of the 1 346 tokens in that answer),
  but it is pseudo-random: with a fixed seed every draw repeats. *Temperature decides what is sampled; the seed
  makes the sampling repeat.*
- **Every test so far was repeatable:** 3 models (Gemma 4 26B, Qwen3 0.6B in Q8_0 and Q4_K_M), thinking on and off,
  2 machines (Windows with 2× RTX 5090, Apple M5 Pro).

What "frozen" means: the same model file, llama.cpp build, settings and machine, one request at a time. Each of
these changed the text when it changed:

| what changed | result |
|---|---|
| llama.cpp build (Ollama update 0.40.6 → 0.40.7) | a different text, still repeatable (the update 0.40.4 → 0.40.6 changed nothing) |
| quantization (Q4_K_M instead of Q8_0) | a different text – effectively a different model |
| thinking on / off | a different prompt, so a different text |

The input means the exact tokens: any change in the prompt – even one extra space – makes it a different input
and can change the whole answer.

Computing several requests together (batching) can also change the text – this is how cloud APIs work, so even
temperature 0 does not guarantee the same answer twice there. Locally, one request at a time, the answer repeats;
`parallel_test.py` shows the effect on your own server.

*Repeatable ≠ correct:* a broken answer repeats just as faithfully – on the Mac a tiny model looped until the
token limit, identically in 10 of 10 runs.

All numbers: [FINDINGS.md](FINDINGS.md); every test: [RESULTS.md](RESULTS.md).

## How the test works

- One prompt (a short request for a story, in Polish), seed 42, temperatures 0 and 1, the same number of runs at
  each (quick 10, standard 100, long 1000). One request at a time. Model profiles and server
  arguments: `reference.json`.
- Reference hashes come from the reference PC and are stored in `reference.json` **per model, llama.cpp build and
  thinking mode**; the texts are in `reference/texts/`. A result is compared with the reference of the same build.

### How the hash is computed

1. The answer text exactly as returned by the server (`choices[0].message.content`), no reasoning, no prompt.
   With thinking on, the thinking (`reasoning_content`) gets its own hash, computed the same way.
2. No normalisation: no trimming, no Unicode normalisation, no newline conversion.
3. UTF-8 without BOM → SHA-256 → lowercase hex; the short hash is the first 12 characters (the full digest is stored
   too).

`text_hash` in `run_test.py` is the only place that computes hashes. SHA-256 is the same on every operating system,
so a hash can only differ if the text differs. `pytest` re-hashes all reference texts in `reference/texts/` on your
machine – if that fails, something changed the files (e.g. line endings), not the model.

## Running it with Claude Code

Requirements:
- [Ollama](https://ollama.com) – the model and `llama-server` come from it. Model profiles:
  - `qwen3:0.6b-q8_0` – **default**: ~0.8 GB download, ~2 GB of memory, runs on almost any machine;
  - `qwen3:0.6b-q4_K_M` (the same file as the default tag `qwen3:0.6b`) – ~0.5 GB, a different text than Q8;
  - `gemma4:26b-a4b-it-q8_0` – the main experiment: ~28 GB download, ~30 GB of GPU memory, Apple unified memory,
    or GPU + RAM together.
- Python ≥ 3.10.
- [Claude Code](https://claude.com/claude-code). The kit was built and run with Claude Opus 5.5.

Open this folder in Claude Code and say:
- *"run the determinism test"* – skill `determinism-test`: checks Ollama and the hardware, asks for the machine
  label, the model and thinking, starts `llama-server` from Ollama's installation, runs the test and reports;
- *"save the result"* – skill `determinism-save`: rebuilds `RESULTS.md`, adds the findings, makes a local commit.

Every step is written out in [.claude/skills/determinism-test/SKILL.md](.claude/skills/determinism-test/SKILL.md).
The batching test (10 requests at once, `parallel_test.py`) is optional – ask Claude for it.

## Results

- `results/<label>/<start>_<model>[_thinking]_<length>.json` – one file per test: machine and software description,
  hashes per temperature, repeatable / equal to the reference, warnings, every unique text once.
- `RESULTS.md` – a table of all results, generated by `python results_table.py`.
- `FINDINGS.md` – what the results show so far.

## License

MIT – see [LICENSE](LICENSE). Use it, copy it, change it and share it, for any purpose; just keep the
copyright notice.

# Findings

Facts come from result files in `results/` unless a section names another source; hypotheses are marked as such.

## Results so far

### origin-pc-2x5090-win11 (reference machine, 19–22 Oct 2026)

Windows 11 Pro, AMD Ryzen 9 9950X, 61.6 GB RAM, 2× RTX 5090 (driver 610.88, CUDA 13.3), `llama-server` from Ollama
0.40.4 (`b1-b42b7e6d3`), 0.40.6 (`b1-620d2225c`) and 0.40.7 (`b1-f174553e1`), one request at a time, prompt cache off.

**gemma4:26b-a4b-it-q8_0** (the main experiment):

- **Temperature 0:** 100 runs on GPU 0 + 5 on GPU 1 (Ollama 0.40.4), 3 after a restart and an automatic Ollama
  update (0.40.6), then 1000 + 1000 runs on both GPUs at the same time (one server per GPU) – **all
  `138e66a98238`**. The long run crossed midnight, a full hour, and the PC went to sleep in the middle of a
  generation – no change.
- **Temperature 1:** 100 runs on GPU 0 + 5 on GPU 1, 3 after the restart, then 5000 + 5000 runs on both GPUs at
  the same time (20 Oct, 08:11 → 19:41, ~11.5 h, no pauses) – **all 10 108 runs `0e661f14a480`**. The model really
  samples: 106 of 1 346 tokens are not the most likely token
  (e.g. one with p = 0.038), and every one of these draws repeated identically. *Temperature decides what is
  sampled; the seed makes the sampling repeat.*
- **Near-ties:** the temperature-0 answer has 8 tokens where the top two candidates are within 0.05 nat; the closest:
  "…Golden Retrieverem, którego" → "główn" p = 0.4210 vs "ul" p = 0.4205. In single requests it is always resolved
  the same way.
- The restart and the first silent Ollama update (0.40.4 → 0.40.6, new llama.cpp build) did not change either hash;
  the second update (0.40.7) did – see below.

### Ollama update 0.40.6 → 0.40.7 (21 Oct): new text for both models

- Ollama updated itself to 0.40.7: new llama.cpp build `b1-f174553e1` and new `ggml-cuda.dll`, `libllama.dll`,
  `libllama-server-impl.dll`. Driver, model files, sampling parameters and the 29 prompt tokens unchanged.
- **gemma4:26b-a4b-it-q8_0:** still repeatable (10/10 at both temperatures, in two separate tests), but a **different text**:
  temperature 0 `179e3c96dff3` (1 276 tokens) instead of `138e66a98238`, temperature 1 `d4afe73e6d3c` (981 tokens)
  instead of `0e661f14a480`. Both split from the old text at character 133, in the title: "Wielka Przygoda na
  **Podwórku**" → "na **Słonecznej Polanie**" – token 43 ("Pod" 0.44 vs "S" 0.37, a margin of 0.17 nat). It is
  **not a near-tie**: 25 tokens of the answer have a smaller margin, e.g. token
  37 ("Wiel" vs "Tr", 0.11 nat), earlier in the text, did not flip. The text splits at the first token where the
  change of the numbers exceeds the margin, not at the closest one – and the update moved them by more than
  0.17 nat there, far more than rounding noise.
- **qwen3:0.6b-q8_0:** also a different text, also repeatable: `23cd03276e07` / `599eb85dd1d7` instead of
  `8cc886e38b1a` / `e94074591fd4`.
- **Not caused by CPU load:** the first 0.40.7 test ran while a CPU-only `llama-server` test (the same Gemma model,
  `-ngl 0`, ~80 % CPU) was running in another process; a clean rerun the next day (CPU ~3 %) gave exactly the same
  new hashes for both models. The CPU load only slowed the GPU generation (114 instead of ~174 tok/s).
- *An update can change the text, but does not have to* (0.40.4 → 0.40.6 did not). Therefore `reference.json`
  stores reference hashes **per model and per llama.cpp build**.

### mac-m5pro-64gb (first machine outside the origin PC, 22 Oct 2026)

macOS (Darwin 27.0.0, arm64), Apple M5 Pro, 64 GB unified memory, Metal (all layers on the GPU), `llama-server` from
Ollama 0.40.4 (`b1-b42b7e6d3`), the model file of the `qwen3:0.6b` tag (= `qwen3:0.6b-q4_K_M`).

- **qwen3:0.6b-q4_K_M, thinking off:** `631ab4fa9203` (498 tokens) / `ccacc3b1968b` (568 tokens), 10/10 each,
  ~1.7–1.9 s per generation. **Not comparable with the PC:** the Q4_K_M references exist only for build
  `b1-f174553e1` (Ollama 0.40.7); on another build a different hash is expected and says nothing about the hardware.
- **Thinking on – a repetition loop, reproduced exactly:** temperature 0 `4c655c49d7b9` + thinking `c2e8cd6366bc`,
  10/10, but **every run hit the 8 000-token limit**. The thinking ends normally (~1 400 characters; the prompt is
  misunderstood again: "a fox, a rabbit, and a rabbit"), then the answer starts "W bosom boskiego…" and repeats
  "boskiego" until the limit – ~40 s per run. Temperature 1: `1f65549427df` + thinking `e922edce0c4d` (1 041 tokens,
  ~4 s), 10/10, ended normally. On the origin PC (Ollama 0.40.7) the same profile ended normally at both temperatures
  (768 / 795 tokens).
- *Repeatable ≠ correct, again:* determinism reproduces a broken answer as faithfully as a good one – the same loop,
  the same 8 000 tokens, ten times. And a different build can change not only the hash but the quality of the answer.
- *Hypothesis:* a tiny Q4 model with greedy decoding (temperature 0) can lock into a repetition loop; at
  temperature 1 this seed happened to avoid it. Whether the loop comes from the build (0.40.4) or from the hardware
  (Metal vs CUDA) cannot be told from these data.

## Second model: qwen3:0.6b-q8_0 (default since 22 Oct)

`qwen3:0.6b-q8_0` (752M parameters, 0.8 GB download): ~500 tok/s on an RTX 5090, ~1 s per generation, runs on almost
any machine.

- Ollama 0.40.6: `8cc886e38b1a` (536 tokens) / `e94074591fd4` (434 tokens), 10/10 each.
  Ollama 0.40.7: `23cd03276e07` (370 tokens) / `599eb85dd1d7` (90 tokens), 10/10 each, in two separate tests.
- **Thinking on** (`--thinking`; Ollama 0.40.6 and 0.40.7): also repeatable – both the answer and the thinking
  10/10 at both temperatures, on both versions (0.40.7: `93da943bed61` + thinking `52499a3e123b` /
  `fa20d1ae8cef` + thinking `4074d4788a6f`). The text differs from the thinking-off text: the prompt differs –
  25 tokens instead of 29, because with thinking off the chat template appends an empty thinking block. The thinking shows
  that the model misunderstood the Polish prompt, differently at each temperature ("a story about a fox, a rabbit,
  and a cat" / "a wolf and a mouse, plus a cat") – and repeats the misunderstanding exactly. Repeatable ≠ correct.
- The Polish text quality is very poor (invented words) – irrelevant for the test.
- Its default sampling differs from Gemma's (`top_k` 40 instead of 64) – stored per profile in `reference.json`.

## Quantization: Q4_K_M vs Q8_0 (qwen3:0.6b, Ollama 0.40.7)

`qwen3:0.6b` (the default tag) is `qwen3:0.6b-q4_K_M` – the same file (`sha256-7f4030143c1c…`, 522 MB vs 832 MB
for Q8_0). Profile `qwen3:0.6b-q4_K_M`.

- **Repeatable, but a different text than Q8 in all four combinations** (temperature 0 / 1, thinking off / on):
  e.g. temperature 0 – Q4 `0ac50154a638` (446 tokens) vs Q8 `23cd03276e07` (370 tokens). The quantization rounds
  the weights differently, which shifts the probabilities; at the first token where the shift exceeds the margin
  between the two most likely candidates the choice flips, and the rest of the text follows – a different
  quantization is effectively a different model.
- **Hardly faster:** ~557–567 tok/s (Q4) vs ~528–543 tok/s (Q8) on an RTX 5090 – about 5 %, although the file is
  37 % smaller. For such a small model the speed is limited by the fixed cost per token (launching GPU work,
  choosing the token on the CPU), not by reading the weights. *(Hypothesis for the reason; the numbers are
  measured.)*

## Thinking on: gemma4:26b-a4b-it-q8_0 (Ollama 0.40.7)

- **Repeatable:** answer and thinking 10/10 at both temperatures – temperature 0 `29ea83a13637` + thinking
  `2d68efe195a0` (1 843 tokens incl. thinking), temperature 1 `f975c9bddcfd` + thinking `1d183e394b78`
  (2 143 tokens); ~11–12 s per generation, ~175 tok/s; all runs ended normally (within the 8 000-token limit).
- **The prompt gets longer with thinking (32 tokens instead of 29)** – the Gemma 4 template adds a system turn
  with `<|think|>`; with Qwen3 it gets shorter (25 instead of 29). The expected prompt length depends on the
  model and the mode, so each reference stores its own.

## The seed at temperature 0 does nothing (25 Oct)

Origin PC, gemma4:26b-a4b-it-q8_0, Ollama 0.40.7 (`b1-f174553e1`); for each seed a freshly started `llama-server`
on GPU 0 and the kit's own request – only the seed differs.

| seed | temperature 0 (3 runs) | temperature 1 (1 control run) |
|---|---|---|
| 42 | `179e3c96dff3` ×3 | `d4afe73e6d3c` |
| 7 | `179e3c96dff3` ×3 | `e8ca311a05d4` |
| 1234 | `179e3c96dff3` ×3 | `1c94b00f867b` |

- **Temperature 0: three seeds, all 9 runs – one text**, the reference `179e3c96dff3` (1 276 tokens). At
  temperature 0 llama.cpp takes the most likely token every time; nothing is drawn, so the seed has nothing to
  act on.
- **Temperature 1: every seed a different text**; seed 42 gives the reference `d4afe73e6d3c`. The seed does reach
  the server – it decides the draw.
- So the two temperatures test different things: repeatability at temperature 0 shows that the *computation* is
  stable (the same numbers, the same most likely token every time); at temperature 1, that the *seeded
  sampling* repeats.
- Ad-hoc test, not in `results/`; a script for it is planned.

## Batching: several requests computed together

Computing several requests together in one batch (a server with several slots, e.g. `-np 10`) can change the text
even at temperature 0 – this is how cloud APIs serve requests, so they usually cannot guarantee determinism. To
check it on your own server: `parallel_test.py` (10 requests at once, server started with `-np 10 -c 20480`).

## Related observations outside this kit

From the origin project (Ollama's own API, not `llama-server` directly):
- With Ollama's prefix cache, the first call and later calls of the same prompt at temperature 0 gave different
  texts (e4b Q4_K_M).
- After the 0.40.4 → 0.40.6 update, one model (e4b Q4_K_M with speculative decoding, in Ollama) changed its
  first-call text while the 26B Q8_0 in `llama-server` did not; after 0.40.7 the 26B Q8_0 changed too.

## What determinism is good for – and its limits

Useful wherever a result must be *reproduced* or a *change detected*: debugging (reproduce exactly the answer a
user got – if seed, version and environment are logged), a "canary" after updates (fixed prompts with known hashes –
exactly what caught the 0.40.7 change), regression tests of prompts and pipelines, fair A/B comparisons (same seed on
both sides), audit, reproducible synthetic data, safe caching and resuming of pipelines, red teaming, demos.

Limits: repeatable ≠ correct (a wrong answer is repeatably wrong); one seed is one sample, not the model's
variability; a tiny prompt change gives a different text; computing requests together (batching, for
throughput) can break it; versions must be frozen (an update can change the text); cloud APIs usually cannot guarantee it.

## Changelog

- 2026-10-21: kit created; origin PC results imported (temperature 0: 2108 runs; temperature 1: 108 runs).
- 2026-10-21: origin PC temperature 1 – 5000 + 5000 runs added (10 108 runs in total, all identical).
- 2026-10-21: end-to-end test of the kit on the origin PC (quick, 10 + 10, GPU 0): both reference hashes reproduced.
- 2026-10-21: origin PC results split into one result file per test session (8 rows in RESULTS.md, with the context of each session in the note); `run_test.py --gpu` records which GPU was used.
- 2026-10-22: Ollama 0.40.7 changed the text of both models (still repeatable; not caused by a parallel CPU load –
  clean rerun). Reference hashes are now stored per model and llama.cpp build; `qwen3:0.6b-q8_0` added as a second
  model profile and made the default (`run_test.py --profile`).
- 2026-10-22: thinking as an option (`--thinking`; answer and thinking hashed separately), Qwen references with
  thinking for 0.40.6 and 0.40.7; RESULTS.md shows where the model ran (`--device`: GPU / CPU / mix).
- 2026-10-22: `qwen3:0.6b-q4_K_M` profile (Q4_K_M = the default tag `qwen3:0.6b`); references for it (thinking off
  and on) and for gemma4:26b-a4b-it-q8_0 with thinking (Ollama 0.40.7).
- 2026-10-22: first machine outside the origin PC: `mac-m5pro-64gb` (Apple M5 Pro, Ollama 0.40.4),
  qwen3:0.6b-q4_K_M thinking off and on – repeatable, no reference for its build; with thinking at temperature 0 a
  repetition loop up to max_tokens, identical in 10/10 runs.
- 2026-10-25: the seed at temperature 0 does nothing – seeds 42, 7 and 1234 (fresh server for each) gave the same
  text; at temperature 1 every seed gave a different text.
- 2026-10-25: token 43 is not a near-tie (margin 0.17 nat): the update to 0.40.7 moved the numbers by tenths of
  a nat; the text splits where the change exceeds the margin. Batching numbers from the origin project removed;
  no expectations about other machines (after a review).

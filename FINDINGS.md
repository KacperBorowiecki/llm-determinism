# Findings

Facts come from result files in `results/`; hypotheses are marked as such.

## Results so far

### origin-pc-2x5090-win11 (reference machine, 19–21 Oct 2026)

Windows 11 Pro, AMD Ryzen 9 9950X, 61.6 GB RAM, 2× RTX 5090 (driver 610.88, CUDA 13.3), `llama-server` from Ollama
0.40.4 (`b1-b42b7e6d3`) and 0.40.6 (`b1-620d2225c`), one request at a time, prompt cache off.

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
- The restart and the silent Ollama update (new llama.cpp build) did not change either hash for this model.

## Curiosity: 10 requests at once (batching)

Same machine and model, one server with `-np 10 -c 20480`, prompt cache off, seed 42.

- **10 copies of the prompt, 10 rounds:** temperature 0 → 15 different texts, **0/100 equal to the single-request
  text**; temperature 1 → 20 different texts, 0/100.
- **10 different prompts at once, 10 rounds:** every prompt changed its answer – 0/100 at temperature 0 and 0/100 at
  temperature 1; some already at the 6th token.
- The single request on the same 10-slot server still gave the reference hash – the change comes from computing
  requests together, not from the configuration.
- At temperature 0 the texts split exactly at near-ties (token 117 "główn"/"ul", token 91 "nie"/"był").
  *Hypothesis:* computing a request together with others changes the numbers slightly (the model is a mixture of
  experts, so a small change can also change which experts are used); at a near-tie that flips the token.
- Rounds with the same batch layout gave the same set of texts, but which request got which text changed from round
  to round – it depends on the position in the batch, i.e. on arrival order. This is the mechanism behind
  non-deterministic cloud APIs at temperature 0: your request is computed together with other people's requests.
- Throughput: 10 stories at once took ~25 s (~2.5 s per story) vs ~8 s one at a time.

## Related observations outside this kit

From the origin project (Ollama's own API, not `llama-server` directly):
- With Ollama's prefix cache, the first call and later calls of the same prompt at temperature 0 gave different
  texts (e4b Q4_K_M).
- After the automatic Ollama update, one model (e4b Q4_K_M with speculative decoding, in Ollama) changed its
  first-call text; the 26B Q8_0 in `llama-server` did not. An update *can* change the text – not necessarily.

## What determinism is good for – and its limits

Useful wherever a result must be *reproduced* or a *change detected*: debugging (reproduce exactly the answer a
user got – if seed, version and environment are logged), a "canary" after updates (fixed prompts with known hashes),
regression tests of prompts and pipelines, fair A/B comparisons (same seed on both sides), audit, reproducible
synthetic data, safe caching and resuming of pipelines, red teaming, demos.

Limits: repeatable ≠ correct (a wrong answer is repeatably wrong); one seed is one sample, not the model's
variability; a tiny prompt change gives a different text; batching for throughput breaks it (here ~3× more text
per second); versions must be frozen; cloud APIs usually cannot guarantee it.

## Changelog

- 2026-10-21: kit created; origin PC results imported (temperature 0: 2108 runs; temperature 1: 108 runs).
- 2026-10-21: origin PC temperature 1 – 5000 + 5000 runs added (10 108 runs in total, all identical).
- 2026-10-21: end-to-end test of the kit on the origin PC (quick, 10 + 10, GPU 0): both reference hashes reproduced.

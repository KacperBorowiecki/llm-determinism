"""Determinism test for llama-server: the same prompt and seed, N runs at temperature 0 and N at temperature 1.

Talks to an already running llama-server (started from Ollama's installation, see the skill determinism-test). Prints a summary and
writes one result file: results/<label>/<start time>_<profile>[_thinking]_<length>_<id>.json
(':' in the profile name becomes '-'; <id> is a short random code, so tests started in the same minute never
overwrite each other). Model profiles and their reference
hashes (one set per llama.cpp build) are in reference.json.

Next to the text, every run hashes the numbers behind it (logprobs_hash): for every generated token the chosen token
and its logprob, the most likely candidate and the most likely other candidate with its logprob – equal hashes
mean the probabilities were identical bit for bit. With
--make-reference (origin PC only, on the user's request) a repeatable test without warnings becomes the reference
for its build, logprobs included. Every run records when it started and finished and the id the server gave
the response; with --save-raw every run is also kept locally (results/<label>/raw/, not in git): the server's
response byte for byte and the final answer as text.

  python run_test.py --label my-pc                       # default model, quick: 10 + 10
  python run_test.py --label my-pc --profile gemma4:26b-a4b-it-q8_0 --length standard     # 100 + 100
  python run_test.py --label my-pc --thinking            # with the model's thinking (hashed separately)
  python run_test.py --label my-pc --runs 3 --note "-ngl 30 (partial offload)"
  python run_test.py --label my-pc --runs 3 --seed none   # demo: no seed – temperature 0 still repeats, 1 does not
  python run_test.py --label origin-pc --runs 3 --make-reference   # origin PC: this test becomes the reference
"""
import argparse
import collections
import hashlib
import json
import platform
import re
import secrets
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LENGTHS = {"quick": 10, "standard": 100, "long": 1000}


def text_hash(text: str) -> tuple[str, str]:
    """The only hash function of the kit (README: "How the hash is computed"): the exact text, UTF-8, SHA-256.

    Returns (short hash = first 12 hex characters, full hex digest)."""
    full = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return full[:12], full


def logprobs_tokens(content: list[dict]) -> list[dict]:
    """The probabilities behind a generation, from the server's `choices[0].logprobs.content` (one entry per
    generated token; with thinking it covers the thinking and the answer), in the format of the reference files:
    the chosen token and its logprob, the most likely candidate (`top1`) and the most likely candidate other than
    the chosen token (`alt`). At temperature 1 the chosen token may be neither of the two candidates."""
    tokens = []
    for e in content:
        top = e.get("top_logprobs") or []
        alt = next((t for t in top if t.get("token") != e.get("token")), None)
        tokens.append({"token": e.get("token"), "logprob": e.get("logprob"),
                       "top1": top[0].get("token") if top else None,
                       "alt": alt.get("token") if alt else None, "alt_logprob": alt.get("logprob") if alt else None})
    return tokens


def logprobs_hash(tokens: list[dict]) -> str | None:
    """The hash of the probabilities, computed like the text hash: text_hash of the token list as compact JSON.
    Equal hashes = every logprob identical bit for bit. None if the server sent no logprobs."""
    return text_hash(json.dumps(tokens, ensure_ascii=False, separators=(",", ":")))[0] if tokens else None


def dump_reference_text(data: dict) -> str:
    """A reference text file: indent=1 as before, but `tokens` last and one token per line (~1 000 tokens)."""
    text = json.dumps({k: v for k, v in data.items() if k != "tokens"}, ensure_ascii=False, indent=1)
    if "tokens" not in data:
        return text + "\n"
    lines = ",\n".join("  " + json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in data["tokens"])
    return text[:-2] + ",\n \"tokens\": [\n" + lines + "\n ]\n}\n"


def load_reference() -> dict:
    return json.loads((ROOT / "reference.json").read_text(encoding="utf-8"))


def get_profile(reference: dict, name: str | None = None) -> tuple[str, dict]:
    """Model profile from reference.json (default: reference["default_profile"])."""
    name = name or reference["default_profile"]
    return name, reference["profiles"][name]


def pick_reference(profile: dict, build: str | None, thinking: bool = False) -> tuple[dict, bool]:
    """Reference hashes for this llama.cpp build and thinking mode. Unknown build -> the newest reference of the
    same thinking mode (exact=False); none at all -> an empty reference."""
    entries = [e for e in profile["references"] if e.get("thinking", False) == thinking]
    for entry in entries:
        if build in entry["llama_cpp"]:
            return entry, True
    return (entries[-1] if entries else {"llama_cpp": [], "thinking": thinking, "hashes": {}}), False


def reference_text(short_hash: str | None) -> str | None:
    path = ROOT / "reference" / "texts" / f"{short_hash}.json"
    return json.loads(path.read_text(encoding="utf-8"))["text"] if short_hash and path.exists() else None


def divergence(reference: str, text: str, context: int = 40) -> dict:
    """First character where `text` differs from `reference`, with a short excerpt of both."""
    # plain zip on purpose: the texts may differ in length
    i = next((i for i, (a, b) in enumerate(zip(reference, text)) if a != b), min(len(reference), len(text)))
    return {"char": i, "reference": reference[max(0, i - context):i + context],
            "this": text[max(0, i - context):i + context]}


def summarize(results: list[dict], reference: dict) -> dict:
    """Per-temperature statistics. `results`: outputs of ask() with an added 'temperature' key. With thinking, the
    thinking text is hashed and compared separately; repeatable / equal to the reference only if both match."""
    out = {}
    for temp in sorted({r["temperature"] for r in results}):
        key = f"{temp:g}"
        rs = [r for r in results if r["temperature"] == temp]
        counts = collections.Counter(r["hash"] for r in rs)
        main_hash = counts.most_common(1)[0][0]
        ref_hash = reference["hashes"].get(key)
        speeds = [r["tok_per_s"] for r in rs if r.get("tok_per_s")]
        stats = {"runs": len(rs), "hashes": dict(counts.most_common()), "repeatable": len(counts) == 1,
                 "reference_hash": ref_hash, "equals_reference": set(counts) == {ref_hash},
                 "answer_tokens": sorted({r["answer_tokens"] for r in rs}),
                 "prompt_tokens": sorted({r["prompt_tokens"] for r in rs}),
                 "tok_per_s_median": round(statistics.median(speeds), 1) if speeds else None,
                 "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in rs if r.get("finish_reason"))),
                 "divergence_vs_reference": None}
        if any(r.get("thinking_hash") is not None for r in rs):
            thinking_counts = collections.Counter(r["thinking_hash"] for r in rs)
            thinking_ref = reference.get("thinking_hashes", {}).get(key)
            stats["thinking_hashes"] = dict(thinking_counts.most_common())
            stats["thinking_reference_hash"] = thinking_ref
            stats["repeatable"] = stats["repeatable"] and len(thinking_counts) == 1
            stats["equals_reference"] = stats["equals_reference"] and set(thinking_counts) == {thinking_ref}
            # where the thinking split from the reference; the answer's split is computed separately below
            thinking_main = thinking_counts.most_common(1)[0][0]
            thinking_ref_text = reference_text(thinking_ref)
            stats["thinking_divergence_vs_reference"] = None
            if thinking_main != thinking_ref and thinking_ref_text is not None:
                thought = next(r["thinking_text"] for r in rs if r["thinking_hash"] == thinking_main)
                stats["thinking_divergence_vs_reference"] = divergence(thinking_ref_text, thought)
        # logprobs: like the text; None = some run has no logprobs (missing data, not "the numbers differ")
        lp_counts = collections.Counter(r.get("logprobs_hash") for r in rs)
        lp_ref = reference.get("logprobs_hashes", {}).get(key)
        missing = None in lp_counts
        stats["logprobs_hashes"] = {h: c for h, c in lp_counts.most_common() if h is not None}
        stats["logprobs_repeatable"] = None if missing else len(lp_counts) == 1
        stats["logprobs_reference_hash"] = lp_ref
        stats["logprobs_equals_reference"] = None if missing or lp_ref is None else set(lp_counts) == {lp_ref}
        ref_text = reference_text(ref_hash)
        if main_hash != ref_hash and ref_text is not None:
            main_text = next(r["text"] for r in rs if r["hash"] == main_hash)
            stats["divergence_vs_reference"] = divergence(ref_text, main_text)
        out[key] = stats
    return out


def warnings_for(machine: dict, summary: dict, reference: dict, build_after: str | None) -> list[str]:
    """Things that change the hash by themselves – check them before blaming the hardware."""
    warn = []
    server = machine.get("server", {})
    if server.get("model") != reference["model_file"]:
        warn.append(f"model file {server.get('model')} differs from the reference {reference['model_file']}")
    for name, expected in reference["sampling"].items():
        got = server.get("sampling", {}).get(name)
        if not isinstance(got, (int, float)) or abs(got - expected) > 1e-6:
            warn.append(f"sampling {name} = {got}, reference {expected}")
    for key, s in summary.items():
        if s["prompt_tokens"] != [reference["prompt_tokens"]]:
            warn.append(f"temperature {key}: prompt tokens {s['prompt_tokens']}, reference {reference['prompt_tokens']}")
    for key, s in summary.items():
        if s.get("finish_reasons", {}).get("length"):
            warn.append(f"temperature {key}: {s['finish_reasons']['length']} generation(s) hit max_tokens "
                        f"(text cut off)")
    for key, s in summary.items():
        if "logprobs_repeatable" in s and s["logprobs_repeatable"] is None:
            warn.append(f"temperature {key}: the server returned no logprobs for some runs (numbers not checked)")
    if build_after and build_after != server.get("build_info"):
        warn.append(f"llama-server build changed during the test: {server.get('build_info')} -> {build_after}")
    return warn


def ask(url: str, protocol: dict, temperature: float, thinking: bool = False, timeout_s: float = 600) -> dict:
    """One generation, exactly as in the reference experiments – do not change the request body.
    With thinking the server returns the thinking separately (`reasoning_content`); it gets its own hash."""
    body = {"messages": [{"role": "user", "content": protocol["prompt"]}], "temperature": temperature,
            "seed": protocol["seed"],
            "max_tokens": protocol["max_tokens_thinking"] if thinking else protocol["max_tokens"],
            "logprobs": True, "top_logprobs": 2, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if protocol["seed"] is None:   # --seed none: the server draws a new seed for every request
        del body["seed"]
    started, t0 = datetime.now().astimezone(), time.time()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=(10, timeout_s))
    r.raise_for_status()
    finished = datetime.now().astimezone()
    d = r.json()
    message = d["choices"][0]["message"]
    text = message.get("content") or ""
    short, full = text_hash(text)
    tokens = logprobs_tokens((d["choices"][0].get("logprobs") or {}).get("content") or [])
    out = {"hash": short, "sha256": full, "text": text, "answer_tokens": d["usage"]["completion_tokens"],
           "prompt_tokens": d["usage"]["prompt_tokens"], "finish_reason": d["choices"][0].get("finish_reason"),
           "tok_per_s": d.get("timings", {}).get("predicted_per_second"), "seconds": round(time.time() - t0, 2),
           "thinking_hash": None, "logprobs_hash": logprobs_hash(tokens), "logprobs": tokens,
           "started": started.isoformat(timespec="milliseconds"),
           "finished": finished.isoformat(timespec="milliseconds"),
           "response_id": d.get("id"), "created": d.get("created"),
           "raw": r.content}   # the response bytes exactly as they came; only for --save-raw
    if thinking:
        thought = message.get("reasoning_content") or ""
        out["thinking_text"] = thought
        out["thinking_hash"], out["thinking_sha256"] = text_hash(thought)
    return out


def write_result(base: Path, record: dict, suffix: str | None = None) -> Path:
    """Write `record` to `<base>_<id>.json` and return the path. <id> is a short random code (not a hash of the
    content – result files are edited later, e.g. when a result becomes a reference): two tests started in the
    same minute (two GPUs, GPU and CPU, ...) get different files, and mode "x" never overwrites one. `suffix`:
    the id chosen at the start (--save-raw names its folder with it), taken if the name is free."""
    base.parent.mkdir(parents=True, exist_ok=True)
    while True:
        out = base.with_name(f"{base.name}_{suffix or secrets.token_hex(3)}.json")
        suffix = None   # taken: the next try gets a new random id
        try:
            with out.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(record, f, ensure_ascii=False, indent=1)
            return out
        except FileExistsError:
            continue


def save_raw_run(raw_dir: Path, r: dict, raw: bytes, width: int) -> None:
    """--save-raw: one run as two files in raw_dir – t<temperature>_run<N>.json, the server's response byte for
    byte as it came (tokens and logprobs included), and the same name .txt, the final answer (with thinking: the
    thinking first) to read and compare. N has `width` digits, so the files sort in run order."""
    name = f"t{r['temperature']:g}_run{r['run']:0{width}d}"
    (raw_dir / f"{name}.json").write_bytes(raw)
    head = [f"temperature {r['temperature']:g}, run {r['run']}",
            f"started  {r['started']}   finished {r['finished']}",
            f"response {r['response_id']}   created {r['created']}",
            f"{r['answer_tokens']} tokens, {r['finish_reason']}   text {r['hash']}   logprobs {r['logprobs_hash']}"
            + (f"   thinking {r['thinking_hash']}" if "thinking_text" in r else "")]
    body = (["=== thinking ===", r["thinking_text"], ""] if "thinking_text" in r else []) + ["=== answer ===", r["text"]]
    (raw_dir / f"{name}.txt").write_text("\n".join(head + [""] + body) + "\n", encoding="utf-8", newline="\n")


def reference_blockers(summary: dict, problems: list[str], device: str) -> list[str]:
    """Why a test may not become a reference (--make-reference): warnings other than "no reference for this
    build", text or logprobs not repeatable (or missing), the model not entirely on the GPU. Empty = it may."""
    blockers = list(problems) + [f"temperature {k}: text or logprobs not repeatable" for k, st in summary.items()
                                 if not (st["repeatable"] and st["logprobs_repeatable"])]
    if device != "gpu":
        blockers.append(f"--device {device}: references come from runs with all layers on the GPU")
    return blockers


def save_reference(root: Path, reference: dict, profile_name: str, machine: dict, thinking: bool, summary: dict,
                   results: list[dict], sequences: dict, label: str, started: datetime) -> str:
    """--make-reference: store a repeatable test (text and logprobs, no warnings – checked by the caller) as the
    reference for its model profile, llama.cpp build and thinking mode, logprobs included; writes reference.json
    and reference/texts/. Returns what was done. ValueError if this build already has a reference with other
    hashes – nothing is overwritten."""
    build, profile = machine["server"]["build_info"], reference["profiles"][profile_name]
    hashes = {k: next(iter(st["hashes"])) for k, st in summary.items()}
    thinking_hashes = {k: next(iter(st["thinking_hashes"])) for k, st in summary.items()} if thinking else None
    logprobs_hashes = {k: next(iter(st["logprobs_hashes"])) for k, st in summary.items()}
    entry, exact = pick_reference(profile, build, thinking)
    if exact:
        if entry["hashes"] != hashes or (thinking and entry.get("thinking_hashes") != thinking_hashes):
            raise ValueError(f"build {build} already has a reference with other texts – not overwritten")
        if entry.get("logprobs_hashes"):
            if entry["logprobs_hashes"] != logprobs_hashes:
                raise ValueError(f"build {build} already has a reference with other logprobs – not overwritten")
            return "Already the reference for this build."
        entry["logprobs_hashes"] = logprobs_hashes
        done = "Reference logprobs added for this build."
    else:
        prompt_tokens = {t for st in summary.values() for t in st["prompt_tokens"]}
        if len(prompt_tokens) != 1:
            raise ValueError(f"prompt tokens differ between the temperatures: {sorted(prompt_tokens)}")
        entry = {"thinking": thinking, "llama_cpp": [build], "ollama": [machine.get("ollama")],
                 "prompt_tokens": prompt_tokens.pop(), "hashes": hashes}
        if thinking:
            entry["thinking_hashes"] = thinking_hashes
        entry["answer_tokens"] = {k: st["answer_tokens"][0] for k, st in summary.items()}
        entry["logprobs_hashes"] = logprobs_hashes   # last, as when added to an existing reference
        profile["references"].append(entry)
        done = "Became the reference for this build."
    protocol = reference["protocol"]
    source = (f"{profile_name}, thinking {'on' if thinking else 'off'}, Ollama {machine.get('ollama')} ({build}); "
              f"{label}, {started:%Y-%m-%dT%H%M}")
    for key in summary:
        files = [("answer", hashes[key], next(r["text"] for r in results if r["hash"] == hashes[key]))]
        if thinking:
            files.append(("thinking", thinking_hashes[key],
                          next(r["thinking_text"] for r in results if r["thinking_hash"] == thinking_hashes[key])))
        for kind, short, text in files:
            path = root / "reference" / "texts" / f"{short}.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "hash": short, "sha256": text_hash(text)[1], "kind": kind, "temperature": int(float(key)),
                "seed": protocol["seed"], "prompt": protocol["prompt"], "thinking": thinking, "source": source,
                "text": text}
            if kind == "answer" and "tokens" not in data:   # tokens already there: the same text in another build
                data["tokens_source"] = source
                data["tokens_cover"] = "thinking + answer" if thinking else "answer"
                data["tokens"] = sequences[logprobs_hashes[key]]
            elif path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump_reference_text(data), encoding="utf-8", newline="\n")
    (root / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    return done


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    if sys.platform == "darwin":
        return run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_ext")]
        m = MemoryStatus()
        m.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.total / 2**30, 1)
    if sys.platform == "darwin":
        out = run_cmd(["sysctl", "-n", "hw.memsize"])
        return round(int(out) / 2**30, 1) if out.isdigit() else None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 2**20, 1)
    return None


def gpus() -> list[dict]:
    out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if out:
        cuda = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", run_cmd(["nvidia-smi"]))
        return [{"name": name.strip(), "memory": memory.strip(), "driver": driver.strip(),
                 "cuda": cuda.group(1) if cuda else None}
                for name, memory, driver in (line.split(",") for line in out.splitlines())]
    if sys.platform == "darwin":
        return [{"name": cpu_name(), "memory": "unified"}]
    return []


def machine_info(url: str) -> dict:
    """Hardware and software description. No hostname, no user name, no paths."""
    version = re.search(r"\d+\.\d+\.\d+", run_cmd(["ollama", "--version"]))
    props = requests.get(f"{url}/props", timeout=10).json()
    settings = props.get("default_generation_settings", {})
    params = settings.get("params", {})
    return {"os": f"{platform.system()} {platform.release()} ({platform.version()})", "arch": platform.machine(),
            "cpu": cpu_name(), "ram_gb": ram_gb(), "gpus": gpus(), "ollama": version.group(0) if version else None,
            "python": platform.python_version(),
            "server": {"build_info": props.get("build_info"),
                       "model": re.split(r"[\\/]", props.get("model_path", ""))[-1],
                       "n_ctx": settings.get("n_ctx"), "slots": props.get("total_slots"),
                       "sampling": {k: params.get(k) for k in ("top_k", "top_p", "min_p", "samplers")}}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="machine label – Claude always asks the user for it")
    ap.add_argument("--profile", help="model profile from reference.json (default: its default_profile)")
    ap.add_argument("--thinking", action="store_true", help="enable the model's thinking (default: off)")
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="llama-server address (default %(default)s)")
    ap.add_argument("--length", choices=LENGTHS, default="quick", help="quick 10, standard 100, long 1000 runs")
    ap.add_argument("--runs", type=int, help="runs per temperature (overrides --length)")
    ap.add_argument("--gpu", type=int, help="index of the GPU the server runs on (as in CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--device", choices=["gpu", "cpu", "mix"], default="gpu",
                    help="where the model runs: gpu (all layers, -ngl 999), cpu (-ngl 0), mix (partial offload)")
    ap.add_argument("--note", default="", help="free text, e.g. the GPU/RAM split or anything unusual")
    ap.add_argument("--seed", help="a demo of what the seed does, instead of the protocol's seed: a whole number, or "
                                   "'none' (the server draws a new one for every request); never a reference")
    ap.add_argument("--save-raw", action="store_true",
                    help="keep every run locally in results/<label>/raw/ (not in git): the server's response "
                         "byte for byte and the final answer as text")
    ap.add_argument("--make-reference", action="store_true",
                    help="origin PC only, on the user's request: a repeatable test without warnings becomes the "
                         "reference for its build (texts, logprobs)")
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.label):
        ap.error("label: lowercase letters, digits, '.', '_' and '-' only")
    if args.seed is not None and args.seed != "none" and not re.fullmatch(r"\d+", args.seed):
        ap.error("seed: a whole number or 'none'")

    reference = load_reference()
    if args.profile and args.profile not in reference["profiles"]:
        ap.error(f"profile: one of {', '.join(reference['profiles'])}")
    protocol = reference["protocol"]
    protocol_seed = protocol["seed"]
    if args.seed is not None:   # --seed: a demo; the references use the protocol's seed
        protocol = {**protocol, "seed": None if args.seed == "none" else int(args.seed)}
    seed_text = "none" if protocol["seed"] is None else str(protocol["seed"])
    profile_name, profile = get_profile(reference, args.profile)
    n = args.runs or LENGTHS[args.length]
    length = f"custom{n}" if args.runs else args.length
    machine = machine_info(args.url)
    build = machine["server"]["build_info"]
    ref, exact = pick_reference(profile, build, args.thinking)
    started = datetime.now().astimezone()
    known = ", ".join(ref["llama_cpp"]) or "none"
    print(f"{args.label}: {profile_name}, thinking {'on' if args.thinking else 'off'}, seed {seed_text}, "
          f"llama-server {build}, "
          f"{n} runs per temperature; reference: {'this build' if exact else 'none for this build (known: ' + known + ')'}")

    mode = "_thinking" if args.thinking else ""
    seed_tag = "" if args.seed is None else ("_noseed" if protocol["seed"] is None else f"_seed{protocol['seed']}")
    base_name = f"{started:%Y-%m-%dT%H%M}_{profile_name.replace(':', '-')}{mode}{seed_tag}_{length}"
    run_id, raw_dir, raw_error, width = secrets.token_hex(3), None, None, len(str(n))
    if args.save_raw:   # the folder gets the result file's name; created exclusively, like the result file
        (ROOT / "results" / args.label / "raw").mkdir(parents=True, exist_ok=True)
        while True:
            raw_dir = ROOT / "results" / args.label / "raw" / f"{base_name}_{run_id}"
            try:
                raw_dir.mkdir()
                break
            except FileExistsError:
                run_id = secrets.token_hex(3)
        print(f"raw runs: {raw_dir.relative_to(ROOT).as_posix()}")

    results, sequences = [], {}   # sequences (--make-reference only): one logprobs token list per unique hash
    for temp in protocol["temperatures"]:
        ref_hash = ref["hashes"].get(f"{temp:g}")
        thinking_ref = ref.get("thinking_hashes", {}).get(f"{temp:g}")
        for i in range(1, n + 1):
            r = ask(args.url, protocol, float(temp), args.thinking)
            raw = r.pop("raw")             # never kept in memory; written at once with --save-raw
            logprobs = r.pop("logprobs")   # the hash stays in r; the numbers only for --make-reference
            if args.make_reference:
                sequences.setdefault(r["logprobs_hash"], logprobs)
            r.update(temperature=float(temp), run=i, time=datetime.now().astimezone().isoformat(timespec="seconds"))
            results.append(r)
            if raw_dir and not raw_error:
                try:
                    save_raw_run(raw_dir, r, raw, width)
                except OSError as e:   # e.g. a full disk: the test goes on without the raw files
                    raw_error = f"--save-raw stopped at temperature {temp:g}, run {i}: {e}"
            same = r["hash"] == ref_hash and (not args.thinking or r["thinking_hash"] == thinking_ref)
            thought = f" + thinking {r['thinking_hash']}" if args.thinking else ""
            print(f"  temp {temp:g}  run {i:>4}/{n}: {r['hash']}{thought} {'=' if same else '≠'} reference  "
                  f"logprobs {r['logprobs_hash']}  "
                  f"{r['answer_tokens']} tok  {r['seconds']} s", flush=True)
    build_after = requests.get(f"{args.url}/props", timeout=10).json().get("build_info")

    def evaluate(ref: dict, exact: bool) -> tuple[dict, list[str], list[str]]:
        """Summary, all warnings, and the warnings other than \"no reference for this build\"."""
        summary = summarize(results, ref)
        # the reference entry may carry its own prompt token count (e.g. thinking on renders a shorter prompt)
        expected = {**profile, **({"prompt_tokens": ref["prompt_tokens"]} if "prompt_tokens" in ref else {})}
        problems = warnings_for(machine, summary, expected, build_after)
        warnings = list(problems)
        if not exact:
            warnings.append(f"no reference hashes for llama.cpp build {build} ({profile_name}, thinking "
                            f"{'on' if args.thinking else 'off'}); known: {', '.join(ref['llama_cpp']) or 'none'}")
        if args.seed is not None:
            warnings.append(f"seed {seed_text} instead of the protocol's {protocol_seed}: at temperature 0 the "
                            f"reference still applies, at temperature 1 a different text is expected")
        return summary, warnings, problems

    summary, warnings, problems = evaluate(ref, exact)
    note, not_saved, done = args.note, [], ""
    if args.make_reference:
        not_saved = reference_blockers(summary, problems, args.device)
        if args.seed is not None:
            not_saved.append(f"--seed {seed_text}: references use the protocol's seed")
        if not not_saved:
            try:
                reference = load_reference()   # fresh from disk: another test may have saved a reference meanwhile
                profile = reference["profiles"][profile_name]
                done = save_reference(ROOT, reference, profile_name, machine, args.thinking, summary, results,
                                      sequences, args.label, started)
                ref, exact = pick_reference(profile, build, args.thinking)   # compare with it from now on
                summary, warnings, problems = evaluate(ref, exact)
                note = f"{note} {done}".strip()
            except (ValueError, OSError) as e:   # the result is written anyway
                not_saved = [str(e)]
        if not_saved:
            note = f"{note} Not saved as reference: {'; '.join(not_saved)}.".strip()
    record = {"schema": 1, "label": args.label, "profile": profile_name, "thinking": args.thinking,
              "seed": protocol["seed"],
              "reference_llama_cpp": ref["llama_cpp"],
              "reference_exact": exact, "started": started.isoformat(timespec="seconds"),
              "finished": datetime.now().astimezone().isoformat(timespec="seconds"), "length": length,
              "runs_per_temperature": n, "device": args.device, "gpu": args.gpu, "note": note,
              # where the raw runs are on this machine, relative to the result file (not in git)
              **({"raw_dir": raw_dir.relative_to(ROOT / "results" / args.label).as_posix()} if raw_dir else {}),
              "machine": machine, "temperatures": summary,
              "warnings": warnings,
              "texts": {r["hash"]: r["text"] for r in results},
              "runs": [{k: r[k] for k in ("temperature", "run", "time", "started", "finished", "response_id", "hash",
                                          "thinking_hash", "logprobs_hash", "answer_tokens", "finish_reason",
                                          "seconds")} for r in results]}
    if args.thinking:
        record["thinking_texts"] = {r["thinking_hash"]: r["thinking_text"] for r in results}
    if raw_error:
        record["warnings"].append(raw_error)
    out = write_result(ROOT / "results" / args.label / base_name, record, run_id)

    print()
    for key, s in summary.items():
        print(f"temperature {key}: {len(s['hashes'])} unique of {s['runs']} – repeatable: {s['repeatable']}, "
              f"logprobs repeatable: {s['logprobs_repeatable']}, equals reference: {s['equals_reference']}, "
              f"logprobs equal reference: {s['logprobs_equals_reference']}  {s['hashes']}"
              + (f"  thinking {s['thinking_hashes']}" if "thinking_hashes" in s else ""))
    for w in record["warnings"]:
        print(f"WARNING: {w}")
    print(f"Result: {out.relative_to(ROOT)}")
    if raw_dir:
        print(f"Raw runs: {raw_dir.relative_to(ROOT).as_posix()}")
    if args.make_reference:
        if not_saved:
            sys.exit("NOT SAVED AS REFERENCE: " + "; ".join(not_saved))
        print(f"Reference: {done}")


if __name__ == "__main__":
    main()

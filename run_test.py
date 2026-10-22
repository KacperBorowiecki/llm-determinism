"""Determinism test for llama-server: the same prompt and seed, N runs at temperature 0 and N at temperature 1.

Talks to an already running llama-server (started from Ollama's installation, see README). Prints a summary and
writes one result file: results/<label>/<start time>_<profile>_<length>.json. Model profiles and their reference
hashes (one set per llama.cpp build) are in reference.json.

  python run_test.py --label my-pc                       # default model, quick: 10 + 10
  python run_test.py --label my-pc --profile gemma4-26b-q8 --length standard     # 100 + 100
  python run_test.py --label my-pc --thinking            # with the model's thinking (hashed separately)
  python run_test.py --label my-pc --runs 3 --note "-ngl 30 (partial offload)"
"""
import argparse
import collections
import hashlib
import json
import platform
import re
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
    t0 = time.time()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=(10, timeout_s))
    r.raise_for_status()
    d = r.json()
    message = d["choices"][0]["message"]
    text = message.get("content") or ""
    short, full = text_hash(text)
    out = {"hash": short, "sha256": full, "text": text, "answer_tokens": d["usage"]["completion_tokens"],
           "prompt_tokens": d["usage"]["prompt_tokens"], "finish_reason": d["choices"][0].get("finish_reason"),
           "tok_per_s": d.get("timings", {}).get("predicted_per_second"), "seconds": round(time.time() - t0, 2),
           "thinking_hash": None}
    if thinking:
        thought = message.get("reasoning_content") or ""
        out["thinking_text"] = thought
        out["thinking_hash"], out["thinking_sha256"] = text_hash(thought)
    return out


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
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.label):
        ap.error("label: lowercase letters, digits, '.', '_' and '-' only")

    reference = load_reference()
    if args.profile and args.profile not in reference["profiles"]:
        ap.error(f"profile: one of {', '.join(reference['profiles'])}")
    protocol = reference["protocol"]
    profile_name, profile = get_profile(reference, args.profile)
    n = args.runs or LENGTHS[args.length]
    length = f"custom{n}" if args.runs else args.length
    machine = machine_info(args.url)
    build = machine["server"]["build_info"]
    ref, exact = pick_reference(profile, build, args.thinking)
    started = datetime.now().astimezone()
    known = ", ".join(ref["llama_cpp"]) or "none"
    print(f"{args.label}: {profile_name}, thinking {'on' if args.thinking else 'off'}, llama-server {build}, "
          f"{n} runs per temperature; reference: {'this build' if exact else 'none for this build (known: ' + known + ')'}")

    results = []
    for temp in protocol["temperatures"]:
        ref_hash = ref["hashes"].get(f"{temp:g}")
        thinking_ref = ref.get("thinking_hashes", {}).get(f"{temp:g}")
        for i in range(1, n + 1):
            r = ask(args.url, protocol, float(temp), args.thinking)
            r.update(temperature=float(temp), run=i, time=datetime.now().astimezone().isoformat(timespec="seconds"))
            results.append(r)
            same = r["hash"] == ref_hash and (not args.thinking or r["thinking_hash"] == thinking_ref)
            thought = f" + thinking {r['thinking_hash']}" if args.thinking else ""
            print(f"  temp {temp:g}  run {i:>4}/{n}: {r['hash']}{thought} {'=' if same else '≠'} reference  "
                  f"{r['answer_tokens']} tok  {r['seconds']} s", flush=True)
    build_after = requests.get(f"{args.url}/props", timeout=10).json().get("build_info")

    summary = summarize(results, ref)
    # the reference entry may carry its own prompt token count (e.g. thinking on renders a shorter prompt)
    expected = {**profile, **({"prompt_tokens": ref["prompt_tokens"]} if "prompt_tokens" in ref else {})}
    warnings = warnings_for(machine, summary, expected, build_after)
    if not exact:
        warnings.append(f"no reference hashes for llama.cpp build {build} ({profile_name}, thinking "
                        f"{'on' if args.thinking else 'off'}); known: {known}")
    record = {"schema": 1, "label": args.label, "profile": profile_name, "thinking": args.thinking,
              "reference_llama_cpp": ref["llama_cpp"],
              "reference_exact": exact, "started": started.isoformat(timespec="seconds"),
              "finished": datetime.now().astimezone().isoformat(timespec="seconds"), "length": length,
              "runs_per_temperature": n, "device": args.device, "gpu": args.gpu, "note": args.note, "machine": machine, "temperatures": summary,
              "warnings": warnings,
              "texts": {r["hash"]: r["text"] for r in results},
              "runs": [{k: r[k] for k in ("temperature", "run", "time", "hash", "thinking_hash", "answer_tokens",
                                          "finish_reason", "seconds")} for r in results]}
    if args.thinking:
        record["thinking_texts"] = {r["thinking_hash"]: r["thinking_text"] for r in results}
    mode = "_thinking" if args.thinking else ""
    out = ROOT / "results" / args.label / f"{started:%Y-%m-%dT%H%M}_{profile_name}{mode}_{length}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)

    print()
    for key, s in summary.items():
        print(f"temperature {key}: {len(s['hashes'])} unique of {s['runs']} – repeatable: {s['repeatable']}, "
              f"equals reference: {s['equals_reference']}  {s['hashes']}"
              + (f"  thinking {s['thinking_hashes']}" if "thinking_hashes" in s else ""))
    for w in record["warnings"]:
        print(f"WARNING: {w}")
    print(f"Result: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

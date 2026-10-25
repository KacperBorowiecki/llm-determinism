"""Optional curiosity: 10 requests at the same time to one llama-server started with -np 10 -c 20480.

Checks whether computing requests together in one batch changes the text: compares every simultaneous answer with
a single request on the same server – the text and the logprobs (as run_test.py hashes them). Every unique text is
saved once, so you can see where the answers split. Writes results/<label>/parallel/<start>_t<temperature>_<id>.json
(with the machine and server description, as run_test.py).

  python parallel_test.py --label my-pc --rounds 3 --temperature 0
"""
import argparse
import collections
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

from run_test import ROOT, ask, load_reference, machine_info, write_result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    protocol = load_reference()["protocol"]
    slots = requests.get(f"{args.url}/props", timeout=10).json().get("total_slots", 1)
    if slots < args.parallel:
        sys.exit(f"The server has {slots} slot(s); start it with -np {args.parallel} -c 20480 for this test.")
    machine = machine_info(args.url)
    started = datetime.now().astimezone()
    solo = ask(args.url, protocol, args.temperature)
    texts = {solo["hash"]: solo["text"]}   # every unique text once
    print(f"single request: {solo['hash']}  logprobs {solo['logprobs_hash']}")
    rounds = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for rnd in range(1, args.rounds + 1):
            answers = list(pool.map(lambda _: ask(args.url, protocol, args.temperature), range(args.parallel)))
            texts.update((a["hash"], a["text"]) for a in answers)
            counts = collections.Counter(a["hash"] for a in answers)
            lp_counts = collections.Counter(a["logprobs_hash"] for a in answers)
            rounds.append({"round": rnd, "hashes": dict(counts.most_common()),
                           "logprobs_hashes": dict(lp_counts.most_common())})
            print(f"round {rnd}: equal to the single request – text {counts.get(solo['hash'], 0)}/{args.parallel}, "
                  f"logprobs {lp_counts.get(solo['logprobs_hash'], 0)}/{args.parallel}  {dict(counts.most_common())}",
                  flush=True)
    equal = sum(r["hashes"].get(solo["hash"], 0) for r in rounds)
    lp_equal = sum(r["logprobs_hashes"].get(solo["logprobs_hash"], 0) for r in rounds)
    record = {"schema": 1, "kind": "parallel", "label": args.label, "started": started.isoformat(timespec="seconds"),
              "temperature": args.temperature, "parallel": args.parallel, "single_request_hash": solo["hash"],
              "single_request_logprobs_hash": solo["logprobs_hash"], "rounds": rounds, "equal_to_single": equal,
              "logprobs_equal_to_single": lp_equal, "total": args.rounds * args.parallel, "machine": machine,
              "texts": texts}
    out = write_result(ROOT / "results" / args.label / "parallel" /
                       f"{started:%Y-%m-%dT%H%M}_t{args.temperature:g}", record)
    print(f"\nequal to the single request: text {equal}/{record['total']}, logprobs {lp_equal}/{record['total']}\n"
          f"Result: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

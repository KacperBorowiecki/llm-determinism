import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import results_table
from run_test import (divergence, dump_reference_text, get_profile, load_reference, logprobs_hash,
                      logprobs_tokens, pick_reference, reference_blockers, save_raw_run, save_reference, summarize,
                      text_hash, warnings_for, write_result)

ROOT = Path(__file__).resolve().parents[1]


def all_references():
    reference = load_reference()
    for name, profile in reference["profiles"].items():
        for entry in profile["references"]:
            yield name, entry


def test_reference_texts_rehash_to_reference_hashes():
    for name, entry in all_references():
        for key, short in entry["hashes"].items():
            data = json.loads((ROOT / "reference" / "texts" / f"{short}.json").read_text(encoding="utf-8"))
            assert text_hash(data["text"]) == (short, data["sha256"]), f"{name} temperature {key}"
            assert "\r" not in data["text"]


def test_reference_logprobs_rehash_to_reference_hashes():
    for name, entry in all_references():
        for key, lp in entry.get("logprobs_hashes", {}).items():
            path = ROOT / "reference" / "texts" / f"{entry['hashes'][key]}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert logprobs_hash(data["tokens"]) == lp, f"{name} temperature {key}"
            assert len(data["tokens"]) == entry["answer_tokens"][key] and list(data)[-1] == "tokens"


def test_text_hash_uses_exact_bytes():
    assert text_hash("a\n")[0] != text_hash("a\r\n")[0]
    assert text_hash("a")[0] != text_hash("a ")[0]
    short, full = text_hash("x")
    assert len(short) == 12 and len(full) == 64 and full.startswith(short)


def test_profiles_and_reference_per_build():
    reference = load_reference()
    name, profile = get_profile(reference)
    assert name == reference["default_profile"] and profile["references"]
    _, gemma = get_profile(reference, "gemma4:26b-a4b-it-q8_0")
    entry, exact = pick_reference(gemma, "b1-620d2225c")
    assert exact and entry["hashes"]["0"] == "138e66a98238"
    entry, exact = pick_reference(gemma, "b1-unknown")
    newest_off = [e for e in gemma["references"] if not e.get("thinking", False)][-1]
    assert not exact and entry is newest_off   # unknown build -> newest reference of the same thinking mode


def fake(temp, text, prompt_tokens=29):
    short, full = text_hash(text)
    return {"temperature": temp, "hash": short, "sha256": full, "text": text, "answer_tokens": len(text),
            "prompt_tokens": prompt_tokens, "tok_per_s": 170.0}


REFERENCE = {"hashes": {"0": text_hash("same story")[0], "1": "000000000000"}, "prompt_tokens": 29,
             "model_file": "sha256-abc", "sampling": {"top_k": 64, "top_p": 0.95, "min_p": 0.05}}


def test_summarize_repeatable_and_equal_to_reference():
    s = summarize([fake(0.0, "same story")] * 3, REFERENCE)["0"]
    assert s["repeatable"] and s["equals_reference"] and s["runs"] == 3
    assert s["divergence_vs_reference"] is None and s["tok_per_s_median"] == 170.0


def test_summarize_not_repeatable_and_not_reference():
    s = summarize([fake(1.0, "story A"), fake(1.0, "story A"), fake(1.0, "story B")], REFERENCE)["1"]
    assert not s["repeatable"] and not s["equals_reference"]
    assert list(s["hashes"].values()) == [2, 1]


def test_divergence_against_real_reference_text():
    _, gemma = get_profile(load_reference(), "gemma4:26b-a4b-it-q8_0")
    entry = gemma["references"][0]
    text = json.loads((ROOT / "reference" / "texts" / f"{entry['hashes']['0']}.json").read_text(encoding="utf-8"))["text"]
    changed = text[:100] + "X" + text[101:]
    s = summarize([fake(0.0, changed)], entry)["0"]
    assert s["divergence_vs_reference"]["char"] == 100


def test_divergence_points_at_first_different_character():
    d = divergence("The dog was happy.", "The dog was sad.")
    assert d["char"] == 12 and d["reference"].endswith("happy.") and d["this"].endswith("sad.")
    assert divergence("abc", "abcd")["char"] == 3


def test_warnings_flag_factors_that_change_the_hash():
    machine = {"server": {"model": "sha256-other", "build_info": "b1",
                          "sampling": {"top_k": 40, "top_p": 0.949999988079071, "min_p": 0.05}}}
    summary = summarize([fake(0.0, "same story", prompt_tokens=31)], REFERENCE)
    w = warnings_for(machine, summary, REFERENCE, build_after="b2")
    assert any("model file" in x for x in w)
    assert any("top_k" in x for x in w) and not any("top_p" in x for x in w)
    assert any("prompt tokens" in x for x in w) and any("build changed" in x for x in w)


def test_results_table_compares_with_the_reference_of_the_same_build(tmp_path):
    def record(label, build, hashes0):
        return {"label": label, "profile": "gemma4:26b-a4b-it-q8_0", "started": "2026-10-21T10:00:00+02:00", "note": "",
                "gpu": 0, "machine": {"os": "Linux", "gpus": [{"name": "GPU X"}], "ollama": "0.40.6",
                                      "server": {"build_info": build}},
                "temperatures": {"0": {"runs": 10, "hashes": hashes0, "repeatable": len(hashes0) == 1},
                                 "1": {"runs": 10, "hashes": {"aaa": 6, "bbb": 4}, "repeatable": False}},
                "warnings": ["x"]}
    (tmp_path / "results" / "pc").mkdir(parents=True)
    for i, (build, h) in enumerate([("b1-620d2225c", {"138e66a98238": 10}), ("b1-unknown", {"138e66a98238": 10})]):
        (tmp_path / "results" / "pc" / f"r{i}.json").write_text(json.dumps(record("pc", build, h)), encoding="utf-8")
    (tmp_path / "reference.json").write_text((ROOT / "reference.json").read_text(encoding="utf-8"), encoding="utf-8")
    md = results_table.build(tmp_path)
    assert "| gemma4:26b-a4b-it-q8_0 | off | pc |" in md and "`138e66a98238`×10 – repeatable, = ref" in md
    assert "not repeatable" in md and "≠ ref" in md and "no ref for this build" in md
    assert "| GPU 0: GPU X |" in md and "| 10 / 10 |" in md


def fake_thinking(temp, text, thought):
    r = fake(temp, text)
    r["thinking_hash"] = text_hash(thought)[0]
    return r


def test_pick_reference_respects_the_thinking_mode():
    profile = {"references": [{"llama_cpp": ["b1"], "thinking": False, "hashes": {"0": "off"}},
                              {"llama_cpp": ["b1"], "thinking": True, "hashes": {"0": "on"}}]}
    assert pick_reference(profile, "b1")[0]["hashes"]["0"] == "off"
    assert pick_reference(profile, "b1", thinking=True)[0]["hashes"]["0"] == "on"
    assert pick_reference({"references": profile["references"][:1]}, "b1", thinking=True) == (
        {"llama_cpp": [], "thinking": True, "hashes": {}}, False)


def test_summarize_with_thinking_needs_answer_and_thinking():
    ref = {"hashes": {"0": text_hash("story")[0]}, "thinking_hashes": {"0": text_hash("plan")[0]}}
    s = summarize([fake_thinking(0.0, "story", "plan")] * 2, ref)["0"]
    assert s["repeatable"] and s["equals_reference"] and s["thinking_hashes"] == {text_hash("plan")[0]: 2}
    s = summarize([fake_thinking(0.0, "story", "plan"), fake_thinking(0.0, "story", "other plan")], ref)["0"]
    assert not s["repeatable"] and not s["equals_reference"]


def test_warning_when_the_text_was_cut_off():
    r = dict(fake(0.0, "same story"), finish_reason="length")
    summary = summarize([r], REFERENCE)
    machine = {"server": {"model": "sha256-abc", "sampling": {"top_k": 64, "top_p": 0.95, "min_p": 0.05}}}
    assert any("hit max_tokens" in w for w in warnings_for(machine, summary, REFERENCE, None))


def test_results_table_thinking_column_and_both_hashes(tmp_path):
    reference = {"profiles": {"m": {"references": [
        {"llama_cpp": ["b1"], "thinking": True, "hashes": {"0": "aaa"}, "thinking_hashes": {"0": "ttt"}}]}}}
    record = {"label": "pc", "profile": "m", "thinking": True, "started": "2026-10-22T10:00:00+02:00", "note": "",
              "gpu": 0, "machine": {"os": "Linux", "gpus": [], "ollama": "0.40.7", "server": {"build_info": "b1"}},
              "temperatures": {"0": {"runs": 3, "hashes": {"aaa": 3}, "thinking_hashes": {"ttt": 3},
                                     "repeatable": True}},
              "warnings": []}
    (tmp_path / "results" / "pc").mkdir(parents=True)
    (tmp_path / "results" / "pc" / "r.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    md = results_table.build(tmp_path)
    assert "| m | on | pc |" in md and "`aaa`×3 · thinking `ttt`×3 – repeatable, = ref" in md


def test_results_table_is_sorted_by_start_time(tmp_path):
    def record(profile, started):
        return {"label": "pc", "profile": profile, "started": started, "note": "", "gpu": 0,
                "machine": {"os": "Linux", "gpus": [], "ollama": "0.40.7", "server": {"build_info": "b1"}},
                "temperatures": {"0": {"runs": 1, "hashes": {"aaa": 1}, "repeatable": True}}, "warnings": []}
    (tmp_path / "results" / "pc").mkdir(parents=True)
    # file names and profiles in the opposite order; 10:00+01:00 (09:00 UTC) is earlier than 10:30+02:00 (08:30 UTC)
    for name, profile, started in [("a", "a-model", "2026-10-22T10:00:00+01:00"),
                                   ("b", "b-model", "2026-10-22T10:30:00+02:00"),
                                   ("c", "c-model", "2026-10-21T23:00:00+02:00")]:
        (tmp_path / "results" / "pc" / f"{name}.json").write_text(json.dumps(record(profile, started)),
                                                                 encoding="utf-8")
    (tmp_path / "reference.json").write_text(json.dumps({"profiles": {}}), encoding="utf-8")
    rows = [line for line in results_table.build(tmp_path).splitlines() if line.startswith("| ") and "-model" in line]
    assert [row.split(" | ")[0] for row in rows] == ["| c-model", "| b-model", "| a-model"]


def test_write_result_never_overwrites(tmp_path):
    base = tmp_path / "results" / "pc" / "2026-10-25T1200_qwen3-0.6b-q8_0_quick"
    first, second = write_result(base, {"gpu": 0}), write_result(base, {"gpu": 1})   # same minute, two GPUs
    assert first != second and first.name.startswith(base.name + "_") and first.suffix == ".json"
    assert [json.loads(p.read_text(encoding="utf-8"))["gpu"] for p in (first, second)] == [0, 1]


def test_compute_column_shows_gpu_cpu_or_mix():
    machine = {"cpu": "CPU Y", "gpus": [{"name": "GPU A"}, {"name": "GPU B"}]}
    assert results_table.compute({"machine": machine, "gpu": 1, "device": "gpu"}) == "GPU 1: GPU B"
    assert results_table.compute({"machine": machine, "device": "cpu"}) == "CPU: CPU Y"
    assert results_table.compute({"machine": machine, "gpu": 0, "device": "mix"}) == "mix – GPU 0: GPU A + CPU: CPU Y"
    assert results_table.compute({"machine": machine, "gpu": 0}) == "GPU 0: GPU A"   # before --device existed


def lp_entry(token, logprob, top):
    """One entry of choices[0].logprobs.content as llama-server sends it."""
    return {"id": 7, "token": token, "bytes": [], "logprob": logprob,
            "top_logprobs": [{"id": i, "token": t, "bytes": [], "logprob": p} for i, (t, p) in enumerate(top)]}


def test_logprobs_tokens_like_the_reference_files():
    top = [("O", -0.79), ("**", -1.85)]
    first, second, outside, empty = logprobs_tokens([lp_entry("O", -0.79, top), lp_entry("**", -1.85, top),
                                                     lp_entry("W", -2.27, top), lp_entry("x", -0.1, [])])
    assert first == {"token": "O", "logprob": -0.79, "top1": "O", "alt": "**", "alt_logprob": -1.85}
    assert second["top1"] == "O" and second["alt"] == "O"     # temperature 1: the second candidate was drawn
    assert outside["top1"] == "O" and outside["alt"] == "O"   # drawn from outside both candidates
    assert empty["top1"] is None and empty["alt"] is None and empty["alt_logprob"] is None
    old = json.loads((ROOT / "reference" / "texts" / "138e66a98238.json").read_text(encoding="utf-8"))["tokens"][0]
    assert list(first) == list(old)                           # the same keys in the same order as the old references


def test_logprobs_hash_is_the_text_hash_of_the_numbers():
    entries = [lp_entry("O", -0.7933734655380249, [("O", -0.7933734655380249), ("**", -1.85)])]
    tokens = logprobs_tokens(entries)
    h = logprobs_hash(tokens)
    assert h == text_hash(json.dumps(tokens, ensure_ascii=False, separators=(",", ":")))[0]
    assert logprobs_hash(logprobs_tokens(entries)) == h
    assert logprobs_hash([dict(tokens[0], logprob=-0.79337346553802)]) != h   # one digit less
    assert logprobs_hash([]) is None


def test_summarize_logprobs_like_the_text():
    ref = dict(REFERENCE, logprobs_hashes={"0": "lp0"})
    same = [dict(fake(0.0, "same story"), logprobs_hash="lp0")] * 3
    s = summarize(same, ref)["0"]
    assert s["logprobs_repeatable"] and s["logprobs_equals_reference"] and s["logprobs_hashes"] == {"lp0": 3}
    s = summarize(same[:2] + [dict(same[0], logprobs_hash="lp1")], ref)["0"]
    assert s["repeatable"] and s["logprobs_repeatable"] is False and s["logprobs_equals_reference"] is False
    s = summarize(same[:2] + [fake(0.0, "same story")], ref)["0"]     # one run without logprobs
    assert s["logprobs_repeatable"] is None and s["logprobs_equals_reference"] is None
    assert summarize(same, REFERENCE)["0"]["logprobs_equals_reference"] is None   # reference without logprobs
    machine = {"server": {"model": "sha256-abc", "sampling": {"top_k": 64, "top_p": 0.95, "min_p": 0.05}}}
    assert any("no logprobs" in w for w in warnings_for(machine, summarize([fake(0.0, "same story")], REFERENCE),
                                                        REFERENCE, None))
    assert not any("no logprobs" in w for w in warnings_for(machine, summarize(same, ref), REFERENCE, None))


def test_reference_text_file_one_token_per_line():
    data = {"hash": "abc", "text": "O to", "tokens": [{"token": "O", "logprob": -0.1}, {"token": " to", "logprob": 0.0}]}
    text = dump_reference_text(data)
    assert json.loads(text) == data and list(json.loads(text)) == ["hash", "text", "tokens"]
    assert '  {"token":"O","logprob":-0.1},' in text.splitlines()
    assert json.loads(dump_reference_text({"hash": "abc"})) == {"hash": "abc"}


MACHINE = {"server": {"build_info": "b9-new"}, "ollama": "0.99.0"}
STARTED = datetime(2026, 10, 25, 12, 0)


def reference_root(tmp_path, entries):
    """A copy of the reference files in tmp_path: one profile "m" with the given reference entries."""
    reference = {"protocol": load_reference()["protocol"], "profiles": {"m": {"references": entries}}}
    (tmp_path / "reference" / "texts").mkdir(parents=True)
    (tmp_path / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    return reference


def repeatable_test():
    """summary, results and logprobs sequences of a repeatable test (3 runs per temperature)."""
    texts = {"0": "story zero", "1": "story one"}
    results = [dict(fake(float(k), t), logprobs_hash=f"lp{k}") for k, t in texts.items()]
    summary = {k: {"hashes": {text_hash(t)[0]: 3}, "logprobs_hashes": {f"lp{k}": 3}, "prompt_tokens": [29],
                   "answer_tokens": [100]} for k, t in texts.items()}
    sequences = {f"lp{k}": [{"token": k, "logprob": -0.1, "top1": k, "alt": "t", "alt_logprob": -2.0}] for k in texts}
    return summary, results, sequences


def test_save_reference_for_a_new_build(tmp_path):
    reference = reference_root(tmp_path, [])
    summary, results, sequences = repeatable_test()
    done = save_reference(tmp_path, reference, "m", MACHINE, False, summary, results, sequences, "pc", STARTED)
    assert done.startswith("Became the reference")
    entry = json.loads((tmp_path / "reference.json").read_text(encoding="utf-8"))["profiles"]["m"]["references"][0]
    assert entry["llama_cpp"] == ["b9-new"] and entry["ollama"] == ["0.99.0"] and entry["prompt_tokens"] == 29
    assert entry["logprobs_hashes"] == {"0": "lp0", "1": "lp1"}
    data = json.loads((tmp_path / "reference" / "texts" / f"{entry['hashes']['0']}.json").read_text(encoding="utf-8"))
    assert data["text"] == "story zero" and data["tokens"] == sequences["lp0"] and data["tokens_cover"] == "answer"
    assert text_hash(data["text"]) == (entry["hashes"]["0"], data["sha256"])


def test_save_reference_adds_logprobs_and_never_overwrites(tmp_path):
    summary, results, sequences = repeatable_test()
    hashes = {k: next(iter(s["hashes"])) for k, s in summary.items()}
    reference = reference_root(tmp_path, [{"thinking": False, "llama_cpp": ["b9-new"], "hashes": hashes}])
    for k, h in hashes.items():   # the existing reference texts, without logprobs
        (tmp_path / "reference" / "texts" / f"{h}.json").write_text(
            json.dumps({"hash": h, "text": results[int(k)]["text"]}), encoding="utf-8")
    done = save_reference(tmp_path, reference, "m", MACHINE, False, summary, results, sequences, "pc", STARTED)
    assert done.startswith("Reference logprobs added")
    data = json.loads((tmp_path / "reference" / "texts" / f"{hashes['1']}.json").read_text(encoding="utf-8"))
    assert data["text"] == "story one" and data["tokens"] == sequences["lp1"]
    done = save_reference(tmp_path, reference, "m", MACHINE, False, summary, results, sequences, "pc", STARTED)
    assert done.startswith("Already the reference")
    before = (tmp_path / "reference.json").read_text(encoding="utf-8")
    other_logprobs = {k: dict(s, logprobs_hashes={"lpX": 3}) for k, s in summary.items()}
    with pytest.raises(ValueError, match="other logprobs"):
        save_reference(tmp_path, reference, "m", MACHINE, False, other_logprobs, results, sequences, "pc", STARTED)
    other_texts = {k: dict(s, hashes={"000000000000": 3}) for k, s in summary.items()}
    with pytest.raises(ValueError, match="other texts"):
        save_reference(tmp_path, reference, "m", MACHINE, False, other_texts, results, sequences, "pc", STARTED)
    assert (tmp_path / "reference.json").read_text(encoding="utf-8") == before


def test_reference_blockers():
    ok = {"0": {"repeatable": True, "logprobs_repeatable": True}, "1": {"repeatable": True, "logprobs_repeatable": True}}
    assert reference_blockers(ok, [], "gpu") == []
    assert reference_blockers(ok, ["sampling top_k = 40, reference 64"], "gpu") == ["sampling top_k = 40, reference 64"]
    no_logprobs = dict(ok, **{"1": {"repeatable": True, "logprobs_repeatable": None}})
    other_text = dict(ok, **{"0": {"repeatable": False, "logprobs_repeatable": True}})
    assert any("not repeatable" in b for b in reference_blockers(no_logprobs, [], "gpu"))
    assert any("not repeatable" in b for b in reference_blockers(other_text, [], "gpu"))
    assert any("--device mix" in b for b in reference_blockers(ok, [], "mix"))


def test_save_reference_with_thinking(tmp_path):
    reference = reference_root(tmp_path, [])
    summary, results, sequences = repeatable_test()
    for r in results:
        r["thinking_text"] = "thought: " + r["text"]
        r["thinking_hash"] = text_hash(r["thinking_text"])[0]
    for k, st in summary.items():
        st["thinking_hashes"] = {results[int(k)]["thinking_hash"]: 3}
    save_reference(tmp_path, reference, "m", MACHINE, True, summary, results, sequences, "pc", STARTED)
    entry = reference["profiles"]["m"]["references"][0]
    texts = tmp_path / "reference" / "texts"
    answer = json.loads((texts / f"{entry['hashes']['0']}.json").read_text(encoding="utf-8"))
    thought = json.loads((texts / f"{entry['thinking_hashes']['0']}.json").read_text(encoding="utf-8"))
    assert entry["thinking"] and list(entry)[-1] == "logprobs_hashes"
    assert answer["tokens_cover"] == "thinking + answer" and answer["tokens"] == sequences["lp0"]
    assert answer["tokens_source"].startswith("m, thinking on")
    assert thought["kind"] == "thinking" and thought["text"] == "thought: story zero" and "tokens" not in thought


def raw_run(**extra):
    """One run as main() holds it, with the fields save_raw_run writes."""
    r = dict(fake(0.0, "Zażółć gęślą jaźń.\nDruga linia."), run=3, finish_reason="stop", logprobs_hash="lp0",
             thinking_hash=None, started="2026-10-25T11:18:33.412+01:00", finished="2026-10-25T11:18:34.205+01:00",
             response_id="chatcmpl-abc", created=1792923852)
    r.update(extra)
    return r


def test_save_raw_run_keeps_the_response_byte_for_byte(tmp_path):
    raw = '{"id":"chatcmpl-abc","choices":[{"message":{"content":"Zażółć gęślą jaźń."}}]}'.encode("utf-8")
    save_raw_run(tmp_path, raw_run(), raw, 2)
    assert (tmp_path / "t0_run03.json").read_bytes() == raw
    text = (tmp_path / "t0_run03.txt").read_text(encoding="utf-8")
    assert "started  2026-10-25T11:18:33.412+01:00" in text and "finished 2026-10-25T11:18:34.205+01:00" in text
    assert "response chatcmpl-abc" in text and "created 1792923852" in text
    assert text.endswith("=== answer ===\nZażółć gęślą jaźń.\nDruga linia.\n") and "=== thinking ===" not in text
    save_raw_run(tmp_path, raw_run(temperature=1.0, run=12, thinking_text="Myślę.", thinking_hash="th"), raw, 4)
    thought = (tmp_path / "t1_run0012.txt").read_text(encoding="utf-8")
    assert thought.index("=== thinking ===\nMyślę.") < thought.index("=== answer ===") and "thinking th" in thought


def test_write_result_takes_the_given_id(tmp_path):
    base = tmp_path / "results" / "pc" / "2026-10-25T1200_qwen3-0.6b-q8_0_quick"
    first = write_result(base, {"n": 1}, "abc123")
    second = write_result(base, {"n": 2}, "abc123")   # taken: a new random id
    assert first.name.endswith("_abc123.json") and second != first and second.exists()
    assert json.loads(first.read_text(encoding="utf-8")) == {"n": 1}   # the first file is untouched


def test_raw_folders_are_not_in_git():
    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
    assert ignored("results/x/raw/y/t0_run01.json") and not ignored("results/x/2026-10-25T1200_x_quick_abc123.json")


class FakeResponse:
    """What requests returns: the body as bytes and as JSON."""
    def __init__(self, data):
        self.data, self.content = data, json.dumps(data, ensure_ascii=False).encode("utf-8")

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_main_with_save_raw_against_a_fake_server(tmp_path, monkeypatch):
    import run_test
    (tmp_path / "reference.json").write_text((ROOT / "reference.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(run_test, "ROOT", tmp_path)
    profile = load_reference()["profiles"]["qwen3:0.6b-q8_0"]
    props = {"build_info": "b9-test", "model_path": "models/" + profile["model_file"], "total_slots": 1,
             "default_generation_settings": {"n_ctx": 8192, "params": dict(profile["sampling"])}}
    sent = []

    def post(url, json=None, timeout=None):
        body = {"id": f"chatcmpl-{len(sent)}", "created": 1792924458 + len(sent),
                "usage": {"completion_tokens": 1, "prompt_tokens": 29}, "timings": {"predicted_per_second": 100.0},
                "choices": [{"finish_reason": "stop", "message": {"content": f"historia {json['temperature']:g}"},
                             "logprobs": {"content": [lp_entry("historia", -0.1, [("historia", -0.1), ("bajka", -2.0)])]}}]}
        sent.append(body)
        return FakeResponse(body)

    monkeypatch.setattr(run_test.requests, "get", lambda url, timeout=None: FakeResponse(props))
    monkeypatch.setattr(run_test.requests, "post", post)
    monkeypatch.setattr(run_test, "run_cmd", lambda cmd: "ollama version is 0.99.0" if cmd[0] == "ollama" else "")
    monkeypatch.setattr(sys, "argv", ["run_test.py", "--label", "pc", "--runs", "2", "--save-raw"])
    run_test.main()
    result = next((tmp_path / "results" / "pc").glob("*.json"))
    r = json.loads(result.read_text(encoding="utf-8"))
    raw_dir = result.parent / r["raw_dir"]
    assert raw_dir.name == result.stem and list(r).index("raw_dir") == list(r).index("note") + 1
    assert sorted(p.name for p in raw_dir.iterdir()) == [f"t{t}_run{i}.{e}" for t in (0, 1) for i in (1, 2)
                                                        for e in ("json", "txt")]
    assert (raw_dir / "t0_run1.json").read_bytes() == json.dumps(sent[0], ensure_ascii=False).encode("utf-8")
    assert [x["response_id"] for x in r["runs"]] == [f"chatcmpl-{i}" for i in range(4)]
    assert all(x["started"] <= x["finished"] and "raw" not in x and x["logprobs_hash"] for x in r["runs"])
    assert all(s["repeatable"] and s["logprobs_repeatable"] for s in r["temperatures"].values())

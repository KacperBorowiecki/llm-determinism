import json
from pathlib import Path

import results_table
from run_test import (divergence, get_profile, load_reference, pick_reference, summarize, text_hash,
                      warnings_for)

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


def test_text_hash_uses_exact_bytes():
    assert text_hash("a\n")[0] != text_hash("a\r\n")[0]
    assert text_hash("a")[0] != text_hash("a ")[0]
    short, full = text_hash("x")
    assert len(short) == 12 and len(full) == 64 and full.startswith(short)


def test_profiles_and_reference_per_build():
    reference = load_reference()
    name, profile = get_profile(reference)
    assert name == reference["default_profile"] and profile["references"]
    _, gemma = get_profile(reference, "gemma4-26b-q8")
    entry, exact = pick_reference(gemma, "b1-620d2225c")
    assert exact and entry["hashes"]["0"] == "138e66a98238"
    entry, exact = pick_reference(gemma, "b1-unknown")
    assert not exact and entry is gemma["references"][-1]


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
    _, gemma = get_profile(load_reference(), "gemma4-26b-q8")
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
        return {"label": label, "profile": "gemma4-26b-q8", "started": "2026-10-21T10:00:00+02:00", "note": "",
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
    assert "| gemma4-26b-q8 | pc |" in md and "`138e66a98238`×10 – repeatable, = ref" in md
    assert "not repeatable" in md and "≠ ref" in md and "no ref for this build" in md
    assert "| 0 |" in md and "| 10 / 10 |" in md

import json
from pathlib import Path

import results_table
from run_test import divergence, load_reference, summarize, text_hash, warnings_for

ROOT = Path(__file__).resolve().parents[1]


def test_reference_texts_rehash_to_reference_hashes():
    reference = load_reference()
    for key, short in reference["hashes"].items():
        data = json.loads((ROOT / "reference" / "texts" / f"{short}.json").read_text(encoding="utf-8"))
        assert text_hash(data["text"]) == (short, data["sha256"]), f"temperature {key}"
        assert "\r" not in data["text"]


def test_text_hash_uses_exact_bytes():
    assert text_hash("a\n")[0] != text_hash("a\r\n")[0]
    assert text_hash("a")[0] != text_hash("a ")[0]
    short, full = text_hash("x")
    assert len(short) == 12 and len(full) == 64 and full.startswith(short)


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
    reference = load_reference()
    ref_hash = reference["hashes"]["0"]
    text = json.loads((ROOT / "reference" / "texts" / f"{ref_hash}.json").read_text(encoding="utf-8"))["text"]
    changed = text[:100] + "X" + text[101:]
    s = summarize([fake(0.0, changed)], reference)["0"]
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


def test_results_table_lists_every_result(tmp_path):
    record = {"label": "test-pc", "started": "2026-10-21T10:00:00+02:00", "note": "", "gpu": 0,
              "machine": {"os": "Linux", "gpus": [{"name": "GPU X"}], "ollama": "0.40.6",
                          "server": {"build_info": "b1-abc"}},
              "temperatures": {"0": {"runs": 10, "hashes": {"138e66a98238": 10}, "repeatable": True,
                                     "equals_reference": True},
                               "1": {"runs": 10, "hashes": {"aaa": 6, "bbb": 4}, "repeatable": False,
                                     "equals_reference": False}},
              "warnings": ["x"]}
    (tmp_path / "results" / "test-pc").mkdir(parents=True)
    (tmp_path / "results" / "test-pc" / "r.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "reference.json").write_text((ROOT / "reference.json").read_text(encoding="utf-8"), encoding="utf-8")
    md = results_table.build(tmp_path)
    assert "| test-pc |" in md and "`138e66a98238`×10" in md and "not repeatable" in md and "≠ ref" in md
    assert "| 0 |" in md and "| 10 / 10 |" in md

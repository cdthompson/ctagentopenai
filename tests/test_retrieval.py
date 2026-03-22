import json
from pathlib import Path

from ctagentopenai.retrieval import (
    LabelCorpus,
    compare_methods,
    latest_documents_by_drug,
    load_eval_cases,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "inputs"


def test_rebuilds_corpus_from_sample_dataset(tmp_path):
    source = tmp_path / "labels.json"
    source.write_text((FIXTURE_DIR / "fda-label-sample.json").read_text(encoding="utf-8"), encoding="utf-8")
    db_path = tmp_path / "labels.sqlite"

    corpus = LabelCorpus(db_path)
    stats = corpus.rebuild(latest_documents_by_drug([source]))

    assert stats["labels"] == 3
    assert stats["chunks"] >= 18


def test_grep_and_bm25_return_expected_sections_for_sample_dataset(tmp_path):
    source_path = tmp_path / "labels.json"
    fixture_path = FIXTURE_DIR / "fda-label-sample.json"
    source_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
    db_path = tmp_path / "labels.sqlite"

    corpus = LabelCorpus(db_path)
    corpus.rebuild(latest_documents_by_drug([source_path]))

    grep_result = corpus.query_label(
        drug_name="ibuprofen",
        question="What does the label say about pregnancy?",
        method="grep",
        top_k=3,
    )
    bm25_result = corpus.query_label(
        drug_name="ibuprofen",
        question="What does the label say about pregnancy?",
        method="bm25",
        top_k=3,
    )

    assert grep_result.matches[0].section_name == "use in specific populations"
    assert bm25_result.matches[0].section_name == "use in specific populations"


def test_compare_methods_reports_expected_section_hits(tmp_path):
    source_path = tmp_path / "labels.json"
    fixture_path = FIXTURE_DIR / "fda-label-sample.json"
    eval_path = FIXTURE_DIR / "fda-label-eval.json"
    source_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
    db_path = tmp_path / "labels.sqlite"

    corpus = LabelCorpus(db_path)
    corpus.rebuild(latest_documents_by_drug([source_path]))
    results = compare_methods(corpus, load_eval_cases(eval_path))

    assert len(results) == 6
    assert results[0]["bm25"]["expected_in_top_k"] is True
    assert results[0]["grep"]["expected_in_top_k"] is True


def test_query_result_serializes_to_json(tmp_path):
    source_path = tmp_path / "labels.json"
    fixture_path = FIXTURE_DIR / "fda-label-sample.json"
    source_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
    db_path = tmp_path / "labels.sqlite"

    corpus = LabelCorpus(db_path)
    corpus.rebuild(latest_documents_by_drug([source_path]))
    payload = json.loads(
        corpus.query_label(
            drug_name="warfarin",
            question="What monitoring is recommended?",
            method="bm25",
        ).to_json()
    )

    assert payload["drug_name_resolved"] == "warfarin"
    assert payload["retrieval_method"] == "bm25"
    assert payload["matches"][0]["section"] == "warnings and precautions"

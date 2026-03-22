from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


SECTION_SPECS = [
    ("boxed warning", ["boxed_warning"]),
    ("indications and usage", ["indications_and_usage"]),
    ("dosage and administration", ["dosage_and_administration"]),
    ("contraindications", ["contraindications"]),
    ("warnings and precautions", ["warnings_and_precautions", "warnings", "warnings_and_cautions"]),
    ("adverse reactions", ["adverse_reactions"]),
    ("drug interactions", ["drug_interactions"]),
    ("use in specific populations", ["use_in_specific_populations", "pregnancy"]),
]
SECTION_PRIORITY = {
    "boxed warning": 0,
    "contraindications": 1,
    "warnings and precautions": 2,
    "adverse reactions": 3,
    "drug interactions": 4,
    "use in specific populations": 5,
    "dosage and administration": 6,
    "indications and usage": 7,
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_CHUNK_CHAR_LIMIT = 1200
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class LabelDocument:
    label_id: str
    drug_name_primary: str
    drug_name_aliases: list[str]
    product_type: str
    effective_date: str
    source_path: str
    sections: list[tuple[str, str]]


@dataclass(frozen=True)
class RetrievalMatch:
    section_name: str
    chunk_id: str
    score: float
    chunk_text: str
    label_id: str
    drug_name_primary: str


@dataclass(frozen=True)
class RetrievalResult:
    drug_name_resolved: str
    label_id: str
    retrieval_method: str
    matches: list[RetrievalMatch]

    def to_json(self) -> str:
        payload = {
            "drug_name_resolved": self.drug_name_resolved,
            "label_id": self.label_id,
            "retrieval_method": self.retrieval_method,
            "match_count": len(self.matches),
            "matches": [
                {
                    "section": match.section_name,
                    "chunk_id": match.chunk_id,
                    "score": round(match.score, 6),
                    "text": match.chunk_text,
                    "label_id": match.label_id,
                    "drug_name_primary": match.drug_name_primary,
                }
                for match in self.matches
            ],
        }
        return json.dumps(payload, indent=2)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def normalized_key(text: str) -> str:
    return normalize_text(text).casefold()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def choose_primary_name(names: Iterable[str]) -> str:
    clean = [normalize_text(name) for name in names if normalize_text(name)]
    if not clean:
        return ""
    return clean[0]


def unique_names(*sources: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for source in sources:
        if source is None:
            continue
        values = source if isinstance(source, list) else [source]
        for value in values:
            cleaned = normalize_text(str(value))
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
    return result


def chunk_text(section_name: str, text: str, chunk_char_limit: int, overlap_chars: int) -> list[str]:
    cleaned = normalize_text(text)
    if len(cleaned) <= chunk_char_limit:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > chunk_char_limit:
            chunks.append(current)
            overlap = current[-overlap_chars:].strip()
            current = f"{overlap} {sentence}".strip() if overlap else sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    if not chunks:
        return [cleaned]
    return chunks


def section_text_from_record(record: dict[str, Any], field_names: list[str]) -> str:
    parts: list[str] = []
    for field_name in field_names:
        value = record.get(field_name)
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(normalize_text(str(item)) for item in value if normalize_text(str(item)))
        else:
            cleaned = normalize_text(str(value))
            if cleaned:
                parts.append(cleaned)
    return "\n\n".join(parts)


def extract_label_document(record: dict[str, Any], source_path: str) -> LabelDocument | None:
    openfda = record.get("openfda") or {}
    aliases = unique_names(
        openfda.get("generic_name"),
        openfda.get("brand_name"),
        openfda.get("substance_name"),
        record.get("brand_name"),
        record.get("generic_name"),
    )
    primary_name = choose_primary_name(aliases)
    label_id = normalize_text(
        str(
            record.get("set_id")
            or record.get("id")
            or record.get("spl_set_id")
            or record.get("application_number")
            or ""
        )
    )
    if not label_id or not primary_name:
        return None

    sections = []
    for section_name, field_names in SECTION_SPECS:
        section_text = section_text_from_record(record, field_names)
        if section_text:
            sections.append((section_name, section_text))
    if not sections:
        return None

    product_type = choose_primary_name(
        unique_names(openfda.get("product_type"), record.get("product_type"))
    )
    effective_date = normalize_text(str(record.get("effective_time") or record.get("effective_date") or ""))
    return LabelDocument(
        label_id=label_id,
        drug_name_primary=primary_name,
        drug_name_aliases=aliases,
        product_type=product_type,
        effective_date=effective_date,
        source_path=source_path,
        sections=sections,
    )


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        for record in payload["results"]:
            if isinstance(record, dict):
                yield record
        return
    if isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                yield record
        return
    if isinstance(payload, dict):
        yield payload


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def iter_zip_records(path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            lowered = member.casefold()
            if lowered.endswith(".json"):
                with archive.open(member) as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                    for record in payload["results"]:
                        if isinstance(record, dict):
                            yield record
                elif isinstance(payload, list):
                    for record in payload:
                        if isinstance(record, dict):
                            yield record
                elif isinstance(payload, dict):
                    yield payload
            elif lowered.endswith(".jsonl"):
                with archive.open(member) as handle:
                    for raw_line in handle:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if isinstance(record, dict):
                            yield record


def iter_source_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        yield from iter_json_records(path)
        return
    if suffix == ".jsonl":
        yield from iter_jsonl_records(path)
        return
    if suffix == ".zip":
        yield from iter_zip_records(path)
        return
    raise ValueError(f"Unsupported source format: {path}")


def latest_documents_by_drug(paths: list[Path]) -> list[LabelDocument]:
    latest_by_name: dict[str, LabelDocument] = {}
    for path in paths:
        for record in iter_source_records(path):
            document = extract_label_document(record, str(path))
            if document is None:
                continue
            key = normalized_key(document.drug_name_primary)
            existing = latest_by_name.get(key)
            if existing is None or document.effective_date >= existing.effective_date:
                latest_by_name[key] = document
    return sorted(latest_by_name.values(), key=lambda item: item.drug_name_primary.casefold())


class LabelCorpus:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    label_id TEXT PRIMARY KEY,
                    drug_name_primary TEXT NOT NULL,
                    product_type TEXT NOT NULL,
                    effective_date TEXT NOT NULL,
                    source_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT NOT NULL,
                    alias_key TEXT NOT NULL,
                    label_id TEXT NOT NULL REFERENCES labels(label_id),
                    UNIQUE(alias_key, label_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    label_id TEXT NOT NULL REFERENCES labels(label_id),
                    drug_name_primary TEXT NOT NULL,
                    section_name TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    label_id UNINDEXED,
                    drug_name_primary,
                    section_name,
                    chunk_text
                );
                """
            )

    def rebuild(self, documents: list[LabelDocument]) -> dict[str, int]:
        self.initialize()
        label_count = 0
        chunk_count = 0
        with self.connect() as connection:
            connection.execute("DELETE FROM aliases")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM labels")
            for document in documents:
                label_count += 1
                connection.execute(
                    """
                    INSERT INTO labels (label_id, drug_name_primary, product_type, effective_date, source_path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document.label_id,
                        document.drug_name_primary,
                        document.product_type,
                        document.effective_date,
                        document.source_path,
                    ),
                )
                for alias in document.drug_name_aliases:
                    connection.execute(
                        "INSERT OR IGNORE INTO aliases (alias, alias_key, label_id) VALUES (?, ?, ?)",
                        (alias, normalized_key(alias), document.label_id),
                    )
                for section_name, section_text in document.sections:
                    for chunk_order, chunk in enumerate(
                        chunk_text(
                            section_name,
                            section_text,
                            chunk_char_limit=DEFAULT_CHUNK_CHAR_LIMIT,
                            overlap_chars=DEFAULT_CHUNK_OVERLAP,
                        ),
                        start=1,
                    ):
                        chunk_id = f"{document.label_id}:{section_name}:{chunk_order}"
                        chunk_count += 1
                        connection.execute(
                            """
                            INSERT INTO chunks (
                                chunk_id, label_id, drug_name_primary, section_name, chunk_order, chunk_text
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chunk_id,
                                document.label_id,
                                document.drug_name_primary,
                                section_name,
                                chunk_order,
                                chunk,
                            ),
                        )
                        connection.execute(
                            """
                            INSERT INTO chunks_fts (
                                chunk_id, label_id, drug_name_primary, section_name, chunk_text
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                chunk_id,
                                document.label_id,
                                document.drug_name_primary,
                                section_name,
                                chunk,
                            ),
                        )
            connection.commit()
        return {"labels": label_count, "chunks": chunk_count}

    def resolve_drug(self, drug_name: str) -> sqlite3.Row | None:
        key = normalized_key(drug_name)
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT labels.*
                FROM aliases
                JOIN labels ON labels.label_id = aliases.label_id
                WHERE aliases.alias_key = ?
                ORDER BY labels.effective_date DESC, labels.drug_name_primary ASC
                LIMIT 1
                """,
                (key,),
            ).fetchone()
            if row is not None:
                return row
            return connection.execute(
                """
                SELECT *
                FROM labels
                WHERE lower(drug_name_primary) = lower(?)
                ORDER BY effective_date DESC, drug_name_primary ASC
                LIMIT 1
                """,
                (normalize_text(drug_name),),
            ).fetchone()

    def query_label(self, drug_name: str, question: str, top_k: int = 5, method: str = "bm25") -> RetrievalResult:
        label = self.resolve_drug(drug_name)
        if label is None:
            raise LookupError(f"Unknown drug label: {drug_name}")
        if method == "grep":
            matches = self._query_grep(label["label_id"], question, top_k)
        elif method == "bm25":
            matches = self._query_bm25(label["label_id"], question, top_k)
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")
        return RetrievalResult(
            drug_name_resolved=label["drug_name_primary"],
            label_id=label["label_id"],
            retrieval_method=method,
            matches=matches,
        )

    def _query_grep(self, label_id: str, question: str, top_k: int) -> list[RetrievalMatch]:
        normalized_question = normalized_key(question)
        query_terms = [term for term in tokenize(question) if len(term) > 1]
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, label_id, drug_name_primary, section_name, chunk_text
                FROM chunks
                WHERE label_id = ?
                """,
                (label_id,),
            ).fetchall()
        ranked: list[tuple[tuple[Any, ...], RetrievalMatch]] = []
        for row in rows:
            haystack = normalized_key(row["chunk_text"])
            exact_phrase = normalized_question in haystack
            hit_count = sum(haystack.count(term) for term in query_terms)
            earliest = min(
                (haystack.find(term) for term in query_terms if term in haystack),
                default=10**9,
            )
            if not exact_phrase and hit_count == 0:
                continue
            score = float(exact_phrase) * 1000.0 + float(hit_count)
            match = RetrievalMatch(
                section_name=row["section_name"],
                chunk_id=row["chunk_id"],
                score=score,
                chunk_text=row["chunk_text"],
                label_id=row["label_id"],
                drug_name_primary=row["drug_name_primary"],
            )
            ranked.append(
                (
                    (
                        0 if exact_phrase else 1,
                        -hit_count,
                        earliest,
                        SECTION_PRIORITY.get(row["section_name"], 99),
                        row["chunk_id"],
                    ),
                    match,
                )
            )
        ranked.sort(key=lambda item: item[0])
        return [item[1] for item in ranked[:top_k]]

    def _query_bm25(self, label_id: str, question: str, top_k: int) -> list[RetrievalMatch]:
        query_terms = [term for term in tokenize(question) if len(term) > 1]
        if not query_terms:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunks.chunk_id,
                    chunks.label_id,
                    chunks.drug_name_primary,
                    chunks.section_name,
                    chunks.chunk_text
                FROM chunks
                WHERE chunks.label_id = ?
                """,
                (label_id,),
            ).fetchall()

        documents = [tokenize(f"{row['section_name']} {row['chunk_text']}") for row in rows]
        avg_doc_len = sum(len(doc) for doc in documents) / len(documents) if documents else 0.0
        document_frequency = Counter()
        for document in documents:
            for term in set(document):
                document_frequency[term] += 1

        matches: list[RetrievalMatch] = []
        for row, document in zip(rows, documents):
            term_frequency = Counter(document)
            doc_len = len(document) or 1
            score = 0.0
            for term in query_terms:
                tf = term_frequency.get(term, 0)
                if tf == 0:
                    continue
                df = document_frequency.get(term, 0)
                numerator = len(documents) - df + 0.5
                denominator = df + 0.5
                idf = math.log(1 + (numerator / denominator))
                k1 = 1.5
                b = 0.75
                tf_weight = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / (avg_doc_len or 1))))
                score += idf * tf_weight
            if score == 0.0:
                continue
            if any(term in tokenize(row["section_name"]) for term in query_terms):
                score += 1.5
            matches.append(
                RetrievalMatch(
                    section_name=row["section_name"],
                    chunk_id=row["chunk_id"],
                    score=score,
                    chunk_text=row["chunk_text"],
                    label_id=row["label_id"],
                    drug_name_primary=row["drug_name_primary"],
                )
            )
        matches.sort(
            key=lambda match: (
                -match.score,
                SECTION_PRIORITY.get(match.section_name, 99),
                match.chunk_id,
            )
        )
        return matches[:top_k]


def compare_methods(corpus: LabelCorpus, eval_cases: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    results = []
    for case in eval_cases:
        drug_name = case["drug_name"]
        question = case["question"]
        expected_section = case.get("expected_section")
        case_result = {"drug_name": drug_name, "question": question, "expected_section": expected_section}
        for method in ("grep", "bm25"):
            retrieval = corpus.query_label(drug_name=drug_name, question=question, top_k=top_k, method=method)
            sections = [match.section_name for match in retrieval.matches]
            case_result[method] = {
                "sections": sections,
                "top_score": retrieval.matches[0].score if retrieval.matches else None,
                "expected_in_top_k": expected_section in sections if expected_section else None,
            }
        results.append(case_result)
    return results


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Eval file must contain a JSON array")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FDA label corpus utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a local SQLite corpus from FDA label source files.")
    build_parser.add_argument("--db", required=True, help="Output SQLite database path.")
    build_parser.add_argument("--source", action="append", required=True, help="Input source path (.json, .jsonl, .zip).")

    query_parser = subparsers.add_parser("query", help="Query a built label corpus.")
    query_parser.add_argument("--db", required=True, help="SQLite database path.")
    query_parser.add_argument("--drug", required=True, help="Drug name to resolve.")
    query_parser.add_argument("--question", required=True, help="Question to ask against the label.")
    query_parser.add_argument("--method", choices=["grep", "bm25"], default="bm25")
    query_parser.add_argument("--top-k", type=int, default=5)

    eval_parser = subparsers.add_parser("eval", help="Compare retrieval methods against an eval set.")
    eval_parser.add_argument("--db", required=True, help="SQLite database path.")
    eval_parser.add_argument("--eval-file", required=True, help="JSON file with eval cases.")
    eval_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "build":
        corpus = LabelCorpus(Path(args.db))
        stats = corpus.rebuild(latest_documents_by_drug([Path(source) for source in args.source]))
        print(json.dumps(stats, indent=2))
        return

    corpus = LabelCorpus(Path(args.db))
    if args.command == "query":
        result = corpus.query_label(
            drug_name=args.drug,
            question=args.question,
            top_k=args.top_k,
            method=args.method,
        )
        print(result.to_json())
        return

    eval_results = compare_methods(corpus, load_eval_cases(Path(args.eval_file)), top_k=args.top_k)
    print(json.dumps(eval_results, indent=2))


if __name__ == "__main__":
    main()

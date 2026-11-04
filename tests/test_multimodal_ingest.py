import json
import pytest
from pathlib import Path
from src.multimodal_ingest import MultiModalIngestor, MultiModalDocument

def test_multimodal_ingest_csv(tmp_path: Path):
    ingestor = MultiModalIngestor()
    csv_file = tmp_path / "employees.csv"
    csv_content = (
        "Employee_ID,Name,Role,Department,Salary,Location\n"
        "101,Alice Johnson,Senior Architect,Cloud Infrastructure,145000,San Francisco\n"
        "102,Bob Smith,Security Lead,Cybersecurity,135000,New York\n"
        "103,Charlie Davis,ML Engineer,AI Research,150000,Seattle\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    doc: MultiModalDocument = ingestor.process_file(csv_file)
    assert doc.doc_type == "csv"
    assert doc.doc_name == "employees.csv"
    # Must have schema summary + row passages
    assert len(doc.passages) >= 2
    assert "structured tabular dataset containing 3 rows and 6 columns" in doc.passages[0]
    assert "Alice Johnson" in doc.passages[1]
    # Triples must be extracted from entity columns
    assert len(doc.triples) > 0
    # Check that triple has (subject, relation, object)
    first_triple = doc.triples[0]
    assert len(first_triple) == 3
    assert all(isinstance(elem, str) and elem for elem in first_triple)

def test_multimodal_ingest_json(tmp_path: Path):
    ingestor = MultiModalIngestor()
    json_file = tmp_path / "cluster_config.json"
    json_data = {
        "system": "AdaptiveRAG",
        "cluster": {"nodes": 8, "region": "us-west-2", "status": "active"},
        "settings": {"max_tokens": 1024, "enable_cache": True}
    }
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    doc: MultiModalDocument = ingestor.process_file(json_file)
    assert doc.doc_type == "json"
    assert doc.doc_name == "cluster_config.json"
    assert len(doc.passages) >= 1
    # Check that flattened keys appear in passages
    passages_text = " ".join(doc.passages)
    assert "cluster.region = us-west-2" in passages_text and "cluster.nodes = 8" in passages_text
    assert len(doc.triples) > 0

def test_multimodal_empty_csv(tmp_path: Path):
    ingestor = MultiModalIngestor()
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    doc: MultiModalDocument = ingestor.process_file(csv_file)
    assert doc.doc_type == "csv"
    assert doc.passages == ["Empty CSV document."]

def test_multimodal_empty_json(tmp_path: Path):
    ingestor = MultiModalIngestor()
    json_file = tmp_path / "empty.json"
    json_file.write_text("{}", encoding="utf-8")

    doc: MultiModalDocument = ingestor.process_file(json_file)
    assert doc.doc_type == "json"
    assert len(doc.passages) == 0

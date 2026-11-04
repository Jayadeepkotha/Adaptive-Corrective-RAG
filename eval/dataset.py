import json
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

@dataclass
class BenchmarkSample:
    id: str
    question: str
    gold_answers: List[str]
    contexts: List[Dict[str, Any]]  # [{"title": str, "text": str, "is_supporting": bool, "idx": int}]
    gold_supporting_titles: List[str] = field(default_factory=list)
    gold_supporting_texts: List[str] = field(default_factory=list)
    dataset: str = ""
    question_type: str = ""
    level: str = ""

class DatasetLoader:
    """
    Loads official multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue)
    with their full candidate contexts and ground-truth supporting paragraph labels.
    """

    DATASET_MAPPING = {
        "hotpotqa": "hotpotqa",
        "hotpot": "hotpotqa",
        "2wikimultihop": "2wikimultihopqa",
        "2wiki": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "musique": "musique"
    }

    def __init__(self, data_eval_dir: Optional[Path] = None, archive_path: Optional[Path] = None):
        root = Path(__file__).resolve().parent.parent
        self.data_eval_dir = data_eval_dir or root / "data" / "eval"
        self.data_eval_dir.mkdir(parents=True, exist_ok=True)
        self.archive_path = archive_path or root / "data" / "eval" / "processed_data.tar.gz"

    def ensure_dataset(self, dataset_key: str) -> Path:
        canonical = self.DATASET_MAPPING.get(dataset_key.lower().strip())
        if not canonical:
            raise ValueError(f"Unknown dataset '{dataset_key}'. Supported: {list(self.DATASET_MAPPING.keys())}")

        extracted_file = self.data_eval_dir / f"{canonical}_dev_500.jsonl"
        if extracted_file.exists() and extracted_file.stat().st_size > 0:
            return extracted_file

        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive '{self.archive_path}' not found to extract official datasets.")

        logger.info(f"Extracting '{canonical}' from archive '{self.archive_path.name}' to '{extracted_file.name}'...")
        target_in_tar = f"./processed_data/{canonical}/dev_500_subsampled.jsonl"
        with tarfile.open(self.archive_path, "r:gz") as tar:
            member = tar.getmember(target_in_tar)
            f = tar.extractfile(member)
            if f is None:
                raise RuntimeError(f"Could not extract {target_in_tar} from {self.archive_path}")
            with open(extracted_file, "wb") as out:
                out.write(f.read())

        return extracted_file

    def load_samples(
        self,
        dataset_name: str,
        n: Optional[int] = None,
        seed: int = 42,
        split: str = "all",
        split_seed: int = 42
    ) -> List[BenchmarkSample]:
        file_path = self.ensure_dataset(dataset_name)
        canonical = self.DATASET_MAPPING[dataset_name.lower().strip()]

        raw_records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))

        # Sort by question_id for deterministic ordering before seeded sampling
        raw_records.sort(key=lambda x: str(x.get("question_id", "")))

        # Seeded split of dataset (e.g. 500 questions) into tune (first half) and test (second half)
        split_norm = split.lower().strip()
        if split_norm in ["tune", "test"]:
            split_rng = random.Random(split_seed)
            indices = list(range(len(raw_records)))
            split_rng.shuffle(indices)
            mid = len(indices) // 2
            if split_norm == "tune":
                target_records = sorted([raw_records[i] for i in indices[:mid]], key=lambda x: str(x.get("question_id", "")))
            else:
                target_records = sorted([raw_records[i] for i in indices[mid:]], key=lambda x: str(x.get("question_id", "")))
        else:
            target_records = raw_records

        rng = random.Random(seed)
        if n is not None and n < len(target_records):
            sampled_records = rng.sample(target_records, n)
            # Re-sort sampled records for predictable test run order
            sampled_records.sort(key=lambda x: str(x.get("question_id", "")))
        else:
            sampled_records = target_records

        samples: List[BenchmarkSample] = []
        for rec in sampled_records:
            qid = str(rec.get("question_id", ""))
            q_text = rec.get("question_text", "").strip()

            answers = []
            for ao in rec.get("answers_objects", []):
                if ao.get("text") and ao["text"] not in answers:
                    answers.append(ao["text"])
                for sp in ao.get("spans", []):
                    if sp and sp not in answers:
                        answers.append(sp)

            contexts = []
            gold_titles = []
            gold_texts = []
            for ctx in rec.get("contexts", []):
                t = ctx.get("title", "").strip()
                p = ctx.get("paragraph_text", "").strip()
                is_sup = bool(ctx.get("is_supporting", False))
                contexts.append({
                    "idx": ctx.get("idx", len(contexts)),
                    "title": t,
                    "text": p,
                    "is_supporting": is_sup
                })
                if is_sup:
                    gold_titles.append(t)
                    gold_texts.append(p)

            samples.append(BenchmarkSample(
                id=qid,
                question=q_text,
                gold_answers=answers,
                contexts=contexts,
                gold_supporting_titles=gold_titles,
                gold_supporting_texts=gold_texts,
                dataset=canonical,
                question_type=rec.get("type", ""),
                level=rec.get("level", "")
            ))

        logger.info(f"Loaded {len(samples)} samples from '{canonical}' (seed={seed}).")
        return samples

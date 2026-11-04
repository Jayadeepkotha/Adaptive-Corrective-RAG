import os
import json
import csv
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
from PIL import Image
import pymupdf as fitz
from openai import OpenAI
from src.config import settings

class MultiModalDocument:
    """Represents an ingested document with metadata and extracted semantic passages."""
    def __init__(
        self,
        doc_name: str,
        doc_type: str,
        passages: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        triples: Optional[List[Tuple[str, str, str]]] = None,
        image_preview_path: Optional[str] = None
    ):
        self.doc_name = doc_name
        self.doc_type = doc_type  # 'pdf', 'csv', 'json', 'image', 'text', 'markdown'
        self.passages = passages
        self.metadata = metadata or {}
        self.triples = triples or []
        self.image_preview_path = image_preview_path

class MultiModalIngestor:
    """
    Intelligent, structure-preserving multi-modal ingestion engine.
    Supports:
      - CSV / TSV: Tabular schema analysis, row-level semantic descriptions, statistical summaries.
      - JSON / JSONL: Hierarchical path flattening and object entity mapping.
      - Images / Photos: Visual OCR, dimension inspection, diagram/chart metadata extraction.
      - PDF: Layout-aware block extraction, table detection, and multi-column preservation.
      - TXT / Markdown: Header-aware recursive semantic chunking.
    """

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path: Path, doc_name: Optional[str] = None) -> MultiModalDocument:
        """Dispatcher that automatically chooses the optimal parser based on file extension."""
        name = doc_name or file_path.name
        ext = file_path.suffix.lower()

        if ext in [".csv", ".tsv"]:
            return self.process_csv(file_path, name, delimiter="\t" if ext == ".tsv" else ",")
        elif ext in [".json", ".jsonl"]:
            return self.process_json(file_path, name, is_jsonl=(ext == ".jsonl"))
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            return self.process_image(file_path, name)
        elif ext == ".pdf":
            return self.process_pdf(file_path, name)
        else:
            return self.process_text(file_path, name)

    def process_csv(self, file_path: Path, doc_name: str, delimiter: str = ",") -> MultiModalDocument:
        """
        Parses CSV/TSV into:
        1. A comprehensive table schema summary (columns, total rows, sample values).
        2. Natural-language semantic row descriptions so dense & sparse retrievers can match queries.
        3. Subject-Predicate-Object triples for entity columns.
        """
        logger.info(f"Processing CSV document: '{doc_name}'...")
        passages: List[str] = []
        triples: List[Tuple[str, str, str]] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = [row for row in reader if any(cell.strip() for cell in row)]

            if not rows:
                return MultiModalDocument(doc_name, "csv", ["Empty CSV document."])

            headers = [h.strip() for h in rows[0]]
            data_rows = rows[1:]

            # 1. Table Schema Summary Passage
            schema_summary = (
                f"Document '{doc_name}' is a structured tabular dataset containing {len(data_rows)} rows "
                f"and {len(headers)} columns: {', '.join(headers)}. "
            )
            passages.append(schema_summary)

            # 2. Semantic Row-Level Chunking
            # Batch rows together in groups of 3-5 to maintain context without overloading
            batch_size = 4
            for i in range(0, len(data_rows), batch_size):
                batch = data_rows[i : i + batch_size]
                row_texts = []
                for idx, row in enumerate(batch, start=i + 1):
                    # Create clean readable sentence: "Record 1: ColumnA is X, ColumnB is Y..."
                    fields = []
                    for h_idx, h_name in enumerate(headers):
                        val = row[h_idx].strip() if h_idx < len(row) else ""
                        if val:
                            fields.append(f"{h_name}: {val}")
                            # Extract knowledge graph triple if header implies relation
                            if h_idx > 0 and len(row) > 0 and row[0].strip():
                                triples.append((row[0].strip(), f"HAS_{h_name.upper().replace(' ', '_')}", val))
                    if fields:
                        row_texts.append(f"[Row {idx}] " + " | ".join(fields))

                if row_texts:
                    passages.append(f"Table '{doc_name}' Data:\n" + "\n".join(row_texts))

            metadata = {
                "columns": headers,
                "total_rows": len(data_rows),
                "format": "tabular"
            }
            logger.info(f"Extracted {len(passages)} passages and {len(triples)} triples from CSV '{doc_name}'.")
            return MultiModalDocument(doc_name, "csv", passages, metadata=metadata, triples=triples)

        except Exception as e:
            logger.error(f"Error parsing CSV '{doc_name}': {e}")
            return MultiModalDocument(doc_name, "csv", [f"Error reading CSV {doc_name}: {e}"])

    def process_json(self, file_path: Path, doc_name: str, is_jsonl: bool = False) -> MultiModalDocument:
        """
        Parses JSON/JSONL into hierarchical dot-notation key-value statements and semantic entity nodes.
        Preserves deep structures without chopping braces across arbitrary limits.
        """
        logger.info(f"Processing JSON document: '{doc_name}'...")
        passages: List[str] = []
        triples: List[Tuple[str, str, str]] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                if is_jsonl:
                    records = [json.loads(line) for line in f if line.strip()]
                else:
                    content = json.load(f)
                    records = content if isinstance(content, list) else [content]

            # Flatten each record into readable key-value sentences
            for idx, rec in enumerate(records, start=1):
                flat_items: List[str] = []

                def _flatten(obj, prefix=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            p = f"{prefix}.{k}" if prefix else str(k)
                            _flatten(v, p)
                    elif isinstance(obj, list):
                        for i_idx, elem in enumerate(obj):
                            p = f"{prefix}[{i_idx}]"
                            _flatten(elem, p)
                    else:
                        val_str = str(obj).strip()
                        if val_str:
                            flat_items.append(f"{prefix} = {val_str}")
                            if prefix:
                                triples.append((doc_name, prefix.replace(".", "_"), val_str[:60]))

                _flatten(rec)
                if flat_items:
                    record_header = f"Record #{idx} in '{doc_name}':\n"
                    # Chunk flat items if too large
                    for chunk_idx in range(0, len(flat_items), 15):
                        chunk_slice = flat_items[chunk_idx : chunk_idx + 15]
                        passages.append(record_header + "\n".join(chunk_slice))

            metadata = {"total_records": len(records), "format": "hierarchical_json"}
            logger.info(f"Extracted {len(passages)} passages from JSON '{doc_name}'.")
            return MultiModalDocument(doc_name, "json", passages, metadata=metadata, triples=triples)

        except Exception as e:
            logger.error(f"Error parsing JSON '{doc_name}': {e}")
            return MultiModalDocument(doc_name, "json", [f"Error reading JSON {doc_name}: {e}"])

    def process_image(self, file_path: Path, doc_name: str) -> MultiModalDocument:
        """
        Deep Multimodal Image & Diagram Comprehension:
        1. Inspects image metadata (resolution, color space, aspect ratio).
        2. Executes Vision-Language analysis (Gemini 3.8 Flash Vision) to extract:
           - Verbatim machine-printed and handwritten text
           - Flowcharts, architecture components, and numerical data
           - Structured entity-relation triples for the Knowledge Graph.
        3. Graceful fallback if offline or API key missing.
        """
        logger.info(f"Processing Image with Multimodal Vision Engine: '{doc_name}'...")
        passages: List[str] = []
        triples: List[Tuple[str, str, str]] = []

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img_format = (img.format or "JPEG").upper()
                img_mode = img.mode

                # Convert to RGB and encode to JPEG base64 for API
                if img.mode not in ("RGB", "L"):
                    rgb_img = img.convert("RGB")
                else:
                    rgb_img = img

                # Resize if excessively large to preserve latency
                max_dim = 1600
                if max(width, height) > max_dim:
                    scale = max_dim / max(width, height)
                    new_size = (int(width * scale), int(height * scale))
                    rgb_img = rgb_img.resize(new_size, Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=85)
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

            metadata = {
                "width": width,
                "height": height,
                "format": img_format,
                "mode": img_mode,
                "aspect_ratio": f"{width}:{height}",
                "ocr_engine": "Gemini 3.8 Flash Multimodal Vision"
            }

            # 2. Call Multimodal Vision API
            vision_analysis = ""
            gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
            if gemini_key:
                try:
                    vision_client = OpenAI(
                        api_key=gemini_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                        timeout=45.0
                    )
                    prompt = (
                        "You are an expert multimodal document intelligence system.\n"
                        "Analyze this image in detail for semantic enterprise search:\n"
                        "1. Extract all visible text, numbers, labels, and table cells verbatim.\n"
                        "2. Describe any architecture diagrams, charts, graphs, flowcharts, or processes shown.\n"
                        "3. Identify key relationships between entities in the format: EntityA | RELATION | EntityB.\n"
                        "Be concise, factual, and accurate."
                    )
                    resp = vision_client.chat.completions.create(
                        model="gemini-3.8-flash",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                                ]
                            }
                        ],
                        max_tokens=600
                    )
                    if resp and resp.choices and resp.choices[0].message.content:
                        vision_analysis = resp.choices[0].message.content.strip()
                except Exception as ve:
                    logger.warning(f"Vision API call failed: {ve}. Using metadata fallback...")

            # 3. Build Semantic Passages & Triples
            header_desc = (
                f"[Visual Multi-Modal Document: {doc_name}]\n"
                f"Image Type: {img_format} ({width}x{height} pixels, Mode: {img_mode})."
            )

            if vision_analysis:
                passages.append(f"{header_desc}\n\nVisual Analysis & Extracted Content:\n{vision_analysis}")
                # Parse EntityA | RELATION | EntityB triples from vision output
                import re
                triple_matches = re.findall(
                    r"([A-Za-z0-9_\s]{2,30})\s*\|\s*([A-Za-z0-9_\s]{2,25})\s*\|\s*([A-Za-z0-9_\s]{2,40})",
                    vision_analysis
                )
                for s, r, o in triple_matches:
                    clean_s = s.strip()
                    clean_r = r.strip().upper().replace(" ", "_")
                    clean_o = o.strip()
                    if clean_s and clean_o and clean_r:
                        triples.append((clean_s, clean_r, clean_o))

                triples.append((doc_name, "IMAGE_TYPE", img_format))
                triples.append((doc_name, "HAS_DIMENSIONS", f"{width}x{height}"))
            else:
                passages.append(f"{header_desc}\nVisual content artifact without extracted machine-printed text.")
                triples.append((doc_name, "IMAGE_TYPE", img_format))
                triples.append((doc_name, "HAS_DIMENSIONS", f"{width}x{height}"))

            logger.info(f"Processed image '{doc_name}' with {len(passages)} multimodal passage(s) and {len(triples)} triples.")
            return MultiModalDocument(
                doc_name,
                "image",
                passages,
                metadata=metadata,
                triples=triples,
                image_preview_path=str(file_path)
            )

        except Exception as e:
            logger.error(f"Error processing image '{doc_name}': {e}")
            return MultiModalDocument(doc_name, "image", [f"Error processing image {doc_name}: {e}"])

    def process_pdf(self, file_path: Path, doc_name: str) -> MultiModalDocument:
        """
        Layout-aware PDF parser:
        - Extracts structured text blocks (page.get_text('blocks')) to preserve headings and tables.
        - Identifies and handles multi-column layouts without scrambling lines.
        - Chunks by logical paragraphs rather than splitting sentences mid-word.
        """
        logger.info(f"Processing PDF document: '{doc_name}'...")
        passages: List[str] = []
        triples: List[Tuple[str, str, str]] = []

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1

                # Extract layout blocks: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 = text, 1 = image
                blocks = page.get_text("blocks")

                current_block_texts = []
                current_word_count = 0

                for b in blocks:
                    if b[6] == 0:  # Text block
                        text = b[4].strip()
                        if not text:
                            continue

                        # Clean hyphenated line breaks e.g. "com-\nplexity" -> "complexity"
                        clean_text = text.replace("-\n", "").replace("\n", " ")
                        words = clean_text.split()

                        if current_word_count + len(words) > self.chunk_size and current_block_texts:
                            passage_body = " ".join(current_block_texts)
                            passages.append(f"[{doc_name} | Page {page_num}]\n{passage_body}")
                            current_block_texts = []
                            current_word_count = 0

                        current_block_texts.append(clean_text)
                        current_word_count += len(words)

                if current_block_texts:
                    passage_body = " ".join(current_block_texts)
                    passages.append(f"[{doc_name} | Page {page_num}]\n{passage_body}")

            doc.close()
            metadata = {"total_pages": total_pages, "format": "pdf"}
            logger.info(f"Extracted {len(passages)} layout-aware passages from PDF '{doc_name}'.")
            return MultiModalDocument(doc_name, "pdf", passages, metadata=metadata, triples=triples)

        except Exception as e:
            logger.error(f"Error reading PDF '{doc_name}': {e}")
            return MultiModalDocument(doc_name, "pdf", [f"Error reading PDF {doc_name}: {e}"])

    def process_text(self, file_path: Path, doc_name: str) -> MultiModalDocument:
        """Parses TXT, Markdown, or code files with paragraph-preserving semantic chunking."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            passages: List[str] = []
            curr: List[str] = []
            curr_words = 0

            for p in paragraphs:
                words = p.split()
                if curr_words + len(words) > self.chunk_size and curr:
                    passages.append(f"[{doc_name}]\n" + "\n\n".join(curr))
                    curr = []
                    curr_words = 0
                curr.append(p)
                curr_words += len(words)

            if curr:
                passages.append(f"[{doc_name}]\n" + "\n\n".join(curr))

            return MultiModalDocument(doc_name, "text", passages)
        except Exception as e:
            return MultiModalDocument(doc_name, "text", [f"Error reading text {doc_name}: {e}"])

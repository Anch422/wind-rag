from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import re
from functools import lru_cache
import pandas as pd


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    kind: str
    farm: str
    answer: str = ""


def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", encoding="latin1", **kwargs)


def corpus_digest(docs):
    return hashlib.sha256(json.dumps([asdict(d) | {"source": Path(d.source).name}
                                     for d in docs], sort_keys=True).encode()).hexdigest()


def load_report_documents(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Technician reports folder does not exist: {root}")
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in sorted(root.rglob('*.pdf')))
    return list(_load_report_documents_cached(str(root), signature))


@lru_cache(maxsize=8)
def _load_report_documents_cached(root, signature):
    """Read searchable PDFs with file, page and chunk provenance."""
    from pypdf import PdfReader
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Technician reports folder does not exist: {root}")
    docs = []
    for filename, _, _ in signature:
        path = Path(filename)
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(''):
                raise ValueError('Password-protected PDF')
            name_key = hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()[:12]
            extracted = 0
            for page_number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or '').strip()
                if not text:
                    continue
                extracted += 1
                farm_match = re.search(r'Wind Farm [A-Z]', text)
                farm = farm_match.group() if farm_match else 'Technician reports'
                report_match = re.search(r'Report ID:\s*([A-Za-z0-9-]+)', text)
                report_id = report_match.group(1) if report_match else path.stem
                words = text.split()
                for chunk_number, start in enumerate(range(0, len(words), 250), 1):
                    chunk = ' '.join(words[start:start + 300])
                    content = f'Technician report {report_id}. {farm}. PDF page {page_number}. {chunk}'
                    answer_match = re.search(r'(?:Repair record|Possible cause)\s*(.*?)(?:Verification|Cause status|Safety and limitations|Parts and materials|Additional notes|$)', chunk, re.I)
                    answer = answer_match.group(1).strip() if answer_match else chunk
                    docs.append(Document(f'REPORT:{name_key}:p{page_number}:c{chunk_number}',
                                         f'{report_id} · {farm} · page {page_number}', content,
                                         str(path), 'report', farm, answer))
                    if start + 300 >= len(words):
                        break
            if not extracted:
                raise ValueError('No searchable text; scanned PDFs need OCR before import')
        except Exception as exc:
            raise ValueError(f'Cannot read technician PDF {path.name}: {exc}') from exc
    if not docs:
        raise ValueError('No searchable technician PDF reports found in the selected folder.')
    return tuple(docs)


def load_corpus(root, reports_root=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Select the extracted CARE_To_Compare folder.")
    docs, events, features = [], [], []
    for farm_dir in sorted(root.glob("Wind Farm *")):
        farm = farm_dir.name
        event_file = farm_dir / "event_info.csv"
        feature_file = farm_dir / "feature_description.csv"
        if not event_file.exists() or not feature_file.exists():
            raise ValueError(f"Missing metadata files in {farm}.")
        for _, row in read_csv(event_file).fillna("").iterrows():
            event = str(int(row["event_id"]))
            asset = str(row.get("asset_id", row.get("asset", "unknown")))
            label = str(row["event_label"])
            description = str(row["event_description"]).strip()
            answer = description or ("Normal operation" if label == "normal" else "Anomaly recorded; root cause not provided")
            doc_id = f"{farm[-1]}:event:{event}"
            text = (f"{farm}. Event {event}. Turbine asset {asset}. "
                    f"Start: {row['event_start']}. End: {row['event_end']}. "
                    f"Recorded label: {label}. Recorded description: {answer}.")
            docs.append(Document(doc_id, f"{farm} · Event {event}", text,
                                 str(event_file), "event", farm, answer))
            events.append({"document_id": doc_id, "farm": farm, "event_id": event,
                           "asset": asset, "label": label, "start": str(row["event_start"]),
                           "end": str(row["event_end"]), "description": answer,
                           "scada_path": str(farm_dir / "datasets" / f"{event}.csv")})
        for _, row in read_csv(feature_file).fillna("").iterrows():
            sensor = str(row["sensor_name"])
            doc_id = f"{farm[-1]}:sensor:{sensor}"
            description = str(row["description"])
            unit = str(row["unit"]).replace("�C", "°C")
            if unit == "�":
                unit = "°"
            stat = str(row.get("statistics_type", row.get("statistic_type", "")))
            text = (f"{farm}. Sensor {sensor}. Measurement: {description}. Unit: {unit}. "
                    f"Statistics available: {stat}. Angle: {row.get('is_angle', '')}. "
                    f"Counter: {row.get('is_counter', '')}.")
            docs.append(Document(doc_id, f"{farm} · {sensor}", text,
                                 str(feature_file), "sensor", farm, f"{description}; unit {unit}"))
            features.append({"document_id": doc_id, "farm": farm, "sensor": sensor,
                             "description": description, "unit": unit, "statistics": stat})
    if not events:
        raise ValueError("No CARE event metadata found. Select the folder containing Wind Farm A/B/C.")
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("Duplicate document IDs in metadata.")
    if reports_root is not None:
        docs.extend(load_report_documents(reports_root))
    digest = corpus_digest(docs)
    return docs, events, features, digest


def starter_questions(events, features, report_docs=None):
    questions = []
    for e in events:
        questions.append({"id": f"q-{e['document_id']}",
                          "question": f"What was recorded for turbine asset {e['asset']} at {e['farm']} in the event starting {e['start']}?",
                          "reference_answer": e["description"], "relevant_ids": [e["document_id"]],
                          "origin": "metadata-generated starter; requires human review"})
    for f in features[::10]:
        questions.append({"id": f"q-{f['document_id']}",
                          "question": f"What does {f['sensor']} measure at {f['farm']}, and what is its unit?",
                          "reference_answer": f"{f['description']}; unit {f['unit']}",
                          "relevant_ids": [f["document_id"]], "origin": "metadata-generated starter; requires human review"})
    for doc in report_docs or []:
        if doc.kind != 'report' or not any(section in doc.text for section in ('Repair record', 'Possible cause')):
            continue
        subject = 'repair' if 'Repair record' in doc.text else 'possible cause'
        report_id = doc.title.split(' · ')[0]
        questions.append({'id': f'q-{doc.id}',
                          'question': f'According to technician report {report_id} at {doc.farm}, what {subject} was documented?',
                          'reference_answer': doc.answer, 'relevant_ids': [doc.id], 'farm': doc.farm,
                          'origin': 'report-derived question; requires human review'})
    return questions


def load_questions(path, valid_ids):
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        rows = pd.read_csv(path, encoding="utf-8-sig").fillna("").to_dict("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Benchmark must contain a nonempty list of questions.")
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Question {i + 1} must be an object.")
        row.setdefault("id", f"custom-{i + 1}")
        row["id"] = str(row["id"])
        if row["id"] in seen:
            raise ValueError(f"Duplicate question ID: {row['id']}")
        seen.add(row["id"])
        if not str(row.get("question", "")).strip() or "reference_answer" not in row or "relevant_ids" not in row:
            raise ValueError("Each question needs question, reference_answer, and relevant_ids.")
        row["question"] = str(row["question"]).strip()
        row["reference_answer"] = str(row["reference_answer"]).strip()
        if isinstance(row["relevant_ids"], str):
            raw = row["relevant_ids"].strip()
            row["relevant_ids"] = json.loads(raw) if raw.startswith("[") else [x.strip() for x in raw.split("|") if x.strip()]
        if not isinstance(row["relevant_ids"], list):
            raise ValueError("relevant_ids must be a JSON list or pipe-separated IDs.")
        unknown = set(row["relevant_ids"]) - set(valid_ids)
        if unknown:
            raise ValueError(f"Unknown relevant document IDs: {sorted(unknown)}")
        row.setdefault("origin", "user supplied; review status unspecified")
    return rows

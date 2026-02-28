from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd
from cryptography.fernet import Fernet
from PIL import Image

try:
    import pytesseract
except Exception:  # pragma: no cover - optional runtime dependency
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover - optional runtime dependency
    convert_from_bytes = None

REPORT_PATTERN = {
    "Hemoglobin": r"hemoglobin\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "WBC": r"wbc\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "RBC": r"rbc\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "Platelets": r"platelets?\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "Cholesterol": r"cholesterol\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "HDL": r"hdl\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "LDL": r"ldl\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "Triglycerides": r"triglycerides?\s*[:\-]?\s*([0-9]+\.?[0-9]*)",
    "Age": r"age\s*[:\-]?\s*([0-9]+)",
    "Gender": r"gender\s*[:\-]?\s*(male|female|other)",
}


@dataclass
class StorageConfig:
    db_path: str = "data/health_reports.db"
    key_path: str = "data/.fernet.key"


def _load_or_create_key(key_path: str) -> bytes:
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    return key


def encrypt_payload(payload: bytes, config: StorageConfig) -> bytes:
    f = Fernet(_load_or_create_key(config.key_path))
    return f.encrypt(payload)


def decrypt_payload(payload: bytes, config: StorageConfig) -> bytes:
    f = Fernet(_load_or_create_key(config.key_path))
    return f.decrypt(payload)


def init_db(config: StorageConfig) -> None:
    os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
    con = sqlite3.connect(config.db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            test_date TEXT,
            encrypted_csv BLOB
        )
        """
    )
    con.commit()
    con.close()


def save_report(df: pd.DataFrame, patient_id: str, test_date: str, config: StorageConfig) -> None:
    csv_blob = df.to_csv(index=False).encode("utf-8")
    enc_blob = encrypt_payload(csv_blob, config)

    con = sqlite3.connect(config.db_path)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO reports (patient_id, test_date, encrypted_csv) VALUES (?, ?, ?)",
        (patient_id, test_date, enc_blob),
    )
    con.commit()
    con.close()


def load_reports(patient_id: Optional[str], config: StorageConfig) -> pd.DataFrame:
    con = sqlite3.connect(config.db_path)
    query = "SELECT patient_id, test_date, encrypted_csv FROM reports"
    params = ()
    if patient_id:
        query += " WHERE patient_id = ?"
        params = (patient_id,)
    rows = con.execute(query, params).fetchall()
    con.close()

    frames = []
    for pid, test_date, blob in rows:
        raw = decrypt_payload(blob, config)
        frame = pd.read_csv(BytesIO(raw))
        frame["Patient_ID"] = pid
        frame["Test_Date"] = test_date
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def extract_text_from_upload(uploaded_bytes: bytes, filename: str) -> str:
    suffix = filename.lower().split(".")[-1]
    if suffix == "pdf":
        if convert_from_bytes is None or pytesseract is None:
            raise RuntimeError("pdf2image and pytesseract are required for PDF OCR")
        pages = convert_from_bytes(uploaded_bytes)
        return "\n".join(pytesseract.image_to_string(page) for page in pages)

    if suffix in {"png", "jpg", "jpeg"}:
        if pytesseract is None:
            raise RuntimeError("pytesseract is required for image OCR")
        image = Image.open(BytesIO(uploaded_bytes))
        return pytesseract.image_to_string(image)

    return uploaded_bytes.decode("utf-8", errors="ignore")


def parse_report_text(text: str) -> pd.DataFrame:
    normalized = text.lower()
    extracted: dict[str, object] = {
        "Test_Date": datetime.now().date().isoformat(),
        "Symptoms": "",
    }

    for field, pattern in REPORT_PATTERN.items():
        match = re.search(pattern, normalized)
        if not match:
            continue
        value = match.group(1)
        if field in {"Gender"}:
            extracted[field] = value.capitalize()
        elif field in {"Age"}:
            extracted[field] = int(value)
        else:
            extracted[field] = float(value)

    if "gender" not in normalized:
        extracted["Gender"] = "Female"
    if "age" not in normalized:
        extracted["Age"] = 30

    extracted.setdefault("Patient_ID", "P-UNKNOWN")
    return pd.DataFrame([extracted])

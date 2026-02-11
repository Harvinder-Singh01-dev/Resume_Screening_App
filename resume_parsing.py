import os
import re
import json
import threading
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from pypdf import PdfReader
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using pypdf.
    If the PDF is scanned (image-only), this will likely return little/no text.
    """
    reader = PdfReader(pdf_path)
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.replace("\x00", "").strip()
        if txt:
            parts.append(f"\n--- Page {i+1} ---\n{txt}")
    return "\n".join(parts).strip()


# ----------------------------
# BEDROCK / CLAUDE CALL
# ----------------------------
def bedrock_client(region: str):
    cfg = Config(connect_timeout=30, read_timeout=120, retries={"max_attempts": 5})
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def build_prompt(resume_text: str) -> str:
    schema = {
        "candidate": {
            "full_name": None,
            "department": None,  # keep in schema
            "email": None,
            "phone": None,
            "location": None,
            "headline": None,
            "summary": None,
            "total_years_experience": None,
            "links": {
                "linkedin": None,
                "github": None,
                "portfolio": None,
                "other": []
            }
        },
        "skills": {
            "core": [],
            "technical": [],
            "tools": [],
            "soft": [],
            "domains": []
        },
        "experience": [
            {
                "company": None,
                "job_title": None,
                "employment_type": None,
                "location": None,
                "start_date": None,
                "end_date": None,
                "is_current": None,
                "bullets": []
            }
        ],
        "education": [
            {
                "institution": None,
                "degree": None,
                "field_of_study": None,
                "location": None,
                "start_date": None,
                "end_date": None,
                "grade_or_gpa": None,
                "notes": []
            }
        ],
        "projects": [
            {
                "name": None,
                "role": None,
                "start_date": None,
                "end_date": None,
                "description": None,
                "technologies": [],
                "links": []
            }
        ],
        "certifications": [
            {
                "name": None,
                "issuer": None,
                "issue_date": None,
                "expiry_date": None,
                "credential_id": None,
                "credential_url": None
            }
        ],
        "awards": [
            {
                "name": None,
                "issuer": None,
                "date": None,
                "description": None
            }
        ],
        "publications": [
            {
                "title": None,
                "publisher_or_venue": None,
                "date": None,
                "url": None
            }
        ],
        "languages": [
            {
                "language": None,
                "proficiency": None
            }
        ],
        "metadata": {
            "source": "resume_pdf",
            "extraction_notes": [],
            "confidence": {
                "overall": None,
                "fields_with_low_confidence": []
            }
        }
    }

    instructions = f"""
You are an expert resume information extraction system.

TASK:
Extract all relevant information from the resume text and return it as STRICT JSON ONLY.
No commentary. No markdown. No extra keys beyond the schema.

RULES (VERY IMPORTANT):
1) Output must be valid JSON (double quotes, no trailing commas).
2) Use EXACT strings from the resume whenever possible.
3) If a field is missing, use null (for single values) or [] (for lists).
4) Do NOT invent details. Do NOT guess dates/companies/roles.
5) Normalize dates when possible:
   - Prefer "YYYY-MM" or "YYYY-MM-DD" if present
   - Otherwise keep as written (e.g., "Jun 2022")
6) For experience bullets, capture achievements/responsibilities as separate bullet strings.
7) For skills, de-duplicate and keep concise.
8) If multiple emails/phones exist, use the most prominent one in candidate.email/phone,
   and put others inside candidate.links.other (as strings) if relevant.
9) If you are uncertain about a value, put it in the correct field but add a note in:
   metadata.extraction_notes and list the field path under:
   metadata.confidence.fields_with_low_confidence
10) Extract total_years_experience as a number (integer or float) estimating overall years of professional experience as per conditions.
    - During professional experience period if candidate's degree period is overlapping then include the professional experience in the total experience.
    - Else do not include the the experience of study in total experience.
11) Return JSON that matches this schema exactly.

SCHEMA (copy structure exactly, fill values):
{json.dumps(schema, indent=2)}

RESUME TEXT:
\"\"\"{resume_text}\"\"\"
"""
    return instructions.strip()


def invoke_claude_extract_json(
    client,
    model_id: str,
    resume_text: str,
    max_tokens: int = 3000,
    temperature: float = 0.0,
) -> str:
    prompt = build_prompt(resume_text)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )

    payload = json.loads(resp["body"].read().decode("utf-8"))
    parts = payload.get("content", [])
    return "".join([p.get("text", "") for p in parts if p.get("type") == "text"]).strip()


# ----------------------------
# JSON SAFETY: PARSE + REPAIR
# ----------------------------
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def try_parse_json(model_output: str) -> Dict[str, Any]:
    try:
        return json.loads(model_output)
    except json.JSONDecodeError:
        pass

    m = JSON_BLOCK_RE.search(model_output)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Model output is not valid JSON and no JSON object could be extracted.")


def reask_for_valid_json(
    client,
    model_id: str,
    resume_text: str,
    bad_output: str,
    max_tokens: int = 3000,
) -> str:
    repair_prompt = f"""
You previously returned output that was not valid JSON.

Return STRICT valid JSON ONLY, following the same schema, with no extra text.

Here is the invalid output:
\"\"\"{bad_output}\"\"\"

Here is the resume text again:
\"\"\"{resume_text}\"\"\"
""".strip()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}],
    }

    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )

    payload = json.loads(resp["body"].read().decode("utf-8"))
    parts = payload.get("content", [])
    return "".join([p.get("text", "") for p in parts if p.get("type") == "text"]).strip()


# ----------------------------
# DEPARTMENT FROM SUBFOLDER
# ----------------------------
def get_department_from_path(pdf_path: str, resumes_root: str) -> str:
    rel = os.path.relpath(pdf_path, resumes_root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "Unknown"


# ----------------------------
# PIPELINE: PROCESS ONE PDF
# ----------------------------
def process_resume_pdf(
    client,
    model_id: str,
    pdf_path: str,
    out_dir: str,
    department: str,
) -> Optional[str]:
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        return None

    raw = invoke_claude_extract_json(client, model_id, resume_text)

    try:
        data = try_parse_json(raw)
    except Exception:
        repaired = reask_for_valid_json(client, model_id, resume_text, raw)
        data = try_parse_json(repaired)

    # Inject department
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object at top level.")
    data.setdefault("candidate", {})
    if not isinstance(data["candidate"], dict):
        data["candidate"] = {}
    data["candidate"]["department"] = department

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{base}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path


# ----------------------------
# THREADED MAIN
# ----------------------------
_thread_local = threading.local()
_print_lock = threading.Lock()


def get_thread_client(region: str):
    """Create one Bedrock client per thread (safe + avoids shared state surprises)."""
    c = getattr(_thread_local, "client", None)
    if c is None:
        c = bedrock_client(region)
        _thread_local.client = c
    return c


def main():
    load_dotenv()

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    resumes_dir = os.getenv("RESUMES_DIR", "Fallback_Resumes")
    outputs_dir = os.getenv("OUTPUTS_DIR", "New_Resume_Json")

    max_workers = int(os.getenv("MAX_WORKERS", "8"))

    if not os.path.isdir(resumes_dir):
        raise FileNotFoundError(f"Resumes folder not found: {resumes_dir}")

    pdfs: List[str] = []
    for root, _, files in os.walk(resumes_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    if not pdfs:
        print(f"[INFO] No PDFs found under {resumes_dir}")
        return

    total = len(pdfs)
    print(f"[INFO] Found {total} PDF(s). Using threads: {max_workers}. Model: {model_id}")

    # Worker function for threads
    def worker(pdf_path: str) -> tuple[str, Optional[str], Optional[str]]:
        dept = get_department_from_path(pdf_path, resumes_dir)
        try:
            client = get_thread_client(region)
            out_path = process_resume_pdf(client, model_id, pdf_path, outputs_dir, dept)
            return (pdf_path, out_path, None)
        except Exception as e:
            return (pdf_path, None, str(e))

    done = 0
    ok = 0
    failed = 0
    no_text = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, p) for p in pdfs]

        for fut in as_completed(futures):
            pdf_path, out_path, err = fut.result()
            done += 1

            with _print_lock:
                if err:
                    failed += 1
                    print(f"[{done}/{total}] [ERROR] {pdf_path}: {err}")
                elif out_path is None:
                    no_text += 1
                    print(f"[{done}/{total}] [WARN] No extractable text: {pdf_path}")
                else:
                    ok += 1
                    print(f"[{done}/{total}] [OK] {pdf_path} -> {out_path}")

    print(
        f"\n[SUMMARY] Total: {total} | OK: {ok} | No-text: {no_text} | Failed: {failed}"
    )


if __name__ == "__main__":
    main()

"""
bulk_index.py
─────────────────────────────────────────────────────────────────────────────
Reads ALL parsed-resume JSON files from S3 (PARSED_RESUME_BUCKET) and
indexes them into Qdrant with concurrent parallel processing.

Usage:
    cd /home/ubuntu/Resume_Screening_App
    source venv/bin/activate
    python bulk_index.py

Optional env overrides:
    PROCESS_MAX_WORKERS=8   # parallel threads (default 8)
    DRY_RUN=true            # list files only, no indexing
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── .env loading ─────────────────────────────────────────────────────────────
# Manually parse .env so this works from any working directory
# (find_dotenv() breaks when called outside a project tree)
_env_candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.getcwd(), ".env"),
    os.path.expanduser("~/.env"),
]
for _ep in _env_candidates:
    if os.path.exists(_ep):
        with open(_ep) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
        print(f"✅ .env loaded from: {_ep}")
        break
else:
    print("⚠️  No .env file found — relying on system environment variables")

import boto3
from botocore.config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ── Config from env ──────────────────────────────────────────────────────────
AWS_REGION          = os.getenv("AWS_REGION",           "us-east-1")
S3_REGION           = os.getenv("S3_REGION",            "ap-south-1")
EMBED_MODEL_ID      = os.getenv("EMBED_MODEL_ID",       "amazon.titan-embed-text-v1")
LLM_MODEL_ID        = os.getenv("BEDROCK_MODEL_ID",     "anthropic.claude-3-5-sonnet-20240620-v1:0")
PARSED_BUCKET       = os.getenv("PARSED_RESUME_BUCKET", "")
QDRANT_HOST         = os.getenv("QDRANT_HOST",          "localhost")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT",      "6333"))
COLLECTION          = os.getenv("QDRANT_COLLECTION",    "candidates")
MAX_WORKERS         = int(os.getenv("PROCESS_MAX_WORKERS", "8"))
DRY_RUN             = os.getenv("DRY_RUN", "false").lower() == "true"
JSON_PREFIX         = os.getenv("PARSED_RESUME_PREFIX", "")   # leave blank = entire bucket

if not PARSED_BUCKET:
    raise EnvironmentError(
        "PARSED_RESUME_BUCKET is not set. "
        "Add it to your .env file: PARSED_RESUME_BUCKET=your-bucket-name"
    )

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── AWS / Qdrant clients ─────────────────────────────────────────────────────
_boto_cfg = Config(
    connect_timeout=30,
    read_timeout=120,
    retries={"max_attempts": 5},
)

# No verify=False — it breaks SigV4 signing used by EC2 IAM roles
s3_client       = boto3.client("s3",               region_name=S3_REGION,  config=_boto_cfg)
bedrock_runtime = boto3.client("bedrock-runtime",  region_name=AWS_REGION, config=_boto_cfg)
qdrant_client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

log.info("S3 region      : %s", S3_REGION)
log.info("Bedrock region : %s", AWS_REGION)
log.info("Parsed bucket  : %s", PARSED_BUCKET)
log.info("Qdrant         : %s:%s  collection=%s", QDRANT_HOST, QDRANT_PORT, COLLECTION)
log.info("Workers        : %d", MAX_WORKERS)
log.info("Dry run        : %s", DRY_RUN)

# ── Bedrock embedding ────────────────────────────────────────────────────────
def bedrock_embed(text: str) -> List[float]:
    text = (text or "").strip() or "N/A"
    resp = bedrock_runtime.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


# ── Qdrant collection bootstrap ──────────────────────────────────────────────
def ensure_collection() -> int:
    log.info("Checking Qdrant collection '%s' …", COLLECTION)
    dim = len(bedrock_embed("dimension probe"))

    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION in existing:
        log.info("Collection already exists (dim=%d)", dim)
        return dim

    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "v_profile":    qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            "v_skills":     qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            "v_experience": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            "v_education":  qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        },
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=256),
    )
    log.info("✅ Created collection '%s' (dim=%d)", COLLECTION, dim)
    return dim


# ── Text builders ────────────────────────────────────────────────────────────
def _j(lst: Any) -> str:
    return "; ".join(str(x).strip() for x in (lst or []) if str(x).strip())

def build_profile_text(r: Dict) -> str:
    c = r.get("candidate", {}) or {}
    return (
        f"Department: {c.get('department','')}\n"
        f"Location: {c.get('location','')}\n"
        f"Total experience: {c.get('total_years_experience','')} years\n"
        f"Summary: {c.get('summary','')}"
    )

def build_skills_text(r: Dict) -> str:
    s = r.get("skills", {}) or {}
    return (
        f"Core skills: {_j(s.get('core'))}\n"
        f"Technical skills: {_j(s.get('technical'))}\n"
        f"Tools: {_j(s.get('tools'))}"
    )

def build_experience_text(r: Dict) -> str:
    parts = []
    for e in (r.get("experience", []) or []):
        bullets = " | ".join((e.get("bullets") or [])[:12])
        parts.append(
            f"Role: {e.get('job_title','')} | Company: {e.get('company','')} | "
            f"{e.get('start_date','')} to {e.get('end_date') or 'Present'} | "
            f"Highlights: {bullets}"
        )
    return "\n".join(parts)

def build_education_text(r: Dict) -> str:
    lines = ["Education:"]
    for e in (r.get("education", []) or []):
        lines.append(
            f"{e.get('degree','')} {e.get('field_of_study','') or ''}, "
            f"{e.get('institution','')}, {e.get('end_date','') or ''} "
            f"(Grade: {e.get('grade_or_gpa','')})"
        )
    certs = r.get("certifications", []) or []
    lines.append("Certifications:")
    lines += [str(c) for c in certs] if certs else ["None"]
    return "\n".join(lines)


# ── Payload builder ──────────────────────────────────────────────────────────
def _norm(xs: List) -> List[str]:
    return sorted({str(x).strip().lower() for x in (xs or []) if str(x).strip()})

def build_payload(r: Dict) -> Dict:
    c   = r.get("candidate", {})   or {}
    s   = r.get("skills", {})      or {}
    exp = r.get("experience", [])  or []
    edu = r.get("education", [])   or []

    return {
        "full_name":              c.get("full_name"),
        "email":                  c.get("email"),
        "phone":                  c.get("phone"),
        "location":               c.get("location"),
        "department":             c.get("department"),
        "total_years_experience": c.get("total_years_experience"),
        "original_filename":      c.get("original_filename", ""),
        "raw_resume_bucket":      c.get("raw_resume_bucket"),
        "raw_resume_key":         c.get("raw_resume_key"),
        "parsed_resume_bucket":   c.get("parsed_resume_bucket"),
        "parsed_resume_key":      c.get("parsed_resume_key"),
        "skills_flat":  _norm(
            (s.get("core") or []) + (s.get("technical") or []) +
            (s.get("tools") or []) + (s.get("soft") or []) + (s.get("domains") or [])
        ),
        "companies":   _norm([e.get("company", "") for e in exp]),
        "job_titles":  _norm([e.get("job_title", "") for e in exp]),
        "degrees":     _norm([e.get("degree", "") for e in edu]),
        "candidate_json":       c,
        "skills_json":          s,
        "experience_json":      exp,
        "education_json":       edu,
        "certifications_json":  r.get("certifications", []),
        "metadata":             r.get("metadata", {}),
    }


# ── Stable UUID from email / phone / name ────────────────────────────────────
def stable_point_id(r: Dict) -> str:
    c   = r.get("candidate", {}) or {}
    key = c.get("email") or c.get("phone") or c.get("full_name") or str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(key).strip().lower()))


# ── Upsert one resume JSON into Qdrant ───────────────────────────────────────
def upsert_resume(resume_json: Dict) -> str:
    pid = stable_point_id(resume_json)
    qdrant_client.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(
            id=pid,
            vector={
                "v_profile":    bedrock_embed(build_profile_text(resume_json)),
                "v_skills":     bedrock_embed(build_skills_text(resume_json)),
                "v_experience": bedrock_embed(build_experience_text(resume_json)),
                "v_education":  bedrock_embed(build_education_text(resume_json)),
            },
            payload=build_payload(resume_json),
        )]
    )
    return pid


# ── S3 helpers ───────────────────────────────────────────────────────────────
def list_s3_json_keys(bucket: str, prefix: str = "") -> List[str]:
    """
    Returns all .json keys in the bucket (handles S3 pagination automatically).
    """
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                keys.append(key)
    return keys


def read_s3_json(bucket: str, key: str) -> Dict:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


# ── Worker: index one S3 JSON file ──────────────────────────────────────────
def index_one(bucket: str, key: str) -> Tuple[str, str]:
    """
    Returns ("ok", point_id) or raises on failure.
    """
    resume_json = read_s3_json(bucket, key)
    pid = upsert_resume(resume_json)
    return "ok", pid


# ── Main bulk indexer ────────────────────────────────────────────────────────
def bulk_index():
    # 1. Bootstrap collection
    ensure_collection()

    # 2. List all JSON files in the parsed bucket
    log.info("Listing JSON files in s3://%s/%s …", PARSED_BUCKET, JSON_PREFIX)
    keys = list_s3_json_keys(PARSED_BUCKET, prefix=JSON_PREFIX)

    if not keys:
        log.warning("No .json files found in s3://%s/%s", PARSED_BUCKET, JSON_PREFIX)
        return

    log.info("Found %d JSON file(s) to index", len(keys))

    if DRY_RUN:
        log.info("DRY RUN — listing keys only, no indexing performed:")
        for k in keys:
            print(f"  {k}")
        return

    # 3. Parallel indexing
    ok = failed = 0
    failed_keys: List[Tuple[str, str]] = []   # (key, error_message)
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_key = {
            executor.submit(index_one, PARSED_BUCKET, key): key
            for key in keys
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                _, pid = future.result()
                ok += 1
                log.info("✅ [%d/%d]  %s  →  %s", ok + failed, len(keys), key, pid)
            except Exception as exc:
                failed += 1
                failed_keys.append((key, str(exc)))
                log.error("❌ [%d/%d]  %s  →  %s", ok + failed, len(keys), key, exc)

    elapsed = time.time() - start

    # 4. Summary
    print("\n" + "═" * 60)
    print(f"  Total files  : {len(keys)}")
    print(f"  ✅ Indexed   : {ok}")
    print(f"  ❌ Failed    : {failed}")
    print(f"  ⏱  Time      : {elapsed:.1f}s")
    print("═" * 60)

    if failed_keys:
        print("\nFailed files:")
        for k, err in failed_keys:
            print(f"  • {k}\n    Error: {err}")

    # 5. Final Qdrant count
    try:
        count = qdrant_client.get_collection(COLLECTION).points_count
        print(f"\n✅ Qdrant collection '{COLLECTION}' now has {count} candidates.")
    except Exception as e:
        log.warning("Could not fetch final count: %s", e)


if __name__ == "__main__":
    bulk_index()
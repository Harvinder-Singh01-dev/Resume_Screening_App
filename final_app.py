import tempfile
import os
import re
import json
import uuid
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import boto3
from botocore.config import Config
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ============================================================
# STREAMLIT CONFIG + STYLES (keeps your existing UI intact)
# ============================================================
st.set_page_config(page_title="Resume Screening Portal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stButton > button {
        background-color: #00B4D8;
        color: white;
        border-radius: 20px;
        padding: 0.3rem 1.5rem;
        font-weight: 500;
        border: none;
        font-size: 0.9em;
    }
    .stButton > button:hover { background-color: #0096C7; }
    .candidate-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #00B4D8;
    }
    .skill-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        background-color: #e3f2fd;
        color: #1976d2;
        border-radius: 15px;
        font-size: 0.85em;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .experience-item {
        border-bottom: 1px solid #e0e0e0;
        padding: 10px 0;
    }
    .experience-item:last-child { border-bottom: none; }
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stExpander > div:first-child {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        background-color: transparent;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG (aligned with your current screening app)
# ============================================================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
LLM_MODEL_ID   = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "candidates")

# For "process resumes" throughput
MAX_WORKERS = int(os.getenv("PROCESS_MAX_WORKERS", "4"))

# ============================================================
# CLIENTS (cached)
# ============================================================
@st.cache_resource
def get_clients():
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    cfg = Config(connect_timeout=30, read_timeout=120, retries={"max_attempts": 5})
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=cfg)
    return qdrant, bedrock

client, bedrock_runtime = get_clients()

# ============================================================
# BEDROCK EMBEDDING (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def bedrock_embed(text: str) -> List[float]:
    text = (text or "").strip() or "N/A"
    body = json.dumps({"inputText": text})
    resp = bedrock_runtime.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    return data["embedding"]

# ============================================================
# QDRANT COLLECTION CREATION (no side effects)
# ============================================================
@st.cache_resource
def ensure_collection(collection_name: str) -> int:
    """
    Ensures the collection exists with the required named vectors.
    Returns embedding dimension.
    """
    dim = len(bedrock_embed("dimension check"))
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        return dim

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "v_profile": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            "v_skills": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            "v_experience": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            "v_education": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        },
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=256),
    )
    return dim

DIM = ensure_collection(COLLECTION)

# ============================================================
# RESUME PARSING (PDF -> TEXT -> LLM JSON)
# ============================================================
def extract_text_from_pdf(pdf_path: str) -> str:
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

def build_prompt(resume_text: str) -> str:
    schema = {
        "candidate": {
            "full_name": None,
            "department": None,
            "email": None,
            "phone": None,
            "location": None,
            "headline": None,
            "summary": None,
            "total_years_experience": None,
            "links": {"linkedin": None, "github": None, "portfolio": None, "other": []},
        },
        "skills": {"core": [], "technical": [], "tools": [], "soft": [], "domains": []},
        "experience": [{
            "company": None, "job_title": None, "employment_type": None, "location": None,
            "start_date": None, "end_date": None, "is_current": None, "bullets": []
        }],
        "education": [{
            "institution": None, "degree": None, "field_of_study": None, "location": None,
            "start_date": None, "end_date": None, "grade_or_gpa": None, "notes": []
        }],
        "projects": [{
            "name": None, "role": None, "start_date": None, "end_date": None,
            "description": None, "technologies": [], "links": []
        }],
        "certifications": [{
            "name": None, "issuer": None, "issue_date": None, "expiry_date": None,
            "credential_id": None, "credential_url": None
        }],
        "awards": [{"name": None, "issuer": None, "date": None, "description": None}],
        "publications": [{"title": None, "publisher_or_venue": None, "date": None, "url": None}],
        "languages": [{"language": None, "proficiency": None}],
        "metadata": {
            "source": "resume_pdf",
            "extraction_notes": [],
            "confidence": {"overall": None, "fields_with_low_confidence": []},
        },
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
""".strip()
    return instructions

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def try_parse_json(model_output: str) -> Dict[str, Any]:
    try:
        return json.loads(model_output)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(model_output)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Model output is not valid JSON and no JSON object could be extracted.")

def invoke_claude_extract_json(
    resume_text: str,
    max_tokens: int = 3000,
    temperature: float = 0.0,
) -> str:
    prompt = build_prompt(resume_text)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    resp = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))
    parts = payload.get("content", [])
    return "".join([p.get("text", "") for p in parts if p.get("type") == "text"]).strip()

def reask_for_valid_json(resume_text: str, bad_output: str, max_tokens: int = 3000) -> str:
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
    resp = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))
    parts = payload.get("content", [])
    return "".join([p.get("text", "") for p in parts if p.get("type") == "text"]).strip()

def parse_resume_pdf_to_json(pdf_path: str, department: str) -> Optional[Dict[str, Any]]:
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        return None

    raw = invoke_claude_extract_json(resume_text)
    try:
        data = try_parse_json(raw)
    except Exception:
        repaired = reask_for_valid_json(resume_text, raw)
        data = try_parse_json(repaired)

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object at top level.")

    data.setdefault("candidate", {})
    if not isinstance(data["candidate"], dict):
        data["candidate"] = {}
    data["candidate"]["department"] = department
    return data

# ============================================================
# INDEXING (resume JSON -> Qdrant named vectors)
# ============================================================
def _normalize_list(xs: List[str]) -> List[str]:
    out = []
    for x in xs or []:
        x = str(x).strip().lower()
        if x:
            out.append(x)
    return sorted(list(set(out)))

def _build_profile_text(r: Dict[str, Any]) -> str:
    c = r.get("candidate", {}) or {}
    return "\n".join([
        f"Department: {c.get('department','')}",
        f"Location: {c.get('location','')}",
        f"Total experience: {c.get('total_years_experience','')} years",
        f"Summary: {c.get('summary','')}",
    ]).strip()

def _build_skills_text(r: Dict[str, Any]) -> str:
    s = r.get("skills", {}) or {}
    def join_list(x):
        x = x or []
        return "; ".join([str(i).strip() for i in x if str(i).strip()])
    return "\n".join([
        f"Core skills: {join_list(s.get('core'))}",
        f"Technical skills: {join_list(s.get('technical'))}",
        f"Tools: {join_list(s.get('tools'))}",
    ]).strip()

def _build_experience_text(r: Dict[str, Any]) -> str:
    exp = r.get("experience", []) or []
    parts = []
    for e in exp:
        bullets = e.get("bullets", []) or []
        bullet_text = " | ".join(bullets[:12])
        parts.append(
            f"Role: {e.get('job_title','')} | Company: {e.get('company','')} | "
            f"Dates: {e.get('start_date','')} to {e.get('end_date') or 'Present'} | "
            f"Highlights: {bullet_text}"
        )
    return "\n".join(parts).strip()

def _build_education_text(r: Dict[str, Any]) -> str:
    edu = r.get("education", []) or []
    cert = r.get("certifications", []) or []
    lines = ["Education:"]
    for e in edu:
        lines.append(
            f"{e.get('degree','')} {e.get('field_of_study','') or ''}, {e.get('institution','')}, "
            f"{e.get('end_date','') or ''} (Grade: {e.get('grade_or_gpa','')})"
        )
    lines.append("Certifications:")
    if not cert:
        lines.append("None")
    else:
        for c in cert:
            lines.append(str(c))
    return "\n".join(lines).strip()

def _build_payload(r: Dict[str, Any]) -> Dict[str, Any]:
    c = r.get("candidate", {}) or {}
    s = r.get("skills", {}) or {}
    exp = r.get("experience", []) or []
    edu = r.get("education", []) or []

    skills_flat = _normalize_list(
        (s.get("core") or []) + (s.get("technical") or []) + (s.get("tools") or []) +
        (s.get("soft") or []) + (s.get("domains") or [])
    )
    companies = _normalize_list([e.get("company", "") for e in exp])
    job_titles = _normalize_list([e.get("job_title", "") for e in exp])
    degrees = _normalize_list([e.get("degree", "") for e in edu])

    return {
        "full_name": c.get("full_name"),
        "email": c.get("email"),
        "phone": c.get("phone"),
        "location": c.get("location"),
        "department": c.get("department"),
        "total_years_experience": c.get("total_years_experience"),
        "skills_flat": skills_flat,
        "companies": companies,
        "job_titles": job_titles,
        "degrees": degrees,

        "candidate_json": c,
        "skills_json": s,
        "experience_json": exp,
        "education_json": edu,
        "certifications_json": r.get("certifications", []),
        "metadata": r.get("metadata", {}),
    }

def _stable_point_id(resume_json: Dict[str, Any]) -> str:
    c = resume_json.get("candidate", {}) or {}
    key = c.get("email") or c.get("phone") or c.get("full_name") or str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(key).strip().lower()))

def upsert_resume(resume_json: Dict[str, Any]) -> str:
    pid = _stable_point_id(resume_json)

    profile_text = _build_profile_text(resume_json)
    skills_text = _build_skills_text(resume_json)
    experience_text = _build_experience_text(resume_json)
    education_text = _build_education_text(resume_json)

    vectors = {
        "v_profile": bedrock_embed(profile_text),
        "v_skills": bedrock_embed(skills_text),
        "v_experience": bedrock_embed(experience_text),
        "v_education": bedrock_embed(education_text),
    }

    payload = _build_payload(resume_json)

    client.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(id=pid, vector=vectors, payload=payload)],
    )
    return pid

# ============================================================
# SCREENING (your original logic untouched)
# ============================================================
def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return json.dumps(x, ensure_ascii=False).strip()

def simple_tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\+\#\.\-]{2,}", (text or "").lower())
    stop = {
        "and","or","the","to","of","in","for","with","a","an","is","are","on","as","at","by","be",
        "we","you","our","will","shall","from","this","that","it"
    }
    return [t for t in toks if t not in stop]

def fmt_years(x):
    try:
        return int(x)
    except Exception:
        return 0

@st.cache_data(show_spinner=False)
def bedrock_split_jd_4views(jd_text: str) -> Dict[str, str]:
    prompt = f"""
You are given a Job Description (JD).
Task: Extract ONLY what is explicitly present. Do NOT add any new facts.

Return STRICT JSON with keys:
- "profile_view": role title/seniority/location/department-type info if present
- "skills_view": required/preferred skills/tools/keywords
- "responsibilities_view": responsibilities/tasks/deliverables
- "education_view": degrees/certifications if present

JD:
\"\"\"{jd_text}\"\"\"
"""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 900,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    data = json.loads(resp["body"].read())
    out_text = data["content"][0]["text"].strip()

    try:
        out_json = json.loads(out_text)
        return {
            "profile_view": out_json.get("profile_view", ""),
            "skills_view": out_json.get("skills_view", ""),
            "responsibilities_view": out_json.get("responsibilities_view", ""),
            "education_view": out_json.get("education_view", ""),
        }
    except Exception:
        return {"profile_view": "", "skills_view": "", "responsibilities_view": jd_text, "education_view": ""}

def qdrant_vector_query(collection_name: str, vector_name: str, vector: List[float], limit: int, q_filter=None):
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection_name,
            query=vector,
            using=vector_name,
            limit=limit,
            with_payload=True,
            query_filter=q_filter
        )
        return res.points
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=(vector_name, vector),
            limit=limit,
            with_payload=True,
            query_filter=q_filter
        )
    if hasattr(client, "query"):
        return client.query(
            collection_name=collection_name,
            query_vector=(vector_name, vector),
            limit=limit,
            with_payload=True,
            query_filter=q_filter
        )
    raise RuntimeError("Unsupported qdrant-client API. Upgrade qdrant-client.")

def build_qdrant_filter(min_years: Optional[int], dept_contains: str, loc_contains: str):
    must = []

    if min_years is not None:
        must.append(qmodels.FieldCondition(
            key="total_years_experience",
            range=qmodels.Range(gte=min_years)
        ))

    if dept_contains.strip():
        must.append(qmodels.FieldCondition(
            key="department",
            match=qmodels.MatchText(text=dept_contains.strip())
        ))

    if loc_contains.strip():
        must.append(qmodels.FieldCondition(
            key="location",
            match=qmodels.MatchText(text=loc_contains.strip())
        ))

    if not must:
        return None

    return qmodels.Filter(must=must)

def extract_evidence_by_company(jd_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    jd_tokens = set(simple_tokenize(jd_text))
    skills_tokens = jd_tokens
    resp_tokens = jd_tokens

    skills_flat = payload.get("skills_flat", []) or []
    matched_skills = [s for s in skills_flat if s.lower() in jd_tokens or any(t in s.lower() for t in skills_tokens)]

    company_evidence = {}
    experience_json = payload.get("experience_json", []) or []

    for exp in experience_json:
        company = exp.get("company", "Unknown Company")
        job_title = exp.get("job_title", "")
        bullets = exp.get("bullets", []) or []

        matched_bullets = []
        for bullet in bullets:
            bullet_lower = bullet.lower()
            if any(token in bullet_lower for token in list(jd_tokens)[:100]) or \
               any(token in bullet_lower for token in skills_tokens) or \
               any(token in bullet_lower for token in resp_tokens):
                matched_bullets.append(bullet)

        if matched_bullets:
            if company not in company_evidence:
                company_evidence[company] = []
            company_evidence[company].append({
                "job_title": job_title,
                "matched_bullets": matched_bullets
            })

    return {
        "matched_skills": matched_skills[:20],
        "matched_skills_count": len(matched_skills),
        "company_evidence": company_evidence
    }

def search_candidates_from_jd(
    jd_text: str,
    top_k: int,
    per_vector_limit: int,
    prefer: str,
    q_filter=None,
):
    views = bedrock_split_jd_4views(jd_text)

    skills_q_text = safe_text(views.get("skills_view")) or jd_text
    exp_q_text    = safe_text(views.get("responsibilities_view")) or jd_text
    prof_q_text   = safe_text(views.get("profile_view")) or jd_text

    q_skills = bedrock_embed(skills_q_text)
    q_exp    = bedrock_embed(exp_q_text)
    q_prof   = bedrock_embed(prof_q_text)

    if prefer == "Skills-first":
        w_exp, w_sk = 0.45, 0.55
    elif prefer == "Experience-first":
        w_exp, w_sk = 0.70, 0.30
    else:
        w_exp, w_sk = 0.60, 0.40

    w_prof = 0.10

    res_exp = qdrant_vector_query(COLLECTION, "v_experience", q_exp, limit=per_vector_limit, q_filter=q_filter)
    res_sk  = qdrant_vector_query(COLLECTION, "v_skills", q_skills, limit=per_vector_limit, q_filter=q_filter)
    res_prof = qdrant_vector_query(COLLECTION, "v_profile", q_prof, limit=per_vector_limit, q_filter=q_filter)

    fused: Dict[str, Dict[str, Any]] = {}

    def add_results(results, weight, key):
        for r in results:
            pid = str(r.id)
            if pid not in fused:
                fused[pid] = {"id": pid, "score": 0.0, "payload": r.payload, "scores": {}}
            fused[pid]["score"] += weight * float(r.score)
            fused[pid]["scores"][key] = float(r.score)

    add_results(res_exp, w_exp, "experience_score")
    add_results(res_sk,  w_sk,  "skills_score")
    add_results(res_prof, w_prof, "profile_score")

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return ranked, views

def post_sort_results(results: List[Dict[str, Any]], jd_text: str, sort_mode: str) -> List[Dict[str, Any]]:
    if sort_mode == "Default (Relevance)":
        return results

    enriched = []
    for r in results:
        p = r["payload"]
        ev = extract_evidence_by_company(jd_text, p)
        years = fmt_years(p.get("total_years_experience"))
        enriched.append((r, ev, years))

    if sort_mode == "Years of experience (desc)":
        enriched.sort(key=lambda x: (x[2], x[0]["score"]), reverse=True)
    elif sort_mode == "Skills match count (desc)":
        enriched.sort(key=lambda x: (x[1]["matched_skills_count"], x[0]["score"]), reverse=True)

    return [x[0] for x in enriched]

def render_candidate_card(idx: int, result: Dict[str, Any], jd_text: str):
    p = result["payload"]
    years = fmt_years(p.get("total_years_experience"))
    dept = p.get("department") or "—"
    name = p.get("full_name") or "Candidate"
    score = result["score"]
    evidence = extract_evidence_by_company(jd_text, p)

    with st.container():
        st.markdown(f"""
        <div style="background: linear-gradient(to right, #f8f9fa, #ffffff);
                    border-left: 4px solid #00B4D8;
                    padding: 5px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <h3 style="margin: 0; color: #2c3e50;">{name}</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background: #e3f2fd; border-radius: 8px;">
                <h2 style="margin: 0; color: #1976d2;">{:.1%}</h2>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Score</p>
            </div>
            """.format(score), unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #f3e5f5; border-radius: 8px;">
                <h2 style="margin: 0; color: #7b1fa2;">{years}</h2>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Years Exp.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #e8f5e9; border-radius: 8px;">
                <h2 style="margin: 0; color: #388e3c;">{evidence['matched_skills_count']}</h2>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Matching Skills</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style="text-align: center; padding: 37px; background: #fff3e0; border-radius: 8px;">
                <p style="margin: 0; color: #e65100;font-weight: 600;">{dept}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Department</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

        left_col, right_col = st.columns([1.2, 2])

        with left_col:
            st.markdown("""
            <div style="background: #f5f7fa; padding: 15px; border-radius: 8px;">
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">🎯 Matching Skills</h5>
            </div>
            """, unsafe_allow_html=True)

            if evidence['matched_skills']:
                for skill in evidence['matched_skills'][:10]:
                    skill = skill[0].upper() + skill[1:].lower()
                    st.markdown(f"""
                    <div style="background: #e3f2fd;
                               padding: 6px 12px;
                               margin: 3px 0;
                               border-radius: 15px;
                               font-size: 0.85em;
                               color: #1565c0;
                               display: inline-block;">
                        {skill}
                    </div>
                    """, unsafe_allow_html=True)
                if len(evidence['matched_skills']) > 10:
                    st.caption(f"+ {len(evidence['matched_skills']) - 10} more skills")
            else:
                st.write("No matching skills")

        with right_col:
            st.markdown("""
            <div style="background: #f0f4f8; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">💼 Top Roles</h5>
            </div>
            """, unsafe_allow_html=True)

            exp_list = p.get("experience_json", [])[:3]
            for exp in exp_list:
                job_title = exp.get("job_title", "")
                company = exp.get("company", "")
                duration = f"{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}"
                st.markdown(f"""
                <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 6px;">
                    <strong style="color: #1976d2;">{job_title}</strong><br>
                    <span style="color: #666; font-size: 0.9em;">{company} • {duration}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background: #f0f4f8; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">📋 Relevant Experience</h5>
            </div>
            """, unsafe_allow_html=True)

            if evidence['company_evidence']:
                for company, roles in list(evidence['company_evidence'].items())[:2]:
                    st.markdown(f"""
                    <div style="margin-bottom: 12px; padding: 12px; background: white; border-radius: 6px;">
                        <strong style="color: #e65100;">{company}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    for role in roles[:1]:
                        for bullet in role['matched_bullets'][:2]:
                            st.markdown(f"""
                            <div style="margin-left: 15px; margin-bottom: 5px; color: #555;">
                                • {bullet}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No matching experience found")

        if st.button(f"Show Details", key=f"details_{idx}"):
            with st.expander("Detailed Information", expanded=True):
                tabs = st.tabs(["All Skills", "Full Experience", "Score Breakdown"])

                with tabs[0]:
                    all_skills = p.get("skills_flat", [])
                    if all_skills:
                        st.write(", ".join(all_skills))

                with tabs[1]:
                    for exp in p.get("experience_json", []):
                        st.write(f"**{exp.get('job_title', '')}** at {exp.get('company', '')}")
                        for bullet in exp.get("bullets", []):
                            st.write(f"• {bullet}")

                with tabs[2]:
                    st.json(result.get("scores", {}))

# ============================================================
# NEW: UI UPLOAD + PROCESS (sidebar above Vector Settings)
# ============================================================
def _safe_filename(name: str) -> str:
    name = (name or "resume.pdf").strip()
    name = re.sub(r"[^A-Za-z0-9\.\-_ ]+", "_", name)
    return name[:180] or "resume.pdf"

def process_uploaded_resumes(uploaded_files, department: str) -> Tuple[int, int, int, List[str]]:
    """
    Returns: (ok, no_text, failed, upserted_point_ids)
    """
    if not uploaded_files:
        return 0, 0, 0, []

    tmpdir = os.path.join(tempfile.gettempdir(), "resume_screening_uploads")
    os.makedirs(tmpdir, exist_ok=True)

    # Save PDFs to disk (pypdf works best with paths)
    saved_paths: List[str] = []
    for uf in uploaded_files:
        fn = _safe_filename(uf.name)
        path = os.path.join(tmpdir, f"{uuid.uuid4().hex}_{fn}")
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(path)

    ok = 0
    no_text = 0
    failed = 0
    upserted: List[str] = []

    progress = st.progress(0.0, text="Processing resumes...")
    status = st.empty()

    # Threaded pipeline: parse -> upsert (per resume)
    # Note: keep workers modest to avoid Bedrock throttling.
    def worker(pdf_path: str):
        data = parse_resume_pdf_to_json(pdf_path, department=department)
        if data is None:
            return ("no_text", pdf_path, None)
        pid = upsert_resume(data)
        return ("ok", pdf_path, pid)

    total = len(saved_paths)
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, total))) as ex:
        futures = {ex.submit(worker, p): p for p in saved_paths}
        for fut in as_completed(futures):
            completed += 1
            try:
                tag, pdf_path, pid = fut.result()
                if tag == "ok":
                    ok += 1
                    upserted.append(pid)
                else:
                    no_text += 1
            except Exception as e:
                failed += 1

            progress.progress(completed / total, text=f"Processed {completed}/{total} resumes")
            status.write(f"✅ OK: {ok} | ⚠️ No text: {no_text} | ❌ Failed: {failed}")

    # Cleanup old temp files to keep disk tidy (best-effort)
    try:
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass

    return ok, no_text, failed, upserted

# ============================================================
# MAIN UI (kept same)
# ============================================================
st.markdown('<h1 class="main-header">📋 Resume Intelligence Engine</h1>', unsafe_allow_html=True)
st.markdown("Paste your job description here:")

jd_col, controls_col = st.columns([3, 1])

with jd_col:
    jd_text = st.text_area(
        "Job Description",
        height=200,
        label_visibility="collapsed",
        placeholder="Paste the full job description here..."
    )

with controls_col:
    st.markdown("##### ⚙️ Top Results")
    top_k = st.number_input("", min_value=1, max_value=50, value=10, label_visibility="collapsed")
    st.markdown("")
    search_btn = st.button("🔍 Search Candidates", type="primary", use_container_width=True)

with st.sidebar:
    st.header("📥 Upload Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True
    )
    upload_dept = st.text_input("Department for uploaded resumes", value="Unknown")
    process_btn = st.button("⚡ Process Resumes", use_container_width=True)

    if process_btn:
        if not uploaded_resumes:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Parsing → Indexing into Qdrant..."):
                ok, no_text, failed, upserted = process_uploaded_resumes(uploaded_resumes, upload_dept.strip() or "Unknown")
            if ok:
                st.success(f"✅ Processed & indexed {ok} resume(s).")
            if no_text:
                st.warning(f"⚠️ {no_text} PDF(s) had no extractable text (scanned PDFs).")
            if failed:
                st.error(f"❌ Failed to process {failed} resume(s). Check logs/credentials.")
            if upserted:
                st.caption(f"Upserted IDs: {', '.join(upserted[:5])}{' ...' if len(upserted) > 5 else ''}")

    st.header("🔧 Advanced Filters")

    per_vector_limit = 120

    st.subheader("Search Preference")
    prefer = st.selectbox("Weighting Strategy", ["Balanced", "Skills-relevant", "Experience-relevant"])

    st.subheader("Sort Results By")
    sort_mode = st.selectbox("", ["Default (Relevance)", "Years of experience (desc)", "Skills match count (desc)"], label_visibility="collapsed")

    st.subheader("Optional Search Filters")
    min_years_val = st.number_input("Min. Years Experience", 0, 30, 0, 1)
    dept_contains = st.text_input("Department Contains")
    loc_contains = st.text_input("Location Contains")

    st.subheader("Vector Settings")

    try:
        cols = [c.name for c in client.get_collections().collections]
        if COLLECTION not in cols:
            st.error(f"❌ Collection '{COLLECTION}' not found")
        else:
            st.success("✅ Connected to Qdrant")
    except Exception as e:
        st.error(f"❌ Qdrant connection error: {e}")

# Main search results
if search_btn:
    if not jd_text.strip():
        st.error("⚠️ Please paste a Job Description before searching.")
        st.stop()

    q_filter = build_qdrant_filter(
        min_years=min_years_val if min_years_val > 0 else None,
        dept_contains=dept_contains,
        loc_contains=loc_contains,
    )

    with st.spinner("🔍 Searching candidates..."):
        results, views = search_candidates_from_jd(
            jd_text=jd_text,
            top_k=top_k,
            per_vector_limit=per_vector_limit,
            prefer=prefer,
            q_filter=q_filter,
        )
        results = post_sort_results(results, jd_text, sort_mode)

    if results:
        st.success(f"✅ Found {len(results)} matching candidates")
        st.markdown("---")
        st.markdown("### 👥 Shortlisted Candidates")
        for idx, r in enumerate(results, start=1):
            render_candidate_card(idx, r, jd_text)
    else:
        st.warning("No candidates found matching your criteria. Try adjusting the filters.")

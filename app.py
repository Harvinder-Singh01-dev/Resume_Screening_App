import tempfile
import os
import re
import json
import uuid
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import boto3
from botocore.config import Config
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Resume Intelligence Portal",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOGO HELPER
# ============================================================
def get_logo_base64(logo_path: str = "minda_logo.png") -> Optional[str]:
    try:
        with open(logo_path, "rb") as f:
            data = f.read()
        ext = Path(logo_path).suffix.lower().lstrip(".")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "svg": "image/svg+xml",
            "gif": "image/gif",
        }.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


LOGO_B64 = get_logo_base64("minda_logo.png")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* ─── GLOBAL FONT FIX – keep material icons intact ─── */
.material-icons {
    font-family: 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 24px !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
}

/* ─── CSS VARIABLES ─── */
:root {
    /* Navy-black gradient palette */
    --grad-start:  #0c2340;   /* deep navy */
    --grad-mid:    #1a3a5c;   /* mid navy */
    --grad-end:    #000000;   /* black */

    --sky-light:  #e0f2fe;
    --sky-mid:    #bae6fd;
    --sky-deep:   #7dd3fc;
    --sky-accent: #0ea5e9;
    --sky-dark:   #0369a1;
    --white:      #ffffff;
    --text-dark:  #0c2340;
    --text-mid:   #1e4976;
    --text-soft:  #4a7fa5;
    --text-muted: #7aadc8;
    --border:     rgba(14,165,233,0.20);
    --shadow:     0 4px 24px rgba(3,105,161,0.12);
    --shadow-sm:  0 2px 10px rgba(3,105,161,0.08);
    --r-sm: 8px; --r-md: 12px; --r-lg: 18px;

    /* Navy-black gradient shorthand */
    --navy-grad: linear-gradient(160deg, var(--grad-start) 0%, var(--grad-mid) 55%, var(--grad-end) 100%);
    --navy-shadow: 0 4px 20px rgba(12,35,64,0.45);
}

/* ─── APP BACKGROUND ─── */
.stApp {
    background: linear-gradient(145deg, #cfe8fc 0%, #d9eeff 30%, #c3dff7 60%, #b8d9f5 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    min-height: 100vh;
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(145deg, #cfe8fc 0%, #d9eeff 30%, #c3dff7 60%, #b8d9f5 100%) !important;
    border-right: 1px solid rgba(14,165,233,0.15) !important;
    box-shadow: 3px 0 20px rgba(3,105,161,0.08) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: var(--text-dark) !important; }

/* ── Sidebar section headings — DARK navy color ── */
[data-testid="stSidebar"] h4 {
    color: #0c2340 !important;
    font-weight: 800 !important;
    font-size: 0.80rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    border: none !important;
    border-bottom: 2px solid rgba(12,35,64,0.30) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 0 5px 0 !important;
    margin: 14px 0 10px 0 !important;
    cursor: default !important;
    pointer-events: none !important;
}

/* Kill any expander summary / details styling inside sidebar */
[data-testid="stSidebar"] details,
[data-testid="stSidebar"] summary {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    list-style: none !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] summary::-webkit-details-marker { display: none !important; }
[data-testid="stSidebar"] summary::marker { display: none !important; }

/* ── STOP sidebar color rule from bleeding into stButton text ── */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ─── MAIN CONTENT AREA — zero top padding, flush to header ─── */
.main .block-container {
    background: transparent !important;
    padding: 0rem 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* Remove Streamlit's default top spacer */
.main .block-container > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ─── INPUTS ─── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--white) !important;
    border: 1.5px solid rgba(14,165,233,0.30) !important;
    border-radius: var(--r-md) !important;
    color: var(--text-dark) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 4px rgba(3,105,161,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--sky-accent) !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic; font-weight: 400;
}
.stTextInput > label,.stNumberInput > label,
.stTextArea > label,.stSelectbox > label,.stFileUploader > label {
    color: var(--text-mid) !important; font-size: 0.78rem !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ─── SELECTBOX ─── */
.stSelectbox > div > div {
    background: var(--white) !important;
    border: 1.5px solid rgba(14,165,233,0.30) !important;
    border-radius: var(--r-md) !important; color: var(--text-dark) !important;
    box-shadow: 0 1px 4px rgba(3,105,161,0.06) !important;
}
.stSelectbox > div > div:hover { border-color: var(--sky-accent) !important; }

/* ─── PRIMARY BUTTONS – NAVY-BLACK GRADIENT, WHITE TEXT ─── */
.stButton > button,
.stButton > button *,
.stButton > button p,
.stButton > button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button {
    background: var(--navy-grad) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 800 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: var(--navy-shadow) !important;
    transition: all 0.22s ease !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(160deg, #091b30 0%, #0f2847 55%, #000000 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(12,35,64,0.55) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 3px 12px rgba(12,35,64,0.40) !important;
}

/* ─── DOWNLOAD BUTTONS ─── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #0369a1 100%) !important;
    color: var(--white) !important; border: none !important;
    border-radius: var(--r-sm) !important; padding: 0.45rem 1.2rem !important;
    font-weight: 700 !important; font-size: 0.82rem !important;
    box-shadow: 0 3px 10px rgba(13,148,136,0.25) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(13,148,136,0.40) !important;
}

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(186,230,253,0.30) !important;
    border: 2px dashed rgba(14,165,233,0.40) !important;
    border-radius: var(--r-md) !important; color: var(--text-mid) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(186,230,253,0.50) !important; border-color: var(--sky-accent) !important;
}

/* ─── ALERTS ─── */
.stSuccess > div {
    background: rgba(16,185,129,0.10) !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    border-radius: var(--r-md) !important; color: #065f46 !important;
}
.stWarning > div {
    background: rgba(245,158,11,0.10) !important;
    border: 1px solid rgba(245,158,11,0.35) !important;
    border-radius: var(--r-md) !important; color: #78350f !important;
}
.stError > div {
    background: rgba(239,68,68,0.10) !important;
    border: 1px solid rgba(239,68,68,0.35) !important;
    border-radius: var(--r-md) !important; color: #7f1d1d !important;
}
.stInfo > div {
    background: rgba(14,165,233,0.10) !important;
    border: 1px solid rgba(14,165,233,0.35) !important;
    border-radius: var(--r-md) !important; color: var(--text-mid) !important;
}

/* ─── EXPANDERS ─── */
.stExpander {
    background: rgba(255,255,255,0.70) !important;
    border: 1px solid rgba(14,165,233,0.18) !important;
    border-radius: var(--r-md) !important; margin-bottom: 10px !important;
    box-shadow: var(--shadow-sm) !important; overflow: hidden !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stExpander"] summary {
    background: rgba(186,230,253,0.35) !important;
    color: var(--text-mid) !important; font-weight: 700 !important;
    padding: 11px 16px !important; font-size: 0.88rem !important;
}
[data-testid="stExpander"] summary:hover { background: rgba(186,230,253,0.55) !important; }

/* ─── PROGRESS BAR ─── */
.stProgress > div > div {
    background: var(--navy-grad) !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: rgba(14,165,233,0.15) !important; border-radius: 4px !important;
}

/* ─── MARKDOWN TEXT ─── */
.stMarkdown p,.stMarkdown span,.stMarkdown li { color: var(--text-dark) !important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4 {
    color: var(--sky-dark) !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ─── HIDE STREAMLIT CHROME ─── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── TOP HEADER BAR — matches the light app background ── */
header {
    visibility: visible !important;
    height: 20px !important;
    min-height: 10px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: linear-gradient(135deg, #cfe8fc 0%, #c3dff7 100%) !important;
    border-bottom: 1px solid rgba(14,165,233,0.20) !important;
}

/* ─── DIVIDER ─── */
hr {
    border: none !important;
    border-top: 1.5px solid rgba(14,165,233,0.20) !important;
    margin: 12px 0 !important;
}

/* ─── COMPONENT CLASSES ─── */
.skill-badge {
    display: inline-block; padding: 3px 10px; margin: 2px;
    background: rgba(14,165,233,0.12); color: var(--sky-dark);
    border-radius: 20px; font-size: 0.72em; font-weight: 700;
    border: 1px solid rgba(14,165,233,0.25); white-space: nowrap;
}

.cand-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(14,165,233,0.20);
    border-left: 4px solid var(--sky-accent);
    border-radius: var(--r-lg); padding: 16px 20px; margin-bottom: 10px;
    backdrop-filter: blur(10px); box-shadow: var(--shadow-sm);
    transition: all 0.22s ease;
}
.cand-card:hover {
    background: rgba(255,255,255,0.94); border-left-color: var(--sky-dark);
    box-shadow: 0 6px 28px rgba(3,105,161,0.14); transform: translateX(2px);
}

/* ── Rank badge — navy-black gradient ── */
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--navy-grad);
    color: var(--white); font-weight: 800; font-size: 0.82rem; margin-right: 10px;
    box-shadow: 0 2px 10px rgba(12,35,64,0.40); flex-shrink: 0;
}

.stat-chip {
    text-align: center; padding: 10px 12px;
    background: rgba(186,230,253,0.50); border: 1px solid rgba(14,165,233,0.25);
    border-radius: var(--r-md); min-width: 68px;
}
.stat-chip h4 { margin: 0; color: var(--sky-dark); font-size: 1.2rem; font-weight: 800; }
.stat-chip p {
    margin: 1px 0 0 0; color: var(--text-soft); font-size: 0.65em;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;
}

.inner-sec {
    background: rgba(186,230,253,0.35); border: 1px solid rgba(14,165,233,0.18);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;
}
.inner-sec h5 {
    margin: 0; color: var(--sky-dark) !important; font-size: 0.83rem !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important;
}

.exp-row {
    background: rgba(255,255,255,0.60); border: 1px solid rgba(14,165,233,0.15);
    border-left: 3px solid rgba(14,165,233,0.50); border-radius: 8px;
    padding: 9px 12px; margin-bottom: 7px;
}
.exp-title { margin: 0 0 2px 0 !important; color: var(--text-dark) !important; font-weight: 700 !important; font-size: 0.86rem !important; }
.exp-sub   { margin: 0 !important; color: var(--text-soft) !important; font-size: 0.76rem !important; }

.ev-bullet {
    background: rgba(255,255,255,0.65); border-left: 3px solid rgba(14,165,233,0.45);
    border-radius: 5px; padding: 6px 10px; margin: 4px 0 4px 12px;
    font-size: 0.78rem; line-height: 1.45; color: var(--text-mid);
}

/* ─── HERO HEADER — navy-black gradient text, flush to top ─── */
.hero-wrapper {
    padding: 8px 0 6px 0;
    margin-top: 0 !important;
}
.hero-title {
    margin: 0 0 0px 0;
    font-size: 3.2rem;
    font-weight: 1200;
    font-family: 'Plus Jakarta Sans', sans-serif;
    letter-spacing: 0.5px;
    background: #000052;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    filter: drop-shadow(0 2px 6px rgba(12,35,64,0.18));
    paint-order: stroke fill;
}
.hero-subtitle {
    margin: 0 0 14px 0;
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    background: linear-gradient(135deg, #0c2340 0%, #0a3d6b 55%, #000000 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    opacity: 0.90;
}
</style>
""", unsafe_allow_html=True)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1") # amazon.titan-embed-text-v1
print(f'Embedd model id {EMBED_MODEL_ID}')
LLM_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0") # anthropic.claude-3-5-sonnet-20240620-v1:0
print(f'llm model id {LLM_MODEL_ID}')

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
print(f'Qdrant host {QDRANT_HOST}')
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
print(f'Qdrant port {QDRANT_PORT}')
COLLECTION = os.getenv("QDRANT_COLLECTION", "candidates")
print(f'Collection name {COLLECTION}')

RAW_RESUME_BUCKET = os.getenv("RAW_RESUME_BUCKET", "")
print("Raw Resume Bucket : ", RAW_RESUME_BUCKET)
PARSED_RESUME_BUCKET = os.getenv("PARSED_RESUME_BUCKET", "")
print("Parsed Resume Bucket : ", PARSED_RESUME_BUCKET)
RAW_RESUME_PREFIX = os.getenv("RAW_RESUME_PREFIX", "Resume_Bank")
print("Raw Resume Suffix : ", RAW_RESUME_PREFIX)

MAX_WORKERS = int(os.getenv("PROCESS_MAX_WORKERS", "4"))

MAX_WORKERS = 4

if not RAW_RESUME_BUCKET:
    st.warning("RAW_RESUME_BUCKET is not set.")
if not PARSED_RESUME_BUCKET:
    st.warning("PARSED_RESUME_BUCKET is not set.")


# ============================================================
# CLIENTS
# ============================================================
@st.cache_resource
def get_clients():
    cfg = Config(
        connect_timeout=30,
        read_timeout=120,
        retries={"max_attempts": 5}
    )

    s3 = boto3.client("s3", region_name='ap-south-1')
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    return s3, qdrant, bedrock


s3_client, client, bedrock_runtime = get_clients()

def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

def s3_put_json(bucket: str, key: str, payload: dict) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def s3_get_bytes(bucket: str, key: str) -> bytes:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _safe_fn(name: str) -> str:
    name = (name or "resume.pdf").strip()
    name = re.sub(r"[^A-Za-z0-9\._\- ]+", "_", name)
    return name[:180] or "resume.pdf"


def build_raw_resume_key(department: str, filename: str) -> str:
    dept = (department or "General").strip() or "General"
    return f"{RAW_RESUME_PREFIX}/{dept}/{uuid.uuid4().hex}_{_safe_fn(filename)}"


def build_parsed_resume_key(department: str, filename: str) -> str:
    stem = Path(_safe_fn(filename)).stem
    return f"{uuid.uuid4().hex}_{stem}.json"


# ============================================================
# BEDROCK EMBEDDING
# ============================================================
@st.cache_data(show_spinner=False)
def bedrock_embed(text: str) -> List[float]:
    text = (text or "").strip() or "N/A"
    resp = bedrock_runtime.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


# ============================================================
# QDRANT COLLECTION
# ============================================================
@st.cache_resource
def ensure_collection(collection_name: str) -> int:
    dim = len(bedrock_embed("dimension check"))
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
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
# RESUME PARSING
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
            "links": {
                "linkedin": None,
                "github": None,
                "portfolio": None,
                "other": []
            }
        },
        "skills": {"core": [], "technical": [], "tools": [], "soft": [], "domains": []},
        "experience": [{
            "company": None,
            "job_title": None,
            "employment_type": None,
            "location": None,
            "start_date": None,
            "end_date": None,
            "is_current": None,
            "bullets": []
        }],
        "education": [{
            "institution": None,
            "degree": None,
            "field_of_study": None,
            "location": None,
            "start_date": None,
            "end_date": None,
            "grade_or_gpa": None,
            "notes": []
        }],
        "projects": [{
            "name": None,
            "role": None,
            "start_date": None,
            "end_date": None,
            "description": None,
            "technologies": [],
            "links": []
        }],
        "certifications": [{
            "name": None,
            "issuer": None,
            "issue_date": None,
            "expiry_date": None,
            "credential_id": None,
            "credential_url": None
        }],
        "awards": [{
            "name": None,
            "issuer": None,
            "date": None,
            "description": None
        }],
        "publications": [{
            "title": None,
            "publisher_or_venue": None,
            "date": None,
            "url": None
        }],
        "languages": [{
            "language": None,
            "proficiency": None
        }],
        "metadata": {
            "source": "resume_pdf",
            "extraction_notes": [],
            "confidence": {
                "overall": None,
                "fields_with_low_confidence": []
            }
        },
    }

    return f"""You are an expert resume information extraction system.
TASK: Extract all relevant information from the resume text and return STRICT JSON ONLY.
No commentary. No markdown. No extra keys beyond the schema.

RULES:
1) Valid JSON only (double quotes, no trailing commas).
2) Use EXACT strings from the resume.
3) Missing fields: null (single) or [] (lists).
4) Normalize dates to "YYYY-MM" when possible.
5) total_years_experience must be a number.
6) Return JSON matching the schema exactly.

SCHEMA:
{json.dumps(schema, indent=2)}

RESUME TEXT:
\"\"\"{resume_text}\"\"\"
""".strip()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def try_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_RE.search(text)
    if m:
        return json.loads(m.group(0))

    raise ValueError("No valid JSON found.")


def _llm_call(messages, max_tokens=3000, temperature=0.0) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    resp = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps(body).encode(),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read().decode())
    return "".join(
        p.get("text", "")
        for p in payload.get("content", [])
        if p.get("type") == "text"
    ).strip()


def parse_resume_pdf_to_json(
    pdf_path: str,
    department: str,
    original_filename: str = ""
) -> Optional[Dict[str, Any]]:
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        return None

    raw = _llm_call([{
        "role": "user",
        "content": [{"type": "text", "text": build_prompt(resume_text)}]
    }])

    try:
        data = try_parse_json(raw)
    except Exception:
        repair = (
            f"Return STRICT valid JSON ONLY per the schema.\n"
            f"Invalid output:\n\"\"\"{raw}\"\"\"\n"
            f"Resume:\n\"\"\"{resume_text}\"\"\""
        )
        data = try_parse_json(_llm_call([{
            "role": "user",
            "content": [{"type": "text", "text": repair}]
        }]))

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dict.")

    data.setdefault("candidate", {})
    if not isinstance(data["candidate"], dict):
        data["candidate"] = {}

    data["candidate"]["department"] = department
    data["candidate"]["original_filename"] = original_filename or os.path.basename(pdf_path)
    return data


# ============================================================
# INDEXING
# ============================================================
def _norm(xs):
    return sorted(set(str(x).strip().lower() for x in (xs or []) if str(x).strip()))


def _profile_txt(r):
    c = r.get("candidate", {}) or {}
    return (
        f"Department: {c.get('department', '')}\n"
        f"Location: {c.get('location', '')}\n"
        f"Experience: {c.get('total_years_experience', '')} years\n"
        f"Summary: {c.get('summary', '')}"
    )


def _skills_txt(r):
    s = r.get("skills", {}) or {}
    j = lambda x: "; ".join(str(i).strip() for i in (x or []) if str(i).strip())
    return (
        f"Core: {j(s.get('core'))}\n"
        f"Technical: {j(s.get('technical'))}\n"
        f"Tools: {j(s.get('tools'))}"
    )


def _exp_txt(r):
    rows = []
    for e in (r.get("experience", []) or []):
        rows.append(
            f"Role: {e.get('job_title', '')} | "
            f"Co: {e.get('company', '')} | "
            f"{e.get('start_date', '')}–{e.get('end_date') or 'Present'} | "
            + " | ".join((e.get("bullets") or [])[:10])
        )
    return "\n".join(rows)


def _edu_txt(r):
    lines = ["Education:"]
    for e in (r.get("education", []) or []):
        lines.append(
            f"{e.get('degree', '')} {e.get('field_of_study', '')}, "
            f"{e.get('institution', '')} {e.get('end_date', '')}"
        )
    lines += ["Certifications:"] + [str(c) for c in (r.get("certifications", []) or [])]
    return "\n".join(lines)


def _build_payload(r):
    c = r.get("candidate", {}) or {}
    s = r.get("skills", {}) or {}
    exp = r.get("experience", []) or []
    edu = r.get("education", []) or []

    all_sk = _norm(
        (s.get("core") or [])
        + (s.get("technical") or [])
        + (s.get("tools") or [])
        + (s.get("soft") or [])
        + (s.get("domains") or [])
    )

    return {
        "full_name": c.get("full_name"),
        "email": c.get("email"),
        "phone": c.get("phone"),
        "location": c.get("location"),
        "department": c.get("department"),
        "total_years_experience": c.get("total_years_experience"),
        "original_filename": c.get("original_filename", ""),

        "raw_resume_bucket": c.get("raw_resume_bucket"),
        "raw_resume_key": c.get("raw_resume_key"),
        "parsed_resume_bucket": c.get("parsed_resume_bucket"),
        "parsed_resume_key": c.get("parsed_resume_key"),

        "skills_flat": all_sk,
        "companies": _norm([e.get("company", "") for e in exp]),
        "job_titles": _norm([e.get("job_title", "") for e in exp]),
        "degrees": _norm([e.get("degree", "") for e in edu]),

        "candidate_json": c,
        "skills_json": s,
        "experience_json": exp,
        "education_json": edu,
        "certifications_json": r.get("certifications", []),
        "metadata": r.get("metadata", {}),
    }


def _point_id(r):
    c = r.get("candidate", {}) or {}
    k = c.get("email") or c.get("phone") or c.get("full_name") or str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(k).strip().lower()))


def upsert_resume(resume_json):
    pid = _point_id(resume_json)

    client.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(
            id=pid,
            vector={
                "v_profile": bedrock_embed(_profile_txt(resume_json)),
                "v_skills": bedrock_embed(_skills_txt(resume_json)),
                "v_experience": bedrock_embed(_exp_txt(resume_json)),
                "v_education": bedrock_embed(_edu_txt(resume_json)),
            },
            payload=_build_payload(resume_json),
        )]
    )
    return pid


# ============================================================
# RESUME DOWNLOAD HELPER
# ============================================================
def find_resume_pdf(payload: Dict[str, Any]) -> Optional[bytes]:
    bucket = payload.get("raw_resume_bucket")
    key = payload.get("raw_resume_key")
    if not bucket or not key:
        return None
    try:
        return s3_get_bytes(bucket, key)
    except Exception:
        return None


# ============================================================
# SCREENING LOGIC
# ============================================================
def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return json.dumps(x, ensure_ascii=False).strip()


def simple_tokenize(text: str) -> List[str]:
    stop = {
        "and", "or", "the", "to", "of", "in", "for", "with", "a", "an", "is", "are", "on",
        "as", "at", "by", "be", "we", "you", "our", "will", "shall", "from", "this", "that", "it"
    }
    return [
        t for t in re.findall(r"[A-Za-z0-9\+\#\.\-]{2,}", (text or "").lower())
        if t not in stop
    ]


def fmt_years(x):
    try:
        return int(x)
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def split_jd_4views(jd_text: str) -> Dict[str, str]:
    prompt = (
        f"Given this Job Description return STRICT JSON with keys:\n"
        f"\"profile_view\",\"skills_view\",\"responsibilities_view\",\"education_view\"\n"
        f"Extract ONLY what is present.\n\nJD:\n\"\"\"{jd_text}\"\"\""
    )
    raw = _llm_call([{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }], max_tokens=900)

    try:
        out = json.loads(raw)
        return {
            k: out.get(k, "")
            for k in ("profile_view", "skills_view", "responsibilities_view", "education_view")
        }
    except Exception:
        return {
            "profile_view": "",
            "skills_view": "",
            "responsibilities_view": jd_text,
            "education_view": ""
        }


def qdrant_query(vec_name, vec, limit, q_filter=None):
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            using=vec_name,
            limit=limit,
            with_payload=True,
            query_filter=q_filter,
        )
        return res.points

    return client.search(
        collection_name=COLLECTION,
        query_vector=(vec_name, vec),
        limit=limit,
        with_payload=True,
        query_filter=q_filter,
    )


def build_filter(min_years, dept, loc):
    must = []

    if min_years:
        must.append(
            qmodels.FieldCondition(
                key="total_years_experience",
                range=qmodels.Range(gte=min_years)
            )
        )

    if dept.strip():
        must.append(
            qmodels.FieldCondition(
                key="department",
                match=qmodels.MatchText(text=dept.strip())
            )
        )

    if loc.strip():
        must.append(
            qmodels.FieldCondition(
                key="location",
                match=qmodels.MatchText(text=loc.strip())
            )
        )

    return qmodels.Filter(must=must) if must else None


def extract_evidence_by_company(jd_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    jd_tokens = set(simple_tokenize(jd_text))
    skills_tokens = jd_tokens
    resp_tokens = jd_tokens

    skills_flat = payload.get("skills_flat", []) or []
    matched_skills = [
        s for s in skills_flat
        if s.lower() in jd_tokens or any(t in s.lower() for t in skills_tokens)
    ]

    company_evidence: Dict[str, List] = {}
    experience_json = payload.get("experience_json", []) or []

    for exp in experience_json:
        company = exp.get("company", "Unknown Company")
        job_title = exp.get("job_title", "")
        bullets = exp.get("bullets", []) or []

        matched_bullets = []
        for bullet in bullets:
            bullet_lower = bullet.lower()
            if (
                any(token in bullet_lower for token in list(jd_tokens)[:100])
                or any(token in bullet_lower for token in skills_tokens)
                or any(token in bullet_lower for token in resp_tokens)
            ):
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
        "company_evidence": company_evidence,
    }


def search_candidates_from_jd(
    jd_text: str,
    top_k: int,
    per_vector_limit: int,
    prefer: str,
    q_filter=None,
):
    views = split_jd_4views(jd_text)

    skills_q_text = safe_text(views.get("skills_view")) or jd_text
    exp_q_text = safe_text(views.get("responsibilities_view")) or jd_text
    prof_q_text = safe_text(views.get("profile_view")) or jd_text

    q_skills = bedrock_embed(skills_q_text)
    q_exp = bedrock_embed(exp_q_text)
    q_prof = bedrock_embed(prof_q_text)

    if prefer == "Skills-relevant":
        w_exp, w_sk = 0.45, 0.55
    elif prefer == "Experience-relevant":
        w_exp, w_sk = 0.70, 0.30
    else:
        w_exp, w_sk = 0.60, 0.40

    w_prof = 0.10

    res_exp = qdrant_query("v_experience", q_exp, per_vector_limit, q_filter)
    res_sk = qdrant_query("v_skills", q_skills, per_vector_limit, q_filter)
    res_prof = qdrant_query("v_profile", q_prof, per_vector_limit, q_filter)

    fused: Dict[str, Dict[str, Any]] = {}

    def add_results(results, weight, key):
        for r in results:
            pid = str(r.id)
            if pid not in fused:
                fused[pid] = {
                    "id": pid,
                    "score": 0.0,
                    "payload": r.payload,
                    "scores": {}
                }
            fused[pid]["score"] += weight * float(r.score)
            fused[pid]["scores"][key] = float(r.score)

    add_results(res_exp, w_exp, "experience_score")
    add_results(res_sk, w_sk, "skills_score")
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
        enriched.append((r, ev, fmt_years(p.get("total_years_experience"))))

    if sort_mode == "Years of experience (desc)":
        enriched.sort(key=lambda x: (x[2], x[0]["score"]), reverse=True)
    elif sort_mode == "Skills match count (desc)":
        enriched.sort(key=lambda x: (x[1]["matched_skills_count"], x[0]["score"]), reverse=True)

    return [x[0] for x in enriched]


# ============================================================
# RENDER CANDIDATE CARD
# ============================================================
def render_candidate_card(idx: int, result: Dict[str, Any], jd_text: str):
    p = result["payload"]
    name = p.get("full_name") or "Candidate"
    score = result["score"]
    yrs = fmt_years(p.get("total_years_experience"))
    dept = p.get("department") or "—"
    evidence = extract_evidence_by_company(jd_text, p)
    skc = evidence["matched_skills_count"]

    accent = {1: "#0ea5e9", 2: "#0891b2", 3: "#0d9488"}.get(idx, "#64748b")

    st.markdown(f"""
    <div class="cand-card" style="border-left-color:{accent};">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  flex-wrap:wrap;gap:10px;">
        <div style="flex:1;min-width:220px;">
          <div style="display:flex;align-items:center;margin-bottom:5px;">
            <span class="rank-badge">#{idx}</span>
            <span style="font-size:1.18rem;font-weight:800;color:#0c2340;
                         font-family:'Plus Jakarta Sans',sans-serif;letter-spacing:-0.3px;">{name}</span>
          </div>
          <p style="margin:0 0 3px 40px;color:#4a7fa5;font-size:0.8rem;font-weight:500;">
            📧&nbsp;{p.get('email','N/A')} &nbsp;|&nbsp;
            📱&nbsp;{p.get('phone','N/A')} &nbsp;|&nbsp;
            📍&nbsp;{p.get('location','N/A')}
          </p>
          <p style="margin:0 0 0 40px;color:#0369a1;font-size:0.76rem;font-weight:600;">
            🏢&nbsp;{dept}
          </p>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <div class="stat-chip"><h4>{score:.1%}</h4><p>SCORE</p></div>
          <div class="stat-chip"><h4>{yrs}</h4><p>YEARS</p></div>
          <div class="stat-chip"><h4>{skc}</h4><p>SKILLS</p></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"🎯  View Full Profile — {name}", expanded=(idx == 1)):
        orig_fn = p.get("original_filename", "")
        pdf_bytes = find_resume_pdf(p)

        dl_col, _, _ = st.columns([1, 2, 1])
        with dl_col:
            if pdf_bytes:
                dl_name = (
                    orig_fn if orig_fn.lower().endswith(".pdf")
                    else f"{name.replace(' ', '_')}_resume.pdf"
                )
                st.download_button(
                    label="📥  Download Resume PDF",
                    data=pdf_bytes,
                    file_name=dl_name,
                    mime="application/pdf",
                    key=f"dl_{idx}_{result['id']}",
                    use_container_width=True,
                )
            else:
                st.info("💡 Resume PDF not found in S3 metadata")

        st.markdown("<hr>", unsafe_allow_html=True)

        col_l, col_r = st.columns([1.2, 2])

        with col_l:
            st.markdown(
                f"""<div class="inner-sec"><h5>🎯 Matching Skills ({skc})</h5></div>""",
                unsafe_allow_html=True,
            )

            if evidence["matched_skills"]:
                for skill in evidence["matched_skills"][:10]:
                    skill_fmt = skill[0].upper() + skill[1:].lower()
                    st.markdown(
                        f'<span class="skill-badge">{skill_fmt}</span>',
                        unsafe_allow_html=True,
                    )
                if len(evidence["matched_skills"]) > 10:
                    st.caption(f"+ {len(evidence['matched_skills']) - 10} more skills")
            else:
                st.write("No matching skills")

            all_sk = p.get("skills_flat", [])
            if all_sk:
                with st.expander(f"📚  All Skills ({len(all_sk)})"):
                    st.markdown(
                        "<div style='display:flex;flex-wrap:wrap;gap:3px;'>"
                        + "".join(
                            f'<span class="skill-badge" style="background:rgba(13,148,136,0.10);'
                            f'color:#0f766e;border-color:rgba(13,148,136,0.25);">'
                            f'{s[0].upper()+s[1:]}</span>' for s in all_sk
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

        with col_r:
            st.markdown("""<div class="inner-sec"><h5>💼 Top Roles</h5></div>""", unsafe_allow_html=True)

            for exp in (p.get("experience_json", []) or [])[:3]:
                jt = exp.get("job_title", "")
                co = exp.get("company", "")
                dur = f"{exp.get('start_date','')} – {exp.get('end_date','Present')}"
                st.markdown(
                    f"""<div class="exp-row">
                    <p class="exp-title">{jt}</p>
                    <p class="exp-sub">{co}&nbsp;•&nbsp;{dur}</p></div>""",
                    unsafe_allow_html=True,
                )

            if evidence["company_evidence"]:
                st.markdown(
                    """<div class="inner-sec" style="margin-top:10px;">
                    <h5>📋 Relevant Experience</h5></div>""",
                    unsafe_allow_html=True,
                )
                for co, roles in list(evidence["company_evidence"].items())[:2]:
                    st.markdown(
                        f"<p style='color:#b45309;font-size:0.8rem;font-weight:700;margin:6px 0 3px;'>🏢 {co}</p>",
                        unsafe_allow_html=True,
                    )
                    for role in roles[:1]:
                        for bullet in role["matched_bullets"][:2]:
                            st.markdown(
                                f'<div class="ev-bullet">• {bullet}</div>',
                                unsafe_allow_html=True,
                            )
            else:
                st.info("No matching experience found")


# ============================================================
# PROCESS UPLOADED RESUMES
# ============================================================
def process_uploaded_resumes(uploaded_files, department: str) -> Tuple[int, int, int, List[str]]:
    if not uploaded_files:
        return 0, 0, 0, []

    if not RAW_RESUME_BUCKET or not PARSED_RESUME_BUCKET:
        raise ValueError("RAW_RESUME_BUCKET and PARSED_RESUME_BUCKET must be configured.")

    tmpdir = os.path.join(tempfile.gettempdir(), "resume_screening_uploads")
    os.makedirs(tmpdir, exist_ok=True)

    saved: List[Tuple[str, str, str, str, str]] = []

    for uf in uploaded_files:
        fn = _safe_fn(uf.name)
        data = bytes(uf.getbuffer())

        local_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}_{fn}")
        with open(local_path, "wb") as f:
            f.write(data)

        raw_key = build_raw_resume_key(department, fn)
        parsed_key = build_parsed_resume_key(department, fn)

        s3_put_bytes(RAW_RESUME_BUCKET, raw_key, data, "application/pdf")
        saved.append((local_path, fn, RAW_RESUME_BUCKET, raw_key, parsed_key))

    ok = no_text = failed = 0
    upserted: List[str] = []
    progress = st.progress(0.0, text="Processing resumes…")
    status = st.empty()

    def worker(pdf_path: str, orig_fn: str, raw_bucket: str, raw_key: str, parsed_key: str):
        data = parse_resume_pdf_to_json(
            pdf_path,
            department=department,
            original_filename=orig_fn,
        )
        if data is None:
            return ("no_text", None)

        data.setdefault("candidate", {})
        data.setdefault("metadata", {})

        data["candidate"]["raw_resume_bucket"] = raw_bucket
        data["candidate"]["raw_resume_key"] = raw_key
        data["candidate"]["parsed_resume_bucket"] = PARSED_RESUME_BUCKET
        data["candidate"]["parsed_resume_key"] = parsed_key

        data["metadata"]["storage"] = {
            "raw_bucket": raw_bucket,
            "raw_key": raw_key,
            "parsed_bucket": PARSED_RESUME_BUCKET,
            "parsed_key": parsed_key,
        }

        s3_put_json(PARSED_RESUME_BUCKET, parsed_key, data)

        pid = upsert_resume(data)
        return ("ok", pid)

    total, completed = len(saved), 0

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, total))) as ex:
        futures = {
            ex.submit(worker, p, fn, raw_bucket, raw_key, parsed_key): (p, fn)
            for p, fn, raw_bucket, raw_key, parsed_key in saved
        }

        for fut in as_completed(futures):
            completed += 1
            try:
                tag, pid = fut.result()
                if tag == "ok":
                    ok += 1
                    upserted.append(pid)
                else:
                    no_text += 1
            except Exception as e:
                failed += 1
                st.warning(f"Failed one resume: {e}")

            progress.progress(completed / total, text=f"Processing {completed}/{total} resumes…")
            status.markdown(
                f"✅ Indexed: **{ok}** &nbsp;|&nbsp; "
                f"⚠️ No text: **{no_text}** &nbsp;|&nbsp; "
                f"❌ Failed: **{failed}**"
            )

    try:
        for p, *_ in saved:
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass

    return ok, no_text, failed, upserted


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    if LOGO_B64:
        st.markdown(f"""
        <div style="text-align:center; padding:22px 16px 18px 16px;">
            <img src="{LOGO_B64}" alt="UNO Minda"
                 style="width:170px; height:auto; display:block; margin:0 auto;">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:22px 16px 18px 16px;">
            <p style="font-size:1.05rem;font-weight:800;color:#0369a1;
                      font-family:'Plus Jakarta Sans',sans-serif;margin:0;">🏢 UNO Minda</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("#### 📤 Upload Resumes")
    dept_upload = st.text_input("Department Tag", value="General", key="dept_tag")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("⚡  Process & Index Resumes", use_container_width=True, key="proc_btn"):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            try:
                ok, no_text, failed, _ = process_uploaded_resumes(uploaded_files, dept_upload)
                if ok:
                    st.success(f"✅ Indexed {ok} resume(s)!")
                if no_text:
                    st.warning(f"⚠️ {no_text} had no text.")
                if failed:
                    st.error(f"❌ {failed} failed.")
            except Exception as e:
                st.error(f"Processing failed: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("#### 🔧 Search Filters")
    prefer = st.selectbox("Weighting Strategy", ["Balanced", "Skills-relevant", "Experience-relevant"])
    sort_mode = st.selectbox(
        "Sort Results By",
        ["Default (Relevance)", "Years of experience (desc)", "Skills match count (desc)"]
    )
    min_years_val = st.number_input("Min. Years Experience", 0, 30, 0, 1)
    dept_contains = st.text_input("Department Contains", placeholder="e.g. Engineering")
    loc_contains = st.text_input("Location Contains", placeholder="e.g. Pune")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("#### 🔌 Status")
    try:
        cols_q = [c.name for c in client.get_collections().collections]
        if COLLECTION not in cols_q:
            st.error(f"Collection '{COLLECTION}' missing")
        else:
            count = client.get_collection(COLLECTION).points_count
            st.success(f"✅ Qdrant connected  •  **{count}** candidates")
    except Exception as e:
        st.error(f"Qdrant: {e}")


# ============================================================
# MAIN CONTENT — HERO HEADER (navy-black gradient, flush to top)
# ============================================================
st.markdown("""
<div class="hero-wrapper" style="text-align:center; padding: 8px 0 6px 0; margin-top:0;">
  <h1 class="hero-title">Resume Intelligence Engine</h1><br>
  <p class="hero-subtitle">AI Recruiter Copilot — match the right talent, instantly</p>
</div>
""", unsafe_allow_html=True)

jd_col, ctrl_col = st.columns([3, 1])

with jd_col:
    jd_text = st.text_area(
        "Job Description",
        height=185,
        placeholder="Paste the full job description here — role, responsibilities, required skills…",
    )

with ctrl_col:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    top_k = st.number_input("Top Results", min_value=1, max_value=50, value=10, key="top_k")
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    search_btn = st.button("🔍  Search Candidates", type="primary", use_container_width=True)

if search_btn:
    if not jd_text.strip():
        st.error("⚠️  Please paste a Job Description before searching.")
        st.stop()

    q_filter = build_filter(
        min_years_val if min_years_val > 0 else None,
        dept_contains,
        loc_contains,
    )

    with st.spinner("🔍  Analysing job description and matching candidates…"):
        results, views = search_candidates_from_jd(
            jd_text=jd_text,
            top_k=top_k,
            per_vector_limit=120,
            prefer=prefer,
            q_filter=q_filter,
        )
        results = post_sort_results(results, jd_text, sort_mode)

    if results:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.30);
                    border-radius:10px;padding:11px 18px;margin-bottom:16px;
                    display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.2rem;">✅</span>
            <span style="color:#065f46;font-weight:700;font-size:0.92rem;">
                Found <strong>{len(results)}</strong> matching candidates
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style="color:#0c2340;font-family:'Plus Jakarta Sans',sans-serif;
                    font-weight:800;font-size:1.2rem;margin-bottom:12px;">
                    👥&nbsp; Shortlisted Candidates</h3>
        """, unsafe_allow_html=True)

        for idx, r in enumerate(results, start=1):
            render_candidate_card(idx, r, jd_text)
    else:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.28);
                    border-radius:10px;padding:24px;text-align:center;margin-top:20px;">
            <p style="color:#92400e;font-size:1rem;margin:0;font-weight:700;">
                🔎  No candidates matched your criteria — try adjusting filters or JD text.
            </p>
        </div>
        """, unsafe_allow_html=True)
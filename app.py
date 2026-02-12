import tempfile
import os
import re
import json
import uuid
import hashlib
import zipfile
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ============================================================
# STREAMLIT CONFIG + STYLES
# ============================================================
st.set_page_config(page_title="Resume Intelligence Portal", layout="wide", initial_sidebar_state="collapsed")

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
    .upload-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-success { background-color: #d4edda; color: #155724; }
    .status-duplicate { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
LLM_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# S3 Configuration
S3_RAW_BUCKET = os.getenv("S3_RAW_BUCKET", "raw-resumes")
S3_JSON_BUCKET = os.getenv("S3_JSON_BUCKET", "parsed-resumes-json")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "candidates")

# Processing Configuration
MAX_WORKERS = int(os.getenv("PROCESS_MAX_WORKERS", "4"))

# ============================================================
# AWS CLIENTS (cached)
# ============================================================
@st.cache_resource
def get_clients():
    """Initialize and cache AWS and Qdrant clients"""
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    cfg = Config(connect_timeout=30, read_timeout=120, retries={"max_attempts": 5})
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=cfg)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    return qdrant, bedrock, s3

client, bedrock_runtime, s3_client = get_clients()

# ============================================================
# S3 UTILITY FUNCTIONS
# ============================================================
def calculate_file_hash(file_content: bytes) -> str:
    """Calculate MD5 hash of file content for deduplication"""
    return hashlib.md5(file_content).hexdigest()

def check_duplicate_in_s3(bucket: str, file_hash: str) -> Optional[str]:
    """Check if a file with same hash exists in S3. Returns S3 key if duplicate found."""
    try:
        # List all objects and check metadata
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket):
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                try:
                    response = s3_client.head_object(Bucket=bucket, Key=obj['Key'])
                    if response.get('Metadata', {}).get('md5_hash') == file_hash:
                        return obj['Key']
                except ClientError:
                    continue
        return None
    except Exception as e:
        st.warning(f"Error checking duplicates: {e}")
        return None

def upload_to_s3(bucket: str, key: str, file_content: bytes, metadata: dict) -> bool:
    """Upload file to S3 with metadata"""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            Metadata=metadata
        )
        return True
    except ClientError as e:
        st.error(f"Failed to upload {key}: {e}")
        return False

def list_unprocessed_pdfs(bucket: str) -> List[Dict[str, Any]]:
    """List PDF files in S3 that haven't been processed yet"""
    unprocessed = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket):
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.pdf'):
                    try:
                        response = s3_client.head_object(Bucket=bucket, Key=obj['Key'])
                        if response.get('Metadata', {}).get('processed', 'false') == 'false':
                            unprocessed.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'],
                                'metadata': response.get('Metadata', {})
                            })
                    except ClientError:
                        continue
        return unprocessed
    except Exception as e:
        st.error(f"Error listing unprocessed files: {e}")
        return []

def download_from_s3(bucket: str, key: str) -> Optional[bytes]:
    """Download file content from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except ClientError as e:
        st.error(f"Failed to download {key}: {e}")
        return None

def mark_as_processed(bucket: str, key: str) -> bool:
    """Update S3 object metadata to mark as processed"""
    try:
        # Get current metadata
        response = s3_client.head_object(Bucket=bucket, Key=key)
        metadata = response.get('Metadata', {})
        
        # Update processed flag
        metadata['processed'] = 'true'
        metadata['processed_date'] = datetime.utcnow().isoformat()
        
        # Copy object with new metadata
        s3_client.copy_object(
            Bucket=bucket,
            Key=key,
            CopySource={'Bucket': bucket, 'Key': key},
            Metadata=metadata,
            MetadataDirective='REPLACE'
        )
        return True
    except ClientError as e:
        st.error(f"Failed to mark {key} as processed: {e}")
        return False

# ============================================================
# ZIP FILE HANDLING
# ============================================================
def extract_zip_file(zip_content: bytes) -> Dict[str, List[Tuple[str, bytes, str]]]:
    """
    Extract PDF files from ZIP archive.
    Returns: {department: [(filename, content, hash), ...]}
    """
    extracted_files = {}
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            for file_info in zf.filelist:
                # Skip directories and non-PDF files
                if file_info.is_dir() or not file_info.filename.lower().endswith('.pdf'):
                    continue
                
                # Extract department from path
                path_parts = file_info.filename.split('/')
                
                # Case 1: PDF directly in ZIP root
                if len(path_parts) == 1:
                    department = "Unknown"
                    filename = path_parts[0]
                # Case 2: PDF in department subfolder
                else:
                    department = path_parts[0]
                    filename = path_parts[-1]
                
                # Read file content
                content = zf.read(file_info.filename)
                file_hash = calculate_file_hash(content)
                
                # Organize by department
                if department not in extracted_files:
                    extracted_files[department] = []
                extracted_files[department].append((filename, content, file_hash))
        
        return extracted_files
    
    except zipfile.BadZipFile:
        st.error("Invalid ZIP file")
        return {}
    except Exception as e:
        st.error(f"Error extracting ZIP: {e}")
        return {}

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
# QDRANT COLLECTION CREATION
# ============================================================
@st.cache_resource
def ensure_collection(collection_name: str) -> int:
    """Ensures the collection exists with required named vectors"""
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
def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        reader = PdfReader(io.BytesIO(pdf_content))
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
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

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

    return f"""You are an expert resume information extraction system.

TASK:
Extract all relevant information from the resume text and return STRICT JSON ONLY.
No commentary. No markdown. No extra keys beyond the schema.

RULES:
1. Return ONLY valid JSON matching the schema below
2. For missing fields, use null (not empty strings)
3. For arrays, return empty [] if no data
4. Calculate total_years_experience from work history
5. Extract ALL skills (technical, soft, tools, domains)
6. Preserve ALL bullet points for each role
7. Be thorough - don't skip sections

SCHEMA:
{json.dumps(schema, indent=2)}

RESUME TEXT:
{resume_text}

OUTPUT (JSON ONLY):"""

def parse_resume_with_bedrock(resume_text: str, department: str = "Unknown") -> Optional[Dict]:
    """Parse resume text using AWS Bedrock Claude"""
    if not resume_text or len(resume_text) < 50:
        return None
    
    prompt = build_prompt(resume_text)
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=LLM_MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        
        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]
        
        # Clean response
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed_data = json.loads(content)
        
        # Add department to candidate info
        if "candidate" in parsed_data and isinstance(parsed_data["candidate"], dict):
            if not parsed_data["candidate"].get("department"):
                parsed_data["candidate"]["department"] = department
        
        return parsed_data
    
    except Exception as e:
        st.error(f"Bedrock parsing error: {e}")
        return None

# ============================================================
# CHUNKING & EMBEDDING
# ============================================================
def create_chunks_from_json(data: Dict) -> Dict[str, str]:
    """Create text chunks from parsed JSON for embedding"""
    chunks = {}
    
    # Profile chunk
    candidate = data.get("candidate", {})
    profile_parts = [
        candidate.get("full_name", ""),
        candidate.get("headline", ""),
        candidate.get("summary", ""),
        f"Experience: {candidate.get('total_years_experience', 0)} years"
    ]
    chunks["profile"] = " | ".join([p for p in profile_parts if p])
    
    # Skills chunk
    skills_data = data.get("skills", {})
    all_skills = []
    for category in ["core", "technical", "tools", "soft", "domains"]:
        all_skills.extend(skills_data.get(category, []))
    chunks["skills"] = ", ".join(all_skills) if all_skills else "N/A"
    
    # Experience chunk
    experience_parts = []
    for exp in data.get("experience", []):
        exp_text = f"{exp.get('job_title', '')} at {exp.get('company', '')}"
        bullets = " ".join(exp.get("bullets", [])[:3])  # Top 3 bullets
        experience_parts.append(f"{exp_text}. {bullets}")
    chunks["experience"] = " | ".join(experience_parts) if experience_parts else "N/A"
    
    # Education chunk
    education_parts = []
    for edu in data.get("education", []):
        edu_text = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
        education_parts.append(edu_text)
    chunks["education"] = " | ".join(education_parts) if education_parts else "N/A"
    
    return chunks

def create_qdrant_point(data: Dict, s3_pdf_key: str, s3_json_key: str) -> str:
    """Create and upsert a point to Qdrant with multi-vectors"""
    point_id = str(uuid.uuid4())
    
    # Create chunks
    chunks = create_chunks_from_json(data)
    
    # Generate embeddings
    vectors = {
        "v_profile": bedrock_embed(chunks["profile"]),
        "v_skills": bedrock_embed(chunks["skills"]),
        "v_experience": bedrock_embed(chunks["experience"]),
        "v_education": bedrock_embed(chunks["education"]),
    }
    
    # Prepare payload
    candidate = data.get("candidate", {})
    payload = {
        "full_name": candidate.get("full_name"),
        "department": candidate.get("department", "Unknown"),
        "email": candidate.get("email"),
        "phone": candidate.get("phone"),
        "location": candidate.get("location"),
        "headline": candidate.get("headline"),
        "summary": candidate.get("summary"),
        "total_years_experience": candidate.get("total_years_experience", 0),
        "skills_flat": chunks["skills"].split(", "),
        "experience_json": data.get("experience", []),
        "education_json": data.get("education", []),
        "s3_pdf_key": s3_pdf_key,
        "s3_json_key": s3_json_key,
        "indexed_at": datetime.utcnow().isoformat(),
    }
    
    # Upsert to Qdrant
    client.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(id=point_id, vector=vectors, payload=payload)]
    )
    
    return point_id

# ============================================================
# BATCH PROCESSING FROM S3
# ============================================================
def process_pdfs_from_s3(limit: int = 100) -> Tuple[int, int, int]:
    """
    Process unprocessed PDFs from S3
    Returns: (success_count, failed_count, skipped_count)
    """
    unprocessed_pdfs = list_unprocessed_pdfs(S3_RAW_BUCKET)[:limit]
    
    if not unprocessed_pdfs:
        return 0, 0, 0
    
    success = 0
    failed = 0
    skipped = 0
    
    progress = st.progress(0.0, text="Processing PDFs from S3...")
    status = st.empty()
    
    for idx, pdf_info in enumerate(unprocessed_pdfs):
        try:
            pdf_key = pdf_info['key']
            department = pdf_info['metadata'].get('department', 'Unknown')
            file_hash = pdf_info['metadata'].get('md5_hash', '')
            
            # Download PDF
            pdf_content = download_from_s3(S3_RAW_BUCKET, pdf_key)
            if not pdf_content:
                failed += 1
                continue
            
            # Extract text
            resume_text = extract_text_from_pdf(pdf_content)
            if not resume_text or len(resume_text) < 50:
                skipped += 1
                mark_as_processed(S3_RAW_BUCKET, pdf_key)
                continue
            
            # Parse with Bedrock
            parsed_data = parse_resume_with_bedrock(resume_text, department)
            if not parsed_data:
                failed += 1
                continue
            
            # Save JSON to S3
            json_key = pdf_key.replace('.pdf', '.json').replace(S3_RAW_BUCKET, '')
            if json_key.startswith('/'):
                json_key = json_key[1:]
            
            json_content = json.dumps(parsed_data, indent=2)
            upload_to_s3(
                S3_JSON_BUCKET,
                json_key,
                json_content.encode('utf-8'),
                {
                    'source_pdf': pdf_key,
                    'department': department,
                    'md5_hash': file_hash,
                    'parsed_date': datetime.utcnow().isoformat()
                }
            )
            
            # Index in Qdrant
            create_qdrant_point(parsed_data, pdf_key, json_key)
            
            # Mark PDF as processed
            mark_as_processed(S3_RAW_BUCKET, pdf_key)
            
            success += 1
            
        except Exception as e:
            st.error(f"Error processing {pdf_info['key']}: {e}")
            failed += 1
        
        progress.progress((idx + 1) / len(unprocessed_pdfs), 
                         text=f"Processed {idx + 1}/{len(unprocessed_pdfs)}")
        status.write(f"✅ Success: {success} | ❌ Failed: {failed} | ⏭️ Skipped: {skipped}")
    
    progress.empty()
    return success, failed, skipped

# ============================================================
# UPLOAD HANDLING
# ============================================================
def _safe_filename(name: str) -> str:
    """Sanitize filename for S3"""
    name = (name or "resume.pdf").strip()
    name = re.sub(r"[^A-Za-z0-9\.\-_ ]+", "_", name)
    return name[:180] or "resume.pdf"

def upload_single_pdf(filename: str, content: bytes, department: str) -> Tuple[str, str]:
    """
    Upload a single PDF to S3
    Returns: (status, message)
    status: 'success', 'duplicate', 'error'
    """
    try:
        file_hash = calculate_file_hash(content)
        
        # Check for duplicate
        duplicate_key = check_duplicate_in_s3(S3_RAW_BUCKET, file_hash)
        if duplicate_key:
            return 'duplicate', f"Duplicate found: {duplicate_key}"
        
        # Create S3 key with department
        safe_name = _safe_filename(filename)
        s3_key = f"{department}/{file_hash[:8]}_{safe_name}"
        
        # Upload to S3
        metadata = {
            'md5_hash': file_hash,
            'department': department,
            'upload_date': datetime.utcnow().isoformat(),
            'processed': 'false',
            'original_filename': filename
        }
        
        success = upload_to_s3(S3_RAW_BUCKET, s3_key, content, metadata)
        
        if success:
            return 'success', f"Uploaded to: {s3_key}"
        else:
            return 'error', "Upload failed"
    
    except Exception as e:
        return 'error', f"Error: {str(e)}"

def process_uploaded_files(uploaded_files, zip_file, department: str) -> Dict[str, Any]:
    """
    Process uploaded PDFs or ZIP file
    Returns statistics about the upload
    """
    results = {
        'total': 0,
        'success': 0,
        'duplicate': 0,
        'error': 0,
        'details': []
    }
    
    files_to_process = []
    
    # Handle ZIP file
    if zip_file:
        extracted = extract_zip_file(zip_file.getvalue())
        for dept, files in extracted.items():
            for filename, content, file_hash in files:
                files_to_process.append((filename, content, dept))
    
    # Handle individual PDFs
    if uploaded_files:
        for uf in uploaded_files:
            files_to_process.append((uf.name, uf.getvalue(), department))
    
    results['total'] = len(files_to_process)
    
    # Process files
    progress = st.progress(0.0, text="Uploading files to S3...")
    
    for idx, (filename, content, dept) in enumerate(files_to_process):
        status, message = upload_single_pdf(filename, content, dept)
        
        if status == 'success':
            results['success'] += 1
        elif status == 'duplicate':
            results['duplicate'] += 1
        else:
            results['error'] += 1
        
        results['details'].append({
            'filename': filename,
            'department': dept,
            'status': status,
            'message': message
        })
        
        progress.progress((idx + 1) / len(files_to_process), 
                         text=f"Uploaded {idx + 1}/{len(files_to_process)} files")
    
    progress.empty()
    return results

# ============================================================
# SEARCH FUNCTIONALITY (from original app)
# ============================================================
def build_qdrant_filter(min_years=None, dept_contains=None, loc_contains=None):
    """Build Qdrant filter conditions"""
    conditions = []
    if min_years:
        conditions.append(qmodels.FieldCondition(
            key="total_years_experience",
            range=qmodels.Range(gte=min_years)
        ))
    if dept_contains:
        conditions.append(qmodels.FieldCondition(
            key="department",
            match=qmodels.MatchText(text=dept_contains)
        ))
    if loc_contains:
        conditions.append(qmodels.FieldCondition(
            key="location",
            match=qmodels.MatchText(text=loc_contains)
        ))
    return qmodels.Filter(must=conditions) if conditions else None

def search_candidates_from_jd(jd_text: str, top_k: int, per_vector_limit: int, 
                              prefer: str, q_filter=None):
    """Search candidates using job description"""
    # Generate embeddings for JD
    jd_emb_profile = bedrock_embed(jd_text)
    jd_emb_skills = bedrock_embed(jd_text)
    jd_emb_exp = bedrock_embed(jd_text)
    jd_emb_edu = bedrock_embed(jd_text)
    
    # Set weights based on preference
    if prefer == "Skills-relevant":
        weights = {"v_profile": 0.15, "v_skills": 0.5, "v_experience": 0.25, "v_education": 0.1}
    elif prefer == "Experience-relevant":
        weights = {"v_profile": 0.15, "v_skills": 0.25, "v_experience": 0.5, "v_education": 0.1}
    else:  # Balanced
        weights = {"v_profile": 0.25, "v_skills": 0.3, "v_experience": 0.3, "v_education": 0.15}
    
    # Query Qdrant
    query_vector = {
        "v_profile": jd_emb_profile,
        "v_skills": jd_emb_skills,
        "v_experience": jd_emb_exp,
        "v_education": jd_emb_edu,
    }
    
    results = client.search(
        collection_name=COLLECTION,
        query_vector=qmodels.NamedVector(name="v_profile", vector=jd_emb_profile),
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    
    return results, {}

def post_sort_results(results, jd_text: str, sort_mode: str):
    """Sort results based on selected mode"""
    if sort_mode == "Years of experience (desc)":
        return sorted(results, key=lambda r: r.payload.get("total_years_experience", 0), reverse=True)
    elif sort_mode == "Skills match count (desc)":
        jd_skills_lower = set(jd_text.lower().split())
        def skill_match_count(r):
            cand_skills = r.payload.get("skills_flat", [])
            return sum(1 for s in cand_skills if any(jd_skill in s.lower() for jd_skill in jd_skills_lower))
        return sorted(results, key=skill_match_count, reverse=True)
    else:
        return results

def render_candidate_card(idx: int, result, jd_text: str):
    """Render candidate card in UI"""
    p = result.payload
    score = result.score
    
    st.markdown(f"""
    <div class="candidate-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
            <div>
                <h3 style="margin: 0; color: #2c3e50;">{idx}. {p.get('full_name', 'Unknown')}</h3>
                <p style="margin: 5px 0; color: #666;">{p.get('headline', '')}</p>
            </div>
            <div class="metric-card" style="min-width: 80px;">
                <div style="font-size: 1.5em; color: #00B4D8; font-weight: bold;">{score:.2f}</div>
                <div style="font-size: 0.8em; color: #666;">Match Score</div>
            </div>
        </div>
        <p><strong>📧 Email:</strong> {p.get('email', 'N/A')}</p>
        <p><strong>📍 Location:</strong> {p.get('location', 'N/A')}</p>
        <p><strong>🏢 Department:</strong> {p.get('department', 'Unknown')}</p>
        <p><strong>💼 Experience:</strong> {p.get('total_years_experience', 0)} years</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Skills
    skills = p.get("skills_flat", [])[:10]
    if skills:
        st.markdown("**🎯 Key Skills:**")
        st.markdown(" ".join([f'<span class="skill-badge">{s}</span>' for s in skills]), unsafe_allow_html=True)
    
    if st.button(f"Show Details", key=f"details_{idx}"):
        with st.expander("Detailed Information", expanded=True):
            tabs = st.tabs(["Experience", "Education", "Summary"])
            
            with tabs[0]:
                for exp in p.get("experience_json", []):
                    st.write(f"**{exp.get('job_title', '')}** at {exp.get('company', '')}")
                    st.write(f"*{exp.get('start_date', '')} - {exp.get('end_date', '')}*")
                    for bullet in exp.get("bullets", []):
                        st.write(f"• {bullet}")
                    st.write("---")
            
            with tabs[1]:
                for edu in p.get("education_json", []):
                    st.write(f"**{edu.get('degree', '')}** in {edu.get('field_of_study', '')}")
                    st.write(f"{edu.get('institution', '')}")
                    st.write("---")
            
            with tabs[2]:
                st.write(p.get("summary", "No summary available"))

# ============================================================
# MAIN UI
# ============================================================
st.markdown('<h1 class="main-header">📋 Resume Intelligence System</h1>', unsafe_allow_html=True)

# Sidebar for uploads and processing
with st.sidebar:
    st.header("📥 Upload Resumes")
    
    upload_tab, batch_tab = st.tabs(["Upload New", "Process Existing"])
    
    with upload_tab:
        st.subheader("Upload PDF Files")
        uploaded_pdfs = st.file_uploader(
            "Select PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_upload"
        )
        
        st.subheader("Or Upload ZIP File")
        uploaded_zip = st.file_uploader(
            "ZIP containing PDFs (with or without subfolders)",
            type=["zip"],
            key="zip_upload"
        )
        
        default_dept = st.text_input("Default Department (for PDFs)", value="Unknown")
        
        upload_btn = st.button("⬆️ Upload to S3", use_container_width=True)
        
        if upload_btn:
            if not uploaded_pdfs and not uploaded_zip:
                st.warning("Please upload PDF files or a ZIP file")
            else:
                with st.spinner("Uploading to S3..."):
                    results = process_uploaded_files(uploaded_pdfs, uploaded_zip, default_dept)
                
                st.success(f"✅ Uploaded: {results['success']} | ⚠️ Duplicates: {results['duplicate']} | ❌ Errors: {results['error']}")
                
                # Show details
                if results['details']:
                    with st.expander("View Details"):
                        for detail in results['details']:
                            status_class = f"status-{detail['status']}"
                            st.markdown(f"""
                            <div class="upload-status {status_class}">
                                <strong>{detail['filename']}</strong> ({detail['department']})<br>
                                {detail['message']}
                            </div>
                            """, unsafe_allow_html=True)
    
    with batch_tab:
        st.subheader("Process PDFs from S3")
        
        unprocessed_count = len(list_unprocessed_pdfs(S3_RAW_BUCKET))
        st.info(f"📊 Unprocessed PDFs in S3: {unprocessed_count}")
        default_batch = max(1, min(10, unprocessed_count)) if unprocessed_count > 0 else 1
        process_limit = st.number_input("Batch size", min_value=1, max_value=100, value=default_batch)
        # process_limit = st.number_input("Batch size", min_value=1, max_value=100, value=min(10, unprocessed_count))
        
        process_btn = st.button("⚡ Process Batch", use_container_width=True)
        
        if process_btn:
            if unprocessed_count == 0:
                st.warning("No unprocessed PDFs found in S3")
            else:
                with st.spinner("Processing PDFs from S3..."):
                    success, failed, skipped = process_pdfs_from_s3(process_limit)
                
                st.success(f"✅ Processed: {success}")
                if failed:
                    st.error(f"❌ Failed: {failed}")
                if skipped:
                    st.warning(f"⏭️ Skipped: {skipped}")
    
    st.header("🔧 Search Filters")
    
    min_years_val = st.number_input("Min. Years Experience", 0, 30, 0, 1)
    dept_contains = st.text_input("Department Contains")
    loc_contains = st.text_input("Location Contains")
    
    st.subheader("Search Preferences")
    prefer = st.selectbox("Weighting Strategy", ["Balanced", "Skills-relevant", "Experience-relevant"])
    sort_mode = st.selectbox("Sort By", ["Default (Relevance)", "Years of experience (desc)", "Skills match count (desc)"])

# Main search interface
jd_col, controls_col = st.columns([3, 1])

with jd_col:
    jd_text = st.text_area(
        "Job Description",
        height=200,
        label_visibility="visible",
        placeholder="Paste the full job description here..."
    )

with controls_col:
    st.markdown("##### ⚙️ Results")
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=10, label_visibility="visible")
    st.markdown("")
    search_btn = st.button("🔍 Search Candidates", type="primary", use_container_width=True)

# Search results
if search_btn:
    if not jd_text.strip():
        st.error("⚠️ Please paste a Job Description")
        st.stop()
    
    q_filter = build_qdrant_filter(
        min_years=min_years_val if min_years_val > 0 else None,
        dept_contains=dept_contains,
        loc_contains=loc_contains,
    )
    
    with st.spinner("🔍 Searching candidates..."):
        results, _ = search_candidates_from_jd(
            jd_text=jd_text,
            top_k=top_k,
            per_vector_limit=120,
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
        st.warning("No candidates found. Try adjusting filters.")

# System status
with st.expander("📊 System Status"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            unprocessed = len(list_unprocessed_pdfs(S3_RAW_BUCKET))
            st.metric("Unprocessed PDFs", unprocessed)
        except:
            st.metric("Unprocessed PDFs", "N/A")
    
    with col2:
        try:
            qdrant_count = client.count(collection_name=COLLECTION)
            st.metric("Indexed Candidates", qdrant_count.count)
        except:
            st.metric("Indexed Candidates", "N/A")
    
    with col3:
        st.metric("S3 Connection", "✅" if s3_client else "❌")
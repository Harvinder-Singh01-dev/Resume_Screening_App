import boto3
import io
import os
from dotenv import load_dotenv
import logging
from pypdf import PdfReader
load_dotenv()

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
BUCKET_NAME = os.getenv('RAW_RESUME_BUCKET')
PREFIX      = os.getenv('RAW_RESUME_PREFIX')  
S3_REGION = os.getenv('S3_REGION')    

# ─── S3 Client (uses EC2 IAM Role automatically) ──────────────────────────────
s3 = boto3.client("boto3", region_name=S3_REGION)  


def list_departments(bucket: str, prefix: str) -> list[str]:
    """Return all department sub-folder names under Resume_Bank/."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

    departments = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            departments.append(cp["Prefix"]) 
    return departments


def list_pdfs_in_folder(bucket: str, folder_prefix: str) -> list[str]:
    """Return all PDF object keys inside a given folder."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=folder_prefix)

    pdfs = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                pdfs.append(key)
    return pdfs


def read_pdf_from_s3(bucket: str, key: str) -> str:
    """Download a PDF from S3 into memory and extract its text."""
    response = s3.get_object(Bucket=bucket, Key=key)
    pdf_bytes = response["Body"].read()

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def process_all_resumes(bucket: str, prefix: str):
    """Walk every department folder and read each resume PDF."""
    departments = list_departments(bucket, prefix)

    if not departments:
        logger.warning("No department folders found under '%s'", prefix)
        return

    for dept_prefix in departments:
        dept_name = dept_prefix.rstrip("/").split("/")[-1]
        logger.info("── Department: %s ──────────────────────", dept_name)

        pdf_keys = list_pdfs_in_folder(bucket, dept_prefix)

        if not pdf_keys:
            logger.warning("  No PDFs found in %s", dept_prefix)
            continue

        for key in pdf_keys:
            filename = key.split("/")[-1]
            logger.info("  Reading: %s", filename)

            try:
                text = read_pdf_from_s3(bucket, key)
                word_count = len(text.split())

                # ── Testing output ── replace this block with your actual logic
                logger.info("    ✓ Extracted %d words", word_count)
                logger.info("    Preview: %s", text[:200].replace("\n", " "))

            except Exception as e:
                logger.error("    ✗ Failed to read %s — %s", filename, e)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting S3 Resume Reader")
    logger.info("Bucket : %s", BUCKET_NAME)
    logger.info("Prefix : %s", PREFIX)
    process_all_resumes(BUCKET_NAME, PREFIX)
    logger.info("Done.")
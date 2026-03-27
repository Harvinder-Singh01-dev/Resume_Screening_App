import boto3
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv('RAW_RESUME_BUCKET')   # raw_resume_pool
PREFIX      = os.getenv('RAW_RESUME_PREFIX')   # Resume_Bank/
S3_REGION   = os.getenv('S3_REGION')

s3 = boto3.client("s3", region_name=S3_REGION)


def upload_resume(local_file_path: str, department: str):
    """
    Uploads a PDF to:
    s3://raw_resume_pool/Resume_Bank/General/Rakesh.pdf
    """

    if not os.path.exists(local_file_path):
        logger.error("❌ File not found: %s", local_file_path)
        return

    if not local_file_path.lower().endswith(".pdf"):
        logger.error("❌ Only PDF files are allowed.")
        return

    filename = os.path.basename(local_file_path)
    s3_key   = f"{PREFIX}{department}/{filename}"
    # Resume_Bank/ + General + / + Rakesh.pdf
    # → Resume_Bank/General/Rakesh.pdf ✅

    logger.info("Uploading   : %s", local_file_path)
    logger.info("S3 Location : s3://%s/%s", BUCKET_NAME, s3_key)

    try:
        s3.upload_file(
            Filename  = local_file_path,
            Bucket    = BUCKET_NAME,
            Key       = s3_key,
            ExtraArgs = {"ContentType": "application/pdf"}
        )
        logger.info("✅ Uploaded '%s' → s3://%s/%s", filename, BUCKET_NAME, s3_key)

    except Exception as e:
        logger.error("❌ Upload failed: %s", e)


def list_pdfs_in_department(department: str):
    """
    Lists all PDFs under:
    s3://raw_resume_pool/Resume_Bank/General/
    """
    folder_prefix = f"{PREFIX}{department}/"
    # Resume_Bank/ + General + /
    # → Resume_Bank/General/ ✅

    paginator = s3.get_paginator("list_objects_v2")
    pages     = paginator.paginate(Bucket=BUCKET_NAME, Prefix=folder_prefix)

    pdf_keys = [
        obj["Key"]
        for page in pages
        for obj in page.get("Contents", [])
        if obj["Key"].lower().endswith(".pdf")
    ]

    if not pdf_keys:
        logger.info("No PDFs found in: s3://%s/%s", BUCKET_NAME, folder_prefix)
        return

    logger.info("── PDFs in '%s' ──────────────────────", department)
    for key in pdf_keys:
        logger.info("  📄 %s", key.split("/")[-1])


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    LOCAL_PDF_PATH = "Rakesh.pdf"    # PDF file on EC2
    DEPARTMENT     = "General"       # Department folder name

    upload_resume(LOCAL_PDF_PATH, DEPARTMENT)
    list_pdfs_in_department(DEPARTMENT)
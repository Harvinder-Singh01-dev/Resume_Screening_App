import boto3
import os
from dotenv import load_dotenv
import logging
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
BUCKET_NAME = os.getenv('RAW_RESUME_BUCKET')
PREFIX      = os.getenv('RAW_RESUME_PREFIX')
S3_REGION   = os.getenv('S3_REGION')

s3 = boto3.client("s3", region_name=S3_REGION)

def list_departments(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
    return [cp["Prefix"] for page in pages for cp in page.get("CommonPrefixes", [])]


def list_pdfs_in_folder(bucket, folder_prefix):
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=folder_prefix)
    return [obj["Key"] for page in pages for obj in page.get("Contents", []) if obj["Key"].lower().endswith(".pdf")]


def process_all_resumes(bucket, prefix):
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
            logger.info("  📄 %s", filename)


if __name__ == "__main__":
    process_all_resumes(BUCKET_NAME, PREFIX)
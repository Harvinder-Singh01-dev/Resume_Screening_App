#!/usr/bin/env python3
"""
S3 Utility Script for Resume Intelligence System
Provides CLI tools for managing S3 buckets and testing functionality
"""

import sys
import argparse
import hashlib
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import json
from pathlib import Path

# Configuration
S3_RAW_BUCKET = "raw-resumes"
S3_JSON_BUCKET = "parsed-resumes-json"
AWS_REGION = "ap-south-1"

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

def create_buckets():
    """Create both S3 buckets if they don't exist"""
    buckets = [S3_RAW_BUCKET, S3_JSON_BUCKET]
    
    for bucket in buckets:
        try:
            # Check if bucket exists
            s3_client.head_bucket(Bucket=bucket)
            print(f"✅ Bucket '{bucket}' already exists")
        except ClientError:
            # Create bucket
            try:
                if AWS_REGION == 'ap-south-1':
                    s3_client.create_bucket(Bucket=bucket)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                    )
                print(f"✅ Created bucket: {bucket}")
                
                # Enable versioning
                s3_client.put_bucket_versioning(
                    Bucket=bucket,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                print(f"✅ Enabled versioning for: {bucket}")
                
                # Enable encryption
                s3_client.put_bucket_encryption(
                    Bucket=bucket,
                    ServerSideEncryptionConfiguration={
                        'Rules': [{
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            }
                        }]
                    }
                )
                print(f"✅ Enabled encryption for: {bucket}")
                
            except ClientError as e:
                print(f"❌ Error creating bucket {bucket}: {e}")

def list_unprocessed():
    """List all unprocessed PDF files in raw bucket"""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        unprocessed_count = 0
        
        print(f"\n📋 Unprocessed PDFs in {S3_RAW_BUCKET}:\n")
        print(f"{'File Key':<60} {'Size':<10} {'Department':<15}")
        print("-" * 90)
        
        for page in paginator.paginate(Bucket=S3_RAW_BUCKET):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                if obj['Key'].endswith('.pdf'):
                    try:
                        response = s3_client.head_object(Bucket=S3_RAW_BUCKET, Key=obj['Key'])
                        metadata = response.get('Metadata', {})
                        
                        if metadata.get('processed', 'false') == 'false':
                            size = f"{obj['Size'] / 1024:.1f} KB"
                            dept = metadata.get('department', 'Unknown')
                            print(f"{obj['Key']:<60} {size:<10} {dept:<15}")
                            unprocessed_count += 1
                    except ClientError:
                        continue
        
        print(f"\nTotal unprocessed files: {unprocessed_count}")
        
    except ClientError as e:
        print(f"❌ Error listing files: {e}")

def list_all():
    """List all files in both buckets"""
    for bucket in [S3_RAW_BUCKET, S3_JSON_BUCKET]:
        print(f"\n📦 Files in {bucket}:\n")
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            total_size = 0
            file_count = 0
            
            for page in paginator.paginate(Bucket=bucket):
                if 'Contents' not in page:
                    print("  (empty)")
                    continue
                    
                for obj in page['Contents']:
                    size = obj['Size']
                    total_size += size
                    file_count += 1
                    print(f"  {obj['Key']} ({size / 1024:.1f} KB)")
            
            print(f"\nTotal: {file_count} files, {total_size / (1024*1024):.2f} MB")
            
        except ClientError as e:
            print(f"❌ Error accessing bucket: {e}")

def check_duplicate(file_path: str):
    """Check if a local file already exists in S3"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        file_hash = hashlib.md5(content).hexdigest()
        print(f"File hash: {file_hash}")
        
        # Check in S3
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_RAW_BUCKET):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                try:
                    response = s3_client.head_object(Bucket=S3_RAW_BUCKET, Key=obj['Key'])
                    s3_hash = response.get('Metadata', {}).get('md5_hash')
                    
                    if s3_hash == file_hash:
                        print(f"\n✅ Duplicate found!")
                        print(f"   S3 Key: {obj['Key']}")
                        print(f"   Department: {response.get('Metadata', {}).get('department')}")
                        print(f"   Upload Date: {response.get('Metadata', {}).get('upload_date')}")
                        return True
                except ClientError:
                    continue
        
        print("\n❌ No duplicate found - file is unique")
        return False
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

def upload_test_file(file_path: str, department: str = "Test"):
    """Upload a test PDF file to S3"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        file_hash = hashlib.md5(content).hexdigest()
        filename = Path(file_path).name
        s3_key = f"{department}/{file_hash[:8]}_{filename}"
        
        metadata = {
            'md5_hash': file_hash,
            'department': department,
            'upload_date': datetime.utcnow().isoformat(),
            'processed': 'false',
            'original_filename': filename
        }
        
        s3_client.put_object(
            Bucket=S3_RAW_BUCKET,
            Key=s3_key,
            Body=content,
            Metadata=metadata
        )
        
        print(f"✅ Uploaded successfully!")
        print(f"   S3 Key: {s3_key}")
        print(f"   Hash: {file_hash}")
        print(f"   Size: {len(content) / 1024:.1f} KB")
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except ClientError as e:
        print(f"❌ Upload failed: {e}")

def mark_processed(s3_key: str, processed: bool = True):
    """Mark a file as processed or unprocessed"""
    try:
        # Get current metadata
        response = s3_client.head_object(Bucket=S3_RAW_BUCKET, Key=s3_key)
        metadata = response.get('Metadata', {})
        
        # Update processed flag
        metadata['processed'] = 'true' if processed else 'false'
        if processed:
            metadata['processed_date'] = datetime.utcnow().isoformat()
        
        # Copy object with new metadata
        s3_client.copy_object(
            Bucket=S3_RAW_BUCKET,
            Key=s3_key,
            CopySource={'Bucket': S3_RAW_BUCKET, 'Key': s3_key},
            Metadata=metadata,
            MetadataDirective='REPLACE'
        )
        
        status = "processed" if processed else "unprocessed"
        print(f"✅ Marked {s3_key} as {status}")
        
    except ClientError as e:
        print(f"❌ Error: {e}")

def clear_bucket(bucket_name: str, confirm: bool = False):
    """Clear all files from a bucket (use with caution!)"""
    if not confirm:
        response = input(f"⚠️  Are you sure you want to delete ALL files from {bucket_name}? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        deleted_count = 0
        
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' not in page:
                print(f"Bucket {bucket_name} is already empty")
                return
                
            # Delete in batches
            objects = [{'Key': obj['Key']} for obj in page['Contents']]
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects}
            )
            deleted_count += len(objects)
            print(f"Deleted {deleted_count} files...")
        
        print(f"✅ Cleared {deleted_count} files from {bucket_name}")
        
    except ClientError as e:
        print(f"❌ Error: {e}")

def bucket_stats():
    """Show statistics for both buckets"""
    for bucket in [S3_RAW_BUCKET, S3_JSON_BUCKET]:
        print(f"\n📊 Statistics for {bucket}:")
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            total_size = 0
            file_count = 0
            departments = {}
            processed_count = 0
            unprocessed_count = 0
            
            for page in paginator.paginate(Bucket=bucket):
                if 'Contents' not in page:
                    print("  (empty)")
                    continue
                    
                for obj in page['Contents']:
                    file_count += 1
                    total_size += obj['Size']
                    
                    try:
                        response = s3_client.head_object(Bucket=bucket, Key=obj['Key'])
                        metadata = response.get('Metadata', {})
                        dept = metadata.get('department', 'Unknown')
                        departments[dept] = departments.get(dept, 0) + 1
                        
                        if bucket == S3_RAW_BUCKET:
                            if metadata.get('processed', 'false') == 'true':
                                processed_count += 1
                            else:
                                unprocessed_count += 1
                    except ClientError:
                        continue
            
            print(f"  Total files: {file_count}")
            print(f"  Total size: {total_size / (1024*1024):.2f} MB")
            
            if bucket == S3_RAW_BUCKET:
                print(f"  Processed: {processed_count}")
                print(f"  Unprocessed: {unprocessed_count}")
            
            if departments:
                print(f"  By department:")
                for dept, count in sorted(departments.items()):
                    print(f"    - {dept}: {count}")
            
        except ClientError as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='S3 Utility for Resume Intelligence System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create              Create S3 buckets
  %(prog)s list                List all files in buckets
  %(prog)s unprocessed         List unprocessed PDFs
  %(prog)s stats               Show bucket statistics
  %(prog)s upload test.pdf     Upload a test file
  %(prog)s check test.pdf      Check if file is duplicate
  %(prog)s mark file.pdf       Mark file as processed
  %(prog)s clear-raw           Clear raw bucket (careful!)
        """
    )
    
    parser.add_argument('command', choices=[
        'create', 'list', 'unprocessed', 'stats', 'upload', 
        'check', 'mark', 'unmark', 'clear-raw', 'clear-json'
    ])
    parser.add_argument('file', nargs='?', help='File path (for upload/check/mark)')
    parser.add_argument('--dept', default='Test', help='Department name for upload')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_buckets()
    elif args.command == 'list':
        list_all()
    elif args.command == 'unprocessed':
        list_unprocessed()
    elif args.command == 'stats':
        bucket_stats()
    elif args.command == 'upload':
        if not args.file:
            print("❌ Please specify a file to upload")
            return
        upload_test_file(args.file, args.dept)
    elif args.command == 'check':
        if not args.file:
            print("❌ Please specify a file to check")
            return
        check_duplicate(args.file)
    elif args.command == 'mark':
        if not args.file:
            print("❌ Please specify an S3 key to mark")
            return
        mark_processed(args.file, True)
    elif args.command == 'unmark':
        if not args.file:
            print("❌ Please specify an S3 key to unmark")
            return
        mark_processed(args.file, False)
    elif args.command == 'clear-raw':
        clear_bucket(S3_RAW_BUCKET)
    elif args.command == 'clear-json':
        clear_bucket(S3_JSON_BUCKET)

if __name__ == '__main__':
    main()
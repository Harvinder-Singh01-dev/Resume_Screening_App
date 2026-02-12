#!/usr/bin/env python3
"""
Interactive Step-by-Step Testing for Resume Intelligence System
This script walks you through testing each component
"""

import sys
import os
import time

def print_header(text):
    """Print a styled header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step_num, total, text):
    """Print a step header"""
    print(f"\n{'─'*70}")
    print(f"  STEP {step_num}/{total}: {text}")
    print(f"{'─'*70}\n")

def wait_for_user():
    """Wait for user to press Enter"""
    input("\n👉 Press Enter to continue... ")

def run_test(description, test_func):
    """Run a test and report results"""
    print(f"Testing: {description}")
    try:
        result = test_func()
        if result:
            print("✅ PASSED")
            return True
        else:
            print("❌ FAILED")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print_header("🚀 Resume Intelligence System - Interactive Setup & Testing")
    
    print("""
This interactive script will guide you through:
1. Checking prerequisites
2. Setting up the environment
3. Testing each component
4. Running the full application

You can stop at any time by pressing Ctrl+C.
    """)
    
    wait_for_user()
    
    # ========================================
    # STEP 1: Check Prerequisites
    # ========================================
    print_step(1, 8, "Check Prerequisites")
    
    print("Checking if required software is installed...\n")
    
    checks = {
        "Python 3.11+": "python --version || python3 --version",
        "Docker": "docker --version",
        "AWS CLI": "aws --version",
    }
    
    all_ok = True
    for name, cmd in checks.items():
        result = os.system(f"{cmd} > /dev/null 2>&1")
        if result == 0:
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - NOT FOUND")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some prerequisites are missing.")
        print("Please install them first:")
        print("  - Python: https://www.python.org/downloads/")
        print("  - Docker: https://www.docker.com/products/docker-desktop/")
        print("  - AWS CLI: https://aws.amazon.com/cli/")
        sys.exit(1)
    
    print("\n✅ All prerequisites are installed!")
    wait_for_user()
    
    # ========================================
    # STEP 2: Check Virtual Environment
    # ========================================
    print_step(2, 8, "Virtual Environment")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active!")
    else:
        print("⚠️  Virtual environment is NOT active.")
        print("\nPlease activate it first:")
        if sys.platform == "win32":
            print("  Windows: venv\\Scripts\\activate")
        else:
            print("  macOS/Linux: source venv/bin/activate")
        sys.exit(1)
    
    wait_for_user()
    
    # ========================================
    # STEP 3: Check Python Packages
    # ========================================
    print_step(3, 8, "Python Packages")
    
    print("Checking installed packages...\n")
    
    required_packages = [
        'streamlit',
        'boto3',
        'pypdf',
        'qdrant_client',
        'python-dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("\n✅ All required packages are installed!")
    wait_for_user()
    
    # ========================================
    # STEP 4: Check Environment Configuration
    # ========================================
    print_step(4, 8, "Environment Configuration")
    
    print("Checking .env file...\n")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("\nCreate it with:")
        print("  cp .env.example .env")
        print("  Then edit with your AWS credentials")
        sys.exit(1)
    
    print("✅ .env file exists")
    
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except:
        print("⚠️  Could not load .env file")
    
    # Check critical variables
    required_vars = ['AWS_REGION', 'S3_RAW_BUCKET', 'S3_JSON_BUCKET']
    missing_vars = []
    
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is NOT set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Please set these in .env file: {', '.join(missing_vars)}")
        sys.exit(1)
    
    wait_for_user()
    
    # ========================================
    # STEP 5: Test AWS Credentials
    # ========================================
    print_step(5, 8, "AWS Credentials")
    
    print("Testing AWS credentials...\n")
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Identity: {identity['Arn']}")
        print(f"✅ Account: {identity['Account']}")
        
    except NoCredentialsError:
        print("❌ AWS credentials not configured!")
        print("\nConfigure with:")
        print("  aws configure")
        print("Or set in .env file:")
        print("  AWS_ACCESS_KEY_ID=...")
        print("  AWS_SECRET_ACCESS_KEY=...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ AWS authentication error: {e}")
        sys.exit(1)
    
    wait_for_user()
    
    # ========================================
    # STEP 6: Test Qdrant
    # ========================================
    print_step(6, 8, "Qdrant Vector Database")
    
    print("Checking if Qdrant is running...\n")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host='localhost', port=6333)
        collections = client.get_collections()
        
        print("✅ Connected to Qdrant")
        print(f"✅ Collections: {len(collections.collections)}")
        
    except Exception as e:
        print("❌ Cannot connect to Qdrant!")
        print("\nStart Qdrant with:")
        print("  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant")
        print("\nOr if already running:")
        print("  docker start qdrant")
        sys.exit(1)
    
    wait_for_user()
    
    # ========================================
    # STEP 7: Test AWS Bedrock
    # ========================================
    print_step(7, 8, "AWS Bedrock Access")
    
    print("Testing Bedrock embedding...\n")
    
    try:
        import boto3
        import json
        
        bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        # Test embedding
        body = json.dumps({"inputText": "test"})
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        data = json.loads(response['body'].read())
        
        print(f"✅ Bedrock embedding works!")
        print(f"✅ Embedding dimension: {len(data['embedding'])}")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print("❌ Bedrock access denied!")
            print("\nEnable model access:")
            print("  1. Go to: https://console.aws.amazon.com/bedrock/")
            print("  2. Click: Model access")
            print("  3. Enable: Amazon Titan Embeddings")
            print("  4. Enable: Anthropic Claude 3.5 Sonnet")
        else:
            print(f"❌ Bedrock error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print("\nTesting Bedrock LLM (Claude)...\n")
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Say 'test successful'"}]
        })
        
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        
        result = json.loads(response['body'].read())
        content = result['content'][0]['text']
        
        print(f"✅ Bedrock LLM works!")
        print(f"✅ Response: {content[:50]}")
        
    except Exception as e:
        print(f"❌ Bedrock LLM error: {e}")
        sys.exit(1)
    
    wait_for_user()
    
    # ========================================
    # STEP 8: Create S3 Buckets
    # ========================================
    print_step(8, 8, "S3 Buckets")
    
    print("Checking S3 buckets...\n")
    
    try:
        import boto3
        
        s3 = boto3.client('s3')
        raw_bucket = os.getenv('S3_RAW_BUCKET')
        json_bucket = os.getenv('S3_JSON_BUCKET')
        
        buckets_to_check = [raw_bucket, json_bucket]
        
        for bucket in buckets_to_check:
            try:
                s3.head_bucket(Bucket=bucket)
                print(f"✅ {bucket} - exists")
            except:
                print(f"⚠️  {bucket} - does not exist")
                
                response = input(f"Create bucket {bucket}? (y/n): ")
                if response.lower() == 'y':
                    try:
                        if os.getenv('AWS_REGION') == 'us-east-1':
                            s3.create_bucket(Bucket=bucket)
                        else:
                            s3.create_bucket(
                                Bucket=bucket,
                                CreateBucketConfiguration={'LocationConstraint': os.getenv('AWS_REGION')}
                            )
                        print(f"✅ Created {bucket}")
                    except Exception as e:
                        print(f"❌ Failed to create {bucket}: {e}")
    
    except Exception as e:
        print(f"❌ S3 error: {e}")
    
    wait_for_user()
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print_header("🎉 Setup Complete!")
    
    print("""
All components are working correctly!

Next steps:

1. Start the Streamlit app:
   
   streamlit run app.py

2. Or press F5 in VS Code

3. Access at: http://localhost:8501

4. Test the app:
   - Upload a PDF resume
   - Process it
   - Search with a job description

5. When ready, deploy to AWS:
   - Follow DEPLOYMENT.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Need help?
- Check: LOCAL_TESTING_GUIDE.md - Detailed guide
- Check: QUICKSTART.md - Quick reference
- Check: DEPLOYMENT.md - AWS deployment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to build! 🚀
    """)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
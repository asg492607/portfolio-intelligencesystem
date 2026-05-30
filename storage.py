import os
import boto3
from botocore.client import Config

# Configuration environment variables
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "portfolio-intelligence")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

LOCAL_STORAGE_DIR = "local_storage"
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

class StorageClient:
    def __init__(self):
        self.use_s3 = False
        self.s3_client = None
        
        if S3_ACCESS_KEY and S3_SECRET_KEY:
            try:
                session_args = {
                    "aws_access_key_id": S3_ACCESS_KEY,
                    "aws_secret_access_key": S3_SECRET_KEY,
                    "region_name": S3_REGION
                }
                client_args = {
                    "config": Config(signature_version="s3v4")
                }
                if S3_ENDPOINT:
                    client_args["endpoint_url"] = S3_ENDPOINT

                self.s3_client = boto3.client("s3", **session_args, **client_args)
                self.use_s3 = True
                
                # Check / Create bucket if possible
                try:
                    self.s3_client.create_bucket(Bucket=S3_BUCKET)
                except Exception:
                    # Bucket might already exist or permission restricted
                    pass
            except Exception as e:
                print(f"Failed to initialize S3 client: {e}. Falling back to local storage.")
                self.use_s3 = False

    def upload_file(self, local_path: str, object_name: str) -> str:
        """Upload a file to S3/MinIO or copy to local storage directory. Returns target URL/path."""
        if self.use_s3:
            try:
                self.s3_client.upload_file(local_path, S3_BUCKET, object_name)
                # Formulate S3/MinIO URL
                if S3_ENDPOINT:
                    return f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"
                return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{object_name}"
            except Exception as e:
                print(f"Error uploading file {local_path} to S3: {e}. Falling back to local.")

        # Local fallback
        dest_path = os.path.join(LOCAL_STORAGE_DIR, object_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            import shutil
            shutil.copy2(local_path, dest_path)
            return dest_path
        except Exception as e:
            print(f"Failed to copy file to local storage: {e}")
            return local_path

    def upload_data(self, data: bytes, object_name: str, content_type: str = "application/json") -> str:
        """Upload raw byte data to S3 or write to local storage directory."""
        if self.use_s3:
            try:
                self.s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=object_name,
                    Body=data,
                    ContentType=content_type
                )
                if S3_ENDPOINT:
                    return f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"
                return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{object_name}"
            except Exception as e:
                print(f"Error putting data for {object_name} to S3: {e}. Falling back to local.")

        # Local fallback
        dest_path = os.path.join(LOCAL_STORAGE_DIR, object_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            with open(dest_path, "wb") as f:
                f.write(data)
            return dest_path
        except Exception as e:
            print(f"Failed to write byte data to local storage: {e}")
            return f"local://{object_name}"

# Export instance
storage_client = StorageClient()

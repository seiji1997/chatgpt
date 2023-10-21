import boto3
import os
from google.cloud import storage

def lambda_handler(event, context):
    s3_bucket = event['Records'][0]['s3']['bucket']['name']
    s3_key = event['Records'][0]['s3']['object']['key']
    
    # Initialize GCP Cloud Storage client
    client = storage.Client()
    bucket = client.get_bucket('your-gcp-bucket')
    
    # Copy data from S3 to GCP
    blob = bucket.blob(s3_key)
    blob.upload_from_filename(s3_key)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Data migrated to GCP Cloud Storage')
    }

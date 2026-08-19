import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import boto3

s3 = boto3.client("s3")


def handler(event, context):
    """Validate a minimal envelope and archive it to encrypted S3."""
    required = {"timestamp", "device_id", "telemetry"}
    missing = required.difference(event)
    if missing:
        return {"statusCode": 400, "body": json.dumps({"missing": sorted(missing)})}
    now = datetime.now(timezone.utc)
    key = f"telemetry/year={now:%Y}/month={now:%m}/day={now:%d}/{uuid4()}.json"
    s3.put_object(
        Bucket=os.environ["BUCKET_NAME"],
        Key=key,
        Body=json.dumps(event).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return {"statusCode": 202, "body": json.dumps({"object_key": key})}


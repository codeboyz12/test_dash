import boto3
import config

bedrock_rt = boto3.client("bedrock-runtime", region_name=config.REGION)
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=config.REGION)
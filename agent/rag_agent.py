from agent.bedrock_client import bedrock_agent
from agent import config as config
from agent import prompt_templates as prompt_templates

def ask_rag(query: str) -> str:
    kb_conf = {
        "knowledgeBaseId": config.KB_ID,
        "modelArn": config.MODEL_ARN,
    }

    kb_conf.update(prompt_templates.template)

    resp = bedrock_agent.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "knowledgeBaseConfiguration": kb_conf,
            "type": "KNOWLEDGE_BASE"
        }
    )
    return resp["output"]["text"]

def normalize_output(text: str) -> str:
    # ตัด space เกิน และรวมบรรทัด
    lines = [line.strip("•- ") for line in text.splitlines() if line.strip()]
    return "\n".join(f"- {line}" for line in lines)


def ask_alert(query: str) -> str:
    kb_conf = {
        "knowledgeBaseId": config.KB_ID,
        "modelArn": config.MODEL_ARN,
    }

    kb_conf.update(prompt_templates.alert_card)

    resp = bedrock_agent.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "knowledgeBaseConfiguration": kb_conf,
            "type": "KNOWLEDGE_BASE"
        }
    )

    rest_text = normalize_output(resp["output"]["text"].strip())
    print(f"[ask_alert][opt]: {rest_text}")
    return rest_text
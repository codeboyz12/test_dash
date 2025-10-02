from bedrock_client import bedrock_agent
import config
import prompt_templates

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

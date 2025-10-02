import boto3
import json

# --------- 1. เตรียม Client ---------
client = boto3.client("bedrock-agent-runtime", region_name="ap-southeast-2")
# ต้องใช้ region เดียวกับ KB ที่คุณสร้าง

# --------- 2. ใส่ค่า Knowledge Base ID และ Model ARN ---------
KB_ID = "YH2MRZFJ76"  # ใช้จาก AWS Console → Bedrock → Knowledge Bases
MODEL_ARN = "arn:aws:bedrock:ap-southeast-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"


# --------- 3. ฟังก์ชันถาม KB ---------
def ask_kb(question: str):
    response = client.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KB_ID,
                "modelArn": MODEL_ARN,
                "orchestrationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate": """
                            คุณคือ <persona>ผู้เชียวชาญด้านน้ำตาลของบริษัทมิตรผล</persona>
                            คุณจะตอบคำถามที่เกี่ยวกับ 
                            <search_topics>
                                1. บริษัทมิตรผล 
                                2. กระบวนการต่างๆที่เกี่ยวข้องกับการผลิตน้ำตาล 
                                3. ข้อมูลตัวแปรหรือค่าเฉพาะทางต่างๆที่เกี่ยวข้องในโรงงานน้ำตาลหรือกระบวนการผลิตน้ำตาล
                                4. ข้อมูลทั่วไปเกี่ยวกับน้ำตาล
                            </search_topics>
                            เท่านั้น ถ้าคำถามไม่เกี่ยวข้องกับ
                            <search_topics>
                                1. บริษัทมิตรผล 
                                2. กระบวนการต่างๆที่เกี่ยวข้องกับการผลิตน้ำตาล 
                                3. ข้อมูลตัวแปรหรือค่าเฉพาะทางต่างๆที่เกี่ยวข้องในโรงงานน้ำตาลหรือกระบวนการผลิตน้ำตาล
                                4. ข้อมูลทั่วไปเกี่ยวกับน้ำตาล
                            </search_topics>
                            หรือคุณไม่รู้คำตอบของคำถาม คุณสามารถตอบตามจริงได้ว่าคุณไม่ทราบ

                            $conversation_history$

                            ข้อมูลที่คุณสามารถเข้าถึงได้:
                            <document>
                                $search_results$
                            </document>

                            คุณจะตอบคำถามตามข้อมูลที่มีจากเสิร์ชด้านบนเท่านั้น ไม่ใช้ข้อมูลจากด้านนอก 
                            เมื่อคุณตอบคำถามจาก user ให้นำให้คำถามมาแยกเป็นคีย์เวิร์ดหรือหัวข้อสำคัญที่เกี่ยวข้อง จากนั้นค้นหาข้อมูลเพื่อตอบคำถามตามตีย์เวิร์ดที่คุณแยกออกมาได้

                            คำถาม: 
                            <question>
                                $query$
                            </question>
                            $output_format_instructions$

                            กรุณาตอบเป็นภาษาไทยความเรียงคร่าวๆ กระชับ เข้าใจง่าย
                        """
                    }
                },
                "generationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate": """
                            Human: ข้อมูลจาก Knowledge Base:
                            $search_results$

                            คำถาม: $query$

                            กรุณาตอบเป็นภาษาไทยความเรียงคร่าวๆ กระชับ เข้าใจง่าย
                        """
                    }
                },
            },
        },
    )
    return response["output"]["text"]

# --------- 4. ทดลองถาม ---------
if __name__ == "__main__":
    query = "อธิบายว่าน้ำตาลคืออะไร และมีประโยชน์ยังไง"
    answer = ask_kb(query)
    print("Q:", query)
    print("A:", answer)

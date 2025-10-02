from rag_agent import ask_rag, ask_alert

q = "ถ้าเกิด Fermentation infection ขึ้นจริงตามที่ตารางบอก ควรจัดการขั้นตอนแรกยังไง และทำไมต้องทำแบบนั้น?"
print(ask_alert(q))
# infer_onnx.py   ← 部署到8核CPU服务器
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained("./onnx_ner_model")
ort_session = ort.InferenceSession("./onnx_ner_model/model.onnx")

label_list = ["O", "B-NUM", "I-NUM", "B-TYP", "I-TYP", "B-AMT", "I-AMT"]

def parse_text(text: str):
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=64)
    
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0][0]  # [seq_len, num_labels]
    predictions = np.argmax(logits, axis=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    preds = [label_list[p] for p in predictions]
    print(tokens)
    print(preds)
    
    # 简单后处理：提取号码、玩法、金额
    # numbers, play_type, amount = [], "", ""
    # current = ""
    
    # for token, pred in zip(tokens, preds):
    #     if pred in ["B-NUM", "I-NUM"]:
    #         current += token.replace("##", "")
    #     elif current:
    #         numbers.append(current)
    #         current = ""
            
    #     if pred.startswith("B-TYP") or pred.startswith("I-TYP"):
    #         play_type += token
            
    #     if pred.startswith("B-AMT") or pred.startswith("I-AMT"):
    #         amount += re.sub(r"[^\d]", "", token)  # 只留数字
            
    # play_type = play_type.strip()
    # amount = int(amount) if amount.isdigit() else 0
    
    # return {
    #     "numbers": numbers,
    #     "play_type": play_type or "未知",
    #     "amount": amount
    # }

# 测试
print(parse_text("123 456 组选10元"))
# 预期输出类似：{'numbers': ['123', '456'], 'play_type': '组选', 'amount': 10}
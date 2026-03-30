import numpy as np
import conf
conf.model_checkpoint = "./best_ner_model"
import train_ner

# ====================== 快速評估 ======================

# eval_results = train_ner.trainer.evaluate()
# from pprint import pprint
# print("\n未 fine-tune 的驗證集效果：")
# pprint(eval_results)

# ====================== 验证集预测结果调试 ======================

sample_texts = train_ner.dataset["validation"]["text"][:5]
# 把验证集转成 tensor 格式进行预测
eval_dataset = train_ner.tokenized_datasets["validation"]
predictions = train_ner.trainer.predict(eval_dataset)   # 使用 trainer.predict 最方便
raw_output = predictions.predictions   # Trainer 返回的原始预测结果

# 1. 兼容多种返回格式（tuple / JointNEROutput / dict）
if isinstance(raw_output, tuple):
    # 最常见的情况：Trainer 把 JointNEROutput 转成了 tuple
    entity_logits = raw_output[0]      # 第1个永远是 entity_logits
    bet_logits    = raw_output[1]      # 第2个是 bet_logits
elif hasattr(raw_output, "entity_logits") and hasattr(raw_output, "bet_logits"):
    # 如果 Trainer 保留了 ModelOutput 对象
    entity_logits = raw_output.entity_logits
    bet_logits    = raw_output.bet_logits
elif isinstance(raw_output, dict):
    # 某些版本会转成 dict
    entity_logits = raw_output.get("entity_logits")
    bet_logits    = raw_output.get("bet_logits")
else:
    # 兜底（基本不会走到这里）
    entity_logits = raw_output
    bet_logits    = None

# 取出预测结果（argmax）
entity_pred_ids = np.argmax(entity_logits, axis=-1)   # (batch_size, seq_len)
bet_pred_ids = np.argmax(bet_logits, axis=-1)

print("=== 验证集预测结果调试 ===")

for i in range(len(sample_texts)):
    tokens = train_ner.tokenizer.tokenize(sample_texts[i])
    
    # entity 部分
    true_entity_label_ids = train_ner.tokenized_datasets["validation"][i]["entity_labels"]
    pred_entity_label_ids = entity_pred_ids[i]
    
    true_entity_labels = [conf.id2entity[l] for l in true_entity_label_ids if l != -100]
    pred_entity_labels = [conf.id2entity[p] for p, l in zip(pred_entity_label_ids, true_entity_label_ids) if l != -100]

    # bet 部分
    true_bet_label_ids = train_ner.tokenized_datasets["validation"][i]["bet_labels"]
    pred_bet_label_ids = bet_pred_ids[i]                    # 注意：这里应该用 bet 的预测！（当前代码有bug）
    
    true_bet_labels = [conf.id2bet[l] for l in true_bet_label_ids if l != -100]
    pred_bet_labels = [conf.id2bet[p] for p, l in zip(pred_bet_label_ids, true_bet_label_ids) if l != -100]

    print(f"\n样本 {i+1}:")
    print("輸入文本 :", sample_texts[i])
    print("分词结果 :", tokens)
    print("真實 Entity :", true_entity_labels)
    print("預測 Entity :", pred_entity_labels)
    print("真實 BET    :", true_bet_labels)
    print("預測 BET    :", pred_bet_labels)
    
    # 额外显示非 O 的数量，帮助判断模型是否在学习
    non_o_count = sum(1 for label in pred_entity_labels if label != "O")
    print(f"預測非 O 實體數量: {non_o_count}")
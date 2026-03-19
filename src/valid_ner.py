from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from pprint import pprint

# ====================== 配置 ======================
model_checkpoint = "distilbert/distilbert-base-multilingual-cased"   # 或 "cycloneboy/chinese_mobilebert_base_f2"
model_checkpoint = "./best_ner_model"
label_list = ["O", "B-NUM", "I-NUM", "B-TYP", "I-TYP", "B-AMT", "I-AMT"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

num_labels = len(label_list)

# ====================== 查看模型下载地址 ======================
# from huggingface_hub import list_repo_files, hf_hub_url
# files = list_repo_files(model_checkpoint)
# pprint(files)
# print(hf_hub_url(model_checkpoint, 'pytorch_model.bin'))
# exit(0)

# ====================== 数据加载 ======================
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=64, is_split_into_words=False)
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)          # special token忽略
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith("B-") else -100)  # I标签只第一个
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ====================== 模型 ======================
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# ====================== 快速評估 ======================




# 定義評估指標（用 seqeval）
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除 ignored_index (-100)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=16,  # 依記憶體調整
        report_to="none",
    ),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# 直接評估驗證集（不進行訓練）
eval_results = trainer.evaluate()

print("\n未 fine-tune 的驗證集效果：")
pprint(eval_results)


# 随便取验证集前几条作为示例
sample_texts = dataset["validation"]["text"][:5]
sample_labels = dataset["validation"]["labels"][:5]

# 预测
predictions = trainer.predict(tokenized_datasets["validation"])

# 取出预测结果（argmax）
pred_ids = np.argmax(predictions.predictions, axis=2)

# 对齐标签，去掉 -100
for i in range(len(sample_texts)):
    tokens = tokenizer.tokenize(sample_texts[i])
    true_label_ids = tokenized_datasets["validation"][i]["labels"]
    pred_label_ids = pred_ids[i]

    true_labels = [label_list[l] for l in true_label_ids if l != -100]
    pred_labels = [label_list[p] for (p, l) in zip(pred_label_ids, true_label_ids) if l != -100]

    print(f"\n樣本 {i+1}:")
    print("輸入文本:", sample_texts[i])
    print("真實標籤:", true_labels)
    print("模型輸出:", pred_labels)

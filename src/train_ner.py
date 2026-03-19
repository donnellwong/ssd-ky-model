# train_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
# ====================== 配置 ======================
model_checkpoint = "distilbert/distilbert-base-multilingual-cased"   # 或 "cycloneboy/chinese_mobilebert_base_f2"
model_checkpoint = "./best_ner_model"
label_list = ["O", "B-NUM", "I-NUM", "B-TYP", "I-TYP", "B-AMT", "I-AMT"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

num_labels = len(label_list)

# ====================== 查看模型下载地址 ======================
# from huggingface_hub import list_repo_files, hf_hub_url
# from pprint import pprint
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

# ====================== 训练 ======================
training_args = TrainingArguments(
    output_dir="./results_ner",
    eval_strategy="epoch",          # ← 這裡改成 eval_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,     # 依 GPU 顯存調整
    per_device_eval_batch_size=64,
    num_train_epochs=5,                 # 建議先固定一個數字，例如 5～10 之間選一個
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",    # 注意：eval_f1 需要 compute_metrics 回傳 "f1"
    greater_is_better=True,
    fp16=True,                          # 如果是 GPU 訓練，開啟混合精度
    report_to="none",                   # 不上傳到 wandb/tensorboard 等
    # 如果你想每 N 步評估一次，可以加：
    # eval_steps=500,                   # 搭配 eval_strategy="steps"
)

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)
    
    # 移除 ignored_index (-100)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # 返回整体指标
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# trainer.train()
trainer.train(resume_from_checkpoint=True) # 接著上次的 checkpoint

# 保存最好模型
trainer.save_model("./best_ner_model")
tokenizer.save_pretrained("./best_ner_model")
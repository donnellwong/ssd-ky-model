import torch
import torch.nn as nn
import numpy as np
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from datasets import load_dataset
from dataclasses import dataclass
import conf

# ====================== 自定义联合模型 ======================
@dataclass
class JointNEROutput(ModelOutput):
    """
    自定义输出类，用于同时返回 entity_logits 和 bet_logits
    """
    loss: Optional[torch.FloatTensor] = None
    entity_logits: torch.FloatTensor = None      # (batch_size, seq_len, num_entity_labels)
    bet_logits: torch.FloatTensor = None         # (batch_size, seq_len, num_bet_labels)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
class JointNERModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 标签数量
        self.num_entity_labels = len(conf.entity_labels)
        self.num_bet_labels = len(conf.bet_labels)
        
        # 共享的BERT编码器
        self.bert = BertModel(config)
        # self.encoder = AutoModel.from_pretrained(
        #     config._name_or_path, 
        #     config=config
        # )
        # 两个独立的分类头
        self.entity_classifier = nn.Linear(config.hidden_size, self.num_entity_labels)
        self.bet_classifier = nn.Linear(config.hidden_size, self.num_bet_labels)
        
        self.post_init() # Hugging Face 标准初始化

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        entity_labels=None,
        bet_labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state             # (batch_size, seq_len, hidden_size)
        entity_logits = self.entity_classifier(sequence_output) # (batch, seq_len, num_entity_labels)
        bet_logits = self.bet_classifier(sequence_output)       # (batch, seq_len, num_bet_labels)

        loss = None
        if entity_labels is not None and bet_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 把 (batch, seq_len, num_labels) → (batch*seq_len, num_labels)
            entity_loss = loss_fct(
                entity_logits.view(-1, self.num_entity_labels),
                entity_labels.view(-1)
            )
            bet_loss = loss_fct(
                bet_logits.view(-1, self.num_bet_labels),
                bet_labels.view(-1)
            )
            # 联合损失：实体损失 + 0.8 * BET损失（可微调权重，实体通常更重要））
            loss = entity_loss + 0.8 * bet_loss

        # 返回标准 ModelOutput，便于 Trainer 处理
        return JointNEROutput(
            loss=loss,
            entity_logits=entity_logits,
            bet_logits=bet_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
# ====================== 数据加载 ======================
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})

tokenizer = AutoTokenizer.from_pretrained(conf.model_checkpoint, use_fast=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors=None,   # map 时保持 list
    )

    
    # ==================== 调试打印 ====================
    print("=== tokenize_and_align_labels DEBUG ===")
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        ent_len = len(examples["entity_labels"][i]) if examples.get("entity_labels") else 0
        bet_len = len(examples["bet_labels"][i]) if examples.get("bet_labels") else 0
        max_word_idx = max((w for w in word_ids if w is not None), default=-1)
        
        print(f"Text: {text}")
        print(f"  word_ids: {word_ids}")
        print(f"  max_word_idx: {max_word_idx} | entity_labels len: {ent_len} | bet_labels len: {bet_len}")
        print(f"  entity_labels: {examples['entity_labels'][i]}")
        print(f"  bet_labels: {examples['bet_labels'][i]}")
        print("-" * 60)
    # ================================================

    # 对齐 entity_labels
    entity_label_ids = []
    for i, labels in enumerate(examples["entity_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 新词的第一个子词
                label_ids.append(conf.entity2id.get(labels[word_idx], -100))
            else:
                # 同一个词的后续子词 → 如果是 B- 则改为 I-
                lbl = labels[word_idx]
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(conf.entity2id.get(lbl, -100))
            previous_word_idx = word_idx
        entity_label_ids.append(label_ids)

    # 对齐 bet_labels（同样处理）
    bet_label_ids = []
    for i, labels in enumerate(examples["bet_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(conf.bet2id.get(labels[word_idx], -100))
            else:
                lbl = labels[word_idx]
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(conf.bet2id.get(lbl, -100))
            previous_word_idx = word_idx
        bet_label_ids.append(label_ids)

    tokenized_inputs["entity_labels"] = entity_label_ids
    tokenized_inputs["bet_labels"] = bet_label_ids
    return tokenized_inputs

tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ====================== 模型 & Trainer ======================
model = JointNERModel.from_pretrained(
    conf.model_checkpoint,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./results_ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_entity_f1",
    greater_is_better=True,
    fp16=True,
    report_to="none",
)
# ======= 数据量少时，调整一下参数，测试用 =======
# training_args = TrainingArguments(
#     output_dir="./results_ner",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,          # 提高一点
#     per_device_train_batch_size=4,  # 数据少就别用32
#     gradient_accumulation_steps=4,  # 有效 batch_size 变大
#     per_device_eval_batch_size=64,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_entity_f1",
#     greater_is_better=True,
#     fp16=True,
#     report_to="none",
# )

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    """
    自定义评估函数（支持 JointNERModel）
    - p.predictions: 模型返回的 logits（当前模型只返回 entity_logits）
    - p.label_ids:   tuple (entity_label_ids, bet_label_ids) 或单 tensor
    """
    # ====================== 1. 解析 predictions ======================
    predictions = p.predictions
    
    # 情况1：Trainer 返回的是 tuple（entity_logits, bet_logits）—— 未来改成 JointNEROutput 后会这样
    if isinstance(predictions, tuple):
        entity_logits = predictions[0]          # (batch, seq_len, num_entity_labels)
        bet_logits = predictions[1] if len(predictions) > 1 else None
    # 情况2：当前模型只返回 entity_logits（TokenClassifierOutput.logits）
    elif isinstance(predictions, np.ndarray) or torch.is_tensor(predictions):
        entity_logits = predictions
        bet_logits = None
    else:
        raise ValueError(f"Unexpected predictions type: {type(predictions)}")

    entity_preds = np.argmax(entity_logits, axis=-1)   # (batch, seq_len)

    # ====================== 2. 解析 labels ======================
    labels = p.label_ids
    
    # labels 可能是 tuple（两个标签列）或单个 tensor
    if isinstance(labels, tuple) or isinstance(labels, list):
        entity_labels = labels[0]      # entity_labels
        bet_labels = labels[1] if len(labels) > 1 else None
    else:
        entity_labels = labels
        bet_labels = None

    # ====================== 3. 转为 seqeval 需要的 label 列表（过滤 -100） ======================
    # entity 指标
    true_entity = []
    pred_entity = []
    for label_seq, pred_seq in zip(entity_labels, entity_preds):
        true_seq = [conf.id2entity[l] for l in label_seq if l != -100]
        pred_seq = [conf.id2entity[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        true_entity.append(true_seq)
        pred_entity.append(pred_seq)

    # ====================== 4. 计算 entity 指标 ======================
    metrics = {
        "entity_precision": precision_score(true_entity, pred_entity, zero_division=0),
        "entity_recall":    recall_score(true_entity, pred_entity, zero_division=0),
        "entity_f1":        f1_score(true_entity, pred_entity, zero_division=0),
        "entity_accuracy":  accuracy_score(true_entity, pred_entity),
    }

    # ====================== 5. 计算 bet 指标（如果存在） ======================
    if bet_logits is not None and bet_labels is not None:
        bet_preds = np.argmax(bet_logits, axis=-1)
        
        true_bet = []
        pred_bet = []
        for label_seq, pred_seq in zip(bet_labels, bet_preds):
            true_seq = [conf.id2bet[l] for l in label_seq if l != -100]
            pred_seq = [conf.id2bet[p] for p, l in zip(pred_seq, label_seq) if l != -100]
            true_bet.append(true_seq)
            pred_bet.append(pred_seq)
        
        metrics["bet_f1"] = f1_score(true_bet, pred_bet, zero_division=0)
        metrics["bet_precision"] = precision_score(true_bet, pred_bet, zero_division=0)
        metrics["bet_recall"] = recall_score(true_bet, pred_bet, zero_division=0)

    # ====================== 6. 可选：打印调试信息（训练时可注释掉） ======================
    # non_o_count = sum(1 for seq in pred_entity for tag in seq if tag != "O")
    # print(f"[DEBUG] Eval Non-O predictions: {non_o_count} / {sum(len(seq) for seq in pred_entity)}")

    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train() # 從頭開始訓練
    # trainer.train(resume_from_checkpoint=True) # 接著上次的 checkpoint

    # 保存最好模型
    trainer.save_model("./best_ner_model")
    tokenizer.save_pretrained("./best_ner_model")
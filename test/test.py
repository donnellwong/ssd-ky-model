import torch
import transformers
print(torch.__version__)
print(transformers.__version__)


# from transformers import AutoTokenizer
# model_checkpoint = "hfl/chinese-bert-wwm-ext"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenizer.save_pretrained("./onnx_ner_model")
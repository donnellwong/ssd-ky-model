from transformers import AutoTokenizer
model_checkpoint = "hfl/chinese-bert-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

from ky_model import ner_tokens
tokenizer.add_special_tokens(ner_tokens.special_tokens)
tokenizer.save_pretrained("./onnx_ner_model")

from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_checkpoint, use_fast=True)
config.save_pretrained("./onnx_ner_model")
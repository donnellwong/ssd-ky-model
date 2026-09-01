from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.exporters.onnx import export
from optimum.exporters.tasks import TasksManager
from pathlib import Path

# Step 1: 加载原始 Transformers 模型和分词器
model_checkpoint = "./best_ner_model"
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

onnx_config = TasksManager.get_exporter_config_constructor(
    model=model,
    exporter="onnx",
    task="token-classification"
)(model.config)


# # Step 2: 导出为 ONNX
onnx_dir = "./onnx_ner_model"
export(
    model=model,
    config=onnx_config,
    output=Path(onnx_dir) / "model.onnx",
    opset=14,
)

# Step 3: 保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.save_pretrained(onnx_dir)

# 可选：简化 & 量化（进一步提速）
# !onnxsim ./onnx_ner_model/model.onnx ./onnx_ner_model/model_simplified.onnx
# 然后用 optimum-cli onnxruntime quantize --onnx_model ./onnx_ner_model/model_simplified.onnx --avx512 ...

# optimum-cli export onnx --model ./best_ner_model ./onnx_ner_model
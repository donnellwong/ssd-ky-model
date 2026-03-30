### 模型选择
```
方案: 小型双向Transformer，做token classification任务
模型: hfl/chinese-bert-wwm-ext
备用1: hfl/chinese-bert-wwm-tiny
备用2: uer/chinese_roberta_tiny
```
### 安装环境
```
# 可以修改cuda版本，执行nvidia-smi查看
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.57.* datasets==4.8.* accelerate==1.13.* seqeval==1.2.* "optimum[onnxruntime]==2.1.*"
pip install onnxruntime==1.24.* onnxsim

pip install -e .
```
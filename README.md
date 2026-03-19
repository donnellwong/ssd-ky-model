### 模型选择
```
方案: 小型双向Transformer，做token classification任务
首选: distilbert-base-multilingual-cased (支持中文，6层，较轻)
备选: cycloneboy/chinese_mobilebert_base_f2
```
### 安装环境
```
# 可以修改cuda版本，执行nvidia-smi查看
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.57.* datasets==4.8.* accelerate==1.13.* seqeval==1.2.* "optimum[onnxruntime]==2.1.*"
pip install onnxruntime==1.24.* onnxsim
```
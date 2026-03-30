model_checkpoint = "hfl/chinese-bert-wwm-ext"
# model_checkpoint = "./best_ner_model"

# Entity 标签（推荐使用 BIO）
entity_labels = ["O", "B-NUM", "I-NUM", "B-TYPE", "I-TYPE", "B-AMOUNT", "I-AMOUNT"]
# Bet 标签（注分组）—— 必须包含 O
bet_labels = ["O", "B-BET", "I-BET"]

entity2id = {label: i for i, label in enumerate(entity_labels)}
bet2id = {label: i for i, label in enumerate(bet_labels)}
id2entity = {i: label for label, i in entity2id.items()}
id2bet = {i: label for label, i in bet2id.items()}

# ====================== 查看模型下载地址 ======================
# from huggingface_hub import list_repo_files, hf_hub_url
# files = list_repo_files(model_checkpoint)
# pprint(files)
# print(hf_hub_url(model_checkpoint, 'pytorch_model.bin'))
# exit(0)

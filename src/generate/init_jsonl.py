import os
import json
from bson import json_util
import ner_conf

PATH_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = f'{PATH_BASE}/data'

file_name = f'{PATH_DATA}/zz_ssd.zz-ner-token.json'

def load_file():
    train_data = []
    valid_data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f, object_hook=json_util.object_hook)
        len1 = len(data)
        len2 = len1 - int(len1 * 0.1)
        for i in range(len1):
            doc = data[i]
            if any(label not in ner_conf.entity_labels for label in doc['labels']):
                print(set([label for label in doc['labels'] if label not in ner_conf.entity_labels]))
                print(ner_conf.entity_labels)
                raise ValueError('labels 标签不在范围')
            if any(label not in ner_conf.bet_labels for label in doc['bet_labels']):
                print(set([label for label in doc['bet_labels'] if label not in ner_conf.bet_labels]))
                print(ner_conf.bet_labels)
                raise ValueError('bet_labels 标签不在范围')

            if len(doc['tokens']) == len(doc['labels']) and len(doc['labels']) == len(doc['bet_labels']):
                d = {
                    'text': ' '.join(doc['tokens']),
                    'entity_labels': doc['labels'],
                    'bet_labels': doc['bet_labels'],
                }
                if i < len2:
                    train_data.append(d)
                else:
                    valid_data.append(d)
            else:
                print(doc)
                print("-" * 60)

    if train_data:
        with open(f'{PATH_DATA}/train.jsonl', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)

    if valid_data:
        with open(f'{PATH_DATA}/valid.jsonl', 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, ensure_ascii=False, indent=4)

if os.path.exists(file_name):
    load_file()
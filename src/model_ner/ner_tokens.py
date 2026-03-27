MAX_LEN = 10
ERR_TOKEN = '[ERR]'

special_tokens = {
    'additional_special_tokens': [
        ERR_TOKEN,
        "直选",
        "组选"
    ]
}

for mark in ['n','x','b','m','B']:
    special_tokens['additional_special_tokens'] += [f"[{i}{mark}]" for i in range(2,MAX_LEN+1)]
for i in range(1,MAX_LEN+1):
    for j in range(1,MAX_LEN+1):
        special_tokens['additional_special_tokens'].append(f'[{i}nx{j}n]') # 如：12x1234

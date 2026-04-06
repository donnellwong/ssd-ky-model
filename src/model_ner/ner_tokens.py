MAX_LEN = 10
LOTTERY_TOKEN = '[L]'
ERR_TOKEN = '[ERR]'

######### add_special_tokens #########

special_tokens = {
    'additional_special_tokens': [
        LOTTERY_TOKEN,
        ERR_TOKEN,
        "—",
        "直选",
        "组选",
        "复式",
        "复试",
        "百位",
        "十位",
        "个位"
    ]
}

for mark in ['n','x','b','m','B']:
    special_tokens['additional_special_tokens'] += [f"[{i}{mark}]" for i in range(2,MAX_LEN+1)]
for i in range(1,MAX_LEN+1):
    for j in range(1,MAX_LEN+1):
        special_tokens['additional_special_tokens'].append(f'[{i}nx{j}n]') # 如：12x1234

######### add_special_tokens end #########

CN_NUMBER = r'零一二两三四五六七八九十百千万'
OTHON_CODES = [
    r'全',
    r'部',
    r'包',
]

LOTTERY_PATTERN = {
    r'(?<!\d)3d': '福',
    r'(?<!\d)3D': '福',
    r'三d': '福',
    r'三D': '福',
    r'三地': '福',
    r'幅': '福',
    r'p3(?!\d)': '体',
    r'P3(?!\d)': '体',
    r'排列三': '体',
    r'排三': '体',
    r'排3(?!\d)': '体',
    r'排': '体',
    r'休': '体',
}

LOTTERY_NAMES = {k.replace(r'(?<!\d)', '').replace(r'(?!\d)', ''):v for k,v in LOTTERY_PATTERN.items()}

NER_LABELS_GAME_NAMES = {
  'ZD3': '直选',
  'ZX3': '组选',
  'DAO': '全倒',
  'ZD': '定位',
  'G3': '组三',
  'G6': '组六',
  'FS': '复式',
  'ZX1': '独胆',
  'ZX2': '双飞',
  'DZ': '对子',
  'BZ': '豹子',
  'K': '跨',
  'HZ': '和值',
}
NER_LABELS_GAME_KEYS = list(NER_LABELS_GAME_NAMES.keys())

NER_LABELS_CODE_NAMES = {
  'CODE': '号码',
  'CODE-BAI': '百位',
  'CODE-SHI': '十位',
  'CODE-GE': '个位',
  'CODE-ALL': '全包',
}
NER_LABELS_CODE_KEYS = list(NER_LABELS_CODE_NAMES.keys())

NER_LABELS_NAMES = {
  **NER_LABELS_CODE_NAMES,
  
  'AMOUNT': '金额',
  'SUM-AMOUNT': '总金额',
  'LOTTERY': '彩种',
  'LENGTH': '注',
  
  **NER_LABELS_GAME_NAMES,

  'O': '非实体',
  'GAME': '玩法',
}
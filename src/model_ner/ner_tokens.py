import re
from ky_ner.ky_confs import txt_replaces

MAX_LEN = 10
ENTER_TOKEN = '[E]'
LOTTERY_TOKEN = '[L]'
TUO_TOKEN = '[n拖n]'
ERR_TOKEN = '[ERR]'

TO_LIST = [
    [r'X', 'x'],
    [r'➕', '+'],
    [r'＊', '*'],
]
F1 = r'[,_…:/%–	  ：—－～~、，。！]+'
F2 = '-'                          # 符号F1统一替换成F2
F = r'[ .+*' + re.escape(F2) + r']'


n2 = r'\[2(?:n|a|x|b)\]'                # 多个号码（2位）
n3 = r'\[[3456789]{1}(?:n|a|x|b)\]'     # 多个号码（3位）
s = F + r'{,3}'                         # 多个号码中间的符号
MULT_PATTERN = f'(?:(?<={n3})(?:{s}{n3}){{2,}}{s}(?={n3})|(?<={n2})(?:{s}{n2}){{2,}}{s}(?={n2}))'
MULT_TOKEN = '[...]'
MULT_NAME = '[忽略多号码]'

######### add_special_tokens #########

special_tokens = {
    'additional_special_tokens': [
        ENTER_TOKEN,
        LOTTERY_TOKEN,
        TUO_TOKEN,
        ERR_TOKEN,
        "x",
        "直选",
        "组选",
        "和值",
        "[2a]",
    ]
}

for mark in ['n','x','b','m','B']:
    special_tokens['additional_special_tokens'] += [f"[{i}{mark}]" for i in range(2,MAX_LEN+1)]
for i in range(1,MAX_LEN+1):
    for j in range(1,MAX_LEN+1):
        special_tokens['additional_special_tokens'].append(f'[{i}nx{j}n]') # 如：12x1234

######### add_special_tokens end #########

ALL_RE_F = re.escape(
    '.()'
    + F2
    + ENTER_TOKEN
    + ''.join([v[1] for v in TO_LIST])
)
ALL_RE = r'[^' + ALL_RE_F + r'\d零一二三四五六七八九十百千万个各大小双单挑直组选拖沾复倒全包不定注共元毛倍独飞豹跨和和值对码打底防福体]'

TXT_REPLACES = txt_replaces
if '和值' in TXT_REPLACES:
    del TXT_REPLACES['和值']

CN_NUMBER = r'零一二三四五六七八九十百千万'
OTHON_CODES = [
    r'全',
    r'包',
]
CN_NOT_NUMBERS = set([
    '组三',
    '组六',
    '十百',
])
PO_TXTS = {
    '各': '个'
}
PO_MAMES = ['百','十','个'] + list(PO_TXTS.keys())

NER_LABELS_GAME_NAMES = {
    'GAME-1': '玩法',   # 1个玩法：除组三组六，包含直组的所有玩法
    'GAME-2': '直组',   # 2个直组玩法组合：直(单)组倒(复)
    'GAME-3': '组36',   # 组三组六，会和GAME-1的双飞和复试一起使用
    'GAME-PO': '定位',
}
NER_LABELS_GAME_KEYS = list(NER_LABELS_GAME_NAMES.keys())

NER_LABELS_CODE_NAMES = {
    'CODE': '号码',
    'CODE-PO-BAI': '百位',
    'CODE-PO-SHI': '十位',
    'CODE-PO-GE': '个位',
    'CODE-ALL': '全包',
    'CODE-ZHAN': '沾边赖',
}
NER_LABELS_CODE_KEYS = [k for k in NER_LABELS_CODE_NAMES.keys() if 'CODE-PO' not in k]

NER_LABELS_NAMES = {
    **NER_LABELS_GAME_NAMES,
    **NER_LABELS_CODE_NAMES,
    
    'AMOUNT': '金额',
    'SUM-AMOUNT': '总金额',
    'LOTTERY': '彩种',
    'LENGTH': '注',

    'O': '非实体',
    'NOTE': '注释',
    'GAME': '玩法',

    'IGNORE': '忽略',
    'ERR': '错',
    'ROW-TO-COL': '横竖换置',
}
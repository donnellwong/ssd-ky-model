import re
from model_ner import ner_tokens
from sys_zabbix.db import zz_ner_log

class NumberReplace:
    def __init__(self, mark, othen_replacers=[]):
        self.mark = mark # 数字替换的标识
        self.ky_text = ''

        self.numbers = []  # 存储原始数字序列
        self.othen_replacers = othen_replacers # 调整坐标
        self.n = 0

    def update_othen_replacers(self, start, l):
        for i in range(len(self.othen_replacers)-1,-1,-1):
            othen_replacer:NumberReplace = self.othen_replacers[i]
            for span, _, _ in othen_replacer.numbers:
                if start < span[0]:
                    span[0] += l
                    span[1] += l
        
    def placeholder(self, placeholder, number_sequence, start, end):
        '''存储原始数字'''
        if self.n:
            start = start + self.n
            end = end + self.n
        l = len(placeholder) - len(number_sequence)
        if l:
            end = end + l
            self.n += l
        span = [start, end]
        self.numbers.append([span, placeholder, number_sequence])
        self.update_othen_replacers(start, l)
        return placeholder
    
    def replace_numbers(self, match:re.Match):
        '''替换字符串中的数字为 '长度n' 形式'''
        number_sequence = match.group()
        len_num = len(number_sequence)
        len_x = len(re.findall(r'[xX]{1}', number_sequence))
        if len_x > 0:
            # 带X符号的处理
            if len_num == 3 and len_x in [1,2]:
                mark = 'x' # 一二字定
            elif len_x == 1:
                arr = re.findall(r'\d+', number_sequence)
                if len(arr) == 2:
                    l1 = len(arr[0])
                    l2 = len(arr[1])
                    if l1 > ner_tokens.MAX_LEN:
                        l1 = ner_tokens.MAX_LEN
                    if l2 > ner_tokens.MAX_LEN:
                        l2 = ner_tokens.MAX_LEN
                    return self.placeholder(f"[{l1}{self.mark}x{l2}{self.mark}]", number_sequence, *match.span())
                else:
                    return self.placeholder(ner_tokens.ERR_TOKEN, number_sequence, *match.span())
            else:
                return self.placeholder(ner_tokens.ERR_TOKEN, number_sequence, *match.span())
        elif len(set(number_sequence)) == 1 and len_num > 1:
            if self.mark == 'n':
                mark = 'b' # 豹子，如：111
            elif self.mark == 'm':
                mark = 'B' # 如：一一一一一一一一一
        else:
            mark = self.mark

        l = len(number_sequence)
        if l > ner_tokens.MAX_LEN:
            l = ner_tokens.MAX_LEN
        return self.placeholder(f'[{mark}]' if l == 1 else f'[{l}{mark}]', number_sequence, *match.span())
    
    def replace_lottery(self, match:re.Match):
        number_sequence = match.group()
        return self.placeholder(self.mark, number_sequence, *match.span())

class TextSub:
    def __init__(self, ky_LOTTERY_PATTERN=[]):
        self.ky_LOTTERY_PATTERN = list(set(ky_LOTTERY_PATTERN + list(ner_tokens.LOTTERY_PATTERN.keys())))
        self.replacer_l = NumberReplace(ner_tokens.LOTTERY_TOKEN) # 彩种
        # 小写字母
        self.replacer_n = NumberReplace('n', othen_replacers=[
            self.replacer_l
        ])
        # 大写字母
        self.replacer_m = NumberReplace('m', othen_replacers=[
            self.replacer_l,
            self.replacer_n
        ])
        
    def replace(self, text):
        self.replacer_l.ky_text = text
        text = re.sub(r'(?:' + r'|'.join(self.ky_LOTTERY_PATTERN) + r')', self.replacer_l.replace_lottery, text)

        self.replacer_n.ky_text = text
        text = re.sub(r'[\dxX]{2,}', self.replacer_n.replace_numbers, text)

        self.replacer_m.ky_text = text
        text = re.sub(r'['+ner_tokens.CN_NUMBER+r']{2,}', self.replacer_m.replace_numbers, text)

        return text
        
    def restore_original(self, text, tokens, offsets):
        '''复原为原始字符串'''
        numbers = self.replacer_l.numbers + self.replacer_n.numbers + self.replacer_m.numbers
        numbers.sort(key=lambda x:x[0])
        numbers = {tuple(span): (placeholder, number_sequence) for span, placeholder, number_sequence in numbers}

        _tokens, _offsets = [], []
        for token, span in zip(tokens, offsets):
            start, end = span
            # 跳过特殊 token 的偏移量，它们的 start 和 end 通常为 (0, 0)
            if start == 0 and end == 0:
                continue
            _tokens.append(token)
            _offsets.append(span)

        with zz_ner_log.NerLogCollection() as noZ:
            data = noZ.get_labels(_tokens)
        if data is None:
            _labels = ['O'] * len(_tokens)
            _bet_labels = ['B-BET'] + ['I-BET'] * (len(_tokens)-1)
        else:
            _labels = data['labels']
            _bet_labels = data['bet_labels']

        res = []
        for token, span, label, bet_label in zip(_tokens, _offsets, _labels, _bet_labels):
            start, end = span
            if span in numbers and numbers[span][0] == token:
                original_span = numbers[span][1]
            else:
                original_span = text[start:end]
            
            # 替换彩种
            if token == ner_tokens.LOTTERY_TOKEN and original_span in ner_tokens.LOTTERY_NAMES:
                original_span = ner_tokens.LOTTERY_NAMES[original_span]
            
            res.append({
                'text': original_span,
                'token': token,
                'label': label,
                'bet_label': bet_label,
            })
        return res
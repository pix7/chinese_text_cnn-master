import re
from torchtext import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)
#结巴分词不显示日志

#正则表达式
regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')

#分词
def word_cut(text):
    text = regex.sub(' ', text) #替换字符串？
    return [word for word in jieba.cut(text) if word.strip()]


# 读取数据，返回训练集和验证集
def get_dataset(path, text_field, label_field):
    #词法分析器，将编写的文本代码流解析为一个一个的记号
    text_field.tokenize = word_cut
    
    #读取tsv格式的文件得到训练集和验证集
    train, dev = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='dev.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev


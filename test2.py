import math
import jieba
import os

# 读取停用词表
with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])
# 读取中文文档库中的所有文档
corpus_path = 'D:/zongruntang/nlp/jyxstxtqj_downcc.com'
filenames = os.listdir(corpus_path)
docs = []
for filename in filenames:
    with open(os.path.join(corpus_path, filename), 'r', encoding='ansi') as f:
        text = f.read()
        # 分词并过滤停用词
        words = [word for word in jieba.cut(text) if word not in stopwords]
        docs.append(words)
# 将所有文档的分词结果合并为一个列表
all_words = []
for doc in docs:
    all_words.extend(doc)
# 去重
unique_words = set(all_words)

# 定义计算汉字信息熵的函数
def calc_chinese_entropy(text):
    # 统计每个汉字出现的次数
    char_count = {}
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            char_count[char] = char_count.get(char, 0) + 1

    # 计算每个汉字出现的概率
    char_prob = {}
    for char in char_count:
        char_prob[char] = char_count[char] / len(text)

    # 计算信息熵
    entropy = 0
    for prob in char_prob.values():
        entropy -= prob * math.log(prob, 2)

    return entropy


# 定义计算汉字二元词组信息熵的函数
def calc_chinese_bigram_entropy(text):
    # 统计每个汉字二元词组出现的次数
    bigram_count = {}
    bigram_len = 0
    char_count = {}
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            char_count[char] = char_count.get(char, 0) + 1

    for i in range(len(text) - 1):
        if '\u4e00' <= text[i] <= '\u9fa5' and '\u4e00' <= text[i + 1] <= '\u9fa5':
            bigram_count[(text[i], text[i + 1])] = bigram_count.get((text[i], text[i + 1]), 0) + 1
            bigram_len+=1

    # 计算每个汉字二元词组出现的概率
    bigram_prob = {}
    entropy = []
    for bigram in bigram_count.items():
        #bigram_prob[bigram] = bigram_count[bigram] / (bigram_len - 1)
        jp_xy = bigram[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bigram[1] / char_count[bigram[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵

    # 计算信息熵
    return round(sum(entropy), 6)

# 定义计算汉字三元词组信息熵的函数
def calc_chinese_trp_entropy(text):
    # 统计每个汉字三元词组出现的次数
    bigram_count = {}
    bigram_len = 0
    trp_count = {}
    trp_len = 0
    for i in range(len(text) - 2):
        if '\u4e00' <= text[i] <= '\u9fa5' and '\u4e00' <= text[i + 1] <= '\u9fa5' and '\u4e00' <= text[i + 2] <= '\u9fa5':
            trp_count[((text[i], text[i + 1]), text[i + 2])] = trp_count.get(((text[i], text[i + 1]), text[i + 2]), 0) + 1
            trp_len += 1
    for i in range(len(text) - 1):
        if '\u4e00' <= text[i] <= '\u9fa5' and '\u4e00' <= text[i + 1] <= '\u9fa5':
            bigram_count[(text[i], text[i + 1])] = bigram_count.get((text[i], text[i + 1]), 0) + 1
            bigram_len += 1

    # 计算每个汉字三元词组出现的概率
    entropy = []
    for bigram in trp_count.items():
        #bigram_prob[bigram] = bigram_count[bigram] / (bigram_len - 1)
        jp_xy = bigram[1] / trp_len  # 计算联合概率p(x,y)
        cp_xy = bigram[1] / bigram_count[bigram[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵

    # 计算信息熵
    return round(sum(entropy), 6)


#计算所有文档的信息熵
for i, doc in enumerate(docs):
    entropy = calc_chinese_entropy(doc)
    print("文档{}的一元信息熵为：{}".format(i+1, entropy))
for i, doc in enumerate(docs):
    entropy = calc_chinese_bigram_entropy(doc)
    print("文档{}的二元信息熵为：{}".format(i+1, entropy))
for i, doc in enumerate(docs):
    entropy = calc_chinese_trp_entropy(doc)
    print("文档{}的三元信息熵为：{}".format(i+1, entropy))

# 计算整个文库的信息熵和二元词组信息熵
char_entropy = calc_chinese_entropy(all_words)
bigram_entropy = calc_chinese_bigram_entropy(all_words)
trp_entropy = calc_chinese_trp_entropy(all_words)
print("整个文库的汉字信息熵为：", char_entropy)
print("整个文库的汉字二元词组信息熵为：", bigram_entropy)
print("整个文库的汉字三元词组信息熵为：", trp_entropy)
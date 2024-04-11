import jieba
import matplotlib.pyplot as plt

# 读取中文文本文件
novel_file = 'D:\zongruntang\\nlp\\tangzongrun\\.idea\\inf.txt'
with open(novel_file, 'r', encoding='ANSI') as f:
    novel_files = f.read().split(',')
    f.close()

novels = []

for novel_file in novel_files:
    novel_file_name = 'D:/zongruntang/nlp/jyxstxtqj_downcc.com/' + novel_file + '.txt'
    with open(novel_file_name, 'r', encoding='ANSI') as f:
        novel_text = f.read()
        novel_text = novel_text.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        novel_text = novel_text.replace(u'\u3000', u'').replace('\n', '').replace('\r', '').replace(" ", "")
        novel_text = novel_text.replace('[', '').replace(']', '').replace('，','').replace('。','').replace('：','').replace('“','').replace('”','')
        novels.append(novel_text)

# 使用jieba进行分词
novel_text_combined = ''.join(novels) #novels 中的所有文本内容合并为一个字符串 novel_text_combined，然后使用 jieba 库对这个字符串进行分词。
seg_lists = ' '.join(jieba.cut(novel_text_combined)) #jieba.cut() 返回的结果是一个迭代器对象，而不是字符串，因此无法直接进行解码操作.将分词结果转换为空格分隔的字符串来实现迭代器对象到字符串的转换。

# 统计词频

word_freq = {}
for word in seg_lists:
    if  word.strip(): #去除空字符串
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

# 将词频字典转换为列表，并按照词频排序
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)


# 对词频进行排序
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# 提取排名和频率
ranks = list(range(1, len(sorted_word_freq) + 1))
frequencies = [item[1] for item in sorted_word_freq]

# 绘制Zipf's Law
plt.figure(figsize=(10, 6))
plt.plot(ranks, frequencies,  linestyle='-')
plt.xlabel('Rank')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Frequency')
plt.title("Zipf's Law")
plt.xlim(1, 10000)  # 设置横坐标范围为1到100
plt.grid(True)
plt.show()

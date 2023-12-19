import pandas as pd
import jieba
from tqdm import tqdm
file = "../data/lawzhidao_filter.csv"

df = pd.read_csv(file, encoding='utf-8')

d = {
    '<UNK>': 0,
    '<BOS>': 1,
    '<EOS>': 2,
    '<PAD>': 3
}

# 填充空值
df = df.fillna("")

cols = ['title', 'question', 'reply']

for col in df.columns:
    if col in cols:
        for i in tqdm(df[col]):
            for j in jieba.lcut(i):
                    d[j] = d.get(j, len(d))


# 将字典写入文件
f = open("./vocab.txt", encoding='utf-8', mode='w')

kv = [(v, k) for k, v in d.items()]

l = sorted(kv, key=lambda x: x[0])

for k, v in l:
    f.write(v)
    f.write('\n')
f.close()
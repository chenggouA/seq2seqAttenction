import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import numpy as np
import os


# os.chdir("../")
class Tokenizer:
    def __init__(self, path="vocab/vocab.txt"):
        import os
        print(os.getcwd())
        assert os.path.exists(path), "词表文件不存在"
        self.id2word = {}
        self.word2id = {}
        self._build_vocab(path)

    def _build_vocab(self, path):
        f = open(path, encoding='utf-8')
        for i, line in enumerate(f):
            self.id2word[i] = line.strip()
            self.word2id[line.strip()] = i

    def w2i(self, source):
        if isinstance(source, list):
            return [self.w2i(i) for i in source]
        else:
            return self.word2id.get(source, 0)

    def i2w(self, source):
        if isinstance(source, list):
            return [self.i2w(i) for i in source]
        else:
            return self.id2word.get(source, 0)

    def __len__(self):
        return len(self.word2id)


tokenizes = Tokenizer()


def get_train_test():
    if os.path.exists("data/data.npz"):
        loaded_data = np.load("data/data.npz")
        X_train = np.squeeze(loaded_data["X_train"])
        X_test = np.squeeze(loaded_data["X_test"])
        y_train = np.squeeze(loaded_data["y_train"])
        y_test = np.squeeze(loaded_data["y_test"])
        return X_train, X_test, y_train, y_test
    else:
        return new_get_train_test()



def new_get_train_test(input_max_len = 150, output_max_len = 350):
    filePath = "data/lawzhidao_filter.csv"
    df = pd.read_csv(filePath, encoding="utf-8")
    df_best = df[df['is_best'] == 1]
    # print("筛选前数据量: ", len(df))
    # print("筛选后数据量: ", len(df_best))

    # 填充空值
    df_best = df_best.fillna('')
    y = [jieba.lcut(i) for i in df_best['reply'].tolist()]
    x = []
    for title, question in zip(df_best['title'].tolist(), df_best['question'].tolist()):
        if question == "":
            x.append(jieba.lcut(title))
        else:
            x.append(jieba.lcut(question))

    # 加上 <BOS> 
    question = [["<BOS>"] + i  for i in x]
    answer = [["<BOS>"] + i  for i in y]

    question_max_len = min(max(map(len, question)), input_max_len) + 1
    answer_max_len = min(max(map(len, answer)), output_max_len) + 1

    # 做截断
    question = [i[0: question_max_len - 1] for i in question]
    answer = [i[0: answer_max_len -1] for i in answer]
    
    # 加上 <EOS>
    
    question = [i + ["<EOS>"]  for i in question]
    answer = [i + ["<EOS>"]  for i in answer]
    
    # pad
    question_pad = [q + ['<PAD>'] * (question_max_len - len(q)) for q in question]
    answer_pad = [a + ['<PAD>'] * (answer_max_len - len(a)) for a in answer]

    X_train, X_test, y_train, y_test = train_test_split(tokenizes.w2i(question_pad), tokenizes.w2i(answer_pad),
                                                        test_size=0.2)

    np.savez('data/data.npz', X_train=np.array(X_train), X_test=np.array(X_test), y_train=np.array(y_train),
             y_test=np.array(y_test))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_train_test()
    print(len(x_train), len(x_test), len(y_train), len(y_test))

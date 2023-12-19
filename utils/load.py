import pandas as pd
from utils import Tokenizer
from sklearn.model_selection import train_test_split

def get_train_test():
    filePath = "./data/lawzhidao_filter.csv"
    df = pd.read_csv(filePath, encoding="utf-8")
    df_best = df[df['is_best'] == 1]
    # print("筛选前数据量: ", len(df))
    # print("筛选后数据量: ", len(df_best))

    # 填充空值
    df_best = df_best.fillna('')
    y = df_best['reply'].tolist()
    x = []
    for title, question in zip(df_best['title'].tolist(), df_best['question'].tolist()):
        if question == "":
            x.append(title)
        else:
            x.append(question)

    return train_test_split(x, y, test_size=0.2)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_train_test()
    print(len(x_train), len(x_test), len(y_train), len(y_test))



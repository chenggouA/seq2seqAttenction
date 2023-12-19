import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os

# 切换工作目录
# os.chdir("../")
from load import get_train_test

X_train, X_test, y_train, y_test = get_train_test()


def save_len_pic(label: str, data) -> None:
    import numpy as np
    data_len = list(map(len, data))
    # counter = Counter(data_len)
    # e = list(counter.keys())
    # f = list(counter.values())
    sns.boxplot(data_len, color='skyblue')
    plt.title(f'{label} Histogram')
    plt.savefig(f"{label}_frequency_histogram.png")


save_len_pic("test", X_test + y_test)
save_len_pic("train", X_train + y_train)

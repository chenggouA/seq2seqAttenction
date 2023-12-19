from utils.load import x, y
import matplotlib.pyplot as plt


x_len = list(map(len, x))
y_len = list(map(len, y))


# 绘制直方图
plt.hist(x_len, bins=max(x_len)-min(x_len)+1, align='left', edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title('Histogram of Element Counts')
plt.xlabel('Element')
plt.ylabel('Count')

# 显示直方图
plt.show()
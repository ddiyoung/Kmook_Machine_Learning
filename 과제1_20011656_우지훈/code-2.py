import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class logistic:
    def __init__(self):
        self.w = np.random.normal(0, 1, (2, 1))
        self.w_0 = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def h_x(self, x):
        self.z = np.dot(x, self.w) + self.w_0
        return self.sigmoid(self.z)


epoch = 100000
h = [logistic() for _ in range(3)]

def train(n, x_data, y_data):
    for e in range(epoch):
        w_gard = np.zeros((2, ))
        w_0_gard = 0
        for x, y in zip(x_data, y_data):
            w = h[n].h_x(x)
            w_gard += (w - y[n]) * x.T
            w_0_gard += w-y[n]

        h[n].w -= 0.1 * (w_gard / len(x_data)).reshape(-1, 1)
        h[n].w_0 -= 0.1 * w_0_gard / len(x_data)


df = pd.read_csv('과제1.csv')

# X2 Kernel_length Manufacture
df['X2 kernel_length'] = df['X2 kernel_length'].apply(lambda x: x.replace(',', '.')).apply(pd.to_numeric)

# Feature Scaling
x1_max, x1_mean, x1_min = df['X1 kernel_area'].max(), df['X1 kernel_area'].mean(), df['X1 kernel_area'].min()
x2_max, x2_mean, x2_min = df['X2 kernel_length'].max(), df['X2 kernel_length'].mean(), df['X2 kernel_length'].min()

df['X1 FS'] = (df['X1 kernel_area'] - x1_mean) / (x1_max - x1_min)
df['X2 FS'] = (df['X2 kernel_length'] -x2_mean) / (x2_max - x2_min)
Uq = df['Wheat Varieties'].unique()
df['Y'] = df['Wheat Varieties'].apply(lambda x : [int(i==x) for i in Uq])

x_data = np.array([x for x in df[['X1 FS', 'X2 FS']].values])
y_data = np.array([y for y in df['Y'].values])


train(0, x_data, y_data)
train(1, x_data, y_data)
train(2, x_data, y_data)

print("End of Train")

plt.scatter([i[0] for i, _ in zip(x_data, y_data) if _.argmax() ==  0], [i[1] for i, _ in zip(x_data, y_data) if _.argmax() ==  0], alpha=0.3, c='b')
plt.scatter([i[0] for i, _ in zip(x_data, y_data) if _.argmax() ==  1], [i[1] for i, _ in zip(x_data, y_data) if _.argmax() ==  1], alpha=0.3, c='r')
plt.scatter([i[0] for i, _ in zip(x_data, y_data) if _.argmax() ==  2], [i[1] for i, _ in zip(x_data, y_data) if _.argmax() ==  2], alpha=0.3, c='g')

x1 = np.array([i[0] for i, _ in zip(x_data, y_data)])
x2 = np.array([i[1] for i, _ in zip(x_data, y_data)])

W = h[0].w
y = -(W[0] * x1 + h[0].w_0) / W[1]
plt.plot(x1, y, c='b')

W = h[1].w
y = -(W[0] * x1 + h[1].w_0) / W[1]
plt.plot(x1, y, c='r')

W = h[2].w
y = -(W[0] * x1 + h[2].w_0) / W[1]
plt.plot(x1, y, c='g')

plt.ylim((-1.5, 1.5))
plt.show()
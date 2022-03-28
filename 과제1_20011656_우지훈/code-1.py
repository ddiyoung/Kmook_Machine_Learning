import pandas as pd

df = pd.read_csv('과제1.csv')

# X2 Kernel_length Manufacture
df['X2 kernel_length'] = df['X2 kernel_length'].apply(
    lambda x: x.replace(',', '.')).apply(pd.to_numeric)

# Feature Scaling
x1_max, x1_mean, x1_min = df['X1 kernel_area'].max(), df['X1 kernel_area'].mean(), df['X1 kernel_area'].min()
x2_max, x2_mean, x2_min = df['X2 kernel_length'].max(), df['X2 kernel_length'].mean(), df['X2 kernel_length'].min()

df['X1 FS'] = (df['X1 kernel_area'] - x1_mean) / (x1_max - x1_min)
df['X2 FS'] = (df['X2 kernel_length'] -x2_mean) / (x2_max - x2_min)


print(df)

import pandas as pd

df = pd.read_csv('dataset\\train\label\\tap2.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\tap3.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\tap4.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\tap5.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\tap6.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\push1.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\push2.csv')

df = df['touch']
print(df.value_counts())

df = pd.read_csv('dataset\\train\label\\push3.csv')

df = df['touch']
print(df.value_counts())
# df = df.reset_index(drop=True)
# print(df.head(5))
# df.to_csv('dataset\\train\label\\tap.csv', sep=',', na_rep='NaN')
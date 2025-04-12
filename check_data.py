import pandas as pd

# 读取训练数据
train_data = pd.read_excel('policy_data.xlsx')
print("\n训练数据的列名：")
print(train_data.columns.tolist())

# 读取测试数据
test_data = pd.read_excel('policy_test.xlsx')
print("\n测试数据的列名：")
print(test_data.columns.tolist()) 
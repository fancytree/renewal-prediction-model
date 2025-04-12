import pandas as pd

# 读取训练数据
train_data = pd.read_excel('policy_data.xlsx')

# 显示每列的数据类型和一些示例值
print("\n训练数据的详细信息：")
for column in train_data.columns:
    print(f"\n列名: {column}")
    print(f"数据类型: {train_data[column].dtype}")
    print("唯一值示例:")
    print(train_data[column].unique()[:5])

# 读取测试数据
test_data = pd.read_excel('policy_test.xlsx')

print("\n\n测试数据的详细信息：")
for column in test_data.columns:
    print(f"\n列名: {column}")
    print(f"数据类型: {test_data[column].dtype}")
    print("唯一值示例:")
    print(test_data[column].unique()[:5]) 
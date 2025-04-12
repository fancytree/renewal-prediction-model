import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def prepare_data(file_path, is_training=True, scaler_params=None):
    """
    准备数据，包括特征工程
    
    Args:
        file_path: 数据文件路径
        is_training: 是否是训练数据
        scaler_params: 标准化参数字典，用于测试数据
    """
    # 读取数据
    data = pd.read_excel(file_path)
    print(f"\n{file_path} 的列名:")
    print(data.columns.tolist())
    
    # 基础特征工程
    data['premium_per_family_member'] = data['premium_amount'] / data['family_members']
    data['premium_per_age'] = data['premium_amount'] / data['age']
    data['age_family_ratio'] = data['age'] / data['family_members']
    
    # 标准化数值特征
    numeric_features = ['premium_amount', 'premium_per_family_member', 'premium_per_age']
    if is_training:
        # 训练数据：计算并保存标准化参数
        scaler_params = {}
        for feature in numeric_features:
            mean = data[feature].mean()
            std = data[feature].std()
            scaler_params[feature] = {'mean': mean, 'std': std}
            data[f'{feature}_standardized'] = (data[feature] - mean) / std
    else:
        # 测试数据：使用训练数据的标准化参数
        for feature in numeric_features:
            mean = scaler_params[feature]['mean']
            std = scaler_params[feature]['std']
            data[f'{feature}_standardized'] = (data[feature] - mean) / std
    
    # 创建年龄组
    data['age_group'] = pd.cut(data['age'], 
                              bins=[0, 30, 40, 50, 60, float('inf')],
                              labels=['<30', '30-40', '40-50', '50-60', '>60'])
    
    # 创建家庭规模类别
    data['family_size_category'] = pd.cut(data['family_members'],
                                        bins=[0, 1, 2, 4, float('inf')],
                                        labels=['single', 'couple', 'small', 'large'])
    
    # 创建保费分位数
    if is_training:
        # 计算分位数边界
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bins = pd.qcut(data['premium_amount'], q=5, labels=False, retbins=True)[1]
        data['premium_quantile'] = pd.qcut(data['premium_amount'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        # 保存分位数边界
        scaler_params['premium_quantiles'] = bins
    else:
        # 使用训练数据的分位数边界
        bins = scaler_params['premium_quantiles']
        data['premium_quantile'] = pd.cut(data['premium_amount'], 
                                        bins=bins,
                                        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                        include_lowest=True)
    
    # 转换分类特征为哑变量
    categorical_features = ['age_group', 'family_size_category', 'premium_quantile']
    dummy_features = pd.get_dummies(data[categorical_features], prefix=categorical_features)
    
    # 选择最终特征
    final_features = pd.concat([
        data[['age', 'family_members']],  # 基础特征
        data[[col for col in data.columns if '_standardized' in col]],  # 标准化特征
        data[['age_family_ratio']],  # 比率特征
        dummy_features  # 分类特征的哑变量
    ], axis=1)
    
    # 打印特征统计信息
    print(f"\n{file_path} 特征统计信息:")
    print(final_features.describe())
    
    if is_training:
        # 将标签转换为数值
        y = (data['renewal'] == 'Yes').astype(int)
        # 对于训练数据，返回特征、标签、特征名称和标准化参数
        return final_features, y, final_features.columns.tolist(), scaler_params
    else:
        # 对于测试数据，只返回特征
        return final_features

def train_decision_tree(X, y, max_depth=4):
    """
    训练决策树模型
    
    Args:
        X: 特征矩阵
        y: 目标变量
        max_depth: 决策树最大深度
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 计算类别权重
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    class_weights = dict(zip(
        np.unique(y_train),
        n_samples / (n_classes * np.bincount(y_train))
    ))
    
    # 创建并训练模型
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        class_weight=class_weights,
        min_samples_leaf=10  # 减小最小叶子节点样本数
    )
    model.fit(X_train, y_train)
    
    # 打印特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\n特征重要性:")
    print(feature_importance[feature_importance['importance'] > 0])
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 打印评估报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实值')
    plt.xlabel('预测值')
    plt.savefig('decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_decision_tree(model, feature_names, class_names):
    """
    可视化决策树
    
    Args:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        class_names: 类别名称列表
    """
    try:
        # 创建决策树可视化
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=False
        )
        
        # 将决策树保存为PNG文件
        graph = graphviz.Source(dot_data)
        graph.render('decision_tree', format='png', cleanup=True)
        print("决策树可视化已保存为 decision_tree.png")
        
    except Exception as e:
        print(f"可视化过程中发生错误: {str(e)}")

def save_tree_rules(model, feature_names, class_names, output_file='decision_tree_rules.txt'):
    """
    保存决策树规则
    
    Args:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        class_names: 类别名称列表
        output_file: 输出文件路径
    """
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    value = model.tree_.value
    
    rules = []
    
    def build_rules(node_id, rule):
        if children_left[node_id] == children_right[node_id]:  # 叶子节点
            class_id = np.argmax(value[node_id][0])
            class_name = class_names[class_id]
            rules.append(f"{rule} -> {class_name}")
            return
        
        feature_name = feature_names[feature[node_id]]
        threshold_value = threshold[node_id]
        
        # 左子树规则
        left_rule = f"{rule} AND {feature_name} <= {threshold_value:.2f}"
        build_rules(children_left[node_id], left_rule)
        
        # 右子树规则
        right_rule = f"{rule} AND {feature_name} > {threshold_value:.2f}"
        build_rules(children_right[node_id], right_rule)
    
    build_rules(0, "IF")
    
    # 保存规则
    with open(output_file, 'w', encoding='utf-8') as f:
        for rule in rules:
            f.write(rule + '\n')

def print_tree_structure(model, feature_names, class_names, indent=''):
    """
    以文本形式打印决策树结构
    
    Args:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        class_names: 类别名称列表
        indent: 缩进字符串
    """
    tree_ = model.tree_
    
    def recurse(node, depth):
        indent = '    ' * depth
        
        if tree_.feature[node] != -2:  # 非叶子节点
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            print(f"{indent}├── 特征: {name} <= {threshold:.2f}")
            
            # 递归左子树
            print(f"{indent}│   ├── 是:")
            recurse(tree_.children_left[node], depth + 2)
            
            # 递归右子树
            print(f"{indent}│   └── 否:")
            recurse(tree_.children_right[node], depth + 2)
        else:  # 叶子节点
            class_probabilities = tree_.value[node][0] / tree_.value[node][0].sum()
            predicted_class = class_names[np.argmax(class_probabilities)]
            prob = max(class_probabilities) * 100
            samples = tree_.n_node_samples[node]
            print(f"{indent}└── 预测: {predicted_class} (概率: {prob:.1f}%, 样本数: {samples})")
    
    print("\n决策树结构:")
    print("根节点")
    recurse(0, 1)

def predict_test_data(model, feature_names, scaler_params):
    """
    对测试数据集进行预测
    
    Args:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        scaler_params: 标准化参数字典
    """
    try:
        # 读取测试数据
        test_data = pd.read_excel("policy_test.xlsx")
        print("\npolicy_test.xlsx 的列名:")
        print(test_data.columns.tolist())
        
        # 准备测试数据
        test_features = prepare_data("policy_test.xlsx", is_training=False, scaler_params=scaler_params)
        
        # 确保特征列的顺序与训练数据一致
        X_test = test_features[feature_names]
        
        # 进行预测
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'policy_id': test_data['policy_id'],
            'prediction': ['Yes' if p == 1 else 'No' for p in predictions],
            'probability_yes': probabilities[:, 1],
            'probability_no': probabilities[:, 0]
        })
        
        # 添加原始特征到结果中
        results = pd.concat([
            results,
            test_data[['age', 'gender', 'family_members', 'premium_amount']]
        ], axis=1)
        
        # 保存预测结果
        results.to_excel('prediction_results.xlsx', index=False)
        
        # 打印预测结果统计
        print("\n测试数据集预测结果统计:")
        print(f"总样本数: {len(results)}")
        print(f"预测续保客户数: {sum(results['prediction'] == 'Yes')}")
        print(f"预测不续保客户数: {sum(results['prediction'] == 'No')}")
        
        # 打印预测概率分布
        print("\n预测概率分布:")
        print("续保概率分布:")
        print(results['probability_yes'].describe())
        print("\n不续保概率分布:")
        print(results['probability_no'].describe())
        
        # 按年龄段分析预测结果
        print("\n按年龄段的预测结果分布:")
        age_bins = [0, 30, 40, 50, 60, float('inf')]
        age_labels = ['<30', '30-40', '40-50', '50-60', '>60']
        results['age_group'] = pd.cut(results['age'], bins=age_bins, labels=age_labels)
        age_analysis = results.groupby('age_group')['prediction'].value_counts(normalize=True).unstack()
        print(age_analysis)
        
        # 按家庭规模分析预测结果
        print("\n按家庭规模的预测结果分布:")
        family_bins = [0, 1, 2, 4, float('inf')]
        family_labels = ['single', 'couple', 'small', 'large']
        results['family_size_category'] = pd.cut(results['family_members'], 
                                               bins=family_bins, 
                                               labels=family_labels)
        family_analysis = results.groupby('family_size_category')['prediction'].value_counts(normalize=True).unstack()
        print(family_analysis)
        
        return results
        
    except Exception as e:
        print(f"预测过程中发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 准备训练数据
    X, y, feature_names, scaler_params = prepare_data("policy_data.xlsx", is_training=True)
    
    # 训练模型
    model, X_train, X_test, y_train, y_test = train_decision_tree(X, y, max_depth=4)
    
    # 评估模型
    evaluate_model(model, X_test, y_test)
    
    # 可视化决策树
    visualize_decision_tree(model, feature_names, class_names=['No', 'Yes'])
    
    # 保存决策树规则
    save_tree_rules(model, feature_names, class_names=['No', 'Yes'])
    
    # 打印决策树结构
    print_tree_structure(model, feature_names, class_names=['No', 'Yes'])
    
    # 对测试数据集进行预测
    print("\n开始对测试数据集进行预测...")
    prediction_results = predict_test_data(model, feature_names, scaler_params)
    
    print("\n决策树模型训练和评估完成！请查看生成的预测结果文件 prediction_results.xlsx") 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def prepare_data(file_path: str):
    """
    准备模型训练数据，进行特征工程
    
    Args:
        file_path (str): 数据文件路径
    
    Returns:
        tuple: (X, y, feature_names) 特征矩阵、目标变量和特征名称
    """
    # 读取数据
    df = pd.read_excel(file_path)
    
    # 分离特征和目标变量
    X = df.drop(['renewal'], axis=1)
    y = df['renewal']
    
    # 特征工程
    # 1. 基础特征
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].copy()
    
    # 2. 时间相关特征
    if 'policy_start_date' in X.columns:
        X['policy_duration'] = (pd.Timestamp.now() - pd.to_datetime(X['policy_start_date'])).dt.days
        X['policy_duration_years'] = X['policy_duration'] / 365
    
    # 3. 客户价值相关特征
    if 'premium_amount' in X.columns:
        X['premium_per_family_member'] = X['premium_amount'] / X['family_members']
        X['premium_per_age'] = X['premium_amount'] / X['age']
    
    # 4. 客户生命周期特征
    if 'age' in X.columns:
        X['age_group'] = pd.cut(X['age'], 
                               bins=[0, 30, 40, 50, 60, 100],
                               labels=['<30', '30-40', '40-50', '50-60', '>60'])
    
    # 5. 家庭结构特征
    if 'family_members' in X.columns:
        X['family_size_category'] = pd.cut(X['family_members'],
                                         bins=[0, 1, 2, 3, 5, 10],
                                         labels=['single', 'couple', 'small', 'medium', 'large'])
    
    # 6. 交互特征
    if all(col in X.columns for col in ['age', 'family_members']):
        X['age_family_ratio'] = X['age'] / X['family_members']
    
    # 7. 标准化特征
    if 'premium_amount' in X.columns:
        X['premium_standardized'] = (X['premium_amount'] - X['premium_amount'].mean()) / X['premium_amount'].std()
    
    # 8. 分位数特征
    if 'premium_amount' in X.columns:
        X['premium_quantile'] = pd.qcut(X['premium_amount'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # 9. 时间窗口特征
    if 'policy_start_date' in X.columns:
        X['is_recent_policy'] = (pd.Timestamp.now() - pd.to_datetime(X['policy_start_date'])).dt.days < 365
    
    # 10. 客户价值评分
    if all(col in X.columns for col in ['premium_amount', 'policy_duration']):
        X['customer_value_score'] = (X['premium_amount'] * X['policy_duration']) / 1000
    
    # 将分类特征转换为虚拟变量
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y, X.columns.tolist()

def train_logistic_model(X, y):
    """
    训练逻辑回归模型
    
    Args:
        X: 特征矩阵
        y: 目标变量
    
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, scaler) 模型和相关数据
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler

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
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    可视化特征重要性（系数）
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
    """
    # 获取系数
    coefficients = model.coef_[0]
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        '特征': feature_names,
        '系数': coefficients
    })
    
    # 按系数绝对值排序并取前20个
    feature_importance = feature_importance.reindex(
        feature_importance.系数.abs().sort_values(ascending=False).index
    ).head(20)
    
    # 设置颜色
    colors = ['red' if c < 0 else 'green' for c in feature_importance['系数']]
    
    # 绘制水平条形图
    plt.figure(figsize=(15, 10))
    bars = plt.barh(range(len(feature_importance)), feature_importance['系数'], color=colors)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width < 0:
            x = width - 0.02
            ha = 'right'
        else:
            x = width + 0.02
            ha = 'left'
        plt.text(x, i, f'{width:.3f}', va='center', ha=ha)
    
    plt.yticks(range(len(feature_importance)), feature_importance['特征'])
    plt.xlabel('系数值', fontsize=12)
    plt.title('逻辑回归系数Top20可视化\n(红色表示负相关，绿色表示正相关)', fontsize=14, pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='正相关'),
        Patch(facecolor='red', label='负相关')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model_results(model, feature_names, scaler):
    """
    保存模型结果
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        scaler: 标准化器
    """
    # 创建结果字典
    results = {
        'feature_coefficients': dict(zip(feature_names, model.coef_[0].tolist())),
        'intercept': float(model.intercept_[0]),
        'feature_names': feature_names
    }
    
    # 保存为JSON
    with open('model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 准备数据
    X, y, feature_names = prepare_data("policy_data.xlsx")
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, scaler = train_logistic_model(X, y)
    
    # 评估模型
    evaluate_model(model, X_test, y_test)
    
    # 可视化特征重要性
    plot_feature_importance(model, feature_names)
    
    # 保存模型结果
    save_model_results(model, feature_names, scaler)
    
    print("模型训练和评估完成！请查看生成的图表和结果文件。") 
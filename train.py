import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib

def load_data():
    """加载训练数据"""
    df = pd.read_excel('policy_data.xlsx')
    return df

def preprocess_data(df, encoders=None):
    """预处理数据"""
    # 将日期列转换为datetime类型
    date_columns = ['policy_start_date', 'policy_end_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    # 计算保单时长（月）
    df['policy_duration'] = ((df['policy_end_date'] - df['policy_start_date']).dt.days / 30).astype(int)
    
    # 从保单期限中提取数值
    df['policy_term_value'] = df['policy_term'].str.extract('(\d+)').astype(int)
    
    # 删除不需要的列
    df = df.drop(['policy_id', 'policy_start_date', 'policy_end_date', 'policy_term'], axis=1)
    
    # 处理缺失值
    df = df.fillna('unknown')
    
    # 对分类变量进行编码
    categorical_columns = ['gender', 'birth_region', 'insurance_region', 'income_level', 
                         'education_level', 'occupation', 'marital_status', 'policy_type',
                         'claim_history']
    
    if encoders is None:
        # 训练模式：创建新的编码器
        encoders = {}
        for col in categorical_columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    else:
        # 预测模式：使用现有的编码器
        for col in categorical_columns:
            # 处理测试集中可能出现的新类别
            unique_values = set(df[col].unique())
            known_values = set(encoders[col].classes_)
            new_values = unique_values - known_values
            if new_values:
                # 将新类别替换为 'unknown'
                df.loc[df[col].isin(new_values), col] = 'unknown'
            df[col] = encoders[col].transform(df[col])
    
    # 将目标变量转换为数值（如果存在）
    if 'renewal' in df.columns:
        df['renewal'] = df['renewal'].map({'Yes': 1, 'No': 0})
    
    return df, encoders

def train_model():
    """训练模型"""
    try:
        # 加载数据
        df = load_data()
        
        # 预处理数据
        processed_data, encoders = preprocess_data(df)
        
        # 分离特征和目标变量
        X = processed_data.drop('renewal', axis=1)
        y = processed_data['renewal']
        
        # 标准化数值特征
        numeric_features = ['age', 'family_members', 'premium_amount', 'policy_duration', 'policy_term_value']
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        print("\n模型评估报告：")
        print(classification_report(y_test, y_pred))
        
        # 保存模型、标准化器和编码器
        joblib.dump(model, 'decision_tree_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(encoders, 'label_encoders.joblib')
        
        print("\n模型训练完成并已保存！")
        
    except Exception as e:
        print(f"训练过程中出现错误：{str(e)}")

if __name__ == "__main__":
    train_model() 
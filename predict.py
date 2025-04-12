import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_model_and_resources():
    """加载保存的模型、标准化器和编码器"""
    model = joblib.load('decision_tree_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('label_encoders.joblib')
    return model, scaler, encoders

def preprocess_data(df, scaler, encoders):
    """预处理数据，与训练时相同的处理流程"""
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
    
    for col in categorical_columns:
        # 处理测试集中可能出现的新类别
        unique_values = set(df[col].unique())
        known_values = set(encoders[col].classes_)
        new_values = unique_values - known_values
        if new_values:
            # 将新类别替换为 'unknown'
            df.loc[df[col].isin(new_values), col] = 'unknown'
        df[col] = encoders[col].transform(df[col])
    
    # 标准化数值特征
    numeric_features = ['age', 'family_members', 'premium_amount', 'policy_duration', 'policy_term_value']
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    return df

def predict():
    """使用模型进行预测"""
    try:
        # 加载模型和资源
        model, scaler, encoders = load_model_and_resources()
        
        # 读取测试数据
        test_data = pd.read_excel('policy_test.xlsx')
        original_data = test_data.copy()
        
        # 预处理数据
        processed_data = preprocess_data(test_data, scaler, encoders)
        
        # 进行预测
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        # 将预测结果添加到原始数据中
        original_data['renewal_prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
        original_data['renewal_probability'] = probabilities[:, 1]  # 续保的概率
        
        # 保存预测结果
        original_data.to_excel('prediction_results.xlsx', index=False)
        
        print("预测完成！结果已保存到 prediction_results.xlsx")
        print("\n预测结果统计：")
        print(original_data['renewal_prediction'].value_counts())
        print("\n续保概率分布：")
        print(original_data['renewal_probability'].describe())
        
    except Exception as e:
        print(f"预测过程中出现错误：{str(e)}")

if __name__ == "__main__":
    predict() 
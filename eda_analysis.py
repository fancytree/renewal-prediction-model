import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import json
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def convert_to_serializable(obj):
    """
    将对象转换为可JSON序列化的格式
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        可序列化的对象
    """
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj

def perform_eda(file_path: str) -> Dict[str, Any]:
    """
    对Excel文件进行探索性数据分析
    
    Args:
        file_path (str): Excel文件路径
        
    Returns:
        Dict[str, Any]: 包含EDA结果的字典
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 创建结果字典
        eda_results = {}
        
        # 1. 基本信息
        eda_results['基本信息'] = {
            '行数': len(df),
            '列数': len(df.columns),
            '列名': df.columns.tolist(),
            '数据类型': df.dtypes.astype(str).to_dict()
        }
        
        # 2. 描述性统计
        eda_results['描述性统计'] = df.describe().round(2).to_dict()
        
        # 3. 缺失值分析
        eda_results['缺失值分析'] = {
            '每列缺失值数量': df.isnull().sum().to_dict(),
            '每列缺失值比例': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        }
        
        # 4. 唯一值分析
        eda_results['唯一值分析'] = {
            '每列唯一值数量': df.nunique().to_dict(),
            '每列唯一值比例': (df.nunique() / len(df) * 100).round(2).to_dict()
        }
        
        # 5. 数值型变量的相关性分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            eda_results['相关性分析'] = df[numeric_cols].corr().round(2).to_dict()
        
        return eda_results
    
    except Exception as e:
        print(f"执行EDA时发生错误: {str(e)}")
        return {}

def save_eda_results(results: Dict[str, Any], output_file: str = 'eda_results.json'):
    """
    将EDA结果保存到JSON文件
    
    Args:
        results (Dict[str, Any]): EDA结果字典
        output_file (str): 输出文件路径
    """
    try:
        # 转换结果为可序列化的格式
        serializable_results = convert_to_serializable(results)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        print(f"EDA结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存EDA结果时发生错误: {str(e)}")

def plot_numeric_distributions(df: pd.DataFrame):
    """
    绘制数值型变量的分布图
    
    Args:
        df (pd.DataFrame): 数据框
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(15, 5 * len(numeric_cols)))
            for i, col in enumerate(numeric_cols, 1):
                plt.subplot(len(numeric_cols), 1, i)
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'{col} 的分布')
            plt.tight_layout()
            plt.savefig('numeric_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"绘制分布图时发生错误: {str(e)}")

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    绘制相关性热力图
    
    Args:
        df (pd.DataFrame): 数据框
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[numeric_cols].corr().round(2), annot=True, cmap='coolwarm', center=0)
            plt.title('数值型变量相关性热力图')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"绘制热力图时发生错误: {str(e)}")

if __name__ == "__main__":
    # 执行EDA分析
    results = perform_eda("policy_data.xlsx")
    
    # 保存EDA结果
    save_eda_results(results)
    
    # 读取数据用于绘图
    df = pd.read_excel("policy_data.xlsx")
    
    # 绘制分布图
    plot_numeric_distributions(df)
    
    # 绘制相关性热力图
    plot_correlation_heatmap(df)
    
    print("EDA分析完成！请查看生成的JSON文件和图表。") 
import pandas as pd

def read_excel_head(file_path: str, n_rows: int = 5) -> pd.DataFrame:
    """
    读取Excel文件的前n行数据
    
    Args:
        file_path (str): Excel文件路径
        n_rows (int): 要读取的行数，默认为5行
        
    Returns:
        pd.DataFrame: 包含前n行数据的DataFrame
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        # 返回前n行数据
        return df.head(n_rows)
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # 读取policy_data.xlsx的前10行数据
    result = read_excel_head("policy_data.xlsx", n_rows=10)
    print("Excel文件前10行数据：")
    print(result) 
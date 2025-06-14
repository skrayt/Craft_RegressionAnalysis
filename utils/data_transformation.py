import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from db.database import save_dataframe_to_sqlite_with_sanitization


def standardize_data(df: pd.DataFrame, table_name_prefix: str = "") -> pd.DataFrame:
    """
    データフレームを標準化し、データベースに保存する

    Args:
        df (pd.DataFrame): 標準化するデータフレーム
        table_name_prefix (str): テーブル名のプレフィックス

    Returns:
        pd.DataFrame: 標準化されたデータフレーム
    """
    # kijyunnengetuカラムを保持
    kijyunnengetu_col = None
    if "kijyunnengetu" in df.columns:
        kijyunnengetu_col = df["kijyunnengetu"]
        df = df.drop(columns=["kijyunnengetu"])

    # 標準化を適用
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

    # kijyunnengetuカラムを戻す
    if kijyunnengetu_col is not None:
        scaled_df["kijyunnengetu"] = kijyunnengetu_col

    # データベースに保存
    try:
        save_dataframe_to_sqlite_with_sanitization(
            scaled_df, table_name=f"{table_name_prefix}_standardized"
        )
        print(f"DEBUG: {table_name_prefix}の標準化データをデータベースに保存しました。")
    except Exception as e:
        print(f"DEBUG: {table_name_prefix}の標準化データの保存に失敗しました: {e}")

    return scaled_df


def apply_transformations(df: pd.DataFrame, transformations: list) -> dict:
    """
    データフレームに変換を適用する

    Args:
        df (pd.DataFrame): 変換を適用するデータフレーム
        transformations (list): 適用する変換のリスト

    Returns:
        dict: 変換後のデータフレームの辞書
    """
    # kijyunnengetuカラムを保持
    result_df = df.copy()
    kijyunnengetu_col = None
    if "kijyunnengetu" in df.columns:
        kijyunnengetu_col = df["kijyunnengetu"]
        result_df = result_df.drop(columns=["kijyunnengetu"])

    # 各変換を個別に適用し、それぞれのデータフレームを保存
    transformed_dfs = {}

    # 対数変換
    if "対数変換" in transformations:
        log_df = result_df.copy()
        for col in log_df.columns:
            log_df[f"{col}_log"] = np.log1p(log_df[col])
        if kijyunnengetu_col is not None:
            log_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["log_data"] = log_df

    # 差分化
    if "差分化" in transformations:
        diff_df = result_df.copy()
        for col in diff_df.columns:
            diff_df[f"{col}_diff"] = diff_df[col].diff()
        if kijyunnengetu_col is not None:
            diff_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["diff_data"] = diff_df

    # 対数変換後に差分化
    if "対数変換後に差分化" in transformations:
        log_diff_df = result_df.copy()
        for col in log_diff_df.columns:
            log_diff_df[f"{col}_log_diff"] = np.log1p(log_diff_df[col]).diff()
        if kijyunnengetu_col is not None:
            log_diff_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["log_diff_data"] = log_diff_df

    return transformed_dfs

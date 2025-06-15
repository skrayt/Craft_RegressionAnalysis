"""
データ変換用のユーティリティ関数モジュール。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from db.database import (
    save_dataframe_to_sqlite_with_sanitization,
    read_dataframe_from_sqlite,
)


def get_dataframe_for_pattern(
    df: pd.DataFrame, transformation_type: str, is_standardized: bool
) -> pd.DataFrame:
    """
    指定された変換パターンと標準化に基づいてDataFrameを取得する

    Parameters:
    -----------
    df : pd.DataFrame
        元のデータフレーム
    transformation_type : str
        変換タイプ（"none", "log", "diff", "log_diff"）
    is_standardized : bool
        標準化するかどうか

    Returns:
    --------
    pd.DataFrame
        変換・標準化されたデータフレーム
    """
    print(
        f"DEBUG: get_dataframe_for_pattern - 変換タイプ: {transformation_type}, 標準化: {is_standardized}"
    )
    print(f"DEBUG: 入力データフレームの形状: {df.shape}")

    # データフレームのコピーを作成
    transformed_df = df.copy()

    # デバッグ情報：各カラムのデータ型を表示
    print("DEBUG: カラムのデータ型:")
    for col in transformed_df.columns:
        print(f"DEBUG: {col}: {transformed_df[col].dtype}")
        print(f"DEBUG: {col}の欠損値数: {transformed_df[col].isna().sum()}")
        print(f"DEBUG: {col}の最初の5行: {transformed_df[col].head()}")

    # kijyunnengetuカラムを保持
    kijyunnengetu_col = None
    if "kijyunnengetu" in transformed_df.columns:
        kijyunnengetu_col = transformed_df["kijyunnengetu"]
        transformed_df = transformed_df.drop(columns=["kijyunnengetu"])

    # 数値型以外のカラムを除外
    numeric_cols = transformed_df.select_dtypes(include=np.number).columns.tolist()
    print(f"DEBUG: 数値型カラム: {numeric_cols}")
    if not numeric_cols:
        print("DEBUG: 数値型のカラムがありません。")
        return pd.DataFrame()
    transformed_df = transformed_df[numeric_cols]

    # 変換を適用
    try:
        if transformation_type == "log":
            print("DEBUG: 対数変換を適用")
            for col in transformed_df.columns:
                transformed_df[col] = np.log1p(transformed_df[col])
        elif transformation_type == "diff":
            print("DEBUG: 差分化を適用")
            for col in transformed_df.columns:
                transformed_df[col] = transformed_df[col].diff()
            # 差分化後に欠損値を削除
            transformed_df = transformed_df.dropna()
        elif transformation_type == "log_diff":
            print("DEBUG: 対数変換後に差分化を適用")
            for col in transformed_df.columns:
                transformed_df[col] = np.log1p(transformed_df[col]).diff()
            # 差分化後に欠損値を削除
            transformed_df = transformed_df.dropna()
    except Exception as e:
        print(f"DEBUG: 変換適用中にエラー: {str(e)}")
        raise

    # 標準化を適用
    try:
        if is_standardized:
            print("DEBUG: 標準化を適用")
            for col in transformed_df.columns:
                mean = transformed_df[col].mean()
                std = transformed_df[col].std()
                print(f"DEBUG: {col} - 平均: {mean}, 標準偏差: {std}")
                if std != 0:  # 標準偏差が0でない場合のみ標準化
                    transformed_df[col] = (transformed_df[col] - mean) / std
    except Exception as e:
        print(f"DEBUG: 標準化適用中にエラー: {str(e)}")
        raise

    # kijyunnengetuカラムを戻す
    if kijyunnengetu_col is not None:
        transformed_df["kijyunnengetu"] = kijyunnengetu_col

    print(f"DEBUG: 出力データフレームの形状: {transformed_df.shape}")
    return transformed_df


def apply_transformations(
    df: pd.DataFrame, transformations: list[str]
) -> dict[str, pd.DataFrame]:
    """
    指定された変換を適用したデータフレームの辞書を返す

    Parameters:
    -----------
    df : pd.DataFrame
        元のデータフレーム
    transformations : list[str]
        適用する変換のリスト

    Returns:
    --------
    dict[str, pd.DataFrame]
        変換名をキー、変換後のデータフレームを値とする辞書
    """
    result = {}

    for transformation in transformations:
        if transformation == "対数変換":
            transformed_df = get_dataframe_for_pattern(df, "log", False)
            result["log_data"] = transformed_df
        elif transformation == "差分化":
            transformed_df = get_dataframe_for_pattern(df, "diff", False)
            result["diff_data"] = transformed_df
        elif transformation == "対数変換後に差分化":
            transformed_df = get_dataframe_for_pattern(df, "log_diff", False)
            result["log_diff_data"] = transformed_df

    return result


def standardize_data(df: pd.DataFrame, table_name_prefix: str = "") -> pd.DataFrame:
    """
    データフレームを標準化する

    Parameters:
    -----------
    df : pd.DataFrame
        元のデータフレーム
    table_name_prefix : str, optional
        テーブル名のプレフィックス（デフォルトは空文字列）

    Returns:
    --------
    pd.DataFrame
        標準化されたデータフレーム
    """
    return get_dataframe_for_pattern(df, "none", True)

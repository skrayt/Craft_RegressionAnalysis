"""
Database utility functions for RegressionAnalysis project.
Handles SQLite connection and queries.
"""

import os
import pandas as pd
import sqlite3
from typing import List, Tuple

DB_PATH = "db/regression_data.db"  # 主に回帰分析の結果などを保存
DATA_DB_PATH = "database.db"  # ユーザーデータ保存用


def get_connection():
    """
    Establish and return a database connection for regression_data.db.
    """
    return sqlite3.connect(DB_PATH)


def get_data_connection():
    """
    Establish and return a database connection for database.db.
    """
    return sqlite3.connect(DATA_DB_PATH)


def initialize_regression_database():
    """
    Create the necessary tables if they do not exist for regression_data.db.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS regression_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            variable_name TEXT NOT NULL,
            value REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def initialize_user_data_database():
    """
    Create the necessary tables if they do not exist for database.db.
    """
    conn = get_data_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS data (
            kijyunnengetu TEXT,
            column1 REAL,
            column2 REAL
        )
    """
    )  # 初期テーブルスキーマは仮。実際はdf.to_sqlで自動生成される
    conn.commit()
    conn.close()


def insert_data(data: List[Tuple[str, float]]):
    """Insert multiple records into the regression_data table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO regression_data (variable_name, value) VALUES (?, ?)", data
    )
    conn.commit()
    conn.close()


def fetch_all_data() -> List[Tuple[int, str, float]]:
    """Fetch all records from the regression_data table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM regression_data")
    rows = cursor.fetchall()
    conn.close()
    return rows


def clear_data():
    """
    Delete all records from the regression_data table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM regression_data")
    conn.commit()
    conn.close()


def save_dataframe_to_sqlite(df: pd.DataFrame, table_name: str):
    """
    指定されたDataFrameをSQLite（regression_data.db）に保存する。
    """
    os.makedirs("db", exist_ok=True)
    with get_connection() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def read_dataframe_from_sqlite(table_name: str, db_path: str = DATA_DB_PATH):
    """
    SQLiteから指定テーブルのデータを読み込み、DataFrameとして返す。
    デフォルトはユーザーデータ用のdatabase.dbから読み込む。
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except sqlite3.Error:
        df = pd.DataFrame()  # エラー時は空のDataFrameを返す
    finally:
        conn.close()
    return df


def sanitize_column_name(col_name: str) -> str:
    """
    カラム名をSQLiteで使用可能な形式に変換する
    """
    # 特殊文字をアンダースコアに置換（括弧、スペース、その他の特殊文字）
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in str(col_name))
    # 連続するアンダースコアを1つに
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    # 先頭と末尾のアンダースコアを削除
    sanitized = sanitized.strip("_")
    # 先頭が数字の場合は'col_'を付加
    if sanitized and sanitized[0].isdigit():
        sanitized = "col_" + sanitized
    # 空文字列の場合は'col'を返す
    if not sanitized:
        sanitized = "col"
    return sanitized


def save_dataframe_to_sqlite_with_sanitization(
    df: pd.DataFrame, table_name: str = "data"
) -> None:
    """
    データフレームをSQLiteデータベースに保存する関数
    カラム名をSQLiteで使用可能な形式に変換してから保存する

    Args:
        df (pd.DataFrame): 保存するデータフレーム
        table_name (str): 保存先のテーブル名（デフォルトは"data"）
    """
    conn = get_data_connection()
    try:
        # カラム名をSQLiteで使用可能な形式に変換
        df = df.copy()
        original_columns = list(df.columns)
        df.columns = [sanitize_column_name(col) for col in df.columns]
        print(f"変換前のカラム名: {original_columns}")
        print(f"変換後のカラム名: {list(df.columns)}")

        # テーブルが存在する場合は削除
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        # 新しいテーブルを作成してデータを保存
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# === Usage Examples (from other modules) ===
#
# In main.py:
# from db.database import initialize_database
# initialize_database()
#
# In page_analysis.py:
# from db.database import fetch_all_data
# data = fetch_all_data()
# print(data)
#
# In page_regression.py:
# from db.database import insert_data, clear_data
# insert_data([("x1", 1.23), ("x2", 4.56)])
# clear_data()

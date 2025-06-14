"""
時系列データの表示と分析を行うページモジュール。
時系列データの可視化、変換、分析機能を提供する。
"""

import os
import flet as ft
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # GUIバックエンドを使用しない
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tempfile
from datetime import datetime
from db.database import read_dataframe_from_sqlite

# フォントの設定
plt.rcParams["font.family"] = "sans-serif"  # デフォルトのsans-serifフォントを使用
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]  # フォールバックフォントを設定
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止

# 変換タイプの定義
TRANSFORMATION_TYPES = {
    "none": "変換なし",
    "log": "対数変換",
    "diff": "差分化",
    "log_diff": "対数変換後に差分化",
}

# グローバル変数
checkbox_states = {}  # チェックボックスの状態を保持するグローバル辞書
transformation_states = {}  # 各変数の変換タイプを保持するグローバル辞書
standardization_states = {}  # 各変数の標準化状態を保持するグローバル辞書


def get_dataframe_for_pattern(
    transformation: str, is_standardized: bool
) -> pd.DataFrame | None:
    """変換と標準化の設定に応じたデータフレームを取得"""
    try:
        # テーブル名の構築
        if transformation == "none":
            # 変換なしの場合はmerged_dataテーブルを使用
            table_name = "merged_standardized" if is_standardized else "merged_data"
        else:
            # 変換ありの場合は変換タイプに応じたテーブル名を使用
            if transformation == "log":
                table_name = "log_data"
            elif transformation == "diff":
                table_name = "diff_data"
            elif transformation == "log_diff":
                table_name = "log_diff_data"
            else:
                print(f"DEBUG: 不明な変換タイプ: {transformation}")
                return None

            # 標準化が有効な場合は_standardizedを付加
            if is_standardized:
                table_name += "_standardized"

        print(f"DEBUG: データを読み込むテーブル: {table_name}")

        # データベースからデータを読み込む
        df = read_dataframe_from_sqlite(table_name)
        if df is None or df.empty:
            print(f"DEBUG: {table_name}のデータが存在しません。")
            return None

        print(f"DEBUG: 読み込んだデータのカラム: {list(df.columns)}")
        print(f"DEBUG: 読み込んだデータの行数: {len(df)}")

        # 変換後のカラムのみを選択
        if transformation != "none":
            # 変換後のカラム名のサフィックスを決定
            suffix = ""
            if transformation == "log":
                suffix = "_log"
            elif transformation == "diff":
                suffix = "_diff"
            elif transformation == "log_diff":
                suffix = "_log_diff"

            # 標準化が有効な場合は_stdを追加
            if is_standardized:
                suffix += "_std"

            # 変換後のカラムのみを選択
            transformed_columns = [col for col in df.columns if col.endswith(suffix)]
            if not transformed_columns:
                print(
                    f"DEBUG: 変換後のカラム（サフィックス: {suffix}）が見つかりません。"
                )
                return None

            # kijyunnengetuカラムと変換後のカラムのみを選択
            selected_columns = ["kijyunnengetu"] + transformed_columns
            df = df[selected_columns]

            print(f"DEBUG: 選択されたカラム: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"DEBUG: データの読み込みに失敗しました: {str(e)}")
        return None


def plot_single_time_series(
    df: pd.DataFrame, column: str, transformation: str, is_standardized: bool
) -> str:
    """単一の時系列データをプロットし、一時ファイルのパスを返す"""
    try:
        print(
            f"DEBUG: グラフ生成開始 - カラム: {column}, 変換: {transformation}, 標準化: {is_standardized}"
        )
        print(f"DEBUG: データフレームのカラム: {list(df.columns)}")

        # 変換後のカラム名を構築
        if transformation != "none":
            suffix = ""
            if transformation == "log":
                suffix = "_log"
            elif transformation == "diff":
                suffix = "_diff"
            elif transformation == "log_diff":
                suffix = "_log_diff"

            if is_standardized:
                suffix += "_std"

            column = f"{column}{suffix}"

        # データの存在確認
        if column not in df.columns:
            print(f"DEBUG: カラム '{column}' がデータフレームに存在しません。")
            raise ValueError(f"カラム '{column}' がデータフレームに存在しません。")

        # データの有効性確認
        if df[column].isna().all():
            print(f"DEBUG: カラム '{column}' のデータがすべて欠損値です。")
            raise ValueError(f"カラム '{column}' のデータがすべて欠損値です。")

        print(f"DEBUG: カラム '{column}' のデータ型: {df[column].dtype}")
        print(f"DEBUG: カラム '{column}' の欠損値の数: {df[column].isna().sum()}")
        print(f"DEBUG: カラム '{column}' の最初の5行: {df[column].head()}")

        # グラフの作成
        plt.figure(figsize=(8, 6))

        # データのプロット
        plt.plot(df["kijyunnengetu"], df[column], label=column)

        # グラフの設定
        plt.title(
            f"{column}の時系列推移\n"
            + f"変換: {TRANSFORMATION_TYPES[transformation]}"
            + (" (標準化済み)" if is_standardized else "")
        )
        plt.xlabel("年月")
        plt.ylabel("値")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # 一時ファイルの作成
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeseries_{column}_{transformation}_{'std' if is_standardized else 'raw'}_{timestamp}.png"
        filepath = os.path.join(temp_dir, filename)

        print(f"DEBUG: グラフを保存: {filepath}")

        # グラフの保存
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        # ファイルの存在確認
        if not os.path.exists(filepath):
            print(f"DEBUG: グラフファイルの保存に失敗: {filepath}")
            raise FileNotFoundError(f"グラフの保存に失敗しました: {filepath}")

        print(f"DEBUG: グラフ生成完了: {filepath}")
        return filepath

    except Exception as e:
        print(f"DEBUG: グラフ生成エラー ({column}): {str(e)}")
        plt.close()  # エラー時も確実にグラフを閉じる
        raise  # エラーを上位に伝播


def time_series_page(page: ft.Page):
    """時系列データの表示と分析を行うページ"""
    global checkbox_states, transformation_states, standardization_states

    # 変換タイプの選択肢
    transformation_types = {
        "none": "変換なし",
        "log": "対数変換",
        "diff": "差分化",
        "log_diff": "対数変換後に差分化",
    }

    # 変数選択用のチェックボックスと変換タイプ選択用のドロップダウンを保持するコンテナ
    variable_checkboxes = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        height=300,  # 固定高さを設定
        spacing=5,
    )

    # グラフ表示用のコンテナ
    graph_container = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=10,
    )

    # ステータス表示用のテキスト
    status_text = ft.Text("", size=12, color=ft.Colors.RED)

    def initialize_variable_controls():
        """変数選択用のコントロールを初期化"""
        global checkbox_states, transformation_states, standardization_states

        # データベースからカラム一覧を取得
        df = read_dataframe_from_sqlite("merged_data")
        if df is None or df.empty:
            print("DEBUG: データが存在しません。")
            return

        # kijyunnengetu以外のカラムを取得
        columns = [col for col in df.columns if col != "kijyunnengetu"]

        # 各変数のコントロールを生成
        controls = []
        for col in columns:
            # 初期状態の設定
            if col not in checkbox_states:
                checkbox_states[col] = False
            if col not in transformation_states:
                transformation_states[col] = "none"
            if col not in standardization_states:
                standardization_states[col] = False

            # 変数名の表示
            variable_name = ft.Text(col, size=14)

            # チェックボックス
            checkbox = ft.Checkbox(
                value=checkbox_states[col],
                on_change=lambda e, col=col: handle_checkbox_change(e, col),
            )

            # 変換タイプ選択用ドロップダウン
            transformation_dropdown = ft.Dropdown(
                label="変換タイプ",
                options=[
                    ft.dropdown.Option(key, value)
                    for key, value in transformation_types.items()
                ],
                value=transformation_states[col],
                width=150,
                on_change=lambda e, col=col: handle_transformation_change(e, col),
            )

            # 標準化スイッチ
            standardization_switch = ft.Switch(
                label="標準化",
                value=standardization_states[col],
                on_change=lambda e, col=col: handle_standardization_change(e, col),
            )

            # 変数ごとのコントロールを横に並べる
            row = ft.Row(
                [
                    checkbox,
                    variable_name,
                    transformation_dropdown,
                    standardization_switch,
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=10,
            )

            # コントロールをコンテナに追加
            container = ft.Container(
                content=row,
                padding=ft.padding.only(left=10, right=10, top=5, bottom=5),
                border=ft.border.only(
                    bottom=ft.border.BorderSide(1, ft.Colors.GREY_300)
                ),
            )
            controls.append(container)

        variable_checkboxes.controls = controls
        page.update()

    def handle_checkbox_change(e: ft.ControlEvent, column: str):
        """チェックボックスの状態変更を処理"""
        global checkbox_states
        checkbox_states[column] = e.control.value
        print(f"DEBUG: チェックボックス変更 - {column}: {e.control.value}")

    def handle_transformation_change(e: ft.ControlEvent, column: str):
        """変換タイプの変更を処理"""
        global transformation_states
        transformation_states[column] = e.control.value
        print(f"DEBUG: 変換タイプ変更 - {column}: {e.control.value}")

    def handle_standardization_change(e: ft.ControlEvent, column: str):
        """標準化の変更を処理"""
        global standardization_states
        standardization_states[column] = e.control.value
        print(f"DEBUG: 標準化変更 - {column}: {e.control.value}")

    def update_graphs(e: ft.ControlEvent):
        """選択された変数のグラフを更新"""
        # グラフコンテナをクリア
        graph_container.controls.clear()
        status_text.value = ""

        # 選択された変数を取得
        selected_columns = [col for col, checked in checkbox_states.items() if checked]
        if not selected_columns:
            status_text.value = "変数を1つ以上選択してください。"
            page.update()
            return

        try:
            # 現在の行のコンテナ
            current_row = ft.Row(
                controls=[],
                alignment=ft.MainAxisAlignment.START,
                spacing=10,
            )
            row_count = 0

            # 各変数のグラフを生成
            for col in selected_columns:
                # 変換タイプと標準化の設定を取得
                transformation = transformation_states[col]
                is_standardized = standardization_states[col]

                # データを取得
                df = get_dataframe_for_pattern(transformation, is_standardized)
                if df is None:
                    status_text.value = f"エラー: {col}のデータを取得できませんでした。"
                    continue

                try:
                    # グラフを生成
                    graph_path = plot_single_time_series(
                        df, col, transformation, is_standardized
                    )

                    # グラフを表示
                    graph_image = ft.Image(
                        src=graph_path,
                        width=280,
                        height=200,
                        fit=ft.ImageFit.CONTAIN,
                    )

                    # グラフのタイトル
                    graph_title = ft.Text(
                        f"{col}\n{TRANSFORMATION_TYPES[transformation]}"
                        + (" (標準化済み)" if is_standardized else ""),
                        size=12,
                        text_align=ft.TextAlign.CENTER,
                    )

                    # グラフとタイトルを縦に並べる
                    graph_column = ft.Column(
                        [graph_title, graph_image],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=5,
                    )

                    # グラフをコンテナで包む
                    graph_container_item = ft.Container(
                        content=graph_column,
                        width=290,
                        padding=5,
                        border=ft.border.all(1, ft.Colors.GREY_300),
                        border_radius=5,
                    )

                    # 3列ごとに新しい行を作成
                    if row_count % 3 == 0:
                        current_row = ft.Row(
                            controls=[],
                            alignment=ft.MainAxisAlignment.START,
                            spacing=10,
                        )
                        graph_container.controls.append(current_row)

                    current_row.controls.append(graph_container_item)
                    row_count += 1

                except Exception as e:
                    print(f"DEBUG: グラフ生成エラー ({col}): {str(e)}")
                    status_text.value = f"エラー: {col}のグラフ生成に失敗しました。"
                    continue

            if not graph_container.controls:
                status_text.value = "グラフを生成できませんでした。"

        except Exception as e:
            print(f"DEBUG: グラフ更新エラー: {str(e)}")
            status_text.value = f"エラー: グラフの更新に失敗しました。"

        page.update()

    def clear_graphs(e: ft.ControlEvent):
        """グラフをクリア"""
        graph_container.controls.clear()
        status_text.value = "グラフをクリアしました。"
        page.update()

    # 変数選択用のコントロールを初期化
    initialize_variable_controls()

    # グラフ表示ボタン
    show_graph_button = ft.ElevatedButton(
        "グラフを表示",
        on_click=update_graphs,
    )

    # グラフクリアボタン
    clear_graph_button = ft.ElevatedButton(
        "グラフをクリア",
        on_click=clear_graphs,
    )

    # 左側のパネル（変数選択）
    left_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("変数選択", size=16, weight=ft.FontWeight.BOLD),
                variable_checkboxes,
                ft.Row(
                    [show_graph_button, clear_graph_button],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=10,
                ),
                status_text,
            ],
            spacing=10,
        ),
        width=300,
        padding=10,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5,
    )

    # 右側のパネル（グラフ表示）
    right_panel = ft.Container(
        content=graph_container,
        expand=True,
        padding=10,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5,
    )

    # メインのレイアウト
    return ft.Container(
        content=ft.Row(
            [left_panel, right_panel],
            alignment=ft.MainAxisAlignment.START,
            spacing=20,
            expand=True,
        ),
        padding=20,
        expand=True,
    )

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

# チェックボックスの状態を保持するグローバル辞書
checkbox_states = {}


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
    """時系列データ分析ページを構築"""
    # 初期データの読み込み
    initial_df = read_dataframe_from_sqlite("merged_data")
    if initial_df is not None and not initial_df.empty:
        all_columns = [col for col in initial_df.columns if col != "kijyunnengetu"]
    else:
        all_columns = []

    # 変換タイプ選択用ドロップダウン
    transformation_dropdown = ft.Dropdown(
        label="変換タイプ",
        options=[
            ft.dropdown.Option(key, value)
            for key, value in TRANSFORMATION_TYPES.items()
        ],
        value="none",
        width=200,
    )

    # 標準化スイッチ
    standardization_switch = ft.Switch(
        label="標準化",
        value=False,
    )

    # 変数選択用のチェックボックスリスト
    variable_checkboxes = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        height=300,
        spacing=10,
    )

    # グラフ表示用のコンテナ
    graph_container = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        spacing=20,
        expand=True,
    )

    # ステータスメッセージ用のテキスト
    status_text = ft.Text(
        "変数を選択し、「グラフを表示」ボタンを押してください。",
        size=14,
        color=ft.Colors.GREY_700,
    )

    def initialize_checkboxes():
        """チェックボックスの初期化"""
        variable_checkboxes.controls.clear()
        for col in all_columns:
            checkbox = ft.Checkbox(
                label=col,
                value=checkbox_states.get(col, False),
                on_change=lambda e, col=col: handle_checkbox_change(e, col),
            )
            variable_checkboxes.controls.append(checkbox)
        page.update()

    def handle_checkbox_change(e: ft.ControlEvent, column: str):
        """チェックボックスの状態変更を処理"""
        checkbox_states[column] = e.control.value
        # チェックボックスの状態変更時はグラフを更新しない
        status_text.value = "変数の選択が変更されました。「グラフを表示」ボタンを押して更新してください。"
        page.update()

    def update_graphs(e=None):
        """選択された変数のグラフを更新"""
        # 選択された変数を取得
        selected_variables = [
            col for col, is_checked in checkbox_states.items() if is_checked
        ]

        if not selected_variables:
            status_text.value = "表示する変数を選択してください。"
            status_text.color = ft.Colors.RED_400
            page.update()
            return

        status_text.value = "グラフを生成中..."
        status_text.color = ft.Colors.BLUE_400
        page.update()

        try:
            graph_container.controls.clear()

            # 現在の設定を取得
            transformation = transformation_dropdown.value
            is_standardized = standardization_switch.value

            # データフレームを取得
            df = get_dataframe_for_pattern(transformation, is_standardized)
            if df is None or df.empty:
                status_text.value = "データの読み込みに失敗しました。"
                status_text.color = ft.Colors.RED_400
                page.update()
                return

            # グラフを3列で表示するための行コンテナ
            current_row = ft.Row(
                controls=[],
                spacing=10,  # 間隔を縮小
                alignment=ft.MainAxisAlignment.START,
            )
            graph_container.controls.append(current_row)

            # 選択された変数のグラフを生成
            for i, col in enumerate(selected_variables):
                try:
                    # 3列ごとに新しい行を作成
                    if i > 0 and i % 3 == 0:
                        current_row = ft.Row(
                            controls=[],
                            spacing=10,  # 間隔を縮小
                            alignment=ft.MainAxisAlignment.START,
                        )
                        graph_container.controls.append(current_row)

                    # グラフを生成して一時ファイルに保存
                    filepath = plot_single_time_series(
                        df, col, transformation, is_standardized
                    )

                    # グラフを表示（幅を調整して3列に収まるようにする）
                    current_row.controls.append(
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        f"{col}の時系列推移",
                                        size=12,  # フォントサイズを縮小
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Image(
                                        src=filepath,
                                        width=280,  # グラフの幅を縮小
                                        height=200,  # グラフの高さを縮小
                                        fit=ft.ImageFit.CONTAIN,
                                    ),
                                ],
                                spacing=5,  # 間隔を縮小
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            padding=5,  # パディングを縮小
                            border=ft.border.all(1, ft.Colors.GREY_400),
                            border_radius=5,
                            width=290,  # コンテナの幅を調整
                        )
                    )

                except Exception as e:
                    print(f"DEBUG: グラフの生成に失敗しました ({col}): {e}")
                    status_text.value = (
                        f"グラフの生成中にエラーが発生しました: {str(e)}"
                    )
                    status_text.color = ft.Colors.RED_400
                    page.update()
                    return

            # 最後の行が3列未満の場合、空のコンテナを追加してレイアウトを整える
            if len(current_row.controls) < 3:
                remaining_slots = 3 - len(current_row.controls)
                for _ in range(remaining_slots):
                    current_row.controls.append(
                        ft.Container(
                            width=290,
                            height=0,
                        )
                    )

            status_text.value = f"{len(selected_variables)}個のグラフを表示しました。"
            status_text.color = ft.Colors.GREEN_400

        except Exception as e:
            print(f"DEBUG: グラフの更新に失敗しました: {e}")
            status_text.value = f"グラフの更新に失敗しました: {str(e)}"
            status_text.color = ft.Colors.RED_400

        page.update()

    # 表示ボタン
    show_graphs_button = ft.ElevatedButton(
        "グラフを表示",
        icon=ft.Icons.PLAY_ARROW_ROUNDED,
        on_click=update_graphs,
    )

    # クリアボタン
    def clear_graphs(e):
        graph_container.controls.clear()
        status_text.value = "グラフをクリアしました。"
        status_text.color = ft.Colors.GREY_700
        page.update()

    clear_button = ft.ElevatedButton(
        "グラフをクリア",
        icon=ft.Icons.CLEAR_ALL_ROUNDED,
        on_click=clear_graphs,
    )

    # チェックボックスの初期化
    initialize_checkboxes()

    # ページのレイアウトを構築
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        # 左側の設定パネル（固定幅）
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "変換設定", size=16, weight=ft.FontWeight.BOLD
                                    ),
                                    transformation_dropdown,
                                    standardization_switch,
                                    ft.Divider(),
                                    ft.Text(
                                        "変数選択", size=16, weight=ft.FontWeight.BOLD
                                    ),
                                    ft.Container(  # 変数選択部分をスクロール可能に
                                        content=ft.Column(
                                            [variable_checkboxes],
                                            scroll=ft.ScrollMode.AUTO,  # スクロール可能に
                                        ),
                                        height=300,  # 高さを固定
                                        border=ft.border.all(1, ft.Colors.GREY_300),
                                        border_radius=5,
                                    ),
                                    ft.Divider(),
                                    ft.Row(
                                        [show_graphs_button, clear_button],
                                        spacing=10,
                                    ),
                                ],
                                spacing=20,
                            ),
                            padding=20,
                            border=ft.border.all(1, ft.Colors.GREY_400),
                            border_radius=5,
                            width=300,  # 幅を固定
                            alignment=ft.alignment.top_left,  # 上揃え
                        ),
                        # 右側のグラフ表示エリア（スクロール可能）
                        ft.Container(
                            content=ft.Column(
                                [
                                    status_text,
                                    ft.Column(  # グラフコンテナをColumnでラップ
                                        [graph_container],
                                        scroll=ft.ScrollMode.AUTO,  # スクロール可能に
                                        expand=True,
                                    ),
                                ],
                                spacing=20,
                            ),
                            padding=20,
                            expand=True,
                            alignment=ft.alignment.top_left,  # 上揃え
                        ),
                    ],
                    spacing=20,
                    alignment=ft.MainAxisAlignment.START,  # 上揃え
                ),
            ],
            expand=True,
        ),
        expand=True,
    )

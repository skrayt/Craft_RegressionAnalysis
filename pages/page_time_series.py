"""
時系列データの表示と分析を行うページモジュール。
時系列データの可視化、変換、分析機能を提供する。
"""

import os
import flet as ft
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tempfile
from math import ceil
from datetime import datetime
from db.database import (
    read_dataframe_from_sqlite,
    save_dataframe_to_sqlite_with_sanitization,
)
from utils.data_transformation import (
    apply_transformations,
    standardize_data,
    get_dataframe_for_pattern,
)  # 循環参照を避けるため
from components.plot_utils import (
    plot_single_time_series,
    plot_multiple_time_series_grid,
    TRANSFORMATION_TYPES,
)
from components.variable_selector import VariableSelector

matplotlib.use("Agg")  # GUIバックエンドを使用しない

# フォントの設定
plt.rcParams["font.family"] = "sans-serif"  # デフォルトのsans-serifフォントを使用
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
]  # フォールバックフォントを設定
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止

# グローバル変数
checkbox_states = {}  # チェックボックスの状態を保持するグローバル辞書
transformation_states = {}  # 各変数の変換タイプを保持するグローバル辞書
standardization_states = {}  # 各変数の標準化状態を保持するグローバル辞書
variable_types = {}  # 各変数のタイプ（目的変数/説明変数）を保持するグローバル辞書

# UIコントロールの参照を保持するためのグローバル変数
response_variable_checkboxes = ft.Column(
    controls=[], scroll=ft.ScrollMode.AUTO, height=200
)
explanatory_variable_checkboxes = ft.Column(
    controls=[], scroll=ft.ScrollMode.AUTO, height=600
)
graph_container = ft.Column(controls=[], scroll=ft.ScrollMode.AUTO, expand=True)
status_text = ft.Text("", color=ft.Colors.GREEN_700)

# 目的変数選択用のグローバル変数
selected_response_variable = None

# グラフ表示用のコンテナ
response_graph_display_container = ft.Column(
    scroll=ft.ScrollMode.AUTO,
    height=300,
    expand=True,
)

explanatory_graphs_scroll_view = ft.Column(
    scroll=ft.ScrollMode.AUTO,
    height=400,
    expand=True,
)


def plot_single_time_series_to_ax(
    ax, df: pd.DataFrame, column: str, transformation_type: str, is_standardized: bool
) -> None:
    """単一の時系列データを指定されたMatplotlib軸にプロットする"""
    print(
        f"DEBUG: plot_single_time_series_to_ax呼び出し: column={column}, trans={transformation_type}, std={is_standardized}"
    )

    if df is None or df.empty:
        print(f"DEBUG: DataFrameが空のため、{column}のグラフを生成できません。")
        return

    if "kijyunnengetu" not in df.columns:
        print(f"DEBUG: 'kijyunnengetu'カラムがDataFrameにありません。")
        return

    if column not in df.columns:
        print(f"DEBUG: カラム '{column}' がDataFrameにありません。")
        return

    # グラフタイトルを構築
    title_parts = [column]
    if transformation_type != "none":
        title_parts.append(f"({TRANSFORMATION_TYPES[transformation_type]})")
    if is_standardized:
        title_parts.append("(標準化済)")
    plot_title = " ".join(title_parts)

    # データのNaNチェック
    if df[column].isnull().all():
        print(
            f"DEBUG: カラム '{column}' のデータが全てNaNです。グラフを生成できません。"
        )
        return

    ax.plot(
        df["kijyunnengetu"],
        df[column],
        label=column,
        color=sns.color_palette("deep")[0],
    )

    ax.set_title(plot_title, fontsize=12)  # タイトルフォントサイズを調整
    ax.set_xlabel("年月", fontsize=10)
    ax.set_ylabel("値", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)  # ラベルサイズ調整
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=8)
    plt.tight_layout()


def time_series_page(page: ft.Page) -> ft.Container:
    """時系列データ分析ページを構築する関数"""

    # 初期データの読み込み
    initial_df = read_dataframe_from_sqlite("merged_data")
    if initial_df is None or initial_df.empty:
        return ft.Container(
            content=ft.Text(
                "データが読み込まれていません。データ取込み・参照タブでCSVをロードしてください。"
            )
        )

    # kijyunnengetuカラムを日付型に変換
    initial_df["kijyunnengetu"] = pd.to_datetime(
        initial_df["kijyunnengetu"], format="%Y%m"
    )

    # kijyunnengetu以外のカラムを取得
    all_columns = [col for col in initial_df.columns if col != "kijyunnengetu"]

    # グラフ表示用のコンテナ
    response_graph_display_container = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        height=300,
        expand=True,
    )

    explanatory_graphs_scroll_view = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        height=400,
        expand=True,
    )

    # ステータス表示用のテキスト
    status_text = ft.Text("", color=ft.Colors.GREEN_700)

    def run_plot():
        """グラフを描画し、結果を表示する"""
        target, features = variable_selector.get_selected_variables()
        if not target or not features:
            # エラーメッセージを表示
            status_text.value = "目的変数と説明変数を選択してください。"
            status_text.color = ft.Colors.RED_700
            page.update()
            return

        try:
            # 変数の設定を取得
            settings = variable_selector.get_variable_settings()

            # 目的変数のデータを取得（kijyunnengetuを含める）
            target_df = get_dataframe_for_pattern(
                initial_df,
                settings[target]["transformation"],
                settings[target]["standardization"],
            )
            target_df = pd.concat(
                [initial_df[["kijyunnengetu"]], target_df[[target]]], axis=1
            )

            # 説明変数のデータを取得（kijyunnengetuを含める）
            feature_dfs = []
            for feature in features:
                feature_df = get_dataframe_for_pattern(
                    initial_df,
                    settings[feature]["transformation"],
                    settings[feature]["standardization"],
                )
                feature_df = pd.concat(
                    [initial_df[["kijyunnengetu"]], feature_df[[feature]]], axis=1
                )
                feature_dfs.append(feature_df)

            # グラフコンテナをクリア
            response_graph_display_container.controls.clear()
            explanatory_graphs_scroll_view.controls.clear()

            # 目的変数のグラフを生成
            response_img_base64 = plot_single_time_series(
                target_df,
                target,
                settings[target]["transformation"],
                settings[target]["standardization"],
                figsize=(10, 4),
            )

            # 目的変数のグラフをコンテナに追加
            response_graph_display_container.controls.append(
                ft.Container(
                    content=ft.Image(
                        src_base64=response_img_base64, width=800, height=300
                    ),
                    padding=10,
                    border=ft.border.all(2, ft.Colors.BLUE_200),
                    border_radius=10,
                )
            )

            # 説明変数のグラフを生成
            if features:
                # 説明変数のデータを結合
                X = pd.concat(
                    [df[feature] for df, feature in zip(feature_dfs, features)], axis=1
                )
                X["kijyunnengetu"] = initial_df["kijyunnengetu"]

                explanatory_img_base64 = plot_multiple_time_series_grid(
                    X,
                    features,
                    {f: settings[f]["transformation"] for f in features},
                    {f: settings[f]["standardization"] for f in features},
                    n_cols=2,
                    figsize=(12, 4),
                )

                # 説明変数のグラフをコンテナに追加
                explanatory_graphs_scroll_view.controls.append(
                    ft.Container(
                        content=ft.Image(
                            src_base64=explanatory_img_base64,
                            width=800,
                            height=300 * ceil(len(features) / 2),
                        ),
                        padding=10,
                        border=ft.border.all(2, ft.Colors.GREY_300),
                        border_radius=10,
                    )
                )

            status_text.value = "グラフを更新しました。"
            status_text.color = ft.Colors.GREEN_700
            page.update()

        except Exception as e:
            print(f"DEBUG: グラフ描画中にエラーが発生: {str(e)}")
            status_text.value = f"グラフ描画中にエラーが発生しました: {str(e)}"
            status_text.color = ft.Colors.RED_700
            page.update()

    # グラフ描画実行ボタン
    plot_button = ft.ElevatedButton(
        text="グラフ描画実行",
        on_click=lambda _: run_plot(),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE,
        ),
    )

    # 変数選択コンポーネントの初期化
    variable_selector = VariableSelector(
        page=page,
        all_columns=all_columns,
        on_variable_change=None,  # 自動更新を無効化
    )

    # UIコンポーネントを取得
    target_row, feature_container = variable_selector.get_ui_components()

    # ページのレイアウト
    return ft.Container(
        padding=20,
        expand=True,
        content=ft.Column(
            [
                ft.Text("時系列データ分析", size=20, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        ft.Column(
                            [
                                target_row,
                                ft.Text("説明変数を選択：", size=16),
                                feature_container,
                                plot_button,
                                status_text,
                            ],
                            expand=True,
                        ),
                        ft.Column(
                            [
                                ft.Text("目的変数の時系列グラフ：", size=16),
                                response_graph_display_container,
                                ft.Text("説明変数の時系列グラフ：", size=16),
                                explanatory_graphs_scroll_view,
                            ],
                            expand=True,
                        ),
                    ],
                    expand=True,
                ),
            ],
            expand=True,
        ),
    )

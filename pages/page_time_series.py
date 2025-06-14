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
)  # 循環参照を避けるため
from components.plot_utils import (
    plot_single_time_series,
    plot_multiple_time_series_grid,
    TRANSFORMATION_TYPES,
)

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
response_graph_display_container = ft.Column(controls=[], expand=False)  # 目的変数用
explanatory_graphs_scroll_view = ft.Column(
    controls=[], scroll=ft.ScrollMode.AUTO, expand=True
)  # 説明変数用


def get_dataframe_for_pattern(
    df: pd.DataFrame, transformation_type: str, is_standardized: bool
) -> pd.DataFrame | None:
    """指定された変換パターンと標準化に基づいてDataFrameを取得する"""
    print(
        f"DEBUG: get_dataframe_for_pattern呼び出し: transformation_type={transformation_type}, is_standardized={is_standardized}"
    )

    if df.empty:
        print("DEBUG: 入力DataFrameが空です。")
        return None

    transformed_df = df.copy()

    # kijyunnengetuカラムを保持
    kijyunnengetu_col = None
    if "kijyunnengetu" in transformed_df.columns:
        kijyunnengetu_col = transformed_df["kijyunnengetu"]
        transformed_df = transformed_df.drop(columns=["kijyunnengetu"])

    # 数値型以外のカラムを除外
    numeric_cols = transformed_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        print("DEBUG: 数値型のカラムがありません。")
        return None
    transformed_df = transformed_df[numeric_cols]

    # 変換を適用
    if transformation_type == "log":
        for col in transformed_df.columns:
            # 0以下の値はNaNにするか、適切に処理する
            transformed_df[col] = np.log1p(transformed_df[col])
    elif transformation_type == "diff":
        for col in transformed_df.columns:
            transformed_df[col] = transformed_df[col].diff()
    elif transformation_type == "log_diff":
        for col in transformed_df.columns:
            transformed_df[col] = np.log1p(transformed_df[col]).diff()

    # kijyunnengetuカラムを戻す
    if kijyunnengetu_col is not None:
        transformed_df["kijyunnengetu"] = kijyunnengetu_col

    # NaN値の行を削除（特に差分化で発生）
    transformed_df = transformed_df.dropna()

    if transformed_df.empty:
        print(f"DEBUG: 変換後のDataFrameが空です: {transformation_type}")
        return None

    if is_standardized:
        # kijyunnengetuカラムを再度保持
        std_kijyunnengetu_col = None
        if "kijyunnengetu" in transformed_df.columns:
            std_kijyunnengetu_col = transformed_df["kijyunnengetu"]
            transformed_df = transformed_df.drop(columns=["kijyunnengetu"])

        # StandardScalerを適用
        scaler = StandardScaler()
        # NaNを除外してfit_transformを実行し、NaNを埋め戻す
        # transformed_df_numeric = transformed_df.select_dtypes(include=np.number)
        transformed_df_numeric = transformed_df.copy()
        scaled_array = scaler.fit_transform(transformed_df_numeric)
        scaled_df = pd.DataFrame(
            scaled_array,
            columns=transformed_df_numeric.columns,
            index=transformed_df_numeric.index,
        )

        if std_kijyunnengetu_col is not None:
            scaled_df["kijyunnengetu"] = std_kijyunnengetu_col

        return scaled_df

    return transformed_df


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


def time_series_page(page: ft.Page):
    page.title = "時系列データ分析"

    global checkbox_states, transformation_states, standardization_states, variable_types, selected_response_variable

    print("DEBUG: time_series_page関数が呼び出されました。")

    # 初期データの読み込み
    initial_df = read_dataframe_from_sqlite("merged_data")
    if initial_df is None or initial_df.empty:
        status_text.value = "データをロードできませんでした。データ取込み・参照タブでCSVをロードしてください。"
        status_text.color = ft.Colors.RED_500
        page.update()
        return ft.Container(content=status_text)

    # kijyunnengetu以外のカラムを取得
    all_columns = [col for col in initial_df.columns if col != "kijyunnengetu"]
    print(f"DEBUG: initial_dfから取得したカラム: {all_columns}")

    # 目的変数選択用のドロップダウン
    response_variable_dropdown = ft.Dropdown(
        label="目的変数を選択",
        options=[ft.dropdown.Option(col) for col in all_columns],
        width=250,
        on_change=lambda e: handle_response_variable_selection(e, all_columns),
        value=selected_response_variable,  # 初期選択を反映
        text_size=16,  # 文字サイズを大きく
        label_style=ft.TextStyle(
            size=16, weight=ft.FontWeight.BOLD
        ),  # ラベルの文字サイズも大きく
    )

    # 目的変数の変換パターン選択用のドロップダウン
    response_variable_transformation = ft.Dropdown(
        label="目的変数の変換パターン",
        options=[
            ft.dropdown.Option(key, value)
            for key, value in TRANSFORMATION_TYPES.items()
        ],
        value="none",
        width=180,  # 横幅を250から180に縮小
        on_change=lambda e: (
            handle_transformation_change(e, selected_response_variable)
            if selected_response_variable
            else None
        ),
        text_size=12,  # 文字サイズを小さく
        label_style=ft.TextStyle(size=12),  # ラベルの文字サイズも小さく
    )

    # 目的変数の標準化選択用のトグルボタン
    response_variable_standardization = ft.Switch(
        label="標準化",
        value=False,
        on_change=lambda e: (
            handle_standardization_change(e, selected_response_variable)
            if selected_response_variable
            else None
        ),
        label_style=ft.TextStyle(size=12),  # ラベルの文字サイズを小さく
    )

    # 目的変数の選択と変換パターンを横並びに配置
    response_variable_row = ft.Row(
        [
            response_variable_dropdown,  # 目的変数選択を左に
            response_variable_transformation,  # 変換パターンを右に
            response_variable_standardization,  # 標準化トグルボタンをさらに右に
        ],
        alignment=ft.MainAxisAlignment.START,
        spacing=10,
    )

    def handle_response_variable_selection(e: ft.ControlEvent, all_cols: list):
        """目的変数選択時の処理"""
        global selected_response_variable
        selected_response_variable = e.control.value
        print(f"DEBUG: 目的変数が選択されました: {selected_response_variable}")

        # 変数タイプとチェックボックスの状態をリセット
        for col in all_cols:
            if col == selected_response_variable:
                variable_types[col] = "response"
                checkbox_states[col] = True  # 目的変数は自動でチェック
            else:
                variable_types[col] = "explanatory"
                checkbox_states[col] = False  # 説明変数は自動でチェックを外す
            # 目的変数に設定された場合は、変換タイプと標準化も初期化
            transformation_states[col] = "none"
            standardization_states[col] = False

        initialize_variable_controls()  # コントロールを再構築
        page.update()

    def handle_checkbox_change(e: ft.ControlEvent, column: str):
        """チェックボックスの状態変更を処理"""
        global checkbox_states
        checkbox_states[column] = e.control.value
        print(f"DEBUG: チェックボックス変更 - {column}: {checkbox_states[column]}")

    def handle_transformation_change(e: ft.ControlEvent, column: str):
        """変換タイプの変更を処理"""
        global transformation_states
        transformation_states[column] = e.control.value
        print(f"DEBUG: 変換タイプ変更 - {column}: {transformation_states[column]}")

    def handle_standardization_change(e: ft.ControlEvent, column: str):
        """標準化スイッチの変更を処理"""
        global standardization_states
        standardization_states[column] = e.control.value
        print(f"DEBUG: 標準化変更 - {column}: {standardization_states[column]}")

    def update_graphs(e):
        """グラフを更新する"""
        global checkbox_states, transformation_states, standardization_states, variable_types

        if not selected_response_variable:
            status_text.value = "目的変数を選択してください。"
            status_text.color = ft.Colors.RED_500
            page.update()
            return

        # 選択された変数を取得
        selected_variables = []
        for col, is_checked in checkbox_states.items():
            if is_checked:
                selected_variables.append(col)

        if not selected_variables:
            status_text.value = "少なくとも1つの変数を選択してください。"
            status_text.color = ft.Colors.RED_500
            page.update()
            return

        # グラフコンテナをクリア
        response_graph_display_container.controls.clear()
        explanatory_graphs_scroll_view.controls.clear()

        # 目的変数のグラフを先頭行に表示
        response_transformation = transformation_states.get(
            selected_response_variable, "none"
        )
        response_standardization = standardization_states.get(
            selected_response_variable, False
        )
        response_df = get_dataframe_for_pattern(
            initial_df, response_transformation, response_standardization
        )

        # plot_utilsの関数を使用して目的変数のグラフを生成
        response_img_base64 = plot_single_time_series(
            response_df,
            selected_response_variable,
            response_transformation,
            response_standardization,
            figsize=(10, 4),
        )

        # 目的変数のグラフをコンテナに追加
        response_graph_display_container.controls.append(
            ft.Container(
                content=ft.Image(src_base64=response_img_base64, width=800, height=300),
                padding=10,
                border=ft.border.all(
                    2, ft.Colors.BLUE_200
                ),  # 目的変数のグラフを青い枠で囲む
                border_radius=10,
            )
        )
        # 目的変数のグラフが更新されたことを明示するためにupdate
        response_graph_display_container.update()

        # 説明変数のグラフを生成
        explanatory_variables = [
            var for var in selected_variables if var != selected_response_variable
        ]
        if explanatory_variables:
            # plot_utilsの関数を使用して説明変数のグラフを生成
            explanatory_df = get_dataframe_for_pattern(
                initial_df, "none", False
            )  # 変換なしのデータを使用
            explanatory_img_base64 = plot_multiple_time_series_grid(
                explanatory_df,
                explanatory_variables,
                transformation_states,
                standardization_states,
                n_cols=2,
                figsize=(12, 4),
            )

            # 説明変数のグラフをコンテナに追加
            explanatory_graphs_scroll_view.controls.append(
                ft.Container(
                    content=ft.Image(
                        src_base64=explanatory_img_base64,
                        width=800,
                        height=300 * ceil(len(explanatory_variables) / 2),
                    ),
                    padding=10,
                    border=ft.border.all(
                        2, ft.Colors.GREY_300
                    ),  # 説明変数のグラフをグレーの枠で囲む
                    border_radius=10,
                )
            )
            # 説明変数のグラフが更新されたことを明示するためにupdate
            explanatory_graphs_scroll_view.update()

        status_text.value = "グラフを更新しました。"
        status_text.color = ft.Colors.GREEN_700
        page.update()

    def clear_graphs(e):
        """表示されているグラフをクリアする"""
        print("DEBUG: グラフクリア処理を開始します。")
        response_graph_display_container.controls.clear()
        explanatory_graphs_scroll_view.controls.clear()
        status_text.value = "グラフをクリアしました。"
        status_text.color = ft.Colors.BLACK54
        page.update()
        print("DEBUG: グラフクリア処理を完了しました。")

    # グラフ表示ボタン
    show_graph_button = ft.ElevatedButton(
        "グラフを表示", icon=ft.Icons.SHOW_CHART, on_click=update_graphs
    )

    # グラフクリアボタン
    clear_graph_button = ft.ElevatedButton(
        "グラフをクリア", icon=ft.Icons.DELETE, on_click=clear_graphs
    )

    def initialize_variable_controls():
        """変数選択用のコントロールを初期化"""
        global checkbox_states, transformation_states, standardization_states, variable_types, selected_response_variable

        # データベースからカラム一覧を取得
        df = read_dataframe_from_sqlite("merged_data")
        if df is None or df.empty:
            print("DEBUG: データが存在しません。")
            return

        # kijyunnengetu以外のカラムを取得
        columns = [col for col in df.columns if col != "kijyunnengetu"]
        print(f"DEBUG: 利用可能なカラム: {columns}")

        # 各変数のコントロールを生成
        response_controls = []
        explanatory_controls = []

        for col in columns:
            # 初期状態の設定
            if col not in checkbox_states:
                checkbox_states[col] = False
            if col not in transformation_states:
                transformation_states[col] = "none"
            if col not in standardization_states:
                standardization_states[col] = False
            if col not in variable_types:
                variable_types[col] = "explanatory"  # デフォルトは説明変数

            # 目的変数が選択されている場合、その変数のタイプを「response」に強制
            if col == selected_response_variable:
                variable_types[col] = "response"
                checkbox_states[col] = True  # 目的変数は自動でチェック
            else:
                # 選択された目的変数以外の変数は説明変数として扱う
                variable_types[col] = "explanatory"
                # ここではチェックボックスの状態は変更しない（手動選択の余地を残す）

            print(f"DEBUG: 変数 {col} の初期状態:")
            print(f"  - チェックボックス: {checkbox_states[col]}")
            print(f"  - 変換タイプ: {transformation_states[col]}")
            print(f"  - 標準化: {standardization_states[col]}")
            print(f"  - 変数タイプ: {variable_types[col]}")

            # 変数名の表示
            variable_name = ft.Text(col, size=12)

            # チェックボックス
            checkbox = ft.Checkbox(
                key=f"chk_{col}",  # keyを追加
                label="",
                value=checkbox_states[col],
                on_change=lambda e, col=col: handle_checkbox_change(e, col),
            )

            # 変換タイプ選択用ドロップダウン
            transformation_dropdown = ft.Dropdown(
                key=f"transform_dropdown_{col}",  # keyを追加
                label="変換タイプ",
                options=[
                    ft.dropdown.Option(key, value)
                    for key, value in TRANSFORMATION_TYPES.items()
                ],
                value=transformation_states[col],
                width=120,
                on_change=lambda e, col=col: handle_transformation_change(e, col),
            )

            # 標準化トグルボタン
            standardization_switch = ft.Switch(
                label="標準化",
                value=standardization_states[col],
                on_change=lambda e, col=col: handle_standardization_change(e, col),
            )

            # チェックボックスとドロップダウンを横並びに配置（チェックボックスを左に）
            explanatory_variable_checkboxes.controls.append(
                ft.Row(
                    [
                        checkbox,
                        transformation_dropdown,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=10,
                )
            )

            # チェックボックスのスタイルを調整
            # checkbox.height = 35  # チェックボックスの高さを35に設定
            # checkbox.text_style = ft.TextStyle(size=12)  # チェックボックスのテキストサイズを小さく

            # ドロップダウンのスタイルを調整
            # transformation_dropdown.height = 35  # ドロップダウンの高さを35に設定
            # transformation_dropdown.text_size = 12  # ドロップダウンのテキストサイズを小さく
            # transformation_dropdown.label_style = ft.TextStyle(size=12)  # ドロップダウンのラベルサイズを小さく

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

            # 目的変数が選択されている場合は、目的変数セクションには追加しない
            if col == selected_response_variable:
                response_controls.append(container)  # ここには目的変数のみが追加される
            else:
                explanatory_controls.append(container)

        print(f"DEBUG: 目的変数セクションのコントロール数: {len(response_controls)}")
        print(f"DEBUG: 説明変数セクションのコントロール数: {len(explanatory_controls)}")

        response_variable_checkboxes.controls = response_controls
        explanatory_variable_checkboxes.controls = explanatory_controls
        page.update()

    # 初期化処理
    initialize_variable_controls()

    # 左側のパネル（変数選択）
    left_panel = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [show_graph_button, clear_graph_button],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=10,
                ),
                status_text,
                ft.Text("目的変数選択", size=16, weight=ft.FontWeight.BOLD),
                response_variable_row,
                ft.Divider(),
                ft.Text("説明変数", size=16, weight=ft.FontWeight.BOLD),
                explanatory_variable_checkboxes,
                ft.Divider(),
            ],
            spacing=10,
        ),
        width=550,  # 幅を広げる
        padding=10,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5,
    )

    # 右側のパネル（グラフ表示）
    right_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("グラフ表示", size=16, weight=ft.FontWeight.BOLD),
                response_graph_display_container,  # 目的変数のグラフを固定表示
                ft.Divider(),  # 目的変数と説明変数の間に区切り線を追加
                explanatory_graphs_scroll_view,  # 説明変数のグラフをスクロール可能に
            ],
            spacing=10,
            expand=True,
        ),
        expand=True,  # 横方向にexpand
        padding=10,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5,
    )

    return ft.Row(
        [left_panel, right_panel],
        expand=True,
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

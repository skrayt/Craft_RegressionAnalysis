"""
This module provides a page for correlation analysis between
a selected target and multiple features.
"""

# pages/page_analysis.py

import flet as ft
import pandas as pd
import numpy as np
from db.database import read_dataframe_from_sqlite
from components.plot_utils import (
    plot_corr_heatmap,
    plot_vif_heatmap,
    calculate_vif,
    create_vif_table,
)

# 利用可能なデータ変換パターンを定義
TRANSFORMATION_PATTERNS = {
    "none": "変換なし",
    "none_standardized": "変換なし（標準化）",
    "log": "対数変換",
    "log_standardized": "対数変換（標準化）",
    "diff": "差分化",
    "diff_standardized": "差分化（標準化）",
    "log_diff": "対数変換後に差分化",
    "log_diff_standardized": "対数変換後に差分化（標準化）",
}


def get_dataframe_for_pattern(pattern: str) -> pd.DataFrame:
    """
    指定された変換パターンに対応するデータフレームを取得する
    """
    if pattern == "none":
        return read_dataframe_from_sqlite("merged_data")
    elif pattern == "none_standardized":
        return read_dataframe_from_sqlite("merged_standardized")
    elif pattern == "log":
        return read_dataframe_from_sqlite("log_data")
    elif pattern == "log_standardized":
        return read_dataframe_from_sqlite("log_data_standardized")
    elif pattern == "diff":
        return read_dataframe_from_sqlite("diff_data")
    elif pattern == "diff_standardized":
        return read_dataframe_from_sqlite("diff_data_standardized")
    elif pattern == "log_diff":
        return read_dataframe_from_sqlite("log_diff_data")
    elif pattern == "log_diff_standardized":
        return read_dataframe_from_sqlite("log_diff_data_standardized")
    else:
        raise ValueError(f"未知の変換パターン: {pattern}")


def analysis_page(page: ft.Page) -> ft.Container:
    """
    Create the correlation and VIF analysis page with interactive controls.
    """
    # 初期データとして変換なしのデータを使用
    initial_df = get_dataframe_for_pattern("none")
    all_columns = initial_df.columns.tolist()

    # 目的変数の選択と変換パターン
    target_selector = ft.Dropdown(
        label="目的変数",
        options=[ft.dropdown.Option(col) for col in all_columns],
        on_change=lambda e: refresh_feature_options(),
    )
    target_pattern = ft.Dropdown(
        label="目的変数の変換パターン",
        options=[
            ft.dropdown.Option(key, value)
            for key, value in TRANSFORMATION_PATTERNS.items()
        ],
        value="none",
        on_change=lambda e: refresh_feature_options(),
    )

    # 説明変数の選択と変換パターン
    features_available = ft.Column()
    features_selected = ft.Column()
    feature_patterns = {}  # 各説明変数の変換パターンを保持

    corr_heatmap_image = ft.Image()
    vif_table = ft.DataTable(columns=[ft.DataColumn(ft.Text("VIF値"))], rows=[])

    def refresh_feature_options():
        selected = target_selector.value
        features_available.controls.clear()
        features_selected.controls.clear()
        feature_patterns.clear()

        for col in all_columns:
            if col != selected:
                # 説明変数の選択チェックボックス
                checkbox = ft.Checkbox(label=col, value=False)
                # 変換パターン選択ドロップダウン
                pattern_dropdown = ft.Dropdown(
                    label=f"{col}の変換パターン",
                    options=[
                        ft.dropdown.Option(key, value)
                        for key, value in TRANSFORMATION_PATTERNS.items()
                    ],
                    value="none",
                    disabled=True,  # 初期状態は無効
                    width=200,  # ドロップダウンの幅を固定
                )

                # チェックボックスの状態変更時にドロップダウンの有効/無効を切り替え
                def create_on_change_handler(cb, dd):
                    def handler(e):
                        dd.disabled = not cb.value
                        page.update()

                    return handler

                checkbox.on_change = create_on_change_handler(
                    checkbox, pattern_dropdown
                )
                feature_patterns[col] = pattern_dropdown

                # チェックボックスとドロップダウンを横並びに配置（チェックボックスを左に）
                features_available.controls.append(
                    ft.Row(
                        [
                            checkbox,
                            pattern_dropdown,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        spacing=10,
                    )
                )
        page.update()

    def run_correlation():
        selected = target_selector.value
        if not selected:
            return

        # 目的変数のデータを取得
        target_df = get_dataframe_for_pattern(target_pattern.value)
        target_data = target_df[selected]

        # 選択された説明変数とその変換パターンを取得
        selected_features = []
        feature_dfs = []

        for control in features_available.controls:
            if isinstance(control, ft.Column):
                checkbox = control.controls[0]
                pattern_dropdown = control.controls[1]
                if checkbox.value:
                    feature_name = checkbox.label
                    pattern = pattern_dropdown.value
                    feature_df = get_dataframe_for_pattern(pattern)
                    selected_features.append(feature_name)
                    feature_dfs.append(feature_df[feature_name])

        if not selected_features:
            return

        # すべてのデータを結合
        combined_df = pd.concat([target_data] + feature_dfs, axis=1)
        combined_df.columns = [selected] + selected_features

        # 相関ヒートマップを作成
        corr_heatmap_image.src_base64 = plot_corr_heatmap(
            combined_df, selected, selected_features
        )

        # VIFクロステーブルを作成
        vif_table.columns, vif_table.rows = create_vif_table(
            combined_df, selected_features
        )
        page.update()

    return ft.Container(
        content=ft.Row(
            [
                ft.Column(
                    [
                        ft.Text("相関係数分析", size=20, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text("目的変数の設定", size=16),
                        ft.Row(
                            [
                                target_selector,
                                target_pattern,
                            ],
                            alignment=ft.MainAxisAlignment.START,
                            spacing=10,
                        ),
                        ft.Divider(),
                        ft.Text("説明変数の設定", size=16),
                        ft.Text(
                            "※目的変数に選択した項目は説明変数から除外されます",
                            size=12,
                            color=ft.Colors.GREY_600,
                        ),
                        ft.Row(
                            [
                                ft.Column(
                                    [features_available],
                                    scroll=ft.ScrollMode.AUTO,
                                ),
                            ],
                        ),
                        ft.ElevatedButton(
                            "相関係数/VIFを表示",
                            on_click=lambda e: run_correlation(),
                            style=ft.ButtonStyle(
                                color=ft.Colors.WHITE,
                                bgcolor=ft.Colors.BLUE,
                            ),
                        ),
                    ],
                    expand=1,
                    scroll=ft.ScrollMode.AUTO,
                ),
                ft.Column(
                    [
                        ft.Text("相関ヒートマップ", size=16, weight=ft.FontWeight.BOLD),
                        corr_heatmap_image,
                        ft.Divider(),
                        ft.Text("VIF値", size=16, weight=ft.FontWeight.BOLD),
                        vif_table,
                    ],
                    expand=4,
                    spacing=20,
                    scroll=ft.ScrollMode.AUTO,
                ),
            ]
        ),
        expand=True,
        padding=20,
        border=ft.border.all(1, ft.Colors.GREY_400),
    )

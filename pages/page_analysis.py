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
    # plot_vif_heatmap,
    # calculate_vif,
    create_vif_table,
)
from components.variable_selector import VariableSelector
from utils.data_transformation import get_dataframe_for_pattern

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

# 変換パターンに応じたサフィックスを定義
TRANSFORMATION_SUFFIXES = {
    "none": "",
    "none_standardized": "_std",
    "log": "_log",
    "log_standardized": "_log_std",
    "diff": "_diff",
    "diff_standardized": "_diff_std",
    "log_diff": "_log_diff",
    "log_diff_standardized": "_log_diff_std",
}


def analysis_page(page: ft.Page) -> ft.Container:
    """
    Create the correlation and VIF analysis page with interactive controls.
    """
    # 初期データの読み込み
    initial_df = read_dataframe_from_sqlite("merged_data")
    if initial_df is None or initial_df.empty:
        return ft.Container(
            content=ft.Text(
                "データが読み込まれていません。データ取込み・参照タブでCSVをロードしてください。"
            )
        )

    # kijyunnengetu以外のカラムを取得
    all_columns = [col for col in initial_df.columns if col != "kijyunnengetu"]

    # 結果表示用のコンテナ
    result_container = ft.Container(
        content=ft.Column(
            [
                ft.Text("分析結果", size=20, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "変数を選択して「分析実行」ボタンを押してください。",
                    color=ft.Colors.GREY_700,
                ),
            ],
            spacing=10,
        ),
        padding=20,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5,
        expand=True,
    )

    # 分析実行ボタン
    analyze_button = ft.ElevatedButton(
        "分析実行",
        icon=ft.Icons.PLAY_ARROW_ROUNDED,
        on_click=lambda e: run_analysis(),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE,
            padding=10,
        ),
    )

    def validate_data(
        df: pd.DataFrame, column: str, transformation: str
    ) -> tuple[bool, str]:
        """
        データの妥当性をチェックする

        Returns:
            tuple[bool, str]: (データが有効かどうか, エラーメッセージ)
        """
        if df[column].isnull().any():
            return False, f"列 '{column}' に欠損値が含まれています。"

        if transformation in ["log", "log_diff"]:
            if (df[column] <= 0).any():
                return (
                    False,
                    f"列 '{column}' に対数変換できない値（0以下）が含まれています。",
                )

        if transformation in ["diff", "log_diff"]:
            if df[column].isnull().any():
                return False, f"列 '{column}' に差分化できない欠損値が含まれています。"

        return True, ""

    def run_analysis():
        """分析を実行し、結果を表示する"""
        target, features = variable_selector.get_selected_variables()
        if not target or not features:
            # エラーメッセージを表示
            result_container.content = ft.Column(
                [
                    ft.Text("分析結果", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "目的変数と説明変数を選択してください。",
                        color=ft.Colors.RED_700,
                    ),
                ],
                spacing=10,
            )
            page.update()
            return

        # 変数の設定を取得
        settings = variable_selector.get_variable_settings()
        print(f"DEBUG: 変数の設定: {settings}")  # デバッグ用

        try:
            # 目的変数のデータを取得
            target_df = get_dataframe_for_pattern(
                initial_df,
                settings[target]["transformation"],
                settings[target]["standardization"],
            )

            # 目的変数のデータを検証
            is_valid, error_msg = validate_data(
                target_df, target, settings[target]["transformation"]
            )
            if not is_valid:
                raise ValueError(f"目的変数 {target} のデータが無効です: {error_msg}")

            print(f"DEBUG: 目的変数のデータ形状: {target_df.shape}")  # デバッグ用

            # 説明変数のデータを取得
            feature_dfs = []
            for feature in features:
                feature_df = get_dataframe_for_pattern(
                    initial_df,
                    settings[feature]["transformation"],
                    settings[feature]["standardization"],
                )

                # 説明変数のデータを検証
                is_valid, error_msg = validate_data(
                    feature_df, feature, settings[feature]["transformation"]
                )
                if not is_valid:
                    raise ValueError(
                        f"説明変数 {feature} のデータが無効です: {error_msg}"
                    )

                print(
                    f"DEBUG: 説明変数 {feature} のデータ形状: {feature_df.shape}"
                )  # デバッグ用
                feature_dfs.append(feature_df[[feature]])

            # データを結合
            X = pd.concat(feature_dfs, axis=1)
            y = target_df[[target]]

            # 分析用データフレームを作成
            analysis_df = pd.concat([y, X], axis=1)

            # 最終的なデータフレームの検証
            if analysis_df.isnull().any().any():
                raise ValueError("分析用データフレームに欠損値が含まれています。")
            if np.isinf(analysis_df.values).any():
                raise ValueError("分析用データフレームに無限大の値が含まれています。")

            print(
                f"DEBUG: 分析用データフレームの形状: {analysis_df.shape}"
            )  # デバッグ用
            print(
                f"DEBUG: 分析用データフレームのカラム: {analysis_df.columns.tolist()}"
            )  # デバッグ用

            # 相関ヒートマップを作成
            corr_heatmap = ft.Image(
                src_base64=plot_corr_heatmap(analysis_df, target, features),
                width=600,
                height=400,
            )

            # VIFクロステーブルを作成
            vif_headers, vif_rows = create_vif_table(analysis_df, features)
            vif_table = ft.DataTable(
                columns=vif_headers,
                rows=vif_rows,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=5,
            )

            # 変換パターンの情報を表示
            transformation_info = ft.Column(
                [
                    ft.Text("変換パターン", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        f"目的変数 ({target}): {settings[target]['transformation']}"
                        + (" (標準化済)" if settings[target]["standardization"] else "")
                    ),
                ]
            )
            for feature in features:
                transformation_info.controls.append(
                    ft.Text(
                        f"説明変数 ({feature}): {settings[feature]['transformation']}"
                        + (
                            " (標準化済)"
                            if settings[feature]["standardization"]
                            else ""
                        )
                    )
                )

            # 結果を表示
            result_container.content = ft.Column(
                [
                    ft.Text("分析結果", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(f"目的変数: {target}", size=16),
                    ft.Text(f"説明変数: {', '.join(features)}", size=16),
                    ft.Divider(),
                    transformation_info,
                    ft.Divider(),
                    ft.Text("相関行列", size=16, weight=ft.FontWeight.BOLD),
                    corr_heatmap,
                    ft.Divider(),
                    ft.Text("VIF値", size=16, weight=ft.FontWeight.BOLD),
                    vif_table,
                ],
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
            )

        except ValueError as e:
            print(f"DEBUG: データ検証エラー: {str(e)}")  # デバッグ用
            result_container.content = ft.Column(
                [
                    ft.Text("分析結果", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        f"データエラー: {str(e)}",
                        color=ft.Colors.RED_700,
                    ),
                ],
                spacing=10,
            )
        except Exception as e:
            print(f"DEBUG: 分析実行中にエラーが発生: {str(e)}")  # デバッグ用
            result_container.content = ft.Column(
                [
                    ft.Text("分析結果", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        f"分析実行中にエラーが発生しました: {str(e)}",
                        color=ft.Colors.RED_700,
                    ),
                ],
                spacing=10,
            )

        page.update()

    def on_variable_change():
        """変数選択が変更された時の処理"""
        # 変数が選択されていない場合は何もしない
        target, features = variable_selector.get_selected_variables()
        if not target or not features:
            return

        # 分析を実行
        run_analysis()

    # 変数選択コンポーネントの初期化
    variable_selector = VariableSelector(
        page=page,
        all_columns=all_columns,
        on_variable_change=None,  # 変数選択時の自動更新を無効化
    )

    # UIコンポーネントを取得
    target_row, feature_container = variable_selector.get_ui_components()

    # 左側のパネル（変数選択）
    left_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("相関分析・VIF分析", size=20, weight=ft.FontWeight.BOLD),
                target_row,
                ft.Text("説明変数を選択：", size=16),
                feature_container,
                ft.Container(
                    content=analyze_button,
                    alignment=ft.alignment.center,
                    padding=10,
                ),
            ],
            spacing=10,
        ),
        width=550,
        padding=10,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=5,
    )

    return ft.Row(
        [left_panel, result_container],
        expand=True,
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

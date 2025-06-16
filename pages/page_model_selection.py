"""
モデル選択ページモジュール。
目的変数と説明変数候補の総当たり分析を行い、モデル候補を評価する機能を提供する。
"""

import flet as ft
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from db.database import read_dataframe_from_sqlite
from components.variable_selector import VariableSelector
from utils.data_transformation import get_dataframe_for_pattern
from components.plot_utils import TRANSFORMATION_TYPES


def calculate_model_metrics(
    y: pd.Series, X: pd.DataFrame, lag_order: int = 0
) -> Dict[str, float]:
    """
    回帰モデルの各種統計量を計算する

    Parameters:
    -----------
    y : pd.Series
        目的変数
    X : pd.DataFrame
        説明変数
    lag_order : int, default 0
        ラグ次数

    Returns:
    --------
    Dict[str, float]
        各種統計量を含む辞書
    """
    # ラグ付きデータの作成
    if lag_order > 0:
        y_lagged = y.shift(lag_order).dropna()
        X_lagged = X.shift(lag_order).dropna()
        y = y[lag_order:]
        X = X[lag_order:]
        y = y_lagged
        X = X_lagged

    # 定数項を追加
    X = add_constant(X)

    # モデルの推定
    model = OLS(y, X).fit()

    # 各種統計量の計算
    metrics = {
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "AIC": model.aic,
        "BIC": model.bic,
        "DW": durbin_watson(model.resid),
        "F_stat": model.fvalue,
        "F_pvalue": model.f_pvalue,
    }

    return metrics


def calculate_vif(X: pd.DataFrame) -> Dict[str, float]:
    """
    説明変数のVIF値を計算する

    Parameters:
    -----------
    X : pd.DataFrame
        説明変数

    Returns:
    --------
    Dict[str, float]
        各説明変数のVIF値
    """
    vif_data = {}
    for i, col in enumerate(X.columns):
        # 他の説明変数で回帰
        y = X[col]
        X_other = X.drop(col, axis=1)
        X_other = add_constant(X_other)

        # R2を計算
        model = OLS(y, X_other).fit()
        r2 = model.rsquared

        # VIFを計算
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_data[col] = vif

    return vif_data


def model_selection_page(page: ft.Page) -> ft.Container:
    """モデル選択ページを構築する関数"""

    # 初期データの読み込み
    initial_df = read_dataframe_from_sqlite("merged_data")
    if initial_df is None or initial_df.empty:
        return ft.Container(
            content=ft.Text(
                "データが読み込まれていません。データ取込み・参照タブでCSVをロードしてください。"
            )
        )

    # 結果表示用のコンテナ
    result_container = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    # ラグ次数選択用のスライダー
    lag_slider = ft.Slider(
        min=0,
        max=12,
        divisions=12,
        label="ラグ次数: {value}",
        value=0,
        width=200,  # スライダーの幅を調整
    )

    # 説明変数の数を選択するテキストフィールド
    n_features_input = ft.TextField(
        label="使用する説明変数の数",
        value="3",  # デフォルト値
        width=150,
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    # 変換パターン選択用のドロップダウン
    transformation_dropdown = ft.Dropdown(
        label="変換パターン",
        width=200,
        options=[ft.dropdown.Option(key) for key in TRANSFORMATION_TYPES.keys()],
        value="none",  # デフォルト値
    )

    # 利用可能な説明変数を表示するテキスト
    available_features_text = ft.Text(
        "利用可能な説明変数：",
        size=14,
        weight=ft.FontWeight.BOLD,
    )

    def update_available_features():
        """利用可能な説明変数のリストを更新する"""
        target = target_dropdown.value
        if target:
            features = [
                col
                for col in initial_df.columns
                if col != target and col != "kijyunnengetu"
            ]
            available_features_text.value = (
                f"利用可能な説明変数（{len(features)}個）：\n" + ", ".join(features)
            )
            page.update()

    def on_target_change(e):
        """目的変数が変更された時の処理"""
        update_available_features()

    # ソート順選択用のドロップダウン
    sort_dropdown = ft.Dropdown(
        label="ソート順",
        width=200,
        options=[
            ft.dropdown.Option("AIC昇順"),
            ft.dropdown.Option("BIC昇順"),
            ft.dropdown.Option("決定係数降順"),
        ],
        value="AIC昇順",  # デフォルト値
    )

    def run_analysis():
        """総当たり分析を実行する"""
        target = target_dropdown.value
        if not target:
            result_container.controls = [
                ft.Text("目的変数を選択してください。", color=ft.Colors.RED_700)
            ]
            page.update()
            return

        try:
            # 説明変数の数を取得
            n_features = int(n_features_input.value)
            if n_features < 1:
                raise ValueError("説明変数の数は1以上を指定してください。")

            # 目的変数を除いた全ての変数を説明変数候補として使用
            features = [
                col
                for col in initial_df.columns
                if col != target and col != "kijyunnengetu"
            ]
            if n_features > len(features):
                raise ValueError(
                    f"説明変数の数は{len(features)}以下を指定してください。"
                )

            # 変換パターンを取得
            transformation = transformation_dropdown.value

            # 目的変数のデータを取得（標準化済み）
            target_df = get_dataframe_for_pattern(
                initial_df,
                transformation,
                True,  # 標準化
            )
            y = target_df[target]

            # 説明変数のデータを取得（標準化済み）
            feature_dfs = []
            for feature in features:
                feature_df = get_dataframe_for_pattern(
                    initial_df,
                    transformation,
                    True,  # 標準化
                )
                feature_dfs.append(feature_df[[feature]])

            X = pd.concat(feature_dfs, axis=1)

            # 結果を格納するリスト
            results = []

            # 指定された数の説明変数の組み合わせを生成
            for combo in combinations(features, n_features):
                # 選択された説明変数でデータを抽出
                X_subset = X[list(combo)]

                # モデルの統計量を計算
                metrics = calculate_model_metrics(y, X_subset, int(lag_slider.value))

                # VIF値を計算
                vif_values = calculate_vif(X_subset)
                max_vif = max(vif_values.values())

                # 結果を追加
                results.append(
                    {
                        "説明変数": ", ".join(combo),
                        "R2": metrics["R2"],
                        "Adj_R2": metrics["Adj_R2"],
                        "AIC": metrics["AIC"],
                        "BIC": metrics["BIC"],
                        "DW": metrics["DW"],
                        "F統計量": metrics["F_stat"],
                        "F_p値": metrics["F_pvalue"],
                        "最大VIF": max_vif,
                    }
                )

            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)

            # ソート順に応じて結果をソート
            sort_order = sort_dropdown.value
            if sort_order == "AIC昇順":
                results_df = results_df.sort_values("AIC")
            elif sort_order == "BIC昇順":
                results_df = results_df.sort_values("BIC")
            elif sort_order == "決定係数降順":
                results_df = results_df.sort_values("R2", ascending=False)

            # 結果を表示するテーブルを作成
            table = ft.DataTable(
                columns=[
                    ft.DataColumn(
                        ft.Text(col),
                        numeric=col
                        not in ["説明変数"],  # 説明変数以外を数値列として扱う
                    )
                    for col in results_df.columns
                ],
                rows=[
                    ft.DataRow(
                        cells=[
                            ft.DataCell(
                                ft.Text(
                                    f"{val:.4f}" if isinstance(val, float) else str(val)
                                )
                            )
                            for val in row
                        ]
                    )
                    for row in results_df.values
                ],
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=5,
            )

            # テーブルを横スクロール可能なコンテナでラップ
            scrollable_table = ft.Container(
                content=ft.Row(
                    [table],
                    scroll=ft.ScrollMode.AUTO,
                ),
                expand=True,
            )

            # 結果を表示
            result_container.controls = [
                ft.Text("モデル候補の評価結果", size=20, weight=ft.FontWeight.BOLD),
                ft.Text(f"目的変数: {target}"),
                ft.Text(f"説明変数の数: {n_features}"),
                ft.Text(f"変換パターン: {TRANSFORMATION_TYPES[transformation]}"),
                ft.Text(f"ラグ次数: {int(lag_slider.value)}"),
                ft.Text(f"ソート順: {sort_order}"),
                ft.Divider(),
                scrollable_table,
            ]

        except ValueError as e:
            result_container.controls = [ft.Text(str(e), color=ft.Colors.RED_700)]
        except Exception as e:
            print(f"DEBUG: 分析実行中にエラーが発生: {str(e)}")
            result_container.controls = [
                ft.Text(
                    f"分析実行中にエラーが発生しました: {str(e)}",
                    color=ft.Colors.RED_700,
                )
            ]

        page.update()

    # 分析実行ボタン
    analyze_button = ft.ElevatedButton(
        text="総当たり分析実行",
        on_click=lambda _: run_analysis(),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE,
        ),
    )

    # 目的変数選択用のドロップダウン
    target_dropdown = ft.Dropdown(
        label="目的変数を選択",
        width=300,
        options=[
            ft.dropdown.Option(col)
            for col in initial_df.columns
            if col != "kijyunnengetu"
        ],
        on_change=on_target_change,  # 目的変数変更時のコールバックを設定
    )

    # ページのレイアウト
    return ft.Container(
        padding=20,
        expand=True,
        content=ft.Column(
            [
                ft.Text("モデル候補の評価", size=20, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("目的変数の選択：", size=16),
                                target_dropdown,
                                ft.Text("説明変数の設定：", size=16),
                                n_features_input,
                                ft.Text("変換パターンの設定：", size=16),
                                transformation_dropdown,
                                ft.Text("ラグ次数の設定：", size=16),
                                lag_slider,
                                ft.Text("ソート順の設定：", size=16),
                                sort_dropdown,
                                available_features_text,
                                ft.Container(
                                    content=ft.Text(
                                        "",
                                        size=12,
                                        color=ft.Colors.GREY_700,
                                    ),
                                    padding=ft.padding.only(left=10),
                                ),
                                analyze_button,
                            ],
                            width=400,  # 左カラムの幅を固定
                        ),
                        ft.Column(
                            [result_container],
                            expand=True,
                        ),
                    ],
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            expand=True,
        ),
    )

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
import os
from datetime import datetime


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

    # 結果保存用のDataFrame
    current_results_df = None

    # ファイル名入力用のテキストフィールド
    filename_input = ft.TextField(
        label="保存ファイル名（.csvは自動付加）",
        value=f"model_selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        width=300,
    )

    # 保存先ディレクトリの選択
    save_directory = ft.TextField(
        label="保存先ディレクトリ",
        value="./results",
        width=300,
    )

    def save_results_to_csv():
        """分析結果をCSVファイルとして保存する"""
        if current_results_df is None:
            snack = ft.SnackBar(
                content=ft.Text(
                    "保存する結果がありません。先に分析を実行してください。"
                ),
                bgcolor=ft.Colors.RED_400,
                duration=3000,
                show_close_icon=True,
            )
            page.snack_bar = snack
            page.update()
            return

        try:
            # 保存先ディレクトリの作成
            save_dir = save_directory.value.strip()
            if not save_dir:
                save_dir = "./results"

            os.makedirs(save_dir, exist_ok=True)

            # ファイル名の設定
            filename = filename_input.value.strip()
            if not filename:
                filename = f"model_selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if not filename.endswith(".csv"):
                filename += ".csv"

            # ファイルパスの作成
            filepath = os.path.join(save_dir, filename)

            # CSVファイルとして保存
            current_results_df.to_csv(filepath, index=False, encoding="utf-8-sig")

            snack = ft.SnackBar(
                content=ft.Text(
                    f"✅ CSVファイルの保存が完了しました！\n保存先: {filepath}"
                ),
                bgcolor=ft.Colors.GREEN_400,
                duration=5000,
                show_close_icon=True,
            )
            page.snack_bar = snack
            page.update()

        except Exception as e:
            snack = ft.SnackBar(
                content=ft.Text(
                    f"❌ CSVファイル保存中にエラーが発生しました: {str(e)}"
                ),
                bgcolor=ft.Colors.RED_400,
                duration=5000,
                show_close_icon=True,
            )
            page.snack_bar = snack
            page.update()

    def save_results_to_html():
        """分析結果をHTMLファイルとして保存する"""
        if current_results_df is None:
            snack = ft.SnackBar(
                content=ft.Text(
                    "保存する結果がありません。先に分析を実行してください。"
                ),
                bgcolor=ft.Colors.RED_400,
                duration=3000,
                show_close_icon=True,
            )
            page.snack_bar = snack
            page.update()
            return

        try:
            # 保存先ディレクトリの作成
            save_dir = save_directory.value.strip()
            if not save_dir:
                save_dir = "./results"

            os.makedirs(save_dir, exist_ok=True)

            # ファイル名の設定
            filename = filename_input.value.strip()
            if not filename:
                filename = f"model_selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if not filename.endswith(".html"):
                filename += ".html"

            # ファイルパスの作成
            filepath = os.path.join(save_dir, filename)

            # HTMLテーブルのスタイルを定義
            html_style = """
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; font-weight: bold; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                tr:hover { background-color: #e8f4f8; }
                .info { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .metric { font-weight: bold; color: #2980b9; }
                .timestamp { color: #7f8c8d; font-size: 0.9em; }
            </style>
            """

            # 分析情報を取得
            target = target_dropdown.value
            n_features = n_features_input.value
            transformation = transformation_dropdown.value
            lag_order = int(lag_slider.value)
            sort_order = sort_dropdown.value

            # HTMLコンテンツを生成
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>モデル選択分析結果</title>
                {html_style}
            </head>
            <body>
                <h1>モデル選択分析結果</h1>
                
                <div class="info">
                    <h2>分析設定</h2>
                    <p><span class="metric">目的変数:</span> {target}</p>
                    <p><span class="metric">説明変数の数:</span> {n_features}</p>
                    <p><span class="metric">変換パターン:</span> {TRANSFORMATION_TYPES[transformation]}</p>
                    <p><span class="metric">ラグ次数:</span> {lag_order}</p>
                    <p><span class="metric">ソート順:</span> {sort_order}</p>
                    <p><span class="metric">分析結果数:</span> {len(current_results_df)}件</p>
                    <p class="timestamp">生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                </div>

                <h2>モデル候補の評価結果</h2>
                {current_results_df.to_html(index=False, classes='data-table', float_format='%.4f')}
                
                <div class="info">
                    <h2>統計量の説明</h2>
                    <p><span class="metric">R²:</span> 決定係数（1に近いほど良い）</p>
                    <p><span class="metric">Adj_R²:</span> 調整済み決定係数（説明変数の数を考慮）</p>
                    <p><span class="metric">AIC:</span> 赤池情報量基準（小さいほど良い）</p>
                    <p><span class="metric">BIC:</span> ベイズ情報量基準（小さいほど良い）</p>
                    <p><span class="metric">DW:</span> Durbin-Watson統計量（2に近いほど良い）</p>
                    <p><span class="metric">F統計量:</span> F検定統計量（大きいほど良い）</p>
                    <p><span class="metric">F_p値:</span> F検定のp値（0.05未満が有意）</p>
                    <p><span class="metric">最大VIF:</span> 最大分散拡大要因（10未満が望ましい）</p>
                </div>
            </body>
            </html>
            """

            # HTMLファイルとして保存
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            snack = ft.SnackBar(
                content=ft.Text(
                    f"✅ HTMLファイルの保存が完了しました！\n保存先: {filepath}"
                ),
                bgcolor=ft.Colors.GREEN_400,
                duration=5000,
                show_close_icon=True,
            )
            page.snack_bar = snack
            page.update()

        except Exception as e:
            snack = ft.SnackBar(
                content=ft.Text(
                    f"❌ HTMLファイル保存中にエラーが発生しました: {str(e)}"
                ),
                bgcolor=ft.Colors.RED_400,
                duration=5000,
                show_close_icon=True,
            )
            page.snack_bar = snack
            page.update()

    # ファイル出力ボタン
    save_button = ft.ElevatedButton(
        text="結果をCSV保存",
        on_click=lambda _: save_results_to_csv(),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.GREEN,
        ),
    )

    # HTML出力ボタン
    save_html_button = ft.ElevatedButton(
        text="結果をHTML保存",
        on_click=lambda _: save_results_to_html(),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.ORANGE,
        ),
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
        nonlocal current_results_df

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

            # グローバル変数に結果を保存
            current_results_df = results_df.copy()

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
                ft.Text(f"分析結果数: {len(results_df)}件"),
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
                        ft.Container(
                            content=ft.Column(
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
                                    ft.Divider(),
                                    ft.Text("結果の保存設定：", size=16),
                                    save_directory,
                                    filename_input,
                                    save_button,
                                    save_html_button,
                                ],
                                scroll=ft.ScrollMode.AUTO,
                            ),
                            width=400,  # 左カラムの幅を固定
                            height=600,  # 高さを制限
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

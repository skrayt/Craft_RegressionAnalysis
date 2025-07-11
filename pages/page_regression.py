# pages/page_regression.py
# pages/page_regression.py

"""
多変量回帰分析ページの定義モジュール。

このモジュールでは、SQLite から読み込んだデータを元に、
仮の回帰モデル結果と残差プロットを表示する Flet UI を構築する。
"""

import io
import base64
import sqlite3
import flet as ft
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from components.plot_utils import (
    regression_summary_table,
    regression_stats_table,
    regression_diagnostics_table,
    vif_table,
)
from db.database import read_dataframe_from_sqlite
from components.variable_selector import VariableSelector
from utils.data_transformation import get_dataframe_for_pattern


def regression_calc(target: str, features: list[str], X: pd.DataFrame, y: pd.Series):
    """
    回帰分析のモデル、モデルのサマリ、平均二乗誤差（MSE）の平均値、平均二乗誤差（MSE）を返す

    Parameters:
    -----------
    target : str
        目的変数のカラム名
    features : list[str]
        説明変数のカラム名のリスト
    X : pd.DataFrame
        説明変数のデータフレーム
    y : pd.Series
        目的変数のデータ
    """
    # 定数項を追加（Statsmodels用）
    X_with_const = sm.add_constant(X)

    # Statsmodelsで線形回帰モデルを構築
    model = sm.OLS(y, X_with_const).fit()

    # Scikit-learnで交差検証を実施
    lr = LinearRegression()
    cross_val_scores = cross_val_score(lr, X, y, cv=5, scoring="neg_mean_squared_error")

    # 平均二乗誤差（MSE）の平均値
    mean_cross_val_scores = -cross_val_scores.mean()

    # モデルの予測値を計算
    y_pred = model.predict(X_with_const)

    # 平均二乗誤差（MSE）を計算
    mse = mean_squared_error(y, y_pred)

    # VIF値を計算
    vifData = pd.DataFrame()
    vifData["Feature"] = ["const"] + features  # 定数項と説明変数を対応付け
    vifData["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]

    return model, mean_cross_val_scores, mse, vifData


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    残差プロットを作成し、base64形式でエンコードされたPNG画像を返す。
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("予測値")
    plt.ylabel("残差")
    plt.title("残差 vs. 予測値")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def regression_page(page: ft.Page) -> ft.Container:
    """
    Flet UI ページを構築し、ユーザーによる目的変数と説明変数の選択に基づいて回帰モデル結果と残差プロットを表示する。
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
    result_col = ft.Column(scroll=ft.ScrollMode.ALWAYS, expand=1)
    residual_image = ft.Image()
    status_text = ft.Text("", color=ft.Colors.GREEN_700)

    def run_analysis():
        """回帰分析を実行し、結果を表示する"""
        target, features = variable_selector.get_selected_variables()
        if not target or not features:
            status_text.value = "目的変数と説明変数を選択してください。"
            status_text.color = ft.Colors.RED_700
            page.update()
            return

        try:
            # 変数の設定を取得
            settings = variable_selector.get_variable_settings()
            print(f"DEBUG: 変数設定: {settings}")

            # 目的変数のデータを取得
            target_df = get_dataframe_for_pattern(
                initial_df,
                settings[target]["transformation"],
                settings[target]["standardization"],
            )
            print(f"DEBUG: 目的変数データフレームの形状: {target_df.shape}")
            print(
                f"DEBUG: 目的変数データフレームのカラム: {target_df.columns.tolist()}"
            )

            # 説明変数のデータを取得
            feature_dfs = []
            for feature in features:
                feature_df = get_dataframe_for_pattern(
                    initial_df,
                    settings[feature]["transformation"],
                    settings[feature]["standardization"],
                )
                print(
                    f"DEBUG: 説明変数 {feature} のデータフレームの形状: {feature_df.shape}"
                )
                print(
                    f"DEBUG: 説明変数 {feature} のデータフレームのカラム: {feature_df.columns.tolist()}"
                )
                feature_dfs.append(feature_df[[feature]])

            # データを結合
            X = pd.concat(feature_dfs, axis=1)
            y = target_df[target]  # カラム名を指定して取得
            print(f"DEBUG: 結合後のXの形状: {X.shape}")
            print(f"DEBUG: 結合後のXのカラム: {X.columns.tolist()}")
            print(f"DEBUG: yの形状: {y.shape}")
            print(f"DEBUG: yの名前: {y.name}")

            # 回帰分析を実行
            model, mean_cross_val_scores, mse, vifData = regression_calc(
                target, features, X, y
            )

            # 結果を表示
            result_col.controls = [
                ft.Text("【結果】：", size=20),
                ft.Text(
                    value=f"目的変数: {target}",
                ),
                ft.Text(
                    value=f"説明変数: {', '.join(features)}",
                ),
                ft.Text(
                    value=f"交差検証の平均MSE: {mean_cross_val_scores:.2f}",
                    selectable=True,
                    style="monospace",
                    size=14,
                ),
                ft.Text(
                    value=f"モデルのMSE: {mse:.2f}",
                    selectable=True,
                    style="monospace",
                    size=14,
                ),
                ft.Text("VIF値:", size=16),
                vif_table(vifData),
                ft.Row(
                    [
                        regression_summary_table(model),
                        regression_diagnostics_table(model),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                regression_stats_table(model),
            ]

            # 残差プロットを更新
            y_pred = model.predict(sm.add_constant(X))
            residual_image.src_base64 = plot_residuals(y.values, y_pred)

            status_text.value = "分析が完了しました。"
            status_text.color = ft.Colors.GREEN_700
            page.update()

        except Exception as e:
            print(f"DEBUG: 分析実行中にエラーが発生: {str(e)}")
            print(f"DEBUG: エラーの種類: {type(e)}")
            import traceback

            print(f"DEBUG: エラーの詳細:\n{traceback.format_exc()}")
            status_text.value = f"分析実行中にエラーが発生しました: {str(e)}"
            status_text.color = ft.Colors.RED_700
            page.update()

    # 分析実行ボタン
    analyze_button = ft.ElevatedButton(
        text="分析実行",
        on_click=lambda _: run_analysis(),
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

    return ft.Container(
        padding=20,
        expand=True,
        content=ft.Row(
            [
                ft.Column(
                    [
                        ft.Text(
                            "多変量回帰モデル作成",
                            size=20,
                            weight=ft.FontWeight.BOLD,
                        ),
                        target_row,
                        ft.Text("説明変数を選択：", size=16),
                        feature_container,
                        analyze_button,
                        status_text,
                    ],
                    expand=True,
                ),
                ft.Column(
                    [
                        result_col,
                        ft.Text("【残差プロット】：", size=20),
                        residual_image,
                    ],
                    expand=1,
                ),
            ],
            expand=True,
        ),
    )

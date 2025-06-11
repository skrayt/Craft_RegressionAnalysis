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


def regression_calc(target: str, features: list[str], df: pd.DataFrame):
    """
    回帰分析のモデル、モデルのサマリ、平均二乗誤差（MSE）の平均値、平均二乗誤差（MSE）を返す
    """

    # 説明変数と目的変数を設定
    X_selected = df[features]
    y_target = df[target]

    # 定数項を追加（Statsmodels用）
    X_selected_with_const = sm.add_constant(X_selected)

    # Statsmodelsで線形回帰モデルを構築
    model = sm.OLS(y_target, X_selected_with_const).fit()

    # Scikit-learnで交差検証を実施
    lr = LinearRegression()
    cross_val_scores = cross_val_score(
        lr, X_selected, y_target, cv=5, scoring="neg_mean_squared_error"
    )

    # 平均二乗誤差（MSE）の平均値
    mean_cross_val_scores = -cross_val_scores.mean()

    # モデルの予測値を計算
    y_pred = model.predict(X_selected_with_const)

    # 平均二乗誤差（MSE）を計算
    mse = mean_squared_error(y_target, y_pred)

    # VIF値を計算
    vifData = pd.DataFrame()
    vifData["Feature"] = ["const"] + features  # 定数項と説明変数を対応付け
    vifData["VIF"] = [
        variance_inflation_factor(X_selected_with_const.values, i)
        for i in range(X_selected_with_const.shape[1])
    ]

    return model, mean_cross_val_scores, mse, vifData


def plot_residuals(_features: list[str]) -> str:
    """
    ダミーの残差プロットを作成し、base64形式でエンコードされたPNG画像を返す。
    """
    # 残差 vs. 予測値の散布図（ダミー）
    y_true = np.random.normal(size=100)
    y_pred = y_true + np.random.normal(scale=0.5, size=100)
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
    df = read_dataframe_from_sqlite("standardized_data")
    all_columns = df.columns.tolist()

    target_dropdown = ft.Dropdown(
        label="目的変数",
        options=[ft.dropdown.Option(col) for col in all_columns],
        on_change=lambda e: refresh_feature_checkboxes(),
    )

    feature_checkboxes = ft.Column(expand=True)
    result_col = ft.Column(scroll=ft.ScrollMode.ALWAYS, expand=1)
    residual_image = ft.Image()

    def refresh_feature_checkboxes():
        feature_checkboxes.controls.clear()
        target = target_dropdown.value
        for col in all_columns:
            if col != target:
                feature_checkboxes.controls.append(ft.Checkbox(label=col))
        page.update()

    def run_regression(_e):
        target = target_dropdown.value
        features = [
            str(cb.label)
            for cb in feature_checkboxes.controls
            if isinstance(cb, ft.Checkbox) and cb.value
        ]
        lines = [
            f"目的変数: {target}",
            f"説明変数: {', '.join(features)}",
        ]
        if not target or not features:  # targetまたはfeaturesが選択されていなければここで終わり
            return

        model, mean_cross_val_scores, mse, vifData = regression_calc(
            target, features, df
        )

        result_col.controls = [
            ft.Text("【結果】：", size=20),
            ft.Text(
                value="\n".join(lines),
            ),
            ft.Text(
                value=f"交差検証の平均MSE: {mean_cross_val_scores:.2f}",
                selectable=True,
                style="monospace",
                size=14,
            ),
            ft.Text(
                value=f"モデルのMSE: {mse:.2f}", selectable=True, style="monospace", size=14
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
        residual_image.src_base64 = plot_residuals(features)
        page.update()

    return ft.Container(
        padding=20,
        expand=True,
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("多変量回帰モデル作成"),
                            target_dropdown,
                            ft.Text("説明変数を選択："),
                            feature_checkboxes,
                            ft.ElevatedButton("モデル作成", on_click=run_regression),
                        ],
                    )
                ),
                result_col,
                ft.Column(
                    [
                        ft.Text("【残差プロット】：", size=20),
                        residual_image,
                    ],
                    expand=1,
                ),
            ]
        ),
        border=ft.border.all(2, "red"),
    )

"""
ダイナミック回帰分析ページの定義モジュール。

このモジュールでは、SQLiteから読み込んだデータを元に、
ラグ付き回帰モデル（ダイナミック回帰モデル）の分析を実行し、
結果と残差プロットを表示するFlet UIを構築する。
"""

import io
import base64
import flet as ft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def create_lagged_features(
    df: pd.DataFrame, features: list[str], lag_settings: dict[str, int]
) -> pd.DataFrame:
    """
    指定された特徴量に対してラグ変数を作成する

    Parameters:
    -----------
    df : pd.DataFrame
        元のデータフレーム
    features : list[str]
        ラグ変数を作成する特徴量のリスト
    lag_settings : dict[str, int]
        各特徴量のラグ次数設定

    Returns:
    --------
    pd.DataFrame
        ラグ変数を含むデータフレーム
    """
    df_lagged = df.copy()

    # 各特徴量に対してラグ変数を作成
    for feature in features:
        lag = lag_settings[feature]
        if lag > 0:
            for i in range(1, lag + 1):
                df_lagged[f"{feature}_lag{i}"] = df[feature].shift(i)

    return df_lagged


def dynamic_regression_calc(
    target: str,
    features: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    lag_settings: dict[str, int],
) -> tuple:
    """
    ダイナミック回帰分析を実行し、モデルと各種統計量を返す

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
    lag_settings : dict[str, int]
        各説明変数のラグ次数設定

    Returns:
    --------
    tuple
        (モデル, 平均交差検証MSE, モデルMSE, VIF値)
    """
    # ラグ変数の作成
    X_lagged = create_lagged_features(X, features, lag_settings)

    # 欠損値の処理
    X_lagged = X_lagged.dropna()
    y_aligned = y[X_lagged.index]

    # 定数項を追加
    X_with_const = sm.add_constant(X_lagged)

    # Statsmodelsで線形回帰モデルを構築
    model = sm.OLS(y_aligned, X_with_const).fit()

    # Scikit-learnで交差検証を実施
    lr = LinearRegression()
    cross_val_scores = cross_val_score(
        lr, X_lagged, y_aligned, cv=5, scoring="neg_mean_squared_error"
    )

    # 平均二乗誤差（MSE）の平均値
    mean_cross_val_scores = -cross_val_scores.mean()

    # モデルの予測値を計算
    y_pred = model.predict(X_with_const)

    # 平均二乗誤差（MSE）を計算
    mse = mean_squared_error(y_aligned, y_pred)

    # VIF値を計算
    vifData = pd.DataFrame()
    vifData["Feature"] = ["const"] + X_lagged.columns.tolist()
    vifData["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]

    return model, mean_cross_val_scores, mse, vifData


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    残差プロットを作成し、base64形式でエンコードされたPNG画像を返す
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


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series) -> str:
    """
    予測値と実測値の比較プロットを作成し、base64形式でエンコードされたPNG画像を返す
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_true, label="実測値", marker="o")
    plt.plot(dates, y_pred, label="予測値", marker="x", linestyle="--")
    plt.xlabel("日付")
    plt.ylabel("値")
    plt.title("予測値 vs. 実測値")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LagSelector:
    """
    ラグ次数を設定するためのコンポーネント
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.lag_settings = {}
        self.lag_controls = {}

    def update_features(self, features: list[str]):
        """
        特徴量のリストを更新し、UIを再構築する
        """
        self.lag_settings = {feature: 0 for feature in features}
        self.lag_controls = {}

    def on_lag_change(self, feature: str, value: int):
        """
        ラグ次数が変更されたときのコールバック
        """
        self.lag_settings[feature] = int(value)
        # ラグ値の表示を更新
        if feature in self.lag_controls:
            self.lag_controls[feature].value = str(int(value))
            self.page.update()

    def get_ui_components(self) -> ft.Container:
        """
        ラグ設定用のUIコンポーネントを返す
        """
        controls = []
        for feature in self.lag_settings.keys():
            lag_value_text = ft.Text("0", size=14)
            self.lag_controls[feature] = lag_value_text

            row = ft.Row(
                [
                    ft.Text(feature, size=14, width=150),
                    ft.Slider(
                        min=0,
                        max=12,  # 最大12期のラグ
                        value=0,
                        width=200,
                        on_change=lambda e, f=feature: self.on_lag_change(
                            f, e.control.value
                        ),
                    ),
                    lag_value_text,
                ],
                alignment=ft.MainAxisAlignment.START,
            )
            controls.append(row)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("ラグ次数の設定：", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "各説明変数に対して、何期前までのデータを使用するか設定してください。",
                        size=12,
                    ),
                    *controls,
                ],
                spacing=10,
            ),
            padding=10,
            border=ft.border.all(1, ft.Colors.GREY_400),
            border_radius=10,
        )


def dynamic_regression_page(page: ft.Page) -> ft.Container:
    """
    ダイナミック回帰分析ページのUIを構築
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

    # プロット画像用の変数
    residual_image = ft.Image()
    prediction_image = ft.Image()

    # 結果表示用のコンテナ
    result_col = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=10,
        height=page.height - 100,
    )

    # 結果表示用のスクロール可能なコンテナ
    result_scroll_container = ft.Container(
        content=ft.Row(
            [result_col],
            scroll=ft.ScrollMode.AUTO,
            wrap=False,  # 横スクロールを有効にするために必要
        ),
        expand=True,
        border=ft.border.all(1, ft.Colors.GREY_400),
        border_radius=10,
        padding=10,
        width=page.width * 0.8,  # コンテナの幅を明示的に設定
    )

    # プロット表示用のコンテナ
    plot_container = ft.Container(
        content=ft.Column(
            [
                ft.Text("【残差プロット】：", size=20),
                residual_image,
                ft.Text("【予測値vs実測値】：", size=20),
                prediction_image,
            ],
            scroll=ft.ScrollMode.ALWAYS,
            spacing=10,
        ),
        expand=True,
    )

    status_text = ft.Text("", color=ft.Colors.GREEN_700)

    # ラグ選択コンポーネントの初期化
    lag_selector = LagSelector(page)

    def run_analysis():
        """ダイナミック回帰分析を実行し、結果を表示する"""
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

            # 説明変数のデータを取得
            feature_dfs = []
            for feature in features:
                feature_df = get_dataframe_for_pattern(
                    initial_df,
                    settings[feature]["transformation"],
                    settings[feature]["standardization"],
                )
                feature_dfs.append(feature_df[[feature]])

            # データを結合
            X = pd.concat(feature_dfs, axis=1)
            y = target_df[target]

            # ラグ設定を更新
            lag_selector.update_features(features)

            # ダイナミック回帰分析を実行
            model, mean_cross_val_scores, mse, vifData = dynamic_regression_calc(
                target, features, X, y, lag_selector.lag_settings
            )

            # 結果を表示
            result_col.controls = [
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "【分析結果】：", size=20, weight=ft.FontWeight.BOLD
                            ),
                            ft.Text(f"目的変数: {target}"),
                            ft.Text(f"説明変数: {', '.join(features)}"),
                            ft.Text("ラグ設定:"),
                            *[
                                ft.Text(f"  {feature}: {lag}期", size=14)
                                for feature, lag in lag_selector.lag_settings.items()
                            ],
                            ft.Text(
                                f"交差検証の平均MSE: {mean_cross_val_scores:.2f}",
                                selectable=True,
                                style="monospace",
                                size=14,
                            ),
                            ft.Text(
                                f"モデルのMSE: {mse:.2f}",
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
                                scroll=ft.ScrollMode.AUTO,
                                wrap=False,  # 横スクロールを有効にするために必要
                            ),
                            regression_stats_table(model),
                        ],
                        spacing=10,
                    ),
                    padding=10,
                    width=page.width * 0.8,  # コンテナの幅を明示的に設定
                )
            ]

            # 残差プロットを更新
            y_pred = model.predict(sm.add_constant(X))
            residual_image.src_base64 = plot_residuals(y.values, y_pred)

            # 予測値vs実測値プロットを更新
            dates = initial_df["kijyunnengetu"][y.index]
            prediction_image.src_base64 = plot_predictions(y.values, y_pred, dates)

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
                # 左側：変数選択と設定
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "ダイナミック回帰モデル作成",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                            ),
                            target_row,
                            ft.Text("説明変数を選択：", size=16),
                            feature_container,
                            lag_selector.get_ui_components(),
                            analyze_button,
                            status_text,
                        ],
                        spacing=10,
                    ),
                    expand=True,
                    padding=10,
                ),
                # 右側：分析結果
                ft.Container(
                    content=ft.Column(
                        [
                            result_scroll_container,  # スクロール可能な結果コンテナを使用
                            plot_container,
                        ],
                        expand=True,
                        spacing=10,
                    ),
                    expand=1,
                ),
            ],
            expand=True,
            spacing=20,
        ),
    )

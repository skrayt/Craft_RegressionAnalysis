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
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime
from pathlib import Path

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
    lag_settings: dict,
    include_lag: bool = True,  # ラグ使用の有無を追加
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
    lag_settings : dict
        各説明変数のラグ次数設定
    include_lag : bool
        ラグ変数を使用するかどうか

    Returns:
    --------
    tuple
        (モデル, 平均交差検証MSE, モデルMSE, VIF値)
    """
    if include_lag:
        # ラグ変数を作成
        X_with_lags = create_lagged_features(X, features, lag_settings)
    else:
        # ラグを使用しない場合、元のデータをそのまま使用
        X_with_lags = X.copy()

    # 定数項を追加
    X_with_const = sm.add_constant(X_with_lags)

    # Statsmodelsで線形回帰モデルを構築
    model = sm.OLS(y, X_with_const).fit()

    # Scikit-learnで交差検証を実施
    lr = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_scores = cross_val_score(
        lr, X_with_lags, y, cv=tscv, scoring="neg_mean_squared_error"
    )

    # 平均二乗誤差（MSE）の平均値
    mean_cross_val_scores = -cross_val_scores.mean()

    # モデルの予測値を計算
    y_pred = model.predict(X_with_const)

    # 平均二乗誤差（MSE）を計算
    mse = mean_squared_error(y, y_pred)

    # VIF値を計算
    vifData = pd.DataFrame()
    vifData["Feature"] = ["const"] + X_with_lags.columns.tolist()
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

    plt.figure(figsize=(10, 6))

    # 残差の散布図
    plt.scatter(y_pred, residuals, alpha=0.5, label="残差")

    # ゼロライン
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.7, label="ゼロライン")

    # グラフの装飾
    plt.xlabel("予測値")
    plt.ylabel("残差")
    plt.title("残差 vs. 予測値")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # レイアウトの調整
    plt.tight_layout()

    # プロットを画像に変換
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series) -> str:
    """
    予測値と実測値の比較プロットを作成し、base64形式でエンコードされたPNG画像を返す
    """
    plt.figure(figsize=(10, 6))

    # 実測値と予測値をプロット
    plt.plot(dates, y_true, label="実測値", marker="o", markersize=4, alpha=0.7)
    plt.plot(
        dates,
        y_pred,
        label="予測値",
        marker="x",
        markersize=4,
        linestyle="--",
        alpha=0.7,
    )

    # グラフの装飾
    plt.xlabel("日付")
    plt.ylabel("値")
    plt.title("予測値 vs. 実測値")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # レイアウトの調整
    plt.tight_layout()

    # プロットを画像に変換
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_acf(residuals: np.ndarray, max_lag: int = 24) -> str:
    """
    残差の自己相関関数（ACF）プロットを作成し、base64形式でエンコードされたPNG画像を返す

    Parameters:
    -----------
    residuals : np.ndarray
        残差の配列
    max_lag : int
        最大ラグ次数（デフォルト: 24）

    Returns:
    --------
    str
        base64エンコードされたプロット画像
    """
    # 自己相関関数を計算
    acf = sm.tsa.stattools.acf(residuals, nlags=max_lag)

    # 信頼区間の計算（95%信頼区間）
    n = len(residuals)
    ci = 1.96 / np.sqrt(n)  # 95%信頼区間の境界値

    plt.figure(figsize=(10, 6))

    # ACFプロット（use_line_collectionパラメータを削除）
    plt.stem(range(len(acf)), acf)

    # 信頼区間の境界線
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    plt.axhline(y=ci, color="r", linestyle="--", alpha=0.7, label="95%信頼区間")
    plt.axhline(y=-ci, color="r", linestyle="--", alpha=0.7)

    # グラフの装飾
    plt.xlabel("ラグ")
    plt.ylabel("自己相関係数")
    plt.title("残差の自己相関関数（ACF）")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # レイアウトの調整
    plt.tight_layout()

    # プロットを画像に変換
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LagSelector(ft.Container):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.lag_settings = {}  # 各変数のラグ設定を保持
        self.include_lag = False  # デフォルトはFalse
        self.max_lag = 12  # 最大ラグ次数のデフォルト値
        self.optimization_method = "aic"  # デフォルトの最適化方法

        # ラグ設定のUIコンポーネント
        self.include_lag_switch = ft.Switch(
            label="ラグ変数を含める", value=False, on_change=self._on_include_lag_change
        )

        # 最適化方法の選択
        self.optimization_dropdown = ft.Dropdown(
            label="最適化方法",
            options=[
                ft.dropdown.Option("aic", "AIC"),
                ft.dropdown.Option("bic", "BIC"),
                ft.dropdown.Option("cv", "交差検証"),
            ],
            value="aic",
            width=200,
            on_change=self._on_optimization_change,
        )

        # 最適化実行ボタン
        self.optimize_button = ft.ElevatedButton(
            text="ラグ次数を最適化", on_click=self._optimize_lag_order, width=200
        )

        # 最大ラグ次数の設定
        self.max_lag_slider = ft.Slider(
            min=1,
            max=24,
            divisions=23,  # 整数メモリを設定
            value=12,
            label="{value}次",  # スライダーの値ラベル
            width=200,
            on_change=self._on_max_lag_change,
        )
        self.max_lag_text = ft.Text(
            "最大ラグ次数: 12次", size=14
        )  # 現在の値を表示するテキスト

        # ラグ設定の詳細セクション
        self.lag_settings_column = ft.Column(
            controls=[
                ft.Row([self.optimization_dropdown, self.optimize_button]),
                ft.Text("最大ラグ次数の設定："),
                ft.Row(
                    [self.max_lag_slider, self.max_lag_text],
                    alignment=ft.MainAxisAlignment.START,
                ),
                ft.Divider(),
                ft.Text("各変数のラグ次数設定：", size=16),
            ],
            visible=False,  # デフォルトは非表示
        )

        # メインのコンテンツ
        self.content = ft.Column([self.include_lag_switch, self.lag_settings_column])

    def _on_include_lag_change(self, e):
        """ラグ変数を含めるトグルの変更時の処理"""
        self.include_lag = e.control.value
        self.lag_settings_column.visible = self.include_lag
        self.page.update()

    def _on_optimization_change(self, e):
        """最適化方法の変更時の処理"""
        self.optimization_method = e.control.value

    def _on_max_lag_change(self, e):
        """最大ラグ次数の変更時の処理"""
        self.max_lag = int(e.control.value)
        self.max_lag_text.value = f"最大ラグ次数: {self.max_lag}次"
        self._update_lag_ui()
        self.page.update()

    def _optimize_lag_order(self, e):
        """ラグ次数の最適化を実行"""
        try:
            if not self.lag_settings:
                return

            # データを取得
            df = read_dataframe_from_sqlite("merged_data")
            if df is None or df.empty:
                raise ValueError("データが読み込まれていません。")

            # 各変数について最適なラグ次数を計算
            for feature in self.lag_settings.keys():
                data = df[feature]
                optimal_lag = find_optimal_lag_order(
                    data=data,
                    max_lag=self.max_lag,
                    criterion=self.optimization_method,
                    min_lag=1,
                )
                self.lag_settings[feature]["optimal_lag"] = optimal_lag
                self.lag_settings[feature]["lag"] = optimal_lag  # スライダーの値も更新

            # UIを更新
            self._update_lag_ui()

            # 成功メッセージを表示
            method_name = {"aic": "AIC", "bic": "BIC", "cv": "交差検証"}[
                self.optimization_method
            ]
            snack = ft.SnackBar(
                content=ft.Text(f"{method_name}による最適なラグ次数を計算しました。")
            )
            self.page.snack_bar = snack
            snack.open = True
            self.page.update()

        except Exception as e:
            print(f"DEBUG: 最適ラグ計算中にエラーが発生: {str(e)}")
            snack = ft.SnackBar(
                content=ft.Text(f"最適ラグ計算中にエラーが発生しました: {str(e)}")
            )
            self.page.snack_bar = snack
            snack.open = True
            self.page.update()

    def _update_lag_ui(self):
        """ラグ設定のUIを更新"""
        # ラグ設定の詳細部分をクリア
        while len(self.lag_settings_column.controls) > 4:  # 基本要素を除く
            self.lag_settings_column.controls.pop()

        # ラグ設定が有効な場合のみ詳細を表示
        if self.include_lag and self.lag_settings:
            # 各変数のラグ設定を表示
            for feature in self.lag_settings.keys():
                # 現在のラグ次数を表示するテキスト
                lag_text = ft.Text(
                    f"ラグ次数: {self.lag_settings[feature]['lag']}次",
                    size=14,
                )

                # 最適ラグ次数の表示（計算済みの場合）
                optimal_lag = self.lag_settings[feature].get("optimal_lag")
                method_name = {"aic": "AIC", "bic": "BIC", "cv": "交差検証"}[
                    self.optimization_method
                ]
                optimal_text = ft.Text(
                    (
                        f"（{method_name}推奨: {optimal_lag}次）"
                        if optimal_lag is not None
                        else ""
                    ),
                    size=14,
                    color=ft.Colors.BLUE,
                )

                # ラグ次数のスライダー
                lag_slider = ft.Slider(
                    min=0,
                    max=self.max_lag,
                    divisions=self.max_lag,  # 整数メモリを設定
                    value=self.lag_settings[feature]["lag"],
                    label="{value}次",  # スライダーの値ラベル
                    width=200,
                    on_change=lambda e, f=feature, t=lag_text: self._handle_lag_change(
                        e, f, t
                    ),
                )

                # 変数ごとの設定行
                feature_row = ft.Row(
                    [
                        ft.Text(feature, size=14, width=100),
                        ft.Column(
                            [
                                ft.Row(
                                    [lag_slider, lag_text, optimal_text],
                                    alignment=ft.MainAxisAlignment.START,
                                ),
                            ],
                            spacing=5,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                )
                self.lag_settings_column.controls.append(feature_row)

        self.page.update()

    def update_lag_settings(self, features: list[str]):
        """ラグ設定を更新"""
        print(f"DEBUG: ラグ設定を更新中... 選択された特徴量: {features}")
        new_settings = {}
        for feature in features:
            # 既存の設定がある場合はそれを保持、ない場合は新しい設定を作成
            if feature in self.lag_settings:
                new_settings[feature] = self.lag_settings[feature]
            else:
                new_settings[feature] = {
                    "lag": 0,
                    "optimal_lag": None,  # 最適ラグ次数の初期値
                }
        self.lag_settings = new_settings
        self._update_lag_ui()

    def _handle_lag_change(self, e, feature: str, text_control: ft.Text):
        """ラグ次数が変更されたときの処理"""
        try:
            value = int(e.control.value)
            if 0 <= value <= self.max_lag:
                self.lag_settings[feature]["lag"] = value
                text_control.value = f"ラグ次数: {value}次"
            else:
                e.control.value = 0
                self.lag_settings[feature]["lag"] = 0
                text_control.value = "ラグ次数: 0次"
        except ValueError:
            e.control.value = 0
            self.lag_settings[feature]["lag"] = 0
            text_control.value = "ラグ次数: 0次"
        self.page.update()

    def get_lag_settings(self) -> dict:
        """現在のラグ設定を返す"""
        return self.lag_settings.copy()


def find_optimal_lag_order(
    data: pd.Series, max_lag: int, criterion: str = "aic", min_lag: int = 1
) -> int:
    """
    最適なラグ次数を計算する

    Parameters:
    -----------
    data : pd.Series
        時系列データ
    max_lag : int
        最大ラグ次数
    criterion : str
        選択基準 ("aic", "bic", "cv")
    min_lag : int
        最小ラグ次数

    Returns:
    --------
    int
        最適なラグ次数
    """
    print(f"DEBUG: find_optimal_lag_orderが呼び出されました")
    print(f"DEBUG: データの形状: {data.shape}")
    print(f"DEBUG: 選択基準: {criterion}")

    if criterion == "cv":
        # 交差検証による選択
        print("DEBUG: 交差検証による選択を実行")
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = float("inf")
        best_lag = min_lag

        for lag in range(min_lag, max_lag + 1):
            try:
                # ラグ付きデータの作成
                lagged_data = pd.DataFrame(
                    {"y": data[lag:], "x": data[:-lag] if lag > 0 else data}
                ).dropna()

                if len(lagged_data) < 10:  # データが少なすぎる場合はスキップ
                    continue

                X = lagged_data[["x"]]
                y = lagged_data["y"]

                # 交差検証
                model = LinearRegression()
                scores = cross_val_score(
                    model, X, y, cv=tscv, scoring="neg_mean_squared_error"
                )
                score = -scores.mean()

                print(f"DEBUG: ラグ次数 {lag} のスコア: {score}")

                if score < best_score:
                    best_score = score
                    best_lag = lag

            except Exception as e:
                print(f"DEBUG: ラグ次数 {lag} の計算中にエラー: {str(e)}")
                continue

        print(f"DEBUG: 最適なラグ次数: {best_lag} (スコア: {best_score})")
        return best_lag

    else:
        # AIC/BICによる選択
        print("DEBUG: AIC/BICによる選択を実行")
        best_score = float("inf")
        best_lag = min_lag

        for lag in range(min_lag, max_lag + 1):
            try:
                # ラグ付きデータの作成
                lagged_data = pd.DataFrame(
                    {"y": data[lag:], "x": data[:-lag] if lag > 0 else data}
                ).dropna()

                if len(lagged_data) < 10:  # データが少なすぎる場合はスキップ
                    continue

                X = sm.add_constant(lagged_data[["x"]])
                y = lagged_data["y"]

                # モデルの構築
                model = sm.OLS(y, X).fit()

                # スコアの計算
                if criterion == "aic":
                    score = model.aic
                else:  # bic
                    score = model.bic

                print(f"DEBUG: ラグ次数 {lag} のスコア: {score}")

                if score < best_score:
                    best_score = score
                    best_lag = lag

            except Exception as e:
                print(f"DEBUG: ラグ次数 {lag} の計算中にエラー: {str(e)}")
                continue

        print(f"DEBUG: 最適なラグ次数: {best_lag} (スコア: {best_score})")
        return best_lag


def plot_lag_selection_results(
    data: pd.Series, max_lag: int, criterion: str = "aic", min_lag: int = 1
) -> str:
    """
    ラグ選択の結果をプロットする

    Parameters:
    -----------
    data : pd.Series
        時系列データ
    max_lag : int
        最大ラグ次数
    criterion : str
        選択基準 ("aic", "bic", "cv")
    min_lag : int
        最小ラグ次数

    Returns:
    --------
    str
        base64エンコードされたプロット画像
    """
    print(f"DEBUG: plot_lag_selection_resultsが呼び出されました")

    lags = list(range(min_lag, max_lag + 1))
    scores = []

    for lag in lags:
        try:
            # ラグ付きデータの作成
            lagged_data = pd.DataFrame(
                {"y": data[lag:], "x": data[:-lag] if lag > 0 else data}
            ).dropna()

            if len(lagged_data) < 10:
                scores.append(np.nan)
                continue

            X = sm.add_constant(lagged_data[["x"]])
            y = lagged_data["y"]

            if criterion == "cv":
                # 交差検証
                model = LinearRegression()
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, scoring="neg_mean_squared_error"
                )
                score = -cv_scores.mean()
            else:
                # AIC/BIC
                model = sm.OLS(y, X).fit()
                score = model.aic if criterion == "aic" else model.bic

            scores.append(score)
            print(f"DEBUG: ラグ次数 {lag} のスコア: {score}")

        except Exception as e:
            print(f"DEBUG: ラグ次数 {lag} のプロット作成中にエラー: {str(e)}")
            scores.append(np.nan)

    # プロットの作成
    plt.figure(figsize=(10, 6))
    plt.plot(lags, scores, "bo-", label=f"{criterion.upper()}スコア")
    plt.xlabel("ラグ次数")
    plt.ylabel(f"{criterion.upper()}スコア")
    plt.title(f"ラグ次数選択結果 ({criterion.upper()})")
    plt.grid(True)
    plt.legend()

    # 最適なラグ次数を強調
    best_lag = lags[np.nanargmin(scores)]
    best_score = np.nanmin(scores)
    plt.plot(
        best_lag, best_score, "ro", markersize=10, label=f"最適ラグ次数: {best_lag}"
    )
    plt.legend()

    # プロットを画像に変換
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    print("DEBUG: プロットの作成が完了しました")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def dynamic_regression_page(page: ft.Page) -> ft.Container:
    """
    ダイナミック回帰分析ページを構築する関数
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
    result_column = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        expand=1,
        spacing=10,
    )
    detail_column = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        expand=1,
        spacing=10,
    )

    # 分析結果を保持する変数
    analysis_results = {
        "target": None,
        "features": None,
        "model": None,
        "X": None,
        "y": None,
        "y_pred": None,
        "lag_settings": None,
        "include_lag": None,
        "optimization_method": None,
    }

    # 保存ボタンをグローバル変数として保持
    save_button = ft.ElevatedButton(
        text="分析結果を保存",
        on_click=lambda e: save_results(e, analysis_results, page),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.GREEN,
        ),
        disabled=True,  # 初期状態は無効
    )

    def save_results(e, results: dict, page: ft.Page):
        """分析結果をHTMLファイルとして保存"""
        try:
            if results["model"] is None:
                page.show_snack_bar(
                    ft.SnackBar(content=ft.Text("分析が実行されていません。"))
                )
                return

            filepath = save_analysis_results_to_html(
                target=results["target"],
                features=results["features"],
                model=results["model"],
                X=results["X"],
                y=results["y"],
                y_pred=results["y_pred"],
                lag_settings=results["lag_settings"],
                include_lag=results["include_lag"],
                optimization_method=results["optimization_method"],
            )

            # 保存成功メッセージを表示
            snack = ft.SnackBar(content=ft.Text(f"分析結果を保存しました: {filepath}"))
            page.snack_bar = snack
            snack.open = True
            page.update()

        except Exception as e:
            print(f"DEBUG: 結果保存中にエラーが発生: {str(e)}")
            snack = ft.SnackBar(
                content=ft.Text(f"結果の保存中にエラーが発生しました: {str(e)}")
            )
            page.snack_bar = snack
            snack.open = True
            page.update()

    # 変数選択コンポーネントの初期化
    variable_selector = VariableSelector(
        page=page,
        all_columns=all_columns,
        on_variable_change=lambda: on_variable_change(
            page, variable_selector, lag_selector, result_column, detail_column
        ),
    )

    # ラグ設定コンポーネントの初期化
    lag_selector = LagSelector(page)

    # UIコンポーネントを取得
    target_row, feature_container = variable_selector.get_ui_components()

    # 分析実行ボタン
    analyze_button = ft.ElevatedButton(
        text="分析実行",
        on_click=lambda _: run_dynamic_regression(
            page,
            variable_selector,
            lag_selector,
            result_column,
            detail_column,
            analysis_results,
            save_button,
        ),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE,
        ),
    )

    # ボタンを横に並べる
    button_row = ft.Row(
        [analyze_button, save_button],
        alignment=ft.MainAxisAlignment.START,
        spacing=10,
    )

    # 左カラムのコンテンツをスクロール可能なコンテナに配置
    left_column_content = ft.Column(
        [
            ft.Text("ダイナミック回帰分析", size=20, weight=ft.FontWeight.BOLD),
            ft.Container(
                content=target_row,
                padding=10,
                border=ft.border.all(1, ft.Colors.GREY_400),
                border_radius=5,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("説明変数を選択：", size=16),
                        feature_container,
                    ]
                ),
                padding=10,
                border=ft.border.all(1, ft.Colors.GREY_400),
                border_radius=5,
            ),
            ft.Container(
                content=lag_selector,
                padding=10,
                border=ft.border.all(1, ft.Colors.GREY_400),
                border_radius=5,
            ),
            button_row,
        ],
        scroll=ft.ScrollMode.AUTO,
        width=520,
        height=page.height - 40,  # ページの高さからパディングを引く
    )

    # 右カラムのコンテンツ
    right_column_content = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            # ft.Text(
                            #     "【回帰分析結果】：", size=20, weight=ft.FontWeight.BOLD
                            # ),
                            result_column,
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                    expand=True,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            # ft.Text(
                            #     "【詳細分析】：", size=20, weight=ft.FontWeight.BOLD
                            # ),
                            detail_column,
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                    expand=True,
                ),
            ],
            spacing=20,
            expand=True,
        ),
        expand=True,
    )

    return ft.Container(
        padding=20,
        expand=True,
        content=ft.Row(
            [
                left_column_content,
                ft.VerticalDivider(width=1),
                right_column_content,
            ],
            expand=True,
        ),
    )


def on_variable_change(
    page: ft.Page,
    variable_selector: VariableSelector,
    lag_selector: LagSelector,
    result_column: ft.Column,
    detail_column: ft.Column,
):
    """変数選択が変更されたときの処理"""
    # 結果表示をクリア
    result_column.controls.clear()
    detail_column.controls.clear()

    # ラグ設定を更新
    target, features = variable_selector.get_selected_variables()
    if features:
        lag_selector.update_lag_settings(features)

    page.update()


def save_analysis_results_to_html(
    target: str,
    features: list[str],
    model: sm.OLS,
    X: pd.DataFrame,
    y: pd.Series,
    y_pred: np.ndarray,
    lag_settings: dict,
    include_lag: bool,
    optimization_method: str,
) -> str:
    """
    分析結果をHTMLファイルとして保存する

    Parameters:
    -----------
    target : str
        目的変数名
    features : list[str]
        説明変数名のリスト
    model : sm.OLS
        回帰モデル
    X : pd.DataFrame
        説明変数のデータ
    y : pd.Series
        目的変数のデータ
    y_pred : np.ndarray
        予測値
    lag_settings : dict
        ラグ設定
    include_lag : bool
        ラグ変数を含めるかどうか
    optimization_method : str
        最適化方法

    Returns:
    --------
    str
        保存したファイルのパス
    """
    # 現在の日時を取得
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 出力ディレクトリの作成
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    # ファイル名の生成
    filename = f"dynamic_regression_analysis_{timestamp}.html"
    filepath = output_dir / filename

    # 残差の計算
    residuals = y.values - y_pred

    # HTMLテンプレート
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>ダイナミック回帰分析結果</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>ダイナミック回帰分析結果</h1>
        <p>分析日時: {now.strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="section">
            <h2>分析設定</h2>
            <table>
                <tr><th>目的変数</th><td>{target}</td></tr>
                <tr><th>説明変数</th><td>{', '.join(features)}</td></tr>
                <tr><th>サンプル数</th><td>{len(y)}</td></tr>
                <tr><th>ラグ変数の使用</th><td>{'はい' if include_lag else 'いいえ'}</td></tr>
                <tr><th>最適化方法</th><td>{optimization_method.upper()}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>ラグ設定</h2>
            <table>
                <tr>
                    <th>変数名</th>
                    <th>ラグ次数</th>
                    <th>最適ラグ次数</th>
                </tr>
    """

    # ラグ設定の表を追加
    for feature in features:
        if feature in lag_settings:
            settings = lag_settings[feature]
            optimal_lag = settings.get("optimal_lag", "未計算")
            html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{settings['lag']}</td>
                    <td>{optimal_lag}</td>
                </tr>
            """

    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>回帰分析結果</h2>
            <table>
                <tr><th>決定係数 (R²)</th><td>{:.4f}</td></tr>
                <tr><th>調整済み決定係数</th><td>{:.4f}</td></tr>
                <tr><th>F統計量</th><td>{:.4f}</td></tr>
                <tr><th>F検定のp値</th><td>{:.4f}</td></tr>
            </table>
        </div>
    """.format(
        model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue
    )

    # 係数の表を追加
    html_content += """
        <div class="section">
            <h2>係数</h2>
            <table>
                <tr>
                    <th>変数</th>
                    <th>係数</th>
                    <th>標準誤差</th>
                    <th>t値</th>
                    <th>p値</th>
                </tr>
    """

    for i, var in enumerate(model.model.exog_names):
        html_content += f"""
                <tr>
                    <td>{var}</td>
                    <td>{model.params[i]:.4f}</td>
                    <td>{model.bse[i]:.4f}</td>
                    <td>{model.tvalues[i]:.4f}</td>
                    <td>{model.pvalues[i]:.4f}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    """

    # VIF値の表を追加
    vif_data = calculate_vif(X)
    html_content += """
        <div class="section">
            <h2>VIF値</h2>
            <table>
                <tr>
                    <th>変数</th>
                    <th>VIF</th>
                </tr>
    """

    for _, row in vif_data.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['Feature']}</td>
                    <td>{row['VIF']:.4f}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
    """

    # プロットを追加
    html_content += """
        <div class="section">
            <h2>予測値と実測値の比較</h2>
            <div class="plot">
    """
    html_content += f'<img src="data:image/png;base64,{plot_predictions(y.values, y_pred, y.index)}" alt="予測値と実測値の比較">'
    html_content += """
            </div>
        </div>

        <div class="section">
            <h2>残差プロット</h2>
            <div class="plot">
    """
    html_content += f'<img src="data:image/png;base64,{plot_residuals(y.values, y_pred)}" alt="残差プロット">'
    html_content += """
            </div>
        </div>

        <div class="section">
            <h2>残差の自己相関</h2>
            <div class="plot">
    """
    html_content += (
        f'<img src="data:image/png;base64,{plot_acf(residuals)}" alt="残差の自己相関">'
    )
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    # ファイルに保存
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(filepath)


def run_dynamic_regression(
    page: ft.Page,
    variable_selector: VariableSelector,
    lag_selector: LagSelector,
    result_column: ft.Column,
    detail_column: ft.Column,
    analysis_results: dict,
    save_button: ft.ElevatedButton,
):
    """ダイナミック回帰分析を実行し、結果を表示する"""
    try:
        # 選択された変数を取得
        target, features = variable_selector.get_selected_variables()
        if not target or not features:
            page.show_snack_bar(
                ft.SnackBar(content=ft.Text("目的変数と説明変数を選択してください。"))
            )
            return

        # 変数の設定を取得
        settings = variable_selector.get_variable_settings()
        print(f"DEBUG: 変数設定: {settings}")

        # データの準備
        initial_df = read_dataframe_from_sqlite("merged_data")
        if initial_df is None or initial_df.empty:
            page.show_snack_bar(
                ft.SnackBar(content=ft.Text("データが読み込まれていません。"))
            )
            return

        # 目的変数のデータを取得
        target_df = get_dataframe_for_pattern(
            initial_df,
            settings[target]["transformation"],
            settings[target]["standardization"],
        )
        y = target_df[target]

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

        # ラグ設定を取得
        lag_settings = (
            lag_selector.get_lag_settings() if lag_selector.include_lag else {}
        )
        print(f"DEBUG: ラグ設定: {lag_settings}")

        # ラグ変数を追加
        if lag_settings:
            for feature, settings in lag_settings.items():
                if settings["lag"] > 0:
                    for lag in range(1, settings["lag"] + 1):
                        lag_col = f"{feature}_lag{lag}"
                        X[lag_col] = X[feature].shift(lag)

        # 欠損値を削除
        X = X.dropna()
        y = y[X.index]

        # 定数項を追加
        X_with_const = sm.add_constant(X)

        # モデルの構築と学習
        model = sm.OLS(y, X_with_const).fit()

        # 予測値を計算
        y_pred = model.predict(X_with_const)

        # 分析結果を保存
        analysis_results.update(
            {
                "target": target,
                "features": features,
                "model": model,
                "X": X,
                "y": y,
                "y_pred": y_pred,
                "lag_settings": lag_settings,
                "include_lag": lag_selector.include_lag,
                "optimization_method": lag_selector.optimization_method,
            }
        )

        # 結果の表示
        result_column.controls.clear()
        result_column.controls.extend(
            [
                ft.Text("【回帰分析結果】", size=20, weight=ft.FontWeight.BOLD),
                ft.Text(f"目的変数: {target}"),
                ft.Text(f"説明変数: {', '.join(features)}"),
                ft.Text(f"サンプル数: {len(y)}"),
                ft.Text(f"決定係数 (R²): {model.rsquared:.4f}"),
                ft.Text(f"調整済み決定係数: {model.rsquared_adj:.4f}"),
                ft.Text(f"F統計量: {model.fvalue:.4f}"),
                ft.Text(f"F検定のp値: {model.f_pvalue:.4f}"),
                ft.Divider(),
                ft.Text("【係数】", size=16, weight=ft.FontWeight.BOLD),
                regression_summary_table(model),
                ft.Divider(),
                ft.Text("【診断統計量】", size=16, weight=ft.FontWeight.BOLD),
                regression_diagnostics_table(model),
            ]
        )

        # 詳細分析の表示
        detail_column.controls.clear()
        detail_column.controls.extend(
            [
                ft.Text("【詳細分析】", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "予測値と実測値の比較",
                                size=16,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Image(
                                src_base64=plot_predictions(
                                    y.values, y_pred.values, y.index
                                ),
                                width=600,
                                height=400,
                            ),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("残差プロット", size=16, weight=ft.FontWeight.BOLD),
                            ft.Image(
                                src_base64=plot_residuals(y.values, y_pred.values),
                                width=600,
                                height=400,
                            ),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "残差の自己相関", size=16, weight=ft.FontWeight.BOLD
                            ),
                            ft.Image(
                                src_base64=plot_acf(y.values - y_pred.values),
                                width=600,
                                height=400,
                            ),
                            ft.Text(
                                "※ 点線は95%信頼区間を示します。信頼区間内に収まっている場合、そのラグでの自己相関は統計的に有意ではありません。",
                                size=12,
                                italic=True,
                                color=ft.Colors.GREY_700,
                            ),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("VIF値", size=16, weight=ft.FontWeight.BOLD),
                            vif_table(calculate_vif(X)),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                ),
            ]
        )

        # 保存ボタンを有効化
        save_button.disabled = False

        page.update()

    except Exception as e:
        print(f"DEBUG: 分析実行中にエラーが発生: {str(e)}")
        print(f"DEBUG: エラーの種類: {type(e)}")
        import traceback

        print(f"DEBUG: エラーの詳細:\n{traceback.format_exc()}")
        snack = ft.SnackBar(
            content=ft.Text(f"分析実行中にエラーが発生しました: {str(e)}")
        )
        page.snack_bar = snack
        snack.open = True
        page.update()


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF値を計算する"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data

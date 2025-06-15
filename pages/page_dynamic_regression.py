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
            value=12,
            label="最大ラグ次数: {value}",
            width=200,
            on_change=self._on_max_lag_change,
        )

        # ラグ設定の詳細セクション
        self.lag_settings_column = ft.Column(
            controls=[
                ft.Row([self.optimization_dropdown, self.optimize_button]),
                ft.Text("最大ラグ次数の設定："),
                self.max_lag_slider,
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
        self._update_lag_ui()

    def _optimize_lag_order(self, e):
        """ラグ次数の最適化を実行"""
        if not self.lag_settings:
            return

        for feature in self.lag_settings.keys():
            # 最適化方法に基づいてラグ次数を計算
            if self.optimization_method == "aic":
                optimal_lag = self._calculate_optimal_lag_aic(feature)
            elif self.optimization_method == "bic":
                optimal_lag = self._calculate_optimal_lag_bic(feature)
            else:  # cv
                optimal_lag = self._calculate_optimal_lag_cv(feature)

            self.lag_settings[feature]["lag"] = optimal_lag

        self._update_lag_ui()
        self.page.update()

    def _calculate_optimal_lag_aic(self, feature):
        """AICに基づく最適なラグ次数を計算"""
        # TODO: AICに基づく最適化の実装
        return min(5, self.max_lag)  # 仮の実装

    def _calculate_optimal_lag_bic(self, feature):
        """BICに基づく最適なラグ次数を計算"""
        # TODO: BICに基づく最適化の実装
        return min(4, self.max_lag)  # 仮の実装

    def _calculate_optimal_lag_cv(self, feature):
        """交差検証に基づく最適なラグ次数を計算"""
        # TODO: 交差検証に基づく最適化の実装
        return min(6, self.max_lag)  # 仮の実装

    def _update_lag_ui(self):
        """ラグ設定のUIを更新"""
        # ラグ設定の詳細部分をクリア
        while len(self.lag_settings_column.controls) > 4:  # 基本要素を除く
            self.lag_settings_column.controls.pop()

        # ラグ設定が有効な場合のみ詳細を表示
        if self.include_lag and self.lag_settings:
            # 各変数のラグ設定を表示
            for feature in self.lag_settings.keys():
                feature_row = ft.Row(
                    [
                        ft.Text(feature, size=14),
                        ft.Slider(
                            min=0,
                            max=self.max_lag,
                            value=self.lag_settings[feature]["lag"],
                            on_change=lambda e, f=feature: self._handle_lag_change(
                                e, f
                            ),
                            width=200,
                        ),
                        ft.Text(
                            f"ラグ次数: {self.lag_settings[feature]['lag']}",
                            size=14,
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
                new_settings[feature] = {"lag": 0}  # 初期値として辞書を設定
        self.lag_settings = new_settings
        self._update_lag_ui()

    def _handle_lag_change(self, e, feature: str):
        """ラグ次数が変更されたときの処理"""
        try:
            value = int(e.control.value)
            if 0 <= value <= self.max_lag:
                self.lag_settings[feature]["lag"] = value
            else:
                e.control.value = "0"
                self.lag_settings[feature]["lag"] = 0
        except ValueError:
            e.control.value = "0"
            self.lag_settings[feature]["lag"] = 0
        self._update_lag_ui()

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
            page, variable_selector, lag_selector, result_column, detail_column
        ),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE,
        ),
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
            analyze_button,
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
                            ft.Text(
                                "【回帰分析結果】：", size=20, weight=ft.FontWeight.BOLD
                            ),
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
                            ft.Text(
                                "【詳細分析】：", size=20, weight=ft.FontWeight.BOLD
                            ),
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


def run_dynamic_regression(
    page: ft.Page,
    variable_selector: VariableSelector,
    lag_selector: LagSelector,
    result_column: ft.Column,
    detail_column: ft.Column,
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

        # 結果の表示
        result_column.controls.clear()
        result_column.controls.extend(
            [
                ft.Text("【回帰分析結果】：", size=20, weight=ft.FontWeight.BOLD),
                ft.Text(f"目的変数: {target}"),
                ft.Text(f"説明変数: {', '.join(features)}"),
                ft.Text(f"サンプル数: {len(y)}"),
                ft.Text(f"決定係数 (R²): {model.rsquared:.4f}"),
                ft.Text(f"調整済み決定係数: {model.rsquared_adj:.4f}"),
                ft.Text(f"F統計量: {model.fvalue:.4f}"),
                ft.Text(f"F検定のp値: {model.f_pvalue:.4f}"),
                ft.Divider(),
                ft.Text("【係数】：", size=16, weight=ft.FontWeight.BOLD),
                regression_summary_table(model),
                ft.Divider(),
                ft.Text("【診断統計量】：", size=16, weight=ft.FontWeight.BOLD),
                regression_diagnostics_table(model),
            ]
        )

        # 詳細分析の表示
        detail_column.controls.clear()
        detail_column.controls.extend(
            [
                ft.Text("【詳細分析】：", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("残差プロット：", size=16),
                ft.Image(
                    src_base64=plot_residuals(
                        y.values, model.predict(X_with_const).values
                    ),
                    width=600,
                    height=400,
                ),
                ft.Text("VIF値：", size=16),
                vif_table(calculate_vif(X)),
            ]
        )

        page.update()

    except Exception as e:
        print(f"DEBUG: 分析実行中にエラーが発生: {str(e)}")
        print(f"DEBUG: エラーの種類: {type(e)}")
        import traceback

        print(f"DEBUG: エラーの詳細:\n{traceback.format_exc()}")
        page.show_snack_bar(
            ft.SnackBar(content=ft.Text(f"分析実行中にエラーが発生しました: {str(e)}"))
        )


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF値を計算する"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data

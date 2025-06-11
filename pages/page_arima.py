"""
ARIMAモデルの作成画面を提供するFlet UIモジュール。
モデルのパラメータ入力、結果のダミー表示、残差の可視化などを含む。
"""

import io
import sqlite3
import base64

import flet as ft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db.database import read_dataframe_from_sqlite


def trend_check(target: str):
    """
    時系列データのプロット
    """
    plt.figure(figsize=(10, 6))
    plt.plot(target, label="PD")
    plt.title("Time Series of PD")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def dummy_arima_summary(target: str, p: int, d: int, q: int) -> str:
    """
    ダミーのARIMAモデル出力を文字列として返す。
    AIC、BIC、Ljung-Box検定のp値などの架空値を含む。
    """
    return (
        f"ARIMA({p},{d},{q}) モデル（目的変数: {target}）\n\n"
        "AIC = 1234.56（ダミー）\n"
        "BIC = 1278.90（ダミー）\n"
        "残差のLjung-Box検定: p値 = 0.43（ダミー）\n"
        "→ 残差に自己相関なしと判断されます。"
    )


def plot_arima_residuals() -> str:
    """
    ダミーの残差データから自己相関（ACF）プロットを生成し、base64形式で返す。
    """
    residuals = np.random.normal(size=100)
    _, ax = plt.subplots(figsize=(6, 4))
    pd.plotting.autocorrelation_plot(pd.Series(residuals), ax=ax)
    ax.set_title("残差の自己相関（ACF）")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_forecast_series() -> str:
    """
    実測値と予測値をダミーデータで生成し、折れ線グラフをbase64形式で返す。

    Returns:
        str: base64エンコードされたグラフ画像。エラー時は空文字列を返す。
    """
    try:
        t = np.arange(100)
        actual = np.sin(t / 10) + np.random.normal(scale=0.2, size=100)
        forecast = actual + np.random.normal(scale=0.1, size=100)

        plt.figure(figsize=(8, 4))
        plt.plot(t, actual, label="実測値")
        plt.plot(t, forecast, label="予測値", linestyle="--")
        plt.legend()
        plt.title("ARIMAモデルによる予測")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def arima_page(page: ft.Page) -> ft.Container:
    """
    ARIMAモデル作成画面を返す。パラメータの入力、ダミー結果の表示、
    残差と予測値のグラフ描画を行うFletコンテナを構築する。
    """
    df = read_dataframe_from_sqlite("internal_data")
    all_columns = df.columns.tolist()

    target_dropdown = ft.Dropdown(
        label="目的変数（時系列）",
        options=[ft.dropdown.Option(col) for col in all_columns],
    )

    input_p = ft.TextField(label="AR成分（p）", width=100)
    input_d = ft.TextField(label="差分次数（d）", width=100)
    input_q = ft.TextField(label="MA成分（q）", width=100)

    result_text = ft.Text()
    residual_img = ft.Image()
    forecast_img = ft.Image()

    def run_arima():
        target = target_dropdown.value
        if (
            target is None
            or input_p.value is None
            or input_d.value is None
            or input_q.value is None
        ):
            result_text.value = (
                "ARIMAパラメータと目的変数をすべて選択・入力してください。"
            )
            page.update()
            return
        try:
            p = int(input_p.value)
            d = int(input_d.value)
            q = int(input_q.value)
        except ValueError:
            result_text.value = "ARIMAパラメータは整数で入力してください。"
            page.update()
            return

        result_text.value = dummy_arima_summary(target, p, d, q)
        residual_img.src_base64 = plot_arima_residuals()
        forecast_img.src_base64 = plot_forecast_series()
        page.update()

    return ft.Container(
        padding=20,
        expand=True,
        content=ft.Row(
            [
                ft.Column(
                    controls=[
                        ft.Text("ARIMAモデル作成"),
                        target_dropdown,
                        ft.ElevatedButton(
                            "時系列データ確認",
                            on_click=lambda _: (
                                trend_check(target_dropdown.value)
                                if target_dropdown.value
                                else None
                            ),
                        ),
                        ft.Row([input_p, input_d, input_q]),
                        ft.ElevatedButton("モデル作成", on_click=lambda _: run_arima()),
                        ft.Text("【モデル概要】："),
                        result_text,
                    ]
                ),
                ft.Column(
                    [
                        ft.Text("【残差のACF】："),
                        residual_img,
                        ft.Text("【予測グラフ】："),
                        forecast_img,
                    ],
                    expand=True,
                    scroll=ft.ScrollMode.ALWAYS,
                ),
            ]
        ),
        border=ft.border.all(2, "red"),
    )

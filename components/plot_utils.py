"""
# プロット用のユーティリティ関数を提供するモジュール。
#
# 現在は、複数の時系列データをプロットし、base64形式で返す関数を含む。
"""

import flet as ft
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from math import ceil

# フォントの設定
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Hiragino Maru Gothic Pro",
    "Arial",
]
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止

# seabornの設定
sns.set(font="Yu Gothic")
sns.set_style("whitegrid")

figsize = (12, 6)


def plot_multiple_time_series(df: pd.DataFrame, columns: list[str]) -> str:
    """
    指定したカラムを時系列でプロットし、描画させる
    """
    if not columns:
        return ""

    # plt.figure(figsize=(6, 3))
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.plot(df.index, df[col], label=col)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("選択された時系列データ")
    plt.legend(loc="upper right")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")
    # plt.show()


def plot_corr_heatmap(df: pd.DataFrame, target: str, features: list[str]) -> str:
    """
    相関関係のヒートマップを描画
    """
    corr = df[[target] + features].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", ax=ax
    )  # , square=True
    plt.title("相関行列のヒートマップ")
    plt.xticks(rotation=60)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")
    # #ウィンドウサイズが変更されたときに描画をリサイズ
    # def on_resize(e):
    #     plt.tight_layout()
    #     fig.canvas.draw()

    # manager = plt.get_current_fig_manager()
    # manager.window.state('zoomed')
    # fig.canvas.mpl_connect('resize_event', on_resize)
    # plt.show()


def plot_histrical_data(df: pd.DataFrame):
    """
    複数のグラフを並べて描画するプログラム
    """
    # 日本語向けフォントをセット
    plt.rcParams["font.family"] = "MS Gothic"

    df["kijyunnengetu"] = pd.to_datetime(
        df["kijyunnengetu"], format="%Y%m"
    ) + pd.offsets.MonthEnd(0)

    df_title = df.columns

    # figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    fig = plt.figure()

    # add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(3, 4, 1)
    ax2 = fig.add_subplot(3, 4, 5)
    ax3 = fig.add_subplot(3, 4, 6)
    ax4 = fig.add_subplot(3, 4, 7)
    ax5 = fig.add_subplot(3, 4, 8)
    ax6 = fig.add_subplot(3, 4, 9)
    ax7 = fig.add_subplot(3, 4, 10)
    ax8 = fig.add_subplot(3, 4, 11)
    ax9 = fig.add_subplot(3, 4, 12)

    t = df[df_title[0]]
    y1 = df[df_title[9]]
    y2 = df[df_title[1]]
    y3 = df[df_title[2]]
    y4 = df[df_title[3]]
    y5 = df[df_title[4]]
    y6 = df[df_title[5]]
    y7 = df[df_title[6]]
    y8 = df[df_title[7]]
    y9 = df[df_title[8]]

    c1, c2, c3, c4, c5, c6, c7, c8, c9 = (
        "blue",
        "green",
        "red",
        "yellow",
        "purple",
        "brown",
        "orange",
        "pink",
        "black",
    )  # 各プロットの色 ,c5,c6,c7,c8,c9
    _, l2, l3, l4, l5, l6, l7, l8, l9, l10 = df_title  # 各ラベル

    ax1.plot(t, y1, color=c1, label=l10)
    ax2.plot(t, y2, color=c2, label=l2)
    ax3.plot(t, y3, color=c3, label=l3)
    ax4.plot(t, y4, color=c4, label=l4)
    ax5.plot(t, y5, color=c5, label=l5)
    ax6.plot(t, y6, color=c6, label=l6)
    ax7.plot(t, y7, color=c7, label=l7)
    ax8.plot(t, y8, color=c8, label=l8)
    ax9.plot(t, y9, color=c9, label=l9)
    ax1.legend(loc="upper left")  # 凡例
    ax2.legend(loc="upper left")  # 凡例
    ax3.legend(loc="upper left")  # 凡例
    ax4.legend(loc="upper left")  # 凡例
    ax5.legend(loc="upper left")  # 凡例
    ax6.legend(loc="upper left")  # 凡例
    ax7.legend(loc="upper left")  # 凡例
    ax8.legend(loc="upper left")  # 凡例
    ax9.legend(loc="upper left")  # 凡例
    fig.tight_layout()  # レイアウトの設定

    # ウィンドウサイズが変更されたときに描画をリサイズ
    def on_resize(e):
        plt.tight_layout()
        fig.canvas.draw()

    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    fig.canvas.mpl_connect("resize_event", on_resize)
    plt.show()


def calculate_vif(df, features):
    """
    Calculate VIF for all pairs of features.
    """
    vif_matrix = pd.DataFrame(index=features, columns=features, dtype=float)

    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            # if i == j:
            #     vif_matrix.loc[feature_i, feature_j] = np.nan  # 自己相関は計算しない
            # else:
            # 2つの変数間のVIFを計算
            X = df[[feature_i, feature_j]]
            vif = variance_inflation_factor(X.values, 0)  # 1つ目の変数のVIF
            vif_matrix.loc[feature_i, feature_j] = vif

    return vif_matrix


def plot_vif_heatmap(df, selected_features):
    """
    Create a VIF heatmap for the selected features.
    """
    vif_matrix = calculate_vif(df, selected_features)

    # ヒートマップを作成
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        vif_matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        ax=ax,
    )
    plt.title("VIF Heatmap")
    plt.xticks(rotation=75)
    plt.tight_layout()

    # ヒートマップを画像としてエンコード
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    # buffer.seek(0)
    # image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    # buffer.close()
    plt.close()

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_vif_table(df, selected_features):
    """
    Create a VIF cross table for the selected features.
    """

    vif_matrix = calculate_vif(df, selected_features)

    # DataTableのヘッダーを作成
    headers = [ft.DataColumn(ft.Text("features"))] + [
        ft.DataColumn(ft.Text(feature)) for feature in selected_features
    ]
    # DataTableの行を作成
    rows = []
    for feature_i in selected_features:
        row_cells = [ft.DataCell(ft.Text(feature_i))]
        for feature_j in selected_features:
            value = (
                f"{vif_matrix.loc[feature_i,feature_j]:.2f}"
                if not pd.isna(vif_matrix.loc[feature_i, feature_j])
                else "-"
            )
            row_cells.append(ft.DataCell(ft.Text(value)))
        rows.append(ft.DataRow(cells=row_cells))

    return headers, rows


def regression_summary_table(model):
    # summary2().tables[0] を使用してモデル概要情報を取得
    model_summary_table = model.summary2().tables[
        0
    ]  # モデル全体の概要情報が含まれるテーブル

    # 日本語で分かりやすい表形式に変換
    summary_data = []
    for index, row in model_summary_table.iterrows():
        summary_data.append(
            {
                "項目": row[0],  # 項目名（例: R-squared, Adj. R-squared）
                "値": row[1],  # 値
            }
        )

    # 表形式で表示するための DataTable を作成
    summary_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("項目")),
            ft.DataColumn(ft.Text("値")),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(row["項目"])),
                    ft.DataCell(
                        ft.Text(
                            f"{row['値']:.4f}"
                            if isinstance(row["値"], (int, float))
                            else str(row["値"])
                        )
                    ),
                ]
            )
            for row in summary_data
        ],
    )
    return summary_table


def regression_stats_table(model):
    # summary2().tables[1] を使用して統計情報を取得
    coef_table = model.summary2().tables[1]  # 回帰係数などの情報が含まれるテーブル

    # 日本語で分かりやすい表形式に変換
    stats_data = []
    for index, row in coef_table.iterrows():
        stats_data.append(
            {
                "項目": (
                    "定数項" if index == "const" else index
                ),  # const を「定数項」に変換
                "回帰係数": row["Coef."],  # 回帰係数
                "標準誤差": row["Std.Err."],  # 標準誤差
                "t値": row["t"],  # t値
                "P値": row["P>|t|"],  # P値
                "95%信頼区間\n下限": row["[0.025"],  # 95%信頼区間の下限
                "95%信頼区間\n上限": row["0.975]"],  # 95%信頼区間の上限
            }
        )

    # 表形式で表示するための DataTable を作成
    stats_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("項目")),
            ft.DataColumn(ft.Text("回帰係数")),
            ft.DataColumn(ft.Text("標準誤差")),
            ft.DataColumn(ft.Text("t値")),
            ft.DataColumn(ft.Text("P値")),
            ft.DataColumn(ft.Text("95%信頼区間\n下限")),
            ft.DataColumn(ft.Text("95%信頼区間\n上限")),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(row["項目"])),
                    ft.DataCell(ft.Text(f"{row['回帰係数']:.4f}")),
                    ft.DataCell(ft.Text(f"{row['標準誤差']:.4f}")),
                    ft.DataCell(ft.Text(f"{row['t値']:.4f}")),
                    ft.DataCell(ft.Text(f"{row['P値']:.4f}")),
                    ft.DataCell(ft.Text(f"{row['95%信頼区間\n下限']:.4f}")),
                    ft.DataCell(ft.Text(f"{row['95%信頼区間\n上限']:.4f}")),
                ]
            )
            for row in stats_data
        ],
    )
    return stats_table


def regression_diagnostics_table(model):
    # 診断統計量を取得
    dw_stat = durbin_watson(model.resid)  # Durbin-Watson 統計量
    jb_stat, jb_pval = jarque_bera(model.resid)[:2]  # Jarque-Bera 検定
    condition_number = model.condition_number  # 条件数

    # 日本語で分かりやすい表形式に変換
    diagnostics_data = [
        {"項目": "Durbin-Watson", "値": dw_stat},
        {"項目": "Jarque-Bera 検定統計量", "値": jb_stat},
        {"項目": "Jarque-Bera 検定 P値", "値": jb_pval},
        {"項目": "条件数", "値": condition_number},
    ]

    # 表形式で表示するための DataTable を作成
    diagnostics_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("項目")),
            ft.DataColumn(ft.Text("値")),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(row["項目"])),
                    ft.DataCell(ft.Text(f"{row['値']:.4f}")),
                ]
            )
            for row in diagnostics_data
        ],
    )
    return diagnostics_table


def vif_table(vif_data: pd.DataFrame) -> ft.DataTable:
    """
    VIF値を表形式で表示するためのflet DataTableを作成する。
    """
    return ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Feature")),
            ft.DataColumn(ft.Text("VIF")),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(row["Feature"])),
                    ft.DataCell(ft.Text(f"{row['VIF']:.3f}")),
                ]
            )
            for _, row in vif_data.iterrows()
        ],
    )


def plot_single_time_series(
    df: pd.DataFrame,
    column: str,
    transformation_type: str,
    is_standardized: bool,
    figsize: tuple = (10, 4),
) -> str:
    """
    単一の時系列データをプロットし、Base64エンコードされた画像を返す

    Args:
        df (pd.DataFrame): データフレーム
        column (str): プロットするカラム名
        transformation_type (str): 変換タイプ（"none", "log", "diff", "log_diff"）
        is_standardized (bool): 標準化するかどうか
        figsize (tuple): グラフのサイズ

    Returns:
        str: Base64エンコードされた画像データ
    """
    if (
        df is None
        or df.empty
        or "kijyunnengetu" not in df.columns
        or column not in df.columns
    ):
        return ""

    # グラフタイトルを構築
    title_parts = [column]
    if transformation_type != "none":
        title_parts.append(
            f"({TRANSFORMATION_TYPES.get(transformation_type, transformation_type)})"
        )
    if is_standardized:
        title_parts.append("(標準化済)")
    plot_title = " ".join(title_parts)

    # グラフを生成
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        df["kijyunnengetu"],
        df[column],
        label=column,
        color=sns.color_palette("deep")[0],
    )

    ax.set_title(plot_title, fontsize=12)
    ax.set_xlabel("年月", fontsize=10)
    ax.set_ylabel("値", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=8)
    plt.tight_layout()

    # Base64エンコード
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return img_base64


def plot_multiple_time_series_grid(
    df: pd.DataFrame,
    columns: list[str],
    transformation_types: dict[str, str],
    standardization_states: dict[str, bool],
    n_cols: int = 2,
    figsize: tuple = (12, 4),
) -> str:
    """
    複数の時系列データをグリッド形式でプロットし、Base64エンコードされた画像を返す

    Args:
        df (pd.DataFrame): データフレーム
        columns (list[str]): プロットするカラム名のリスト
        transformation_types (dict[str, str]): 各カラムの変換タイプ
        standardization_states (dict[str, bool]): 各カラムの標準化状態
        n_cols (int): グリッドの列数
        figsize (tuple): 1つのグラフのサイズ

    Returns:
        str: Base64エンコードされた画像データ
    """
    if not columns:
        return ""

    n_rows = ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 各カラムのグラフをプロット
    for idx, column in enumerate(columns):
        row = idx // n_cols
        col = idx % n_cols
        transformation_type = transformation_types.get(column, "none")
        is_standardized = standardization_states.get(column, False)

        # グラフタイトルを構築
        title_parts = [column]
        if transformation_type != "none":
            title_parts.append(
                f"({TRANSFORMATION_TYPES.get(transformation_type, transformation_type)})"
            )
        if is_standardized:
            title_parts.append("(標準化済)")
        plot_title = " ".join(title_parts)

        # グラフをプロット
        axes[row, col].plot(
            df["kijyunnengetu"],
            df[column],
            label=column,
            color=sns.color_palette("deep")[0],
        )

        axes[row, col].set_title(plot_title, fontsize=12)
        axes[row, col].set_xlabel("年月", fontsize=10)
        axes[row, col].set_ylabel("値", fontsize=10)
        axes[row, col].tick_params(axis="x", rotation=45, labelsize=8)
        axes[row, col].tick_params(axis="y", labelsize=8)
        axes[row, col].grid(True, linestyle="--", alpha=0.7)
        axes[row, col].legend(fontsize=8)

    # 未使用のサブプロットを非表示
    for idx in range(len(columns), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # Base64エンコード
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return img_base64


# 変換タイプの定義をグローバル変数として追加
TRANSFORMATION_TYPES = {
    "none": "変換なし",
    "log": "対数変換",
    "diff": "差分化",
    "log_diff": "対数変換後に差分化",
    "arcsinh": "逆双曲線正弦変換",
    "arcsinh_diff": "逆双曲線正弦変換後に差分化",
}

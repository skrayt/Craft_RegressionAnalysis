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
    plot_vif_heatmap,
    calculate_vif,
    create_vif_table,
)


def analysis_page(page: ft.Page) -> ft.Container:
    """
    Create the correlation and VIF analysis page with interactive controls.
    """
    df = read_dataframe_from_sqlite("standardized_data")
    all_columns = df.columns.tolist()

    # 選択用リスト
    target_selector = ft.Dropdown(
        label="目的変数",
        options=[ft.dropdown.Option(col) for col in all_columns],
        on_change=lambda e: refresh_feature_options(),
    )

    features_available = ft.Column()
    features_selected = ft.Column()
    corr_heatmap_image = ft.Image()
    # vif_heatmap_image = ft.Image()
    vif_table = ft.DataTable(columns=[ft.DataColumn(ft.Text("VIF値"))], rows=[])

    def refresh_feature_options():
        selected = target_selector.value
        features_available.controls.clear()
        features_selected.controls.clear()
        for col in all_columns:
            if col != selected:
                features_available.controls.append(ft.Checkbox(label=col, value=False))
        page.update()

    def run_correlation():
        selected = target_selector.value
        if not selected:
            return

        selected_features = [
            str(cb.label)
            for cb in features_available.controls
            if isinstance(cb, ft.Checkbox) and cb.value
        ]
        if not selected_features:
            return
        # 相関ヒートマップを作成
        corr_heatmap_image.src_base64 = plot_corr_heatmap(
            df, selected, selected_features
        )
        # VIFヒートマップを作成
        # vif_heatmap_image.src_base64 = plot_vif_heatmap(df, selected_features)
        # VIFクロステーブルを作成
        vif_table.columns, vif_table.rows = create_vif_table(df, selected_features)
        page.update()

    return ft.Container(
        content=ft.Row(
            [
                ft.Column(
                    [
                        ft.Text("相関係数分析"),
                        target_selector,
                        ft.Row(
                            [
                                ft.Column(
                                    [ft.Text("説明変数候補:"), features_available],
                                ),
                            ]
                        ),
                        ft.ElevatedButton(
                            "相関係数/VIFを表示", on_click=lambda e: run_correlation()
                        ),
                    ],
                    expand=1,
                ),
                ft.Column(
                    [
                        corr_heatmap_image,
                        vif_table,
                    ],
                    expand=4,
                    spacing=20,
                ),
            ]
        ),
        expand=True,
        padding=20,
        border=ft.border.all(2, "red"),
    )

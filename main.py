"""
統計解析アプリのエントリーポイントモジュール。
Fletを用いたタブ式UIで各分析機能に画面遷移する。
"""

# main.py

import flet as ft
import pandas as pd
from pages.page_data_load import (
    data_load_page,
    update_data_table,
    initialize_data_from_db_or_csv,
    get_merged_df,
    standardize_data_button_click,
    go_to_column_management,
)
from pages.page_analysis import analysis_page
from pages.page_regression import regression_page
from pages.page_arima import arima_page
from components.data_cleansing import (
    explanatory_variable,
    response_variable,
    merged_variable,
    standardized_variable,
)
from db.database import (
    save_dataframe_to_sqlite,
    initialize_regression_database,
    initialize_user_data_database,
    read_dataframe_from_sqlite,
    save_dataframe_to_sqlite_with_sanitization,
)
from pages.page_column_management import (
    column_management_page,
)  # 新しいファイルからインポート


# アプリケーションデータを保持するクラス
class AppData:
    def __init__(self):
        self.merged_df: pd.DataFrame | None = None
        self.saved_df: pd.DataFrame | None = None


ExternalData = explanatory_variable()
InternalData = response_variable()
MergedData = merged_variable()
StandardizedData = standardized_variable()


def initialize():
    initialize_regression_database()


def main(page: ft.Page):
    """
    Fletアプリを初期化し、各タブ画面を切り替えるUIを構築する。
    """
    print(f"DEBUG: App starting. Initial page route: {page.route}")
    initialize()
    page.title = "統計解析アプリ"
    page.theme_mode = ft.ThemeMode.LIGHT  # ライトモードを設定

    # pageにAppDataインスタンスをアタッチ
    page.app_data = AppData()  # type: ignore

    # アプリケーション起動時にデータをロード
    initialize_user_data_database()  # ユーザーデータDBの初期化 (db/databaseから呼び出し)
    initialize_data_from_db_or_csv(page)  # データロード
    update_data_table(page)  # テーブルの表示を更新

    print(f"DEBUG: Before page.go(page.route) - page.app_data.merged_df type: {type(page.app_data.merged_df)}, is None: {page.app_data.merged_df is None}, is_empty: {page.app_data.merged_df.empty if isinstance(page.app_data.merged_df, pd.DataFrame) else 'N/A'}")  # type: ignore

    def route_change(route):
        print(f"DEBUG: route_change function called. Route: {route}")
        page.views.clear()
        if page.route == "/":
            print("DEBUG: Current route is '/'. Setting up main view with tabs.")

            # タブで画面切替
            def on_tab_change(e):
                print(f"DEBUG: Tab changed. Selected index: {e.control.selected_index}")
                selected_index = e.control.selected_index
                body.controls.clear()
                if selected_index == 0:
                    body.controls.append(data_load_page(page))
                elif selected_index == 1:
                    body.controls.append(analysis_page(page))
                elif selected_index == 2:
                    body.controls.append(regression_page(page))
                elif selected_index == 3:
                    body.controls.append(arima_page(page))
                page.update()

            # タブメニュー
            tabs = ft.Tabs(
                selected_index=0,
                on_change=on_tab_change,
                tabs=[
                    ft.Tab(text="① データ取込み・参照"),
                    ft.Tab(text="② 分析（相関/VIF等）"),
                    ft.Tab(text="③ 多変量回帰分析"),
                    ft.Tab(text="④ ARIMAモデル"),
                ],
            )

            body = ft.Column(expand=True)
            body.controls.append(data_load_page(page))  # 初期ページ

            page.views.append(
                ft.View(
                    "/",
                    [tabs, body],
                )
            )
            print("DEBUG: Main view with tabs appended to page.views.")
        elif page.route == "/column-management":
            print(
                "DEBUG: Current route is '/column-management'. Setting up column management view."
            )
            print(f"DEBUG: route_change - Current app_data.merged_df (before view): {type(page.app_data.merged_df)}, empty: {page.app_data.merged_df.empty if isinstance(page.app_data.merged_df, pd.DataFrame) else 'N/A'}")  # type: ignore
            if page.app_data.merged_df is not None and not page.app_data.merged_df.empty:  # type: ignore
                page.views.append(
                    ft.View(
                        "/column-management",
                        [column_management_page(page, page.app_data.merged_df, lambda: update_data_table(page))],  # type: ignore
                    )
                )
                print("DEBUG: route_change - カラム管理ページのビューを追加しました。")
            else:
                print(
                    "DEBUG: route_change - データが読み込まれていないため、カラム管理ページを表示しません。"
                )
                snack = ft.SnackBar(
                    content=ft.Text(
                        "データが読み込まれていないため、カラム管理を開けません。"
                    )
                )
                page.add(snack)
                snack.open = True
                page.update()
                page.go("/")  # データロードページに戻る
        page.update()
        print("DEBUG: Page updated after route_change.")

    page.on_route_change = route_change
    page.go(page.route)


if __name__ == "__main__":
    ft.app(target=main)

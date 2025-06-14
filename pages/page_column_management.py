import flet as ft
import pandas as pd
from typing import Callable
from db.database import sanitize_column_name, save_dataframe_to_sqlite_with_sanitization
import os
from utils.data_transformation import (
    standardize_data,
    apply_transformations,
)  # 新しいモジュールからインポート
import numpy as np
from sklearn.preprocessing import StandardScaler

# グローバル変数
checkbox_states = {}  # チェックボックスの状態を保持するグローバル辞書
has_unsaved_changes = False


def column_management_page(
    page: ft.Page, current_merged_df: pd.DataFrame, on_save_callback: Callable
):
    global has_unsaved_changes

    print(
        f"DEBUG: column_management_page - current_merged_dfカラム: {list(current_merged_df.columns)}"
    )

    # CSVからデータを読み込む
    IndicatorFolderPath = "Indicator"
    EconomicIndicatorFile = "各種経済指標.csv"
    EconomicIndicatorFilePath = os.path.join(IndicatorFolderPath, EconomicIndicatorFile)
    pdFile = "v_pd.csv"
    pdFilePath = os.path.join(IndicatorFolderPath, pdFile)

    try:
        # 各種経済指標の読み込み
        df1 = pd.read_csv(EconomicIndicatorFilePath)
        df1_conveted = df1[df1["地域"] == "全国"]
        df1_conveted["kijyunnengetu"] = pd.to_datetime(
            df1["時点"], format="%Y年%m月"
        ).dt.strftime("%Y%m")

        # PDデータの読み込み
        df2 = pd.read_csv(pdFilePath, dtype={"kijyunnengetu": str})

        # 両方のデータフレームを単純に結合
        dialog_editing_df = pd.merge(df1_conveted, df2, on="kijyunnengetu", how="inner")
        print(f"DEBUG: dialog_editing_dfカラム: {list(dialog_editing_df.columns)}")
    except Exception as e:
        print(f"DEBUG: CSVからの読み込みに失敗しました: {e}")
        snack = ft.SnackBar(content=ft.Text("データの読み込みに失敗しました。"))
        page.add(snack)
        snack.open = True
        page.update()
        page.go("/")
        return

    column_controls_container = ft.Column()

    def set_unsaved_changes(e=None):
        global has_unsaved_changes
        has_unsaved_changes = True
        print(
            f"DEBUG: set_unsaved_changes - has_unsaved_changes set to {has_unsaved_changes}"
        )

    def initialize_column_controls():
        global checkbox_states
        nonlocal dialog_editing_df

        def handle_checkbox_change(e: ft.ControlEvent, current_col: str):
            global checkbox_states
            checkbox_states.update({current_col: e.control.value})
            set_unsaved_changes()
            print(f"DEBUG: Checkbox changed for {current_col} to {e.control.value}")

        if dialog_editing_df is not None:
            # チェックボックスの初期状態を設定
            for col in dialog_editing_df.columns:
                if col not in checkbox_states:
                    # 現在のmerged_dfに存在するカラムはTrue、それ以外はFalse
                    checkbox_states[col] = col in current_merged_df.columns

            column_controls_list = []
            # 元のCSVファイルのカラム順序を維持
            for col in dialog_editing_df.columns:
                column_controls_list.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Checkbox(
                                    key=f"chk_{col}",
                                    label="",
                                    value=checkbox_states.get(col, False),
                                    on_change=lambda e, current_col=col: handle_checkbox_change(
                                        e, current_col
                                    ),
                                ),
                                ft.Container(
                                    content=ft.Text(col),
                                    width=200,
                                    alignment=ft.alignment.center_left,
                                    padding=ft.padding.only(left=10),
                                ),
                                ft.TextField(
                                    key=f"rename_col_{col}",
                                    label="新しいカラム名",
                                    value=col,  # 元のカラム名をそのまま表示
                                    width=300,
                                    height=40,
                                    text_style=ft.TextStyle(size=14),
                                    label_style=ft.TextStyle(size=12),
                                    on_change=set_unsaved_changes,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.START,
                            spacing=20,
                        ),
                        border=ft.border.only(
                            bottom=ft.border.BorderSide(1, "#E0E0E0")
                        ),
                        padding=ft.padding.only(bottom=5, top=5),
                    )
                )
            column_controls_container.controls = column_controls_list

    def save_data(e: ft.ControlEvent):
        global has_unsaved_changes, checkbox_states
        nonlocal dialog_editing_df
        page = e.page  # ページオブジェクトを保存

        print("DEBUG: save_data関数が呼び出されました。")
        print(f"DEBUG: dialog_editing_dfカラム: {list(dialog_editing_df.columns)}")
        print(f"DEBUG: checkbox_states: {checkbox_states}")

        try:
            # 選択されたカラムのみを抽出
            selected_columns = [
                col
                for col in dialog_editing_df.columns
                if checkbox_states.get(col, False)
            ]
            print(f"DEBUG: 選択されたカラム: {selected_columns}")

            if not selected_columns:
                snack = ft.SnackBar(
                    content=ft.Text("少なくとも1つのカラムを選択してください。")
                )
                page.add(snack)
                snack.open = True
                page.update()
                return

            # 新しいカラム名を取得
            new_column_names = {}
            for col in selected_columns:
                # カラム管理画面のコンテナから直接TextFieldを探す
                for control in column_controls_container.controls:
                    if isinstance(control.content, ft.Row):
                        for row_control in control.content.controls:
                            if (
                                isinstance(row_control, ft.TextField)
                                and row_control.key == f"rename_col_{col}"
                            ):
                                new_name = row_control.value
                                if new_name and new_name != col:
                                    new_column_names[col] = new_name
                                break

            # 選択されたカラムのみを含む新しいデータフレームを作成
            new_df = dialog_editing_df[selected_columns].copy()

            # カラム名を変更
            if new_column_names:
                new_df = new_df.rename(columns=new_column_names)

            # カラム変更後のデータをmerged_dataテーブルに保存
            save_dataframe_to_sqlite_with_sanitization(new_df, table_name="merged_data")
            print("DEBUG: カラム変更後のデータをmerged_dataテーブルに保存しました。")
            print(f"DEBUG: 保存したmerged_dataのカラム: {list(new_df.columns)}")

            # page.app_dataのmerged_dfも更新
            page.app_data.merged_df = new_df.copy()  # type: ignore

            # 変換なしデータの標準化データを生成して保存
            standardized_df = standardize_data(
                new_df.copy(), table_name_prefix="merged"
            )
            print("DEBUG: 変換なしデータの標準化データをデータベースに保存しました。")

            # 変換データの生成と保存
            transformations = ["対数変換", "差分化", "対数変換後に差分化"]
            transformed_dfs = apply_transformations(new_df.copy(), transformations)

            # 各変換データをデータベースに保存
            for table_name, df in transformed_dfs.items():
                try:
                    # 変換データを保存する前に差分化データのna行を削除
                    df = df.dropna()
                    # 変換データを保存
                    save_dataframe_to_sqlite_with_sanitization(
                        df, table_name=table_name
                    )
                    print(f"DEBUG: {table_name}をデータベースに保存しました。")

                    # 標準化データを生成して保存
                    standardized_df = standardize_data(df, table_name_prefix=table_name)
                    print(
                        f"DEBUG: {table_name}の標準化データをデータベースに保存しました。"
                    )
                except Exception as e:
                    print(f"DEBUG: {table_name}の保存に失敗しました: {e}")

            # 保存成功時の処理
            has_unsaved_changes = False
            snack = ft.SnackBar(content=ft.Text("データを保存しました。"))
            page.add(snack)
            snack.open = True
            page.update()

            # コールバックを呼び出してデータテーブルを更新
            if on_save_callback:
                on_save_callback()

            # 元のページに戻る
            page.go("/")

        except Exception as e:
            print(f"DEBUG: データの保存に失敗しました: {e}")
            snack = ft.SnackBar(content=ft.Text("データの保存に失敗しました。"))
            page.add(snack)
            snack.open = True
            page.update()
            page.go("/")  # エラー時もメインページに戻る

    # カラムコントロールの初期化
    initialize_column_controls()

    # 保存ボタン
    save_button = ft.ElevatedButton(
        "変更を保存",
        on_click=save_data,
    )

    # 戻るボタン
    back_button = ft.ElevatedButton(
        "戻る",
        on_click=lambda e: page.go("/"),
    )

    # メインコンテナ
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [save_button, back_button],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                ),
                ft.Divider(),
                ft.Column(
                    controls=[column_controls_container],
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        expand=True,
    )

import flet as ft
import pandas as pd
from typing import Callable
from db.database import sanitize_column_name, save_dataframe_to_sqlite_with_sanitization
import os
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

        print("DEBUG: save_data関数が呼び出されました。")
        print(f"DEBUG: dialog_editing_dfカラム: {list(dialog_editing_df.columns)}")
        print(f"DEBUG: checkbox_states: {checkbox_states}")

        # 選択されたカラムのみを抽出
        selected_columns = [
            col for col in dialog_editing_df.columns if checkbox_states.get(col, False)
        ]
        print(f"DEBUG: 選択されたカラム: {selected_columns}")

        if not selected_columns:
            snack = ft.SnackBar(
                content=ft.Text("少なくとも1つのカラムを選択してください。")
            )
            e.page.add(snack)
            snack.open = True
            e.page.update()
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

        # PDデータのカラムにサフィックスを付ける
        pd_columns = [
            col
            for col in new_df.columns
            if col in df2.columns and col != "kijyunnengetu"
        ]
        if pd_columns:
            new_df = new_df.rename(columns={col: f"{col}_pd" for col in pd_columns})

        try:
            # カラム変更後のデータをmerged_dataテーブルに保存
            try:
                save_dataframe_to_sqlite_with_sanitization(
                    new_df, table_name="merged_data"
                )
                print(
                    "DEBUG: カラム変更後のデータをmerged_dataテーブルに保存しました。"
                )
                print(f"DEBUG: 保存したmerged_dataのカラム: {list(new_df.columns)}")
                # page.app_dataのmerged_dfも更新
                e.page.app_data.merged_df = new_df.copy()  # type: ignore

                # 変換なしデータの標準化データを生成して保存
                standardized_df = standardize_data(
                    new_df.copy(), table_name_prefix="merged"
                )
                print(
                    "DEBUG: 変換なしデータの標準化データをデータベースに保存しました。"
                )
            except Exception as e:
                print(f"DEBUG: merged_dataテーブルへの保存に失敗しました: {e}")
                raise e

            # 変換データの生成と保存
            transformations = ["対数変換", "差分化", "対数変換後に差分化"]
            transformed_dfs = apply_transformations(new_df.copy(), transformations)

            # 各変換データをデータベースに保存
            for table_name, df in transformed_dfs.items():
                try:
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
            e.page.add(snack)
            snack.open = True
            e.page.update()

            # コールバックを呼び出してデータテーブルを更新
            if on_save_callback:
                on_save_callback()

            # 元のページに戻る
            e.page.go("/")
        except Exception as e:
            print(f"DEBUG: データの保存に失敗しました: {e}")
            snack = ft.SnackBar(content=ft.Text("データの保存に失敗しました。"))
            e.page.add(snack)
            snack.open = True
            e.page.update()

    def confirm_back(e):
        global has_unsaved_changes
        print(f"DEBUG: confirm_back called. has_unsaved_changes: {has_unsaved_changes}")
        if has_unsaved_changes:

            def go_back(e):
                global has_unsaved_changes
                has_unsaved_changes = False  # 変更を破棄して戻る
                confirm_dialog.open = False
                page.update()
                page.go("/")
                print("DEBUG: Going back, changes discarded.")

            def stay(e):
                confirm_dialog.open = False
                page.update()
                print("DEBUG: Staying on column management page.")

            confirm_dialog = ft.AlertDialog(
                modal=True,
                title=ft.Text("確認"),
                content=ft.Text("保存していません。よろしいですか？"),
                actions=[
                    ft.ElevatedButton("はい", on_click=go_back),
                    ft.ElevatedButton("いいえ", on_click=stay),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            page.overlay.append(confirm_dialog)
            confirm_dialog.open = True
            page.update()
        else:
            page.go("/")
            print("DEBUG: No unsaved changes, going back.")

    initialize_column_controls()

    # カラム管理ページのレイアウト
    return ft.Container(
        content=ft.Column(
            [
                ft.Text("カラム名の変更と削除", size=20, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        ft.ElevatedButton("保存", on_click=save_data),
                        ft.ElevatedButton("戻る", on_click=confirm_back),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                ),
                ft.Divider(),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("✔", width=30),
                            ft.Text("現在のカラム名", width=200),
                            ft.Text("新しいカラム名", width=300),
                        ],
                        spacing=0,
                    ),
                    padding=ft.padding.only(bottom=5, top=5, left=10),
                    alignment=ft.alignment.center_left,
                ),
                column_controls_container,  # ここでcontrolsを設定
            ],
            scroll=ft.ScrollMode.AUTO,  # Column自体をスクロール可能にする
            expand=True,
        ),
        expand=True,  # Containerもexpandさせる
    )


# 対数変換、差分化、対数変換後に差分化を適用する関数
def apply_transformations(df, transformations):
    # kijyunnengetuカラムを保持
    kijyunnengetu_col = None
    if "kijyunnengetu" in df.columns:
        kijyunnengetu_col = df["kijyunnengetu"]
        df = df.drop(columns=["kijyunnengetu"])

    # 各変換を個別に適用し、それぞれのデータフレームを保存
    transformed_dfs = {}

    # 対数変換
    if "対数変換" in transformations:
        log_df = pd.DataFrame()
        for col in df.columns:
            log_df[f"{col}_log"] = np.log1p(df[col])
        if kijyunnengetu_col is not None:
            log_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["log_data"] = log_df

    # 差分化
    if "差分化" in transformations:
        diff_df = pd.DataFrame()
        for col in df.columns:
            diff_df[f"{col}_diff"] = df[col].diff()
        if kijyunnengetu_col is not None:
            diff_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["diff_data"] = diff_df

    # 対数変換後に差分化
    if "対数変換後に差分化" in transformations:
        log_diff_df = pd.DataFrame()
        for col in df.columns:
            log_diff_df[f"{col}_log_diff"] = np.log1p(df[col]).diff()
        if kijyunnengetu_col is not None:
            log_diff_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["log_diff_data"] = log_diff_df

    return transformed_dfs


# 標準化を適用する関数
def standardize_data(df, table_name_prefix=""):
    # kijyunnengetuカラムを保持
    kijyunnengetu_col = None
    if "kijyunnengetu" in df.columns:
        kijyunnengetu_col = df["kijyunnengetu"]
        df = df.drop(columns=["kijyunnengetu"])

    # 標準化を適用
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

    # kijyunnengetuカラムを戻す
    if kijyunnengetu_col is not None:
        scaled_df["kijyunnengetu"] = kijyunnengetu_col

    # データベースに保存
    try:
        save_dataframe_to_sqlite_with_sanitization(
            scaled_df, table_name=f"{table_name_prefix}_standardized"
        )
        print(f"DEBUG: {table_name_prefix}の標準化データをデータベースに保存しました。")
    except Exception as e:
        print(f"DEBUG: {table_name_prefix}の標準化データの保存に失敗しました: {e}")

    return scaled_df

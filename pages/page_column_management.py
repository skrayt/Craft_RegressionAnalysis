import flet as ft
import pandas as pd
from typing import Callable
from db.database import sanitize_column_name, save_dataframe_to_sqlite_with_sanitization

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

    # original_dfが存在しない場合のエラーハンドリング
    if not hasattr(page.app_data, "original_df") or page.app_data.original_df is None:  # type: ignore
        print("DEBUG: original_dfが存在しないため、merged_dfから作成します。")
        # original_dfが存在しない場合、merged_dfから作成
        if "kijyunnengetu" in current_merged_df.columns:
            original_columns = [
                col for col in current_merged_df.columns if col != "kijyunnengetu"
            ]
            page.app_data.original_df = pd.DataFrame(current_merged_df[original_columns])  # type: ignore
        else:
            page.app_data.original_df = pd.DataFrame(current_merged_df)  # type: ignore
        print(f"DEBUG: original_dfを作成しました。カラム: {list(page.app_data.original_df.columns)}")  # type: ignore

    # オリジナルのデータフレームを使用
    dialog_editing_df = pd.DataFrame(page.app_data.original_df.copy())  # type: ignore
    print(
        f"DEBUG: column_management_page - dialog_editing_dfカラム: {list(dialog_editing_df.columns)}"
    )
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
                    # merged_dfに存在するカラムはTrue、それ以外はFalse
                    checkbox_states[col] = col in current_merged_df.columns

            column_controls_list = []
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
                                    # merged_dfで既にリネームされている場合はその名前を表示
                                    value=(
                                        current_merged_df.columns[
                                            current_merged_df.columns.get_loc(col)
                                        ]
                                        if col in current_merged_df.columns
                                        else col
                                    ),
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

    def save_data(e):
        global has_unsaved_changes, checkbox_states
        nonlocal dialog_editing_df
        print("DEBUG: save_data called")

        new_column_names: dict[str, str] = {}
        for control in column_controls_container.controls:
            if isinstance(control, ft.Container) and isinstance(
                control.content, ft.Row
            ):
                for inner_control in control.content.controls:
                    if isinstance(inner_control, ft.TextField):
                        original_col = inner_control.key.replace("rename_col_", "")
                        new_name = inner_control.value
                        if new_name and new_name != original_col:
                            new_column_names[str(original_col)] = sanitize_column_name(
                                new_name
                            )

        # チェックボックスの状態を元に、選択されたカラムをフィルタリング
        columns_to_keep = [col for col, checked in checkbox_states.items() if checked]
        print(f"DEBUG: Columns to keep: {columns_to_keep}")

        if not columns_to_keep:
            snack = ft.SnackBar(
                content=ft.Text("少なくとも1つのカラムを選択してください。")
            )
            page.add(snack)
            snack.open = True
            page.update()
            print("DEBUG: No columns selected.")
            return

        # 選択されたカラムのみを保持
        filtered_df = dialog_editing_df[columns_to_keep].copy()

        # カラム名の変更を適用
        if new_column_names:
            print(f"DEBUG: Renaming columns: {new_column_names}")
            filtered_df = filtered_df.rename(columns=new_column_names)

        # merged_dfを更新（kijyunnengetuで結合）
        if "kijyunnengetu" in filtered_df.columns:
            # 既存のmerged_dfからkijyunnengetu以外のカラムを取得
            other_columns = [
                col for col in current_merged_df.columns if col != "kijyunnengetu"
            ]
            # 新しいmerged_dfを作成
            new_merged_df = pd.merge(
                filtered_df,
                current_merged_df[["kijyunnengetu"] + other_columns],
                on="kijyunnengetu",
                how="inner",
            )
            # page.app_dataを更新
            page.app_data.merged_df = new_merged_df  # type: ignore
            page.app_data.saved_df = new_merged_df.copy()  # type: ignore
        else:
            # kijyunnengetuがない場合は、filtered_dfをそのまま使用
            page.app_data.merged_df = filtered_df  # type: ignore
            page.app_data.saved_df = filtered_df.copy()  # type: ignore

        print("DEBUG: Data saved to page.app_data.merged_df and saved_df.")

        try:
            save_dataframe_to_sqlite_with_sanitization(page.app_data.saved_df)  # type: ignore
            on_save_callback()
            has_unsaved_changes = False
            page.go("/")
            snack = ft.SnackBar(content=ft.Text("データが保存されました。"))
            page.add(snack)
            snack.open = True
            page.update()
            print("DEBUG: Data saved successfully to DB.")
        except Exception as e:
            snack = ft.SnackBar(content=ft.Text(f"エラーが発生しました: {str(e)}"))
            page.add(snack)
            snack.open = True
            page.update()
            print("エラー詳細:", str(e))
            print("問題のカラム名:", list(filtered_df.columns))

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

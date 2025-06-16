"""
変数選択、変換、標準化のための共通コンポーネントモジュール。
"""

import flet as ft
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple

# 変換タイプの定義
TRANSFORMATION_TYPES = {
    "none": "変換なし",
    "log": "対数変換",
    "diff": "差分化",
    "log_diff": "対数変換後に差分化",
    "arcsinh": "逆双曲線正弦変換",
    "arcsinh_diff": "逆双曲線正弦変換後に差分化",
}


class VariableSelector:
    def __init__(
        self,
        page: ft.Page,
        all_columns: List[str],
        on_variable_change: Optional[Callable] = None,
        initial_target: Optional[str] = None,
    ):
        self.page = page
        self.all_columns = all_columns
        self.on_variable_change = on_variable_change
        self.initial_target = initial_target

        # 状態管理用の辞書
        self.checkbox_states: Dict[str, bool] = {}
        self.transformation_states: Dict[str, str] = {}
        self.standardization_states: Dict[str, bool] = {}
        self.variable_types: Dict[str, str] = {}

        # 選択された目的変数
        self.selected_target = initial_target

        # UIコンポーネントの初期化
        self._initialize_components()

    def _initialize_components(self):
        """UIコンポーネントを初期化"""
        # 目的変数選択用のドロップダウン
        self.target_dropdown = ft.Dropdown(
            label="目的変数を選択",
            options=[ft.dropdown.Option(col) for col in self.all_columns],
            width=150,
            on_change=self._handle_target_selection,
            value=self.initial_target,
            text_size=16,
            label_style=ft.TextStyle(size=16, weight=ft.FontWeight.BOLD),
        )

        # 目的変数の変換パターン選択用のドロップダウン
        self.target_transformation = ft.Dropdown(
            label="目的変数の変換パターン",
            options=[
                ft.dropdown.Option(key, value)
                for key, value in TRANSFORMATION_TYPES.items()
            ],
            value="none",
            width=150,
            on_change=lambda e: self._handle_transformation_change(
                e, self.selected_target
            ),
            text_size=12,
            label_style=ft.TextStyle(size=12),
        )

        # 目的変数の標準化選択用のトグルボタン
        self.target_standardization = ft.Switch(
            label="標準化：",
            value=False,
            on_change=lambda e: self._handle_standardization_change(
                e, self.selected_target
            ),
            label_style=ft.TextStyle(size=12),
        )

        # 説明変数選択用のコンテナ
        self.feature_container = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            height=400,
            expand=True,
        )

        # 目的変数の選択と変換パターンを横並びに配置
        self.target_row = ft.Row(
            [
                self.target_standardization,
                self.target_transformation,
                self.target_dropdown,
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=10,
        )

        # 初期状態の設定
        self._initialize_states()
        self._refresh_feature_controls()

    def _initialize_states(self):
        """各変数の状態を初期化"""
        for col in self.all_columns:
            self.checkbox_states[col] = False
            self.transformation_states[col] = "none"
            self.standardization_states[col] = False
            self.variable_types[col] = "explanatory"

            if col == self.selected_target:
                self.variable_types[col] = "response"
                self.checkbox_states[col] = True

    def _handle_target_selection(self, e: ft.ControlEvent):
        """目的変数選択時の処理"""
        self.selected_target = e.control.value
        print(f"DEBUG: 目的変数が選択されました: {self.selected_target}")

        # 変数タイプとチェックボックスの状態をリセット
        for col in self.all_columns:
            if col == self.selected_target:
                self.variable_types[col] = "response"
                self.checkbox_states[col] = True
            else:
                self.variable_types[col] = "explanatory"
                self.checkbox_states[col] = False
            self.transformation_states[col] = "none"
            self.standardization_states[col] = False

        self._refresh_feature_controls()
        if self.on_variable_change:
            self.on_variable_change()
        self.page.update()

    def _handle_checkbox_change(self, e: ft.ControlEvent, column: str):
        """チェックボックスの状態変更を処理"""
        self.checkbox_states[column] = e.control.value
        print(f"DEBUG: チェックボックス変更 - {column}: {self.checkbox_states[column]}")
        if self.on_variable_change:
            self.on_variable_change()

    def _handle_transformation_change(self, e: ft.ControlEvent, column: str):
        """変換タイプの変更を処理"""
        self.transformation_states[column] = e.control.value
        print(f"DEBUG: 変換タイプ変更 - {column}: {self.transformation_states[column]}")
        if self.on_variable_change:
            self.on_variable_change()

    def _handle_standardization_change(self, e: ft.ControlEvent, column: str):
        """標準化スイッチの変更を処理"""
        self.standardization_states[column] = e.control.value
        print(f"DEBUG: 標準化変更 - {column}: {self.standardization_states[column]}")
        if self.on_variable_change:
            self.on_variable_change()

    def _refresh_feature_controls(self):
        """説明変数のコントロールを更新"""
        self.feature_container.controls.clear()

        for col in self.all_columns:
            if col != self.selected_target:
                # 変数名の表示
                variable_name = ft.Text(col, size=14)

                # チェックボックス
                checkbox = ft.Checkbox(
                    label="",
                    value=self.checkbox_states[col],
                    on_change=lambda e, col=col: self._handle_checkbox_change(e, col),
                )

                # 変換タイプ選択用ドロップダウン
                transformation_dropdown = ft.Dropdown(
                    label="変換タイプ",
                    options=[
                        ft.dropdown.Option(key, value)
                        for key, value in TRANSFORMATION_TYPES.items()
                    ],
                    value=self.transformation_states[col],
                    width=120,
                    text_size=12,
                    on_change=lambda e, col=col: self._handle_transformation_change(
                        e, col
                    ),
                )

                # 標準化トグルボタン
                standardization_switch = ft.Switch(
                    label="標準化：",
                    label_style=ft.TextStyle(size=12),
                    label_position=ft.LabelPosition.LEFT,
                    value=self.standardization_states[col],
                    on_change=lambda e, col=col: self._handle_standardization_change(
                        e, col
                    ),
                )

                # 変数ごとのコントロールを横に並べる
                row = ft.Row(
                    [
                        checkbox,
                        standardization_switch,
                        transformation_dropdown,
                        variable_name,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=10,
                )

                self.feature_container.controls.append(row)

    def get_selected_variables(self) -> Tuple[str, List[str]]:
        """選択された目的変数と説明変数のリストを返す"""
        target = self.selected_target
        features = [
            col
            for col in self.all_columns
            if col != target and self.checkbox_states.get(col, False)
        ]
        return target, features

    def get_variable_settings(self) -> Dict[str, Dict[str, any]]:
        """各変数の設定（変換タイプ、標準化）を返す"""
        settings = {}
        for col in self.all_columns:
            settings[col] = {
                "transformation": self.transformation_states.get(col, "none"),
                "standardization": self.standardization_states.get(col, False),
            }
        return settings

    def get_ui_components(self) -> Tuple[ft.Row, ft.Column]:
        """UIコンポーネントを返す"""
        return self.target_row, self.feature_container

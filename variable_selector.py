"""
This module provides a component for selecting variables using checkboxes in a Flet UI.
"""

from typing import Callable
from flet import Column, Checkbox, Text, Container, ListView, Control, FontWeight


def create_variable_selector(
    title: str,
    columns: list[str],
    selected_vars: list[str],
    on_change: Callable[[list[str]], None],
) -> Control:
    """
    変数選択用のチェックボックスUIを作成する。

    :param title: セクションタイトル（例：「目的変数」「説明変数」）
    :param columns: 選択肢の列名リスト
    :param selected_vars: 初期選択状態の変数リスト
    :param on_change: チェック状態変更時のコールバック（選択済み変数のリストを引数に取る）
    :return: flet.Control (Column)
    """
    checkboxes = []

    def handle_change(_):
        selected = [cb.label for cb in checkboxes if cb.value]
        on_change(selected)

    for col in columns:
        checkbox = Checkbox(
            label=col, value=col in selected_vars, on_change=handle_change
        )
        checkboxes.append(checkbox)

    return Column(
        [
            Text(title, size=16, weight=FontWeight.BOLD),
            Container(
                content=ListView(controls=checkboxes, expand=True),
                height=200,
                border_radius=5,
            ),
        ]
    )

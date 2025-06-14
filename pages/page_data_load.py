import os
import flet as ft
import pandas as pd
import numpy as np

# import sqlite3 # db/database.pyからインポートするため削除
from sklearn.preprocessing import StandardScaler
from utils.data_transformation import (
    standardize_data,
    apply_transformations,
)  # 新しいモジュールからインポート
from .page_column_management import column_management_page
from db.database import (  # db.databaseからインポート
    initialize_user_data_database,
    read_dataframe_from_sqlite,
    save_dataframe_to_sqlite_with_sanitization,
    sanitize_column_name,  # これも一応インポートしておく
)


checkbox_states = {}  # チェックボックスの状態を保持するグローバル辞書
data_table = ft.DataTable(columns=[])  # データ表示用テーブルをグローバルにする
standardized_data_table = ft.DataTable(
    columns=[]
)  # 標準化後のデータ表示用テーブルをグローバルにする


# データテーブルを更新する関数
def update_data_table(page: ft.Page):
    # merged_dfはpage.app_data経由でアクセス
    if page.app_data.merged_df is not None and not page.app_data.merged_df.empty:  # type: ignore
        data_table.columns = [
            ft.DataColumn(ft.Text(col)) for col in page.app_data.merged_df.columns  # type: ignore
        ]
        data_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(str(value))) for value in row])
            for row in page.app_data.merged_df.itertuples(index=False)  # type: ignore
        ]
    else:
        data_table.columns = [ft.DataColumn(ft.Text("No Data"))]
        data_table.rows = []
    page.update()


# 標準化後のデータテーブルを更新する関数
def update_standardized_data_table(page: ft.Page, standardized_df: pd.DataFrame):
    if standardized_df is not None and not standardized_df.empty:
        standardized_data_table.columns = [
            ft.DataColumn(ft.Text(col)) for col in standardized_df.columns
        ]
        standardized_data_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(str(value))) for value in row])
            for row in standardized_df.itertuples(index=False)
        ]
    else:
        standardized_data_table.columns = [ft.DataColumn(ft.Text("No Data"))]
        standardized_data_table.rows = []
    page.update()


# merged_dfの現在の値を返す関数
def get_merged_df(page: ft.Page):
    # merged_dfはpage.app_data経由でアクセス
    print(f"DEBUG: get_merged_df called. page.app_data.merged_df type: {type(page.app_data.merged_df)}, is None: {page.app_data.merged_df is None}")  # type: ignore
    return page.app_data.merged_df  # type: ignore


# 対数変換、差分化、対数変換後に差分化を適用する関数
def apply_transformations(df, transformations):
    # kijyunnengetuカラムを保持
    result_df = df.copy()
    kijyunnengetu_col = None
    if "kijyunnengetu" in df.columns:
        kijyunnengetu_col = df["kijyunnengetu"]
        result_df = result_df.drop(columns=["kijyunnengetu"])

    # 各変換を個別に適用し、それぞれのデータフレームを保存
    transformed_dfs = {}

    # 対数変換
    if "対数変換" in transformations:
        log_df = result_df.copy()
        for col in log_df.columns:
            log_df[f"{col}_log"] = np.log1p(log_df[col])
        if kijyunnengetu_col is not None:
            log_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["log_data"] = log_df

    # 差分化
    if "差分化" in transformations:
        diff_df = result_df.copy()
        for col in diff_df.columns:
            diff_df[f"{col}_diff"] = diff_df[col].diff()
        if kijyunnengetu_col is not None:
            diff_df["kijyunnengetu"] = kijyunnengetu_col
        transformed_dfs["diff_data"] = diff_df

    # 対数変換後に差分化
    if "対数変換後に差分化" in transformations:
        log_diff_df = result_df.copy()
        for col in log_diff_df.columns:
            log_diff_df[f"{col}_log_diff"] = np.log1p(log_diff_df[col]).diff()
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


# 標準化ボタンのクリックイベント
def standardize_data_button_click(e: ft.ControlEvent):
    print(f"DEBUG: standardize_data_button_click関数が呼び出されました。")
    if e.page.app_data.merged_df is None or e.page.app_data.merged_df.empty:  # type: ignore
        snack = ft.SnackBar(content=ft.Text("データが読み込まれていません。"))
        e.page.add(snack)
        snack.open = True
        e.page.update()
        print("DEBUG: standardiz_data_button_click - データが読み込まれていません。")
        return

    transformations = ["対数変換", "差分化", "対数変換後に差分化"]

    # 各変換を適用
    transformed_dfs = apply_transformations(e.page.app_data.merged_df.copy(), transformations)  # type: ignore

    # 各変換データをデータベースに保存
    for table_name, df in transformed_dfs.items():
        try:
            save_dataframe_to_sqlite_with_sanitization(df, table_name=table_name)
            print(f"DEBUG: {table_name}をデータベースに保存しました。")
        except Exception as e:
            print(f"DEBUG: {table_name}の保存に失敗しました: {e}")

    # 各変換データを標準化
    standardized_dfs = {}
    for table_name, df in transformed_dfs.items():
        standardized_df = standardize_data(df, table_name_prefix=table_name)
        standardized_dfs[table_name] = standardized_df

    # 最後に適用した変換のデータを表示用に設定
    last_transformation = list(transformed_dfs.keys())[-1]
    e.page.app_data.standardized_df = standardized_dfs[last_transformation]  # type: ignore

    # 標準化後のデータテーブルを更新
    update_standardized_data_table(e.page, e.page.app_data.standardized_df)  # type: ignore
    e.page.update()
    print("DEBUG: すべての変換と標準化が完了しました。")


# カラム管理ページに遷移する関数 (グローバル関数として移動)
def go_to_column_management(e: ft.ControlEvent):  # イベントオブジェクトeを受け取る
    print("DEBUG: go_to_column_management関数が呼び出されました。")
    # merged_dfはpage.app_data経由でアクセス
    print(f"DEBUG: go_to_column_management - Current app_data.merged_df (before go): {type(e.page.app_data.merged_df)}, empty: {e.page.app_data.merged_df.empty if isinstance(e.page.app_data.merged_df, pd.DataFrame) else 'N/A'}")  # type: ignore
    if e.page.app_data.merged_df is not None and not e.page.app_data.merged_df.empty:  # type: ignore
        print("DEBUG: merged_dfは空ではないため、カラム管理ページへ遷移します。")
        e.page.go("/column-management")  # e.pageを使用
        print("DEBUG: go_to_column_management - カラム管理ページへ遷移します。")
    else:
        snack = ft.SnackBar(
            content=ft.Text("データが読み込まれていないため、カラム管理を開けません。")
        )
        e.page.add(snack)  # e.pageを使用
        snack.open = True
        e.page.update()  # e.pageを使用
        print("DEBUG: go_to_column_management - データが読み込まれていません。")


class AppData:
    def __init__(self):
        self.merged_df: pd.DataFrame | None = None
        self.saved_df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None  # オリジナルのデータフレームを保持
        self.standardized_df: pd.DataFrame | None = (
            None  # 標準化後のデータフレームを保持
        )


# CSVファイルの初期読み込み (グローバル関数として移動)
def load_initial_csv_data(page: ft.Page):  # pageを引数として受け取る
    IndicatorFolderPath = "Indicator"
    EconomicIndicatorFile = "各種経済指標.csv"
    EconomicIndicatorFilePath = os.path.join(IndicatorFolderPath, EconomicIndicatorFile)
    pdFile = "v_pd.csv"
    pdFilePath = os.path.join(IndicatorFolderPath, pdFile)

    try:
        # 各種経済指標の読み込み
        df1 = pd.read_csv(EconomicIndicatorFilePath)
        print(f"df1読み込み完了。シェイプ: {df1.shape}, カラム: {list(df1.columns)}")
        df1_conveted = df1[df1["地域"] == "全国"]
        print(
            f"df1_conveted (地域フィルタ後)。シェイプ: {df1_conveted.shape}, カラム: {list(df1_conveted.columns)}"
        )
        df1_conveted["kijyunnengetu"] = pd.to_datetime(
            df1["時点"], format="%Y年%m月"
        ).dt.strftime("%Y%m")
        print(
            f"df1_conveted (日付変換後)。シェイプ: {df1_conveted.shape}, カラム: {list(df1_conveted.columns)}"
        )

        # PDデータの読み込み
        df2 = pd.read_csv(pdFilePath, dtype={"kijyunnengetu": str})
        print(f"df2読み込み完了。シェイプ: {df2.shape}, カラム: {list(df2.columns)}")

        # 両方のデータフレームを結合してoriginal_dfを作成
        merged_original = pd.merge(df1_conveted, df2, on="kijyunnengetu", how="inner")

        # original_dfとして設定（kijyunnengetuを含める）
        page.app_data.original_df = merged_original.copy()  # type: ignore

        # データベースに保存
        try:
            save_dataframe_to_sqlite_with_sanitization(
                page.app_data.original_df, table_name="original_data"  # type: ignore
            )
            print("DEBUG: original_dfをデータベースに保存しました。")
        except Exception as e:
            print(f"DEBUG: original_dfの保存に失敗しました: {e}")

        print(f"original_df設定完了。シェイプ: {page.app_data.original_df.shape}, カラム: {list(page.app_data.original_df.columns)}")  # type: ignore

        # merged_dfの作成（inner結合）
        page.app_data.merged_df = pd.DataFrame(pd.merge(df1_conveted, df2, on="kijyunnengetu", how="inner"))  # type: ignore

        # merged_dfをデータベースに保存
        try:
            save_dataframe_to_sqlite_with_sanitization(
                page.app_data.merged_df, table_name="merged_data"  # type: ignore
            )
            print("DEBUG: merged_dfをデータベースに保存しました。")
        except Exception as e:
            print(f"DEBUG: merged_dfの保存に失敗しました: {e}")

        print(f"merged_df (結合後)。シェイプ: {page.app_data.merged_df.shape}, カラム: {list(page.app_data.merged_df.columns)}")  # type: ignore

        print("初期CSVロード完了。")
        print(f"merged_dfカラム: {list(page.app_data.merged_df.columns)}")  # type: ignore
        print(f"original_dfカラム: {list(page.app_data.original_df.columns)}")  # type: ignore

    except Exception as e:
        print(f"初期CSVデータのロード中にエラーが発生しました: {e}")
        page.app_data.merged_df = pd.DataFrame()  # type: ignore
        page.app_data.original_df = pd.DataFrame()  # type: ignore
        print("CSVロード失敗。merged_dfとoriginal_dfを空のDataFrameに設定しました。")


# データベースからの初期読み込み (グローバル関数として移動)
def initialize_data_from_db_or_csv(page: ft.Page):
    """データベースまたはCSVからデータを初期化する"""
    try:
        # まずmerged_dataテーブルを確認
        merged_data = read_dataframe_from_sqlite("merged_data")

        if merged_data is not None and not merged_data.empty:
            # merged_dataが存在する場合はそれを使用
            page.app_data.merged_df = pd.DataFrame(merged_data)  # type: ignore
            print("DEBUG: merged_dataテーブルからデータを読み込みました。")
            print(f"DEBUG: merged_dataのカラム: {list(merged_data.columns)}")
        else:
            # merged_dataが存在しない場合はCSVから読み込み
            print("DEBUG: merged_dataテーブルが存在しないため、CSVから読み込みます。")
            load_initial_csv_data(page)

            # 読み込んだデータをmerged_dataとして保存
            if page.app_data.merged_df is not None and not page.app_data.merged_df.empty:  # type: ignore
                try:
                    save_dataframe_to_sqlite_with_sanitization(
                        page.app_data.merged_df, table_name="merged_data"  # type: ignore
                    )
                    print("DEBUG: merged_dfをデータベースに保存しました。")
                    print(f"DEBUG: 保存したmerged_dfのカラム: {list(page.app_data.merged_df.columns)}")  # type: ignore
                except Exception as e:
                    print(f"DEBUG: merged_dfの保存に失敗しました: {e}")

    except Exception as e:
        print(f"DEBUG: データの初期化中にエラーが発生しました: {e}")
        print("データベースの初期化に失敗したため、CSVから読み込みます。")
        load_initial_csv_data(page)


def data_load_page(page: ft.Page):
    page.title = "データベース整備ツール"

    global checkbox_states, data_table, standardized_data_table

    print(f"DEBUG: data_load_page関数が呼び出されました。")

    # 変換タイプの選択肢
    transformation_types = {
        "none": "変換なし",
        "log": "対数変換",
        "diff": "差分化",
        "log_diff": "対数変換後に差分化",
    }

    # ヘッダーとデータ行の表示用コンポーネント
    header_row = ft.Row(
        controls=[],
        alignment=ft.MainAxisAlignment.START,
    )
    data_listview = ft.ListView(
        expand=True,
        spacing=0,
        padding=0,
    )

    # テーブル全体を包むスクロール可能なコンテナ
    scrollable_container = ft.Container(
        content=ft.Column(
            [
                # ヘッダー行（固定）
                ft.Container(
                    content=header_row,
                    border=ft.border.only(
                        left=ft.border.BorderSide(1, ft.Colors.GREY_300),
                        right=ft.border.BorderSide(1, ft.Colors.GREY_300),
                        top=ft.border.BorderSide(1, ft.Colors.GREY_300),
                    ),
                    bgcolor=ft.Colors.GREY_100,  # ヘッダーの背景色
                ),
                # データ行（スクロール可能）
                ft.Container(
                    content=data_listview,
                    border=ft.border.only(
                        left=ft.border.BorderSide(1, ft.Colors.GREY_300),
                        right=ft.border.BorderSide(1, ft.Colors.GREY_300),
                        bottom=ft.border.BorderSide(1, ft.Colors.GREY_300),
                    ),
                    expand=True,
                ),
            ],
            spacing=0,
            expand=True,
        ),
        expand=True,
    )

    # データ表示用のコンテナ構造を更新
    table_container = ft.Container(
        content=ft.Row(
            [scrollable_container],
            scroll=ft.ScrollMode.AUTO,  # 横スクロールを有効化
        ),
        expand=True,
    )

    # コンテンツ表示用のコンテナ（先に定義）
    content_container = ft.Container(
        content=table_container,
        bgcolor=ft.Colors.WHITE,
        expand=True,
    )

    # データテーブルを更新する関数
    def update_data_table_content(df: pd.DataFrame):
        # カラム幅の設定（固定幅）
        column_width = 200  # カラム幅を200pxに増加

        # テーブルの全体幅を計算（余裕を持たせる）
        total_width = max(column_width * len(df.columns) + 50, 800)  # 最小幅800pxを確保

        # スクロール可能なコンテナの幅を設定
        scrollable_container.width = total_width

        # ヘッダー行の更新
        header_controls = []
        for col in df.columns:
            header_controls.append(
                ft.Container(
                    content=ft.Text(
                        col,
                        weight=ft.FontWeight.BOLD,
                        size=14,  # フォントサイズを調整
                    ),
                    width=column_width,
                    padding=ft.padding.only(
                        left=15, right=15, top=8, bottom=8
                    ),  # パディングを増加
                    border=ft.border.only(
                        right=ft.border.BorderSide(1, ft.Colors.GREY_300)
                    ),
                )
            )
        header_row.controls = header_controls

        # データ行の更新
        data_listview.controls.clear()
        for row in df.itertuples(index=False):
            data_row = ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Text(
                            str(value),
                            size=14,  # フォントサイズを調整
                        ),
                        width=column_width,
                        padding=ft.padding.only(
                            left=15, right=15, top=8, bottom=8
                        ),  # パディングを増加
                        border=ft.border.only(
                            right=ft.border.BorderSide(1, ft.Colors.GREY_300)
                        ),
                    )
                    for value in row
                ],
                alignment=ft.MainAxisAlignment.START,
            )
            # 交互の行の背景色を設定
            if len(data_listview.controls) % 2 == 1:
                data_row.bgcolor = ft.Colors.GREY_50
            data_listview.controls.append(data_row)

        page.update()

    # 変換タイプ選択用ドロップダウン
    transformation_dropdown = ft.Dropdown(
        label="変換タイプ",
        options=[
            ft.dropdown.Option(key, value)
            for key, value in transformation_types.items()
        ],
        value="none",
        width=200,
        on_change=lambda e: update_displayed_data(e.page),
    )

    # 標準化スイッチ
    standardization_switch = ft.Switch(
        label="標準化",
        value=False,
        on_change=lambda e: update_displayed_data(e.page),
    )

    def get_selected_dataframe():
        """選択された設定に基づいて表示するデータフレームを取得"""
        transformation = transformation_dropdown.value
        is_standardized = standardization_switch.value

        # データベースからデータを読み込む
        try:
            if transformation == "none":
                # 変換なしの場合
                if is_standardized:
                    # 標準化データを表示
                    df = read_dataframe_from_sqlite("merged_standardized")
                    if df is not None and not df.empty:
                        print(
                            f"DEBUG: merged_standardizedを表示します。カラム: {list(df.columns)}"
                        )
                        return pd.DataFrame(df)

                # 標準化なしのデータを表示
                df = read_dataframe_from_sqlite("merged_data")
                if df is not None and not df.empty:
                    print(f"DEBUG: merged_dataを表示します。カラム: {list(df.columns)}")
                    return pd.DataFrame(df)

                # merged_dataが存在しない場合はoriginal_dataを表示
                df = read_dataframe_from_sqlite("original_data")
                if df is not None and not df.empty:
                    print(
                        f"DEBUG: original_dataを表示します。カラム: {list(df.columns)}"
                    )
                    return pd.DataFrame(df)
                print("DEBUG: 表示するデータが存在しません。")
                return None

            # 変換データの属性名を構築
            table_name = f"{transformation}_data"
            if is_standardized:
                table_name += "_standardized"

            # データベースからデータを読み込む
            df = read_dataframe_from_sqlite(table_name)
            if df is not None and not df.empty:
                print(f"DEBUG: {table_name}を表示します。カラム: {list(df.columns)}")
                return pd.DataFrame(df)
            print(f"DEBUG: {table_name}が存在しません。")
            return None

        except Exception as e:
            print(f"DEBUG: データの読み込みに失敗しました: {e}")
            return None

    def update_displayed_data(page: ft.Page):
        """選択された設定に基づいて表示データを更新"""
        # 標準化スイッチの制御を先に行う
        if transformation_dropdown.value == "none":
            standardized_df = read_dataframe_from_sqlite("merged_standardized")
            standardization_switch.disabled = (
                standardized_df is None or standardized_df.empty
            )
        else:
            table_name = f"{transformation_dropdown.value}_data_standardized"
            standardized_df = read_dataframe_from_sqlite(table_name)
            standardization_switch.disabled = (
                standardized_df is None or standardized_df.empty
            )

        # データの表示を更新
        df = get_selected_dataframe()

        if df is not None and not df.empty:
            # データテーブルの更新
            update_data_table_content(df)
            content_container.content = table_container
            print(f"DEBUG: データテーブルを更新しました。カラム: {list(df.columns)}")
        else:
            # データが存在しない場合
            header_row.controls = [
                ft.Container(
                    content=ft.Text("No Data", weight=ft.FontWeight.BOLD),
                    padding=ft.padding.only(left=10, right=10, top=5, bottom=5),
                )
            ]
            data_listview.controls.clear()
            content_container.content = table_container
            print("DEBUG: データが存在しないため、テーブルをクリアしました。")

        page.update()

    # 初期表示の更新
    update_displayed_data(page)

    # Initialize buttons locally within the data_load_page function
    column_management_button_local = ft.ElevatedButton(
        "カラム名の変更と削除",
        disabled=True,  # 初期状態は無効
        on_click=go_to_column_management,
    )

    # ボタンの有効/無効状態を更新
    is_data_loaded = page.app_data.merged_df is not None and not page.app_data.merged_df.empty  # type: ignore
    column_management_button_local.disabled = not is_data_loaded
    print(f"DEBUG: data_load_page - is_data_loaded: {is_data_loaded}")
    print(
        f"DEBUG: data_load_page - column_management_button_local.disabled: {column_management_button_local.disabled}"
    )

    # 初期化処理
    initialize_user_data_database()

    # データロードの状態を確認し、必要に応じてメッセージを表示
    if page.app_data.merged_df is None or page.app_data.merged_df.empty:  # type: ignore
        snack = ft.SnackBar(
            content=ft.Text(
                "データロードに失敗しました。CSVファイルまたはデータベースを確認してください。"
            ),
            open=True,
        )
        page.add(snack)
        page.update()

    # DataTableのカラムが空の場合はダミーカラムをセット
    if not data_listview.controls:
        data_listview.controls = [
            ft.Container(
                content=ft.Text("No Data"),
                padding=ft.padding.only(left=10, right=10, top=5, bottom=5),
            )
        ]
    if not standardized_data_table.columns:
        standardized_data_table.columns = [ft.DataColumn(ft.Text("No Data"))]
        standardized_data_table.rows = []

    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        column_management_button_local,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                    expand=0.5,
                ),
                ft.Divider(),
                ft.Row(
                    [
                        transformation_dropdown,
                        standardization_switch,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                    expand=0.5,
                ),
                ft.Divider(),
                content_container,  # シンプルにcontent_containerを使用
            ],
            expand=True,
        ),
        expand=True,
    )

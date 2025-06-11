import os
import flet as ft
import pandas as pd
import numpy as np

# import sqlite3 # db/database.pyからインポートするため削除
from sklearn.preprocessing import StandardScaler
from .page_column_management import column_management_page
from db.database import (  # db.databaseからインポート
    initialize_user_data_database,
    read_dataframe_from_sqlite,
    save_dataframe_to_sqlite_with_sanitization,
    sanitize_column_name,  # これも一応インポートしておく
)

# SQLiteデータベースファイルのパス
# DATABASE_FILE = "database.db" # db.databaseで管理するため削除

# グローバル変数
# merged_dfとsaved_dfはAppDataクラスで管理するため削除
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


# SQLiteデータベースを初期化する関数
# initialize_databaseはdb/database.pyにinitialize_user_data_databaseとして移動
# def initialize_database():
#     conn = sqlite3.connect(DATABASE_FILE)
#     cursor = conn.cursor()
#     cursor.execute(
#         """
#         CREATE TABLE IF NOT EXISTS data (
#             kijyunnengetu TEXT,
#             column1 REAL,
#             column2 REAL
#         )
#     """
#     )
#     conn.commit()
#     conn.close()


# SQLiteデータベースからデータを読み込む関数
# load_databaseはdb/database.pyにread_dataframe_from_sqliteとして移動
# def load_database():
#     conn = sqlite3.connect(DATABASE_FILE)
#     try:
#         df = pd.read_sql("SELECT * FROM data", conn)
#     except sqlite3.Error:
#         df = None
#     finally:
#         conn.close()
#     return df


# SQLiteデータベースにデータを保存する関数
# sanitize_column_nameとsave_databaseはdb/database.pyに移動
# def sanitize_column_name(col_name: str) -> str:
#     """
#     カラム名をSQLiteで使用可能な形式に変換する
#     """
#     # 特殊文字をアンダースコアに置換（括弧、スペース、その他の特殊文字）
#     sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in str(col_name))
#     # 連続するアンダースコアを1つに
#     while "__" in sanitized:
#         sanitized = sanitized.replace("__", "_")
#     # 先頭と末尾のアンダースコアを削除
#     sanitized = sanitized.strip("_")
#     # 先頭が数字の場合は'col_'を付加
#     if sanitized and sanitized[0].isdigit():  # sanitizedが空でないことを確認
#         sanitized = "col_" + sanitized
#     # 空文字列の場合は'col'を返す
#     if not sanitized:
#         sanitized = "col"
#     return sanitized


# def save_database(df):
#     conn = sqlite3.connect(DATABASE_FILE)
#     try:
#         # カラム名をSQLiteで使用可能な形式に変換
#         df = df.copy()
#         original_columns = list(df.columns)
#         df.columns = [sanitize_column_name(col) for col in df.columns]
#         print("変換前のカラム名:", original_columns)
#         print("変換後のカラム名:", list(df.columns))

#         # テーブルが存在する場合は削除
#         conn.execute("DROP TABLE IF EXISTS data")
#         # 新しいテーブルを作成してデータを保存
#         df.to_sql("data", conn, if_exists="replace", index=False)
#         conn.commit()
#     except Exception as e:
#         conn.rollback()
#         raise e
#     finally:
#         conn.close()


# 対数変換、差分化、対数変換後に差分化を適用する関数
def apply_transformations(df, transformations):
    # kijyunnengetuカラムを保持
    result_df = df.copy()
    kijyunnengetu_col = None
    if "kijyunnengetu" in df.columns:
        kijyunnengetu_col = df["kijyunnengetu"]
        result_df = result_df.drop(columns=["kijyunnengetu"])

    # 変換を適用
    for col in result_df.columns:
        if "対数変換" in transformations:
            result_df[f"{col}_log"] = np.log1p(result_df[col])
        if "差分化" in transformations:
            result_df[f"{col}_diff"] = result_df[col].diff()
        if "対数変換後に差分化" in transformations:
            result_df[f"{col}_log_diff"] = np.log1p(result_df[col]).diff()

    # kijyunnengetuカラムを戻す
    if kijyunnengetu_col is not None:
        result_df["kijyunnengetu"] = kijyunnengetu_col

    return result_df


# 標準化を適用する関数
def standardize_data(df):
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

    return scaled_df


# 標準化ボタンのクリックイベント
def standardize_data_button_click(e: ft.ControlEvent):
    print(f"DEBUG: standardize_data_button_click関数が呼び出されました。")
    print(f"DEBUG: standardize_data_button_click - Current app_data.merged_df (before check): {type(e.page.app_data.merged_df)}, empty: {e.page.app_data.merged_df.empty if isinstance(e.page.app_data.merged_df, pd.DataFrame) else 'N/A'}")  # type: ignore
    if e.page.app_data.merged_df is None or e.page.app_data.merged_df.empty:  # type: ignore
        snack = ft.SnackBar(content=ft.Text("データが読み込まれていません。"))
        e.page.add(snack)
        snack.open = True
        e.page.update()
        print("DEBUG: standardiz_data_button_click - データが読み込まれていません。")
        return

    transformations = ["対数変換", "差分化", "対数変換後に差分化"]
    # merged_df のコピーに対して変換を適用
    transformed_df = apply_transformations(
        e.page.app_data.merged_df.copy(), transformations  # type: ignore
    )

    # 変換されたデータフレームに対して標準化を適用
    standardized_df = standardize_data(transformed_df)

    # 標準化後のデータをAppDataに保存
    e.page.app_data.standardized_df = standardized_df  # type: ignore

    # 標準化後のデータをデータベースに保存
    try:
        save_dataframe_to_sqlite_with_sanitization(
            standardized_df, table_name="standardized_data"
        )
        print("DEBUG: 標準化後のデータをデータベースに保存しました。")
    except Exception as e:
        print(f"DEBUG: 標準化後のデータの保存に失敗しました: {e}")
        snack = ft.SnackBar(content=ft.Text("標準化後のデータの保存に失敗しました。"))
        e.page.add(snack)
        snack.open = True
        e.page.update()
        return

    # 標準化後のデータテーブルを更新
    update_standardized_data_table(e.page, standardized_df)
    e.page.update()
    print("DEBUG: 標準化が完了しました。")


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
        merged_original = pd.merge(df1_conveted, df2, on="kijyunnengetu", how="outer")

        # kijyunnengetuを除くカラムをoriginal_dfとして設定
        original_columns = [
            col for col in merged_original.columns if col != "kijyunnengetu"
        ]
        page.app_data.original_df = merged_original[original_columns].copy()  # type: ignore

        print(f"original_df設定完了。シェイプ: {page.app_data.original_df.shape}, カラム: {list(page.app_data.original_df.columns)}")  # type: ignore

        # merged_dfの作成（inner結合）
        page.app_data.merged_df = pd.DataFrame(pd.merge(df1_conveted, df2, on="kijyunnengetu", how="inner"))  # type: ignore
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
    # merged_dfはpage.app_data経由でアクセス
    db_data = read_dataframe_from_sqlite("data")  # db.databaseから読み込む
    if db_data is not None and not db_data.empty:
        page.app_data.merged_df = pd.DataFrame(db_data)  # type: ignore

        # 標準化後のデータも読み込む
        try:
            standardized_data = read_dataframe_from_sqlite("standardized_data")  # type: ignore
            if standardized_data is not None and not standardized_data.empty:
                page.app_data.standardized_df = pd.DataFrame(standardized_data)  # type: ignore
                # 標準化後のデータテーブルを更新
                update_standardized_data_table(page, page.app_data.standardized_df)  # type: ignore
        except Exception as e:
            print(f"DEBUG: standardized_dataテーブルの読み込みに失敗しました: {e}")
            # standardized_dataテーブルが存在しない場合は、standardized_dfをNoneに設定
            page.app_data.standardized_df = None  # type: ignore
            # 標準化後のデータテーブルをクリア
            update_standardized_data_table(page, pd.DataFrame())

        # データベースから読み込んだ場合、original_dfは初期CSVから作成
        IndicatorFolderPath = "Indicator"
        EconomicIndicatorFile = "各種経済指標.csv"
        EconomicIndicatorFilePath = os.path.join(
            IndicatorFolderPath, EconomicIndicatorFile
        )
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

            # 両方のデータフレームを結合してoriginal_dfを作成
            merged_original = pd.merge(
                df1_conveted, df2, on="kijyunnengetu", how="outer"
            )

            # kijyunnengetuを除くカラムをoriginal_dfとして設定
            original_columns = [
                col for col in merged_original.columns if col != "kijyunnengetu"
            ]
            page.app_data.original_df = merged_original[original_columns].copy()  # type: ignore

            print(f"データベースからデータロード完了。")
            print(f"merged_dfシェイプ: {page.app_data.merged_df.shape}, カラム: {list(page.app_data.merged_df.columns)}")  # type: ignore
            print(f"original_dfシェイプ: {page.app_data.original_df.shape}, カラム: {list(page.app_data.original_df.columns)}")  # type: ignore
        except Exception as e:
            print(f"初期CSVの読み込みに失敗しました: {e}")
            # CSVの読み込みに失敗した場合、merged_dfからoriginal_dfを作成
            if "kijyunnengetu" in db_data.columns:
                original_columns = [
                    col for col in db_data.columns if col != "kijyunnengetu"
                ]
                page.app_data.original_df = pd.DataFrame(db_data[original_columns])  # type: ignore
            else:
                page.app_data.original_df = pd.DataFrame(db_data)  # type: ignore
            print(f"merged_dfからoriginal_dfを作成しました。")
            print(f"merged_dfシェイプ: {page.app_data.merged_df.shape}, カラム: {list(page.app_data.merged_df.columns)}")  # type: ignore
            print(f"original_dfシェイプ: {page.app_data.original_df.shape}, カラム: {list(page.app_data.original_df.columns)}")  # type: ignore
    else:
        print("データベースにデータがないため、初期CSVをロードします。")
        load_initial_csv_data(page)  # pageを渡す


# FletアプリケーションのUIを構築する関数
def data_load_page(page: ft.Page):
    page.title = "データベース整備ツール"

    global checkbox_states, data_table, standardized_data_table

    print(f"DEBUG: data_load_page関数が呼び出されました。")
    print(
        f"DEBUG: data_load_page - page.app_data.merged_df type: {type(page.app_data.merged_df)}, is None: {page.app_data.merged_df is None}, is_empty: {page.app_data.merged_df.empty if isinstance(page.app_data.merged_df, pd.DataFrame) else 'N/A'}"
    )

    # Initialize buttons locally within the data_load_page function
    column_management_button_local = ft.ElevatedButton(
        "カラム名の変更と削除",
        disabled=True,  # 初期状態は無効
        on_click=go_to_column_management,
    )
    standardize_button_local = ft.ElevatedButton(
        "標準化を適用",
        disabled=True,  # 初期状態は無効
        on_click=standardize_data_button_click,
    )

    # ボタンの有効/無効状態を更新
    is_data_loaded = page.app_data.merged_df is not None and not page.app_data.merged_df.empty  # type: ignore
    column_management_button_local.disabled = not is_data_loaded
    standardize_button_local.disabled = not is_data_loaded
    print(f"DEBUG: data_load_page - is_data_loaded: {is_data_loaded}")
    print(
        f"DEBUG: data_load_page - column_management_button_local.disabled: {column_management_button_local.disabled}"
    )
    print(
        f"DEBUG: data_load_page - standardize_button_local.disabled: {standardize_button_local.disabled}"
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
    if not data_table.columns:
        data_table.columns = [ft.DataColumn(ft.Text("No Data"))]
        data_table.rows = []
    if not standardized_data_table.columns:
        standardized_data_table.columns = [ft.DataColumn(ft.Text("No Data"))]
        standardized_data_table.rows = []

    # タブ切り替え用の関数
    def change_tab(e):
        tabs.selected_index = e.control.selected_index
        # タブ切り替え時にコンテンツを更新
        if tabs.selected_index == 0:
            content_container.content = data_table
        else:
            content_container.content = standardized_data_table
        page.update()

    # コンテンツ表示用のコンテナ
    content_container = ft.Container(
        content=data_table,  # 初期表示は元データ
        width=1200,
        height=300,
        bgcolor=ft.Colors.WHITE,
        border=ft.border.all(1, ft.Colors.GREY_300),
        alignment=ft.alignment.top_left,
        expand=False,
    )

    # タブの作成
    tabs = ft.Tabs(
        selected_index=0,
        on_change=change_tab,
        tabs=[
            ft.Tab(text="元データ"),
            ft.Tab(text="標準化後のデータ"),
        ],
    )

    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        column_management_button_local,
                        standardize_button_local,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                ),
                ft.Divider(),
                tabs,
                content_container,  # タブの下にコンテンツを表示
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        expand=True,
    )

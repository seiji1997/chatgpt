Python is a high-level, dynamically typed programming language known for its readability and simplicity. Here's an overview of Python's grammar and some practical methods and concepts:<br>

### Python Grammar:
Comments: Comments start with a hash symbol (#) and are ignored by the Python interpreter. They are used for adding explanations and notes within your code.<br>

### python
```python
# This is a comment
```
Variables and Data Types:<br>
Variables are used to store data. In Python, you don't need to declare the type of a variable explicitly; it's determined dynamically.<br>

Common data types include integers, floating-point numbers, strings, lists, tuples, dictionaries, sets, and more.<br>

### python
```python
x = 42  # Integer
y = 3.14  # Float
name = "John"  # String
my_list = [1, 2, 3]  # List
my_dict = {"key": "value"}  # Dictionary
```

Indentation:<br>
Python uses indentation to define code blocks (instead of curly braces or other delimiters). Indentation must be consistent within a block.<br>

### python
```python
if condition:
    # Indented block
    do_something()
else:
    # Another indented block
    do_something_else()
```

Conditionals:<br>
if, elif, and else are used for conditional statements.<br>

### python
```python
if condition:
    do_something()
elif another_condition:
    do_something_else()
else:
    do_another_thing()
```

Loops:<br>
for and while loops are used for iteration.<br>

### python
```python
for item in iterable:
    process_item()

while condition:
    do_something()
```
Functions:<br>
Functions are defined using the def keyword and can accept arguments and return values.<br>

### python
```python
def my_function(arg1, arg2):
    # Function body
    return result
```

Lists and Indexing:<br>
Lists are ordered collections of items. Indexing starts from 0.<br>
```python
my_list = [1, 2, 3, 4, 5]
first_element = my_list[0]
```

Strings:<br>
Strings are sequences of characters. You can perform various string operations, like slicing and concatenation.<br>
```python
my_string = "Hello, world!"
substring = my_string[0:5]  # Slicing
concatenated = "Hello" + " " + "world!"  # Concatenation
```
Dictionaries:<br>

Dictionaries are collections of key-value pairs.<br>
```python
my_dict = {"key1": "value1", "key2": "value2"}
value = my_dict["key1"]
```

Classes and Objects:<br>

Python supports object-oriented programming. You can define classes and create objects from them.<br>
```python
class MyClass:
    def __init__(self, param):
        self.data = param

obj = MyClass("Some data")
```
Useful Python Methods and Concepts:<br>
Built-in Functions: Python provides a wide range of built-in functions, such as len(), type(), print(), range(), and input().<br>

List Comprehensions: A concise way to create lists based on existing lists.<br>

```python
squares = [x**2 for x in range(1, 6)]
```
Slicing: Used to extract parts of sequences like strings and lists.<br>

```python
text = "Hello, world!"
sub_text = text[0:5]  # Extracts "Hello"
```
Object-Oriented Programming (OOP): Python supports classes, objects, and inheritance.<br>
Modules and Packages: Python's modular structure allows you to organize code into reusable modules and packages.<br>
Exception Handling: Using try, except, finally, and raise to handle and raise exceptions.<br>
File Handling: Reading and writing files with functions like open(), read(), write(), and close().<br>
Importing Libraries: Importing external libraries and modules using import.<br>
Lambda Functions: Anonymous functions created using the lambda keyword.<br>
Generators: Special functions used to create iterators, often using the yield keyword.<br>
Decorators: Functions that modify the behavior of other functions.<br>
This is just an overview of Python's grammar and some practical methods and concepts. Python's extensive standard library offers many more features and functionalities for various tasks. To use these features effectively, you should explore Python's official documentation and practice writing code.<br>


# simple ver.
```python
# app_roic_min.py
# ------------------------------------------------------------
# ROICミニ分析アプリ（Snowflake + Cortex + Streamlit + Plotly）
# 目的:
#  - 日本語で質問 → Cortexで SELECT SQL を生成 → 実行 → 数表/簡易グラフを表示
#  - 余計な機能は削除（最小限の会話ドリブン分析UI）
# 前提:
#  - FY_Q は 'Q1'～'Q4' の文字列をそのまま使用
#  - Streamlit in Snowflake or ローカルのどちらでも実行可
# 必要パッケージ: streamlit, pandas, plotly, snowflake-snowpark-python
# 接続情報: st.secrets もしくは環境変数（アカウント/ユーザ等）
# ------------------------------------------------------------

import os
import re
import json
import textwrap
import pandas as pd
import plotly.express as px
import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException

# =========================
# Page / Sidebar
# =========================
st.set_page_config(page_title="ROIC Mini — Snowflake + Cortex", layout="wide")
st.title("ROIC Mini — Snowflake + Cortex")
st.caption("日本語で聞く → SQL自動生成 → 実行 → 結果を表とPlotlyで確認（最小構成）")

with st.sidebar:
    st.header("接続 & 対象")
    target_table = st.text_input(
        "対象 (DB.SCHEMA.TABLE / VIEW)",
        value=st.secrets.get("roic_table", "ROIC_DEMO.ROIC_TABLES.ROIC_BASE"),
        help="解析対象の完全修飾名",
    )
    model_name = st.text_input(
        "Cortexモデル", value=st.secrets.get("cortex_model", "mistral-large")
    )
    run_guard = st.checkbox("Cortex Guard（推奨ON）", value=True)
    st.markdown("---")
    st.caption(
        "FY_Q は 'Q1'〜'Q4' をそのまま使います。列名が曖昧な場合はプロンプトで明示してください。"
    )


# =========================
# Snowflake Session
# =========================
@st.cache_resource(show_spinner=False)
def get_session():
    """
    Snowflakeセッションを生成。st.secrets または環境変数を使用。
    必須: account, user, password, role, warehouse, database, schema
    """
    cfg = {
        "account": st.secrets.get("account", os.getenv("SNOWFLAKE_ACCOUNT")),
        "user": st.secrets.get("user", os.getenv("SNOWFLAKE_USER")),
        "password": st.secrets.get("password", os.getenv("SNOWFLAKE_PASSWORD")),
        "role": st.secrets.get("role", os.getenv("SNOWFLAKE_ROLE")),
        "warehouse": st.secrets.get("warehouse", os.getenv("SNOWFLAKE_WAREHOUSE")),
        "database": st.secrets.get("database", os.getenv("SNOWFLAKE_DATABASE")),
        "schema": st.secrets.get("schema", os.getenv("SNOWFLAKE_SCHEMA")),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Snowflake settings: {missing}")
    return Session.builder.configs(cfg).create()


def run_sql_df(session: Session, sql: str) -> pd.DataFrame:
    """SnowflakeでSQLを実行してpandas DataFrameを返す。"""
    return session.sql(sql).to_pandas()


def sanitize_identifier(name: str) -> str:
    """DB.SCHEMA.TABLE 形式を想定。大文字化＋許可文字だけ残す簡易サニタイズ。"""
    return re.sub(r'[^A-Z0-9_.$"]', "", (name or "").upper())


def extract_sql_from_text(text: str) -> str | None:
    """
    Cortex応答からSQL本体を抽出。
    - ```sql ... ``` ブロックがあればその中身
    - なければ 'select ...' から末尾まで
    """
    m = re.search(r"```sql\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";")
    m = re.search(r"(?is)(^|\n)\s*select\s+.+", text or "")
    return m.group(0).strip().rstrip(";") if m else None


def enforce_readonly_single_table(sql: str, allowed_table: str) -> bool:
    """
    安全ガード: SELECT限定 & FROMの参照先が対象テーブル/ビューのみ。
    - ざっくりチェック（正規表現ベース）
    """
    if not re.match(r"(?is)^\s*select\b", sql or ""):
        return False
    froms = re.findall(r"(?is)\bfrom\b\s+([A-Z0-9_.$\"]+)", (sql or "").upper())
    return all(f == allowed_table.upper() for f in froms) if froms else True


def cortex_complete_messages(
    session: Session, messages, model: str, guard: bool = True
) -> str:
    """
    Cortex COMPLETE を呼び出し、messages（Chat形式）に対する応答テキストを取得。
    guard=True の場合は簡易フィルタを有効化。
    """
    options = {"temperature": 0.1}
    if guard:
        options["guard"] = "ON"
    jopt = json.dumps(options, ensure_ascii=False)
    jmsg = json.dumps(messages, ensure_ascii=False)
    sql = f"""
      SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{model}',
        parse_json('{jmsg}'),
        parse_json('{jopt}')
      ) AS response
    """
    try:
        df = run_sql_df(session, sql)
        return df.iloc[0, 0]
    except SnowparkSQLException as e:
        return f"[CORTEX ERROR] {e}"


# =========================
# System Prompts（最小）
# =========================
def system_sql_prompt(qualified_table: str) -> str:
    """
    SQL生成用の最小Systemプロンプト。
    FY_Q は 'Q1'〜'Q4' の文字列をそのまま扱うことを明示。
    """
    return textwrap.dedent(
        f"""
    あなたは Snowflake 上の分析者です。ユーザーの日本語質問に対して
    **実行可能な Snowflake SQL（SELECTのみ）** を1本だけ作成します。

    前提:
    - 解析対象テーブル: {qualified_table}
    - FY_Q は 'Q1'〜'Q4' の文字列をそのまま扱う
    - 出力は ```sql ... ``` の1ブロックのみ
    - DML/DDLは禁止（SELECTのみ）
    - 並べ替えや LIMIT で読みやすい出力にしてよい
    """
    ).strip()


def system_summary_prompt() -> str:
    """生成SQLの簡易サマリ（日本語、箇条書き）を作成するプロンプト。"""
    return textwrap.dedent(
        """
    あなたは SQL の説明者です。
    ユーザーの質問と生成されたSQLを受け取り、日本語で**箇条書きの短い要約**を返してください。
    含める要素:
    - どの列で絞り込み（WHERE）
    - どの軸で集計/比較（GROUP BY, 窓関数）
    - 並べ替え/上位抽出（ORDER BY, LIMIT）
    """
    ).strip()


# =========================
# App State
# =========================
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # [{user, sql?, explanation?, error?}]

# =========================
# Connect
# =========================
try:
    session = get_session()
    session.sql("select 1").collect()
    st.sidebar.success("Snowflake接続OK ✅")
except Exception as e:
    st.sidebar.error(f"接続エラー: {e}")
    st.stop()

# =========================
# Chat UI → SQL生成 → 要約
# =========================
st.subheader("チャットで質問 → SQL生成 → 要約 → 実行")

with st.expander("直近の履歴", expanded=False):
    if not st.session_state["chat"]:
        st.write("（まだ履歴はありません）")
    else:
        for i, turn in enumerate(st.session_state["chat"], 1):
            st.markdown(f"**Q{i}（あなた）:** {turn['user']}")
            if "sql" in turn:
                st.markdown("**生成SQL**")
                st.code(turn["sql"], language="sql")
            if "explanation" in turn:
                st.markdown("**要約**")
                st.write(turn["explanation"])
            if "error" in turn:
                st.error(turn["error"])

user_prompt = st.text_area(
    "日本語で質問（例：2024年度の事業部別ROICの上位10件を表示）",
    height=100,
    placeholder="例：FY=2024 の 'Q1'～'Q4'で、事業部別のROICを平均し、上位10件を見せて",
)

c1, c2 = st.columns([1, 1])
with c1:
    do_generate = st.button("SQL生成（Cortex）")
with c2:
    if st.button("履歴クリア"):
        st.session_state["chat"] = []
        st.experimental_rerun()

qualified = sanitize_identifier(target_table)

if do_generate and user_prompt.strip():
    # SQL生成
    messages_sql = [{"role": "system", "content": system_sql_prompt(qualified)}]
    for turn in st.session_state["chat"]:
        messages_sql.append({"role": "user", "content": turn["user"]})
        if "sql" in turn:
            messages_sql.append(
                {"role": "assistant", "content": f"```sql\n{turn['sql']}\n```"}
            )
    messages_sql.append({"role": "user", "content": user_prompt})

    with st.spinner("CortexがSQLを作成中…"):
        raw_text = cortex_complete_messages(
            session, messages_sql, model_name, guard=run_guard
        )

    gen_sql = extract_sql_from_text(raw_text)
    new_turn = {"user": user_prompt}

    if not gen_sql:
        new_turn["error"] = f"SQLを抽出できませんでした。出力:\n{raw_text}"
    else:
        # サマリ生成
        messages_sum = [
            {"role": "system", "content": system_summary_prompt()},
            {
                "role": "user",
                "content": f"質問: {user_prompt}\n\nSQL:\n```sql\n{gen_sql}\n```",
            },
        ]
        with st.spinner("SQLの要約を作成中…"):
            summary = cortex_complete_messages(
                session, messages_sum, model_name, guard=run_guard
            )
        new_turn["sql"] = gen_sql
        new_turn["explanation"] = summary

    st.session_state["chat"].append(new_turn)
    st.experimental_rerun()

# =========================
# 実行 & 表/グラフ
# =========================
available_sqls = [t["sql"] for t in st.session_state["chat"] if "sql" in t]
selected_sql = st.selectbox(
    "実行するSQLを選択",
    available_sqls,
    index=len(available_sqls) - 1 if available_sqls else 0,
)

if st.button("選択SQLを実行"):
    if not selected_sql:
        st.warning("実行するSQLがありません。")
    elif not enforce_readonly_single_table(selected_sql, qualified):
        st.error("安全ガード：SELECT限定・対象テーブルのみ参照のSQLにしてください。")
    else:
        try:
            with st.spinner("SQL実行中…"):
                df = run_sql_df(session, selected_sql)
            st.success(f"Rows: {len(df):,}")
            st.dataframe(df, use_container_width=True)

            # 簡易Plotly
            if not df.empty:
                cols = list(df.columns)
                # 数値列の推定
                num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
                x_col = st.selectbox("X軸", cols, index=0)
                y_col = (
                    st.selectbox("Y軸（数値）", num_cols, index=0) if num_cols else None
                )
                chart_kind = st.radio("種類", ["棒", "折れ線"], horizontal=True)
                if y_col:
                    fig = (
                        px.bar(df, x=x_col, y=y_col)
                        if chart_kind == "棒"
                        else px.line(df, x=x_col, y=y_col)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("数値列が見つからなかったため、グラフは省略します。")
        except Exception as e:
            st.error(f"実行失敗: {e}")

# =========================
# Footer
# =========================
st.caption(
    "最小構成：チャット→SQL生成→要約→実行→表/簡易Plotlyのみ。FY_Qは'Q1'〜'Q4'をそのまま使用。"
)

```

# full ver.
```python
# app_roic_cortex.py
"""
app_roic_cortex.py — ROIC 対話分析アプリ（Snowflake + Cortex + Streamlit + Plotly）

【目的 / 価値】
- 会話(日本語) → SQL生成 → Warehouse実行 → 可視化 → What-If → ベース/シナリオ比較 → 寄与分解までを
  Snowflake 環境内で完結し、意思決定のスピードを上げる。
- 「説明責任」を確保（生成SQLの提示、日本語サマリー、差分表、係数検証）。

【前提 / データ仕様】
- 既定の解析対象ビュー: ROIC_DEMO.ROIC_TABLES.ROIC_BASE（UI から変更可）
- FY_Q は **元データの英数字大文字 'Q1' / 'Q2' / 'Q3' / 'Q4' をそのまま使用**（変換しない）
- 期間ラベル FYQ_LABEL = CONCAT('FY', FY, '-', FY_Q)
- ROIC の暫定定義（テンプレート・例）:
    NOPAT ≈ (PRICE - COST_PER_UNIT) * QUANTITY * (1 - 0.21)
    投下資本 ≈ (PRICE * QUANTITY) * 0.80
  → 厳密化はビュー側で置換可能（本アプリの SQL 断片は関数で一元化）

【セキュリティ / 信頼境界】
- 生成AIは SNOWFLAKE.CORTEX.COMPLETE() を用い **Snowflake 内**で実行（外部通信なし）
- 実行 SQL は **SELECT 限定**・**単一ビュー参照**の簡易ガードを実装
- 実行前に SQL を UI で明示、日本語サマリーで「どう絞ったか」を説明

【主な機能】
- スキーマ自動プロファイル／列役割推測（製品/事業部/期間/ROIC/都市）
- 初期表示：**最新期の事業部別 ROIC**（会議での起点）
- 日本語チャット → Cortex が SELECT SQL 生成 → 日本語サマリー
- 結果の可視化：数表／ピボット（マトリクス）／Plotly（線・棒）
- What-If：自然言語シナリオ／UIレバーパネル（価格・原価・人件費・物流費）→ 再計算
- ベース vs シナリオ比較：左右表示＋差分表＋比較サマリー＋係数検証
- 感度ヒートマップ（2レバー一括評価）
- ROICツリー（サンバースト／ツリーマップ）＋滝グラフ（寄与分解）
"""

# =========================
# Imports
# =========================
import os
import re
import json
import textwrap
import pandas as pd
import streamlit as st
import plotly.express as px

from typing import Optional, Iterable, Sequence
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException


# =========================
# 0) ページ設定 & エントリポイント
# =========================
def configure_page() -> None:
    """Streamlit ページメタ情報とヘッダを設定する。"""
    st.set_page_config(page_title="ROIC Analyst — Snowflake + Cortex", layout="wide")
    st.title("ROIC Analyst — Snowflake + Cortex AI")
    st.caption(
        "DB/ビューを指定 → 列役割を自動推測 → 日本語で会話しながらSQL生成・実行・可視化・What-If比較。"
    )


# =========================
# 1) 接続 / セッション
# =========================
@st.cache_resource(show_spinner=False)
def get_session() -> Session:
    """
    Snowflake セッションを初期化して返す。
    st.secrets / 環境変数を確認し、欠落があれば例外にする。

    Returns:
        Session: 有効な Snowpark Session
    Raises:
        RuntimeError: 必要な接続設定が不足
    """
    cfg = {
        "account": st.secrets.get("account", os.getenv("SNOWFLAKE_ACCOUNT")),
        "user": st.secrets.get("user", os.getenv("SNOWFLAKE_USER")),
        "password": st.secrets.get("password", os.getenv("SNOWFLAKE_PASSWORD")),
        "role": st.secrets.get("role", os.getenv("SNOWFLAKE_ROLE", "ANALYST")),
        "warehouse": st.secrets.get("warehouse", os.getenv("SNOWFLAKE_WAREHOUSE")),
        "database": st.secrets.get("database", os.getenv("SNOWFLAKE_DATABASE")),
        "schema": st.secrets.get("schema", os.getenv("SNOWFLAKE_SCHEMA")),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Snowflake settings: {missing}")
    return Session.builder.configs(cfg).create()


def render_connection_sidebar() -> tuple[str, str, bool, int, Session]:
    """
    サイドバーで対象テーブルや Cortex モデルなどの基本設定を入力し、
    Snowflake 接続を確立する。

    Returns:
        (roic_table, model_name, run_guard, sample_rows, session)
    """
    with st.sidebar:
        st.header("接続 & 対象テーブル/ビュー")
        roic_table = st.text_input(
            "対象 (DB.SCHEMA.TABLE)",
            value=st.secrets.get("roic_table", "ROIC_DEMO.ROIC_TABLES.ROIC_BASE"),
        )
        model_name = st.text_input(
            "Cortexモデル（例: mistral-large, llama3-70b, snowflake-arctic）",
            value="mistral-large",
        )
        run_guard = st.checkbox("Cortex Guard（安全フィルタ）", value=True)
        sample_rows = st.number_input(
            "プロファイル用サンプル行数",
            min_value=200,
            max_value=200000,
            value=5000,
            step=1000,
        )

    try:
        session = get_session()
        session.sql("select 1").collect()
        st.sidebar.success("Snowflake接続OK ✅")
    except Exception as e:
        st.sidebar.error(f"接続エラー: {e}")
        st.stop()

    return roic_table, model_name, run_guard, sample_rows, session


# =========================
# 2) ユーティリティ（SQL/スキーマ/プロファイル）
# =========================
def sanitize_identifier(name: Optional[str]) -> str:
    """
    DB.SCHEMA.TABLE を完全大文字化し、Snowflake 非対応文字を除去。
    完全修飾識別子の安全な文字列に整形する。

    Args:
        name: 任意のテーブル指定文字列
    Returns:
        str: 大文字＋安全文字のみの識別子
    """
    return re.sub(r'[^A-Z0-9_.$"]', "", (name or "").upper())


def run_sql_df(session: Session, sql: str) -> pd.DataFrame:
    """
    Snowflake SQL を実行して pandas.DataFrame を返す。UI 側は常にこの関数経由。

    Args:
        session: Snowpark Session
        sql: 実行する SELECT 文
    """
    return session.sql(sql).to_pandas()


def get_table_schema(session: Session, qualified_table: str) -> pd.DataFrame:
    """
    INFORMATION_SCHEMA から列名/型/NULLABLE を取得。列役割推測・UI 初期化に利用。

    Args:
        session: Snowpark Session
        qualified_table: DB.SCHEMA.TABLE
    """
    parts = [p.replace('"', "") for p in (qualified_table or "").split(".")]
    if len(parts) != 3:
        raise ValueError("テーブル名は DB.SCHEMA.TABLE の形式で指定してください。")
    db, sch, tbl = parts
    sql = f"""
    select column_name, data_type, is_nullable
      from {db}.information_schema.columns
     where table_schema = '{sch.upper()}'
       and table_name   = '{tbl.upper()}'
     order by ordinal_position
    """
    return run_sql_df(session, sql)


def sample_table(session: Session, qualified_table: str, n: int) -> pd.DataFrame:
    """
    対象テーブルから軽量サンプル。巨大表でも UI 表示と quick_profile が返るよう
    比率サンプル + LIMIT を利用。

    Args:
        session: Snowpark Session
        qualified_table: DB.SCHEMA.TABLE
        n: 取得行数目安
    """
    frac = min(100, max(1, int(100 * n / 2_000_000)))  # 200万行基準
    return run_sql_df(
        session, f"select * from {qualified_table} sample ({frac}) limit {n}"
    )


def quick_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    列ごとの簡易統計（件数/欠損/ユニーク/代表統計）を返す。UI の“クイック・プロファイル”で表示。

    Args:
        df: 対象データフレーム
    """
    prof = []
    for c in df.columns:
        s = df[c]
        d = {"col": c, "dtype": str(s.dtype), "rows": int(len(s))}
        d["nulls"] = int(s.isna().sum())
        d["distinct"] = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s):
            d.update(
                {
                    "min": s.min(skipna=True),
                    "p25": s.quantile(0.25),
                    "median": s.median(),
                    "p75": s.quantile(0.75),
                    "max": s.max(skipna=True),
                    "mean": s.mean(skipna=True),
                    "std": s.std(skipna=True),
                }
            )
        prof.append(d)
    return pd.DataFrame(prof)


# =========================
# 3) LLM（Cortex） / プロンプト契約
# =========================
def cortex_complete_messages(
    session: Session, messages: list[dict], model: str, guard: bool = True
) -> str:
    """
    Cortex COMPLETE を呼び出し、messages(JSON)→応答文字列を取得。

    Args:
        session: Snowpark Session
        messages: OpenAI 互換の messages 配列（role, content）
        model: 使用モデル名
        guard: True の場合、Snowflake 付属のガードを ON
    """
    options = {"temperature": 0.1}
    if guard:
        options["guard"] = "ON"
    jopt = json.dumps(options)
    jmsg = json.dumps(messages, ensure_ascii=False)
    sql = f"""
    select SNOWFLAKE.CORTEX.COMPLETE(
      '{model}',
      parse_json('{jmsg}'),
      parse_json('{jopt}')
    ) as response
    """
    try:
        df = run_sql_df(session, sql)
        return df.iloc[0, 0]
    except SnowparkSQLException as e:
        return f"[CORTEX ERROR] {e}"


def extract_sql_from_text(text: Optional[str]) -> Optional[str]:
    """
    アシスタント応答から ```sql ...``` を抽出。なければ 'select ...' で妥当な末尾まで推定。

    Args:
        text: 応答テキスト
    """
    m = re.search(r"```sql\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";")
    m = re.search(r"(?is)(^|\n)\s*select\s+.+", text or "")
    return m.group(0).strip().rstrip(";") if m else None


def build_system_prompts(
    qtable: str,
    product_col: str,
    bu_col: str,
    period_col: str,
    roic_col: str,
    city_col: Optional[str],
) -> tuple[str, str, str, str]:
    """
    各種 SYSTEM_* プロンプトを構築して返す。FY_Q は 'Q1'〜'Q4' を原文のまま扱う。

    Returns:
        (SYSTEM_SQL, SYSTEM_SUMMARY, SYSTEM_SCENARIO_SQL, SYSTEM_SCENARIO_SUMMARY)
    """
    sys_sql = textwrap.dedent(
        f"""
        あなたは Snowflake 上の ROIC データ分析者です。ユーザーの日本語質問に対して
        **実行可能な Snowflake SQL（SELECTのみ）** を1つだけ生成します。

        FY_Q は 'Q1'/'Q2'/'Q3'/'Q4' を**元データの表記のまま**扱い、必要なら FY と連結して FYQ_LABEL=CONCAT('FY',FY,'-',FY_Q) を使ってよい。

        前提:
        - 解析対象テーブル: {qtable}
        - 主な列: {product_col}, {bu_col}, {period_col}, {roic_col}{(" , "+city_col) if city_col else ""}
        - 出力は **SQL のみ** を ```sql ... ``` の1ブロックで返す
        - 生成SQLは SELECT 限定。DML/DDLは禁止
        - 期間比較（前期比, Δ, %変化）, 製品/事業部{("/都市" if city_col else "")}などの絞り込み・グルーピングに対応
        - 列名が曖昧なときは上記列名を優先して推測
        - 並べ替えや LIMIT 指定で読みやすい結果にする
        """
    ).strip()

    sys_summary = textwrap.dedent(
        """
        あなたは Snowflake SQL の説明者です。
        ユーザーの質問と生成されたSQLを受け取り、**日本語**で **箇条書きの短い要約** を返してください。

        含める要素:
        - どの列で **絞り込み** したか（WHERE）
        - どの軸で **集計/比較** したか（GROUP BY, 窓関数）
        - **並べ替え・上位抽出** の条件（ORDER BY, LIMIT）
        - 結果の **読み方のヒント**（例: “abs_delta が負のほど下落寄与が大きい” など）

        ※ 出力はテキストのみ。SQLは出さない。
        """
    ).strip()

    sys_scn_sql = textwrap.dedent(
        f"""
        あなたは Snowflake 上のROIC分析者です。ユーザーの日本語シナリオ指示を受け、
        既存の「ベース集計ロジック」を踏まえた **仮想シナリオ込みの SELECT SQL** を1つ生成します。

        FY_Q は 'Q1'/'Q2'/'Q3'/'Q4' を**元データの表記のまま**扱います。

        要件:
        - 解析対象テーブル: {qtable}
        - 主要列: {product_col}, {bu_col}, {period_col}, {roic_col}{(" , "+city_col) if city_col else ""}
        - 出力は ```sql ... ``` の **1ブロックのみ**。DML/DDL禁止。SELECTのみ。
        - 可能なら `WITH scenario AS (...)` で係数・閾値を定義し、そこからベース計算に係数を適用した集計を組み立てる。
        - 返す列例: キー（{product_col}/{bu_col}/{period_col}{("/"+city_col) if city_col else ""}）、ベース値、シナリオ値、abs_delta、pct_change など。
        - 読みやすい ORDER BY と LIMIT を付与。
        """
    ).strip()

    sys_scn_summary = textwrap.dedent(
        """
        あなたは SQL結果の説明者です。ユーザーの日本語シナリオ指示とシナリオSQLを受け取り、
        **何をどう変更して（係数/WHERE/JOIN/計算）**、**どのKPIがどう変化**するかを日本語で簡潔に箇条書きでまとめてください。
        含める要素:
        - 調整点（例: 価格=0.5x など）
        - 影響先（例: 原価→利益率→NOPAT→ROIC）
        - 比較軸（例: 期間/事業部/製品/都市）
        - 読み方のヒント（abs_delta が負=下落 など）
        出力はテキストのみ。
        """
    ).strip()

    return sys_sql, sys_summary, sys_scn_sql, sys_scn_summary


# =========================
# 4) ロール推測 / 安全ガード / 基本SQLスニペット
# =========================
def _match(col: str, words: Iterable[str]) -> bool:
    """部分一致の簡易シノニム判定。"""
    c = col.lower()
    return any(w in c for w in words)


def infer_roles_from_schema(
    schema_df: pd.DataFrame, sample_df: Optional[pd.DataFrame]
) -> dict:
    """
    列名/サンプルから {product, bu, city, period, roic} を推測。UI の初期候補として提示。

    Args:
        schema_df: INFORMATION_SCHEMA.columns の結果
        sample_df: サンプルデータ（数値型推定の補助）
    """
    cols = [c for c in schema_df["COLUMN_NAME"].tolist()]
    numeric_cols = []
    if sample_df is not None:
        numeric_cols = [
            c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])
        ]

    syn = {
        "product": [
            "product",
            "product_name",
            "product_id",
            "item",
            "sku",
            "material",
            "part",
            "品目",
            "製品",
            "プロダクト",
        ],
        "bu": [
            "division_name",
            "division_cd",
            "business_unit",
            "bu",
            "division",
            "segment",
            "dept",
            "事業部",
            "部門",
            "セグメント",
        ],
        "city": ["city", "region", "market", "area", "pref", "都市", "地域", "エリア"],
        "period": [
            "fy_q",
            "fyq",
            "achievement_date",
            "achievement_year",
            "achievement_month",
            "fy",
            "period",
            "month",
            "date",
            "fiscal",
            "quarter",
            "year",
            "会計",
            "年月",
            "期間",
        ],
        "roic": ["roic", "roic_value", "roic_pct", "roic_val"],
    }

    guess = {"product": None, "bu": None, "city": None, "period": None, "roic": None}

    for c in cols:
        cl = c.lower()
        if guess["product"] is None and _match(cl, syn["product"]):
            guess["product"] = c
        if guess["bu"] is None and _match(cl, syn["bu"]):
            guess["bu"] = c
        if guess["city"] is None and _match(cl, syn["city"]):
            guess["city"] = c
        if guess["period"] is None and _match(cl, syn["period"]):
            guess["period"] = c

    roic_cands = [c for c in cols if _match(c.lower(), syn["roic"])]
    if roic_cands:
        guess["roic"] = next(
            (c for c in roic_cands if c in numeric_cols), roic_cands[0]
        )
    else:
        n_roic = [c for c in numeric_cols if "roic" in c.lower()]
        if n_roic:
            guess["roic"] = n_roic[0]

    return guess


def enforce_select_single_table(sql: str, allowed_table: str) -> bool:
    """
    SELECT 限定かつ、完全修飾/スキーマ付の FROM 参照が allowed_table のみかを判定（CTE 名は無視）。

    Args:
        sql: 検査対象 SQL
        allowed_table: 許可する単一ビュー（DB.SCHEMA.TABLE）
    """
    if not re.match(r"(?is)^\s*select\b", sql or ""):
        return False
    refs = re.findall(r"(?is)\bfrom\b\s+([A-Z0-9_.$\"]+)", (sql or "").upper())
    refs = [r for r in refs if "." in r]  # CTE名（ドット無し）は除外
    return all(r == allowed_table.upper() for r in refs) if refs else True


def sql_roic_base_cte(qtable: str, tax: float = 0.21, ic_ratio: float = 0.80) -> str:
    """
    ROIC 計算テンプレートを含む base CTE。FY_Q は 'Q1'〜'Q4' を原文のまま使用。

    Args:
        qtable: 解析対象ビュー（DB.SCHEMA.TABLE）
        tax: 税率の近似
        ic_ratio: 投下資本を売上の何割で近似するか
    """
    return f"""
    WITH base AS (
      SELECT
        FY,
        FY_Q,  -- 元データ 'Q1'/'Q2'/'Q3'/'Q4' をそのまま使用
        CONCAT('FY', FY, '-', FY_Q) AS FYQ_LABEL,
        DIVISION_NAME,
        PRODUCT_NAME,
        PRICE,
        QUANTITY,
        COST_PER_UNIT,
        (PRICE - COST_PER_UNIT) * QUANTITY AS GROSS_PROFIT,
        (PRICE * QUANTITY)                  AS REVENUE,
        ((PRICE - COST_PER_UNIT) * QUANTITY) * (1 - {tax}) AS NOPAT,
        (PRICE * QUANTITY) * {ic_ratio}     AS INVESTED_CAPITAL,
        ((PRICE - COST_PER_UNIT) * QUANTITY) * (1 - {tax})
        / NULLIF((PRICE * QUANTITY) * {ic_ratio}, 0) AS ROIC
      FROM {qtable}
    )
    """


def sql_division_roic_latest(qtable: str) -> str:
    """
    初期表示用：最新期（FY降順→FY_Qの順序を CASE で Q4>Q3>Q2>Q1）における
    事業部別の平均 ROIC を返す SQL。

    Args:
        qtable: 解析対象ビュー（DB.SCHEMA.TABLE）
    """
    return f"""
    {sql_roic_base_cte(qtable)}
    , latest AS (
      SELECT FY, FY_Q
      FROM (
        SELECT DISTINCT
               FY, FY_Q,
               ROW_NUMBER() OVER (
                 ORDER BY
                   FY DESC,
                   CASE FY_Q
                     WHEN 'Q4' THEN 4
                     WHEN 'Q3' THEN 3
                     WHEN 'Q2' THEN 2
                     WHEN 'Q1' THEN 1
                     ELSE 0
                   END DESC
               ) AS rn
        FROM base
      )
      WHERE rn = 1
    )
    SELECT
      b.DIVISION_NAME,
      b.FYQ_LABEL AS PERIOD,
      AVG(b.ROIC) AS ROIC
    FROM base b
    JOIN latest l ON b.FY = l.FY AND b.FY_Q = l.FY_Q
    GROUP BY b.DIVISION_NAME, b.FYQ_LABEL
    ORDER BY ROIC DESC
    """


# =========================
# 5) 表示ヘルパ（表/ピボット/Plotly）
# =========================
def render_dataframe(df: pd.DataFrame, caption: Optional[str] = None) -> None:
    """数表を表示する。"""
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True)


def render_plot(
    df: pd.DataFrame, x: str, y: str, color: Optional[str], kind: str
) -> None:
    """Plotly 折れ線/棒グラフを描画する。"""
    if kind == "折れ線":
        fig = px.line(df, x=x, y=y, color=None if not color else color)
    else:
        fig = px.bar(df, x=x, y=y, color=None if not color else color)
    st.plotly_chart(fig, use_container_width=True)


def render_pivot(df: pd.DataFrame) -> None:
    """ピボット（マトリクス）UI を表示する。"""
    cols = list(df.columns)
    row_key = st.selectbox("行（index）", cols, index=0)
    col_key = st.selectbox("列（columns）", cols, index=min(1, len(cols) - 1))
    val_key = st.selectbox(
        "値（values・数値列推奨）", cols, index=min(2, len(cols) - 1)
    )
    if row_key and col_key and val_key:
        try:
            mat = df.pivot_table(
                index=row_key, columns=col_key, values=val_key, aggfunc="sum"
            )
            st.dataframe(mat, use_container_width=True)
        except Exception as e:
            st.error(f"マトリクス作成エラー: {e}")


# =========================
# 6) 比較/検証ヘルパ
# =========================
def pick_keys(df: pd.DataFrame, prefer: Sequence[str]) -> list[str]:
    """候補列から存在するキーを選び、不足時は非数値列を追加して返す。"""
    keys = [c for c in prefer if c and c in df.columns]
    if not keys:
        keys += [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])][:2]
    return list(dict.fromkeys(keys))


def pick_value_col(
    df: pd.DataFrame, candidates: Sequence[str] = ("ROIC", "VALUE", "METRIC")
) -> Optional[str]:
    """代表数値列を推定。候補が無ければ最初の数値列。"""
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in candidates:
        if c in nums:
            return c
    return nums[0] if nums else None


def make_diff_table(
    base_df: pd.DataFrame,
    scn_df: pd.DataFrame,
    join_keys: Sequence[str],
    base_val: str,
    scn_val: str,
) -> pd.DataFrame:
    """ベース vs シナリオの差分表を作成する。"""
    left = base_df[list(join_keys) + [base_val]].copy()
    right = scn_df[list(join_keys) + [scn_val]].copy()
    left.columns = list(join_keys) + ["base_value"]
    right.columns = list(join_keys) + ["scenario_value"]
    diff = pd.merge(left, right, on=list(join_keys), how="inner")
    if diff.empty:
        return diff
    diff["abs_delta"] = diff["scenario_value"] - diff["base_value"]
    diff["pct_change"] = diff["abs_delta"] / diff["base_value"].replace(0, pd.NA)
    return diff


def validate_factor_via_means(
    base_df: pd.DataFrame,
    scn_df: pd.DataFrame,
    target_col: str,
    query_expr: Optional[str],
    expected_factor: Optional[float],
) -> dict:
    """
    「対象列が本当に x倍 になっているか？」を平均値の比で概観チェックする。

    Args:
        base_df: ベース結果
        scn_df: シナリオ結果
        target_col: 検証対象の数値列
        query_expr: pandas.query での絞り込み式（任意）
        expected_factor: 期待係数（任意）
    Returns:
        dict: {base_mean, scn_mean, ratio, match (Optional[bool])}
    """

    def filt(df: pd.DataFrame) -> pd.DataFrame:
        if query_expr and query_expr.strip():
            try:
                return df.query(query_expr)
            except Exception:
                return df
        return df

    fb = filt(base_df)
    fs = filt(scn_df)
    result = {"base_mean": None, "scn_mean": None, "ratio": None, "match": None}

    if target_col in fb.columns and target_col in fs.columns:
        base_mean = pd.to_numeric(fb[target_col], errors="coerce").mean()
        scn_mean = pd.to_numeric(fs[target_col], errors="coerce").mean()
        result["base_mean"] = base_mean
        result["scn_mean"] = scn_mean
        if base_mean not in (0, None, pd.NA):
            ratio = scn_mean / base_mean
            result["ratio"] = ratio
            if expected_factor is not None:
                result["match"] = abs(ratio - expected_factor) / expected_factor <= 0.05
    return result


# =========================
# 7) 画面セクション描画（関数群）
# =========================
def render_default_division_roic(session: Session, qtable: str) -> pd.DataFrame:
    """
    初期表示：最新期の事業部別 ROIC を取得・表示し、base_df/base_sql を初期化する。

    Returns:
        DataFrame: 取得結果（空でない場合）
    """
    st.subheader("デフォルト出力：最新期の『事業部別ROIC』")
    try:
        sql = sql_division_roic_latest(qtable)
        df = run_sql_df(session, sql)
        render_dataframe(df, "最新期（FY と FY_Q='Q1'〜'Q4' を原文使用）の事業部別ROIC")
        if not df.empty:
            try:
                fig = px.bar(df, x="DIVISION_NAME", y="ROIC")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"初期グラフの描画に失敗しました: {e}")
        # セッション保存（比較で利用）
        if "base_df" not in st.session_state:
            st.session_state["base_df"] = df
            st.session_state["base_sql"] = sql
        return df
    except Exception as e:
        st.error(f"デフォルトの事業部別ROIC取得に失敗：{e}")
        return pd.DataFrame()


def render_schema_profile_and_role_guess(
    session: Session, qtable: str
) -> tuple[dict, pd.DataFrame]:
    """
    スキーマ／サンプル／クイック・プロファイルを表示し、列役割推測を行って
    サイドバーのセレクトボックスを初期化する。

    Returns:
        (roles, schema_df)
    """
    st.subheader("1) データの自動プロファイル & 列役割の自動推測")
    try:
        schema_df = get_table_schema(session, qtable)
        samp = sample_table(session, qtable, st.session_state.get("sample_rows", 5000))
        inferred = infer_roles_from_schema(schema_df, samp)

        render_dataframe(schema_df, "スキーマ")
        render_dataframe(samp.head(50), f"サンプル（{len(samp):,}行からの抜粋）")
        render_dataframe(quick_profile(samp), "クイック・プロファイル")

        # サイドバーで列役割を確定
        with st.sidebar:
            st.markdown("---")
            st.header("列役割の確認/修正（自動推測済み）")
            all_cols = schema_df["COLUMN_NAME"].tolist()

            def idx(lst, val, default=0):
                try:
                    return lst.index(val)
                except Exception:
                    return default

            product_col = st.selectbox(
                "製品カラム", options=all_cols, index=idx(all_cols, inferred["product"])
            )
            bu_col = st.selectbox(
                "事業部カラム", options=all_cols, index=idx(all_cols, inferred["bu"])
            )
            city_opt = ["(なし)"] + all_cols
            city_sel = st.selectbox(
                "都市/地域カラム（任意）",
                options=city_opt,
                index=idx(city_opt, inferred["city"], 0),
            )
            period_col = st.selectbox(
                "期間カラム", options=all_cols, index=idx(all_cols, inferred["period"])
            )
            roic_col = st.selectbox(
                "ROIC指標カラム",
                options=all_cols,
                index=idx(all_cols, inferred["roic"]),
            )
            city_col = "" if city_sel == "(なし)" else city_sel

        roles = {
            "product_col": product_col,
            "bu_col": bu_col,
            "city_col": city_col,
            "period_col": period_col,
            "roic_col": roic_col,
        }
        return roles, schema_df
    except Exception as e:
        st.error(f"テーブル読込/プロファイルに失敗: {e}")
        st.stop()


def render_chat_to_sql(
    session: Session,
    system_sql: str,
    system_summary: str,
    model_name: str,
    run_guard: bool,
) -> None:
    """
    チャットで質問 → SQL 生成 → 日本語要約を作成し、履歴に積む。
    """
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    st.subheader("2) チャットで質問 → SQL生成 → 要約（実行は次セクション）")

    with st.expander("チャット履歴（直近）", expanded=False):
        if not st.session_state["chat"]:
            st.write("（まだ履歴はありません）")
        else:
            for i, turn in enumerate(st.session_state["chat"], 1):
                st.markdown(f"**Q{i}（あなた）:** {turn['user']}")
                if "sql" in turn:
                    st.markdown("**生成SQL**:")
                    st.code(turn["sql"], language="sql")
                if "explanation" in turn:
                    st.markdown("**要約（SQLの説明）**:")
                    st.write(turn["explanation"])
                if "error" in turn:
                    st.error(turn["error"])

    user_prompt = st.text_area(
        "日本語で質問（例: 「FY2025のFY_Q='Q3'に限定し、事業部×製品ROICの前期比と下落上位10件」）",
        height=120,
    )
    col_send, col_clear = st.columns([1, 1])
    with col_send:
        send = st.button("SQL生成（Cortex）")
    with col_clear:
        if st.button("履歴クリア"):
            st.session_state["chat"] = []
            st.experimental_rerun()

    if send and user_prompt.strip():
        # 過去の履歴をプロンプトに付与（会話で具体化）
        messages_sql = [{"role": "system", "content": system_sql}]
        for turn in st.session_state["chat"]:
            messages_sql.append({"role": "user", "content": turn["user"]})
            if "sql" in turn:
                messages_sql.append(
                    {"role": "assistant", "content": f"```sql\n{turn['sql']}\n```"}
                )
        messages_sql.append({"role": "user", "content": user_prompt})

        with st.spinner("CortexがSQLを作成中..."):
            raw_sql_text = cortex_complete_messages(
                session, messages_sql, model=model_name, guard=run_guard
            )

        gen_sql = extract_sql_from_text(raw_sql_text)
        new_turn = {"user": user_prompt}

        if not gen_sql:
            new_turn["error"] = f"SQLが抽出できませんでした。出力:\n{raw_sql_text}"
            st.session_state["chat"].append(new_turn)
        else:
            messages_sum = [
                {"role": "system", "content": system_summary},
                {
                    "role": "user",
                    "content": f"質問: {user_prompt}\n\nSQL:\n```sql\n{gen_sql}\n```",
                },
            ]
            with st.spinner("SQLの要約を作成中..."):
                explanation = cortex_complete_messages(
                    session, messages_sum, model=model_name, guard=run_guard
                )

            new_turn["sql"] = gen_sql
            new_turn["explanation"] = explanation
            st.session_state["chat"].append(new_turn)


def render_execute_base(session: Session, qtable: str) -> None:
    """
    履歴中の SQL を選択して実行し、数表/ピボット/Plotly を表示。
    """
    st.subheader("3) 実行結果（ベース）— 数表・マトリクス・Plotly")

    available_sqls = [t["sql"] for t in st.session_state.get("chat", []) if "sql" in t]
    selected_sql = (
        st.selectbox(
            "実行するSQL（ベース）", available_sqls, index=len(available_sqls) - 1
        )
        if available_sqls
        else None
    )

    exec_btn = st.button("選択SQLを実行（ベース）")
    result_df = None
    if exec_btn and selected_sql:
        if not enforce_select_single_table(selected_sql, qtable):
            st.error("安全ガード：SELECT限定・対象テーブルのみ参照で実行してください。")
        else:
            try:
                with st.spinner("SQL実行中..."):
                    result_df = run_sql_df(session, selected_sql)
                st.success(f"Rows: {len(result_df):,}")
                render_dataframe(result_df)
                st.session_state["base_df"] = result_df
                st.session_state["base_sql"] = selected_sql
            except Exception as e:
                st.error(f"実行失敗: {e}")

    if result_df is not None and len(result_df) > 0:
        st.markdown("### 出力オプション")
        tab_tbl, tab_matrix, tab_plot = st.tabs(
            ["数表（そのまま）", "マトリクス（ピボット）", "Plotlyグラフ"]
        )
        with tab_tbl:
            render_dataframe(result_df)
        with tab_matrix:
            render_pivot(result_df)
        with tab_plot:
            cols = list(result_df.columns)
            x_col = st.selectbox("X軸", cols, index=0)
            y_col = st.selectbox("Y軸（数値）", cols, index=min(1, len(cols) - 1))
            color_col = st.selectbox("色（任意）", ["(なし)"] + cols, index=0)
            kind = st.radio("種類", ["折れ線", "棒"], horizontal=True)
            if x_col and y_col:
                try:
                    render_plot(
                        result_df,
                        x_col,
                        y_col,
                        None if color_col == "(なし)" else color_col,
                        kind,
                    )
                except Exception as e:
                    st.error(f"Plotly描画エラー: {e}")


def render_preset_prompts(
    bu_col: str,
    product_col: str,
    roic_col: str,
    city_col: Optional[str],
    system_sql: str,
    system_summary: str,
    model_name: str,
    run_guard: bool,
    session: Session,
) -> None:
    """
    ワンクリック定型プロンプト（下落要因・寄与分解など）を提供し、即座に SQL 生成できる。
    """
    st.subheader("4) ワンクリック定型プロンプト（下落要因・寄与分解など）")

    def prefill(msg: str) -> None:
        st.session_state["prefill"] = msg

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("製品別：最新期 vs 直前期（下落上位10）"):
            prefill(
                f"{roic_col} を {product_col} 単位で最新期と直前期を比較し、latest, previous, abs_delta, pct_change。abs_delta昇順で上位10件。"
            )
    with c2:
        if st.button("事業部×製品：下落寄与TOP"):
            prefill(
                f"{bu_col} × {product_col} 単位で {roic_col} の期次比較（前期比, Δ, %）。下落寄与の大きい順。"
            )
    with c3:
        if st.button("都市指定＋移動平均＋季節性"):
            tgt_city = city_col if city_col else "（都市列が未指定）"
            prefill(
                f"{tgt_city}='Tokyo' に限定し、{product_col}×月の {roic_col} を算出、3ヶ月移動平均と簡易季節性指標、直近24ヶ月。"
            )

    if "prefill" in st.session_state:
        st.info("定型プロンプト（編集して送信可）")
        st.code(st.session_state["prefill"], language="markdown")
        if st.button("この内容でSQL生成"):
            user_prompt = st.session_state["prefill"]
            messages_sql = [{"role": "system", "content": system_sql}]
            for turn in st.session_state.get("chat", []):
                messages_sql.append({"role": "user", "content": turn["user"]})
                if "sql" in turn:
                    messages_sql.append(
                        {"role": "assistant", "content": f"```sql\n{turn['sql']}\n```"}
                    )
            messages_sql.append({"role": "user", "content": user_prompt})

            with st.spinner("CortexがSQLを作成中..."):
                raw_sql_text = cortex_complete_messages(
                    session, messages_sql, model=model_name, guard=run_guard
                )
            gen_sql = extract_sql_from_text(raw_sql_text)
            new_turn = {"user": user_prompt}
            if not gen_sql:
                new_turn["error"] = f"SQLが抽出できませんでした。出力:\n{raw_sql_text}"
            else:
                messages_sum = [
                    {"role": "system", "content": system_summary},
                    {
                        "role": "user",
                        "content": f"質問: {user_prompt}\n\nSQL:\n```sql\n{gen_sql}\n```",
                    },
                ]
                with st.spinner("SQLの要約を作成中..."):
                    explanation = cortex_complete_messages(
                        session, messages_sum, model=model_name, guard=run_guard
                    )
                new_turn["sql"] = gen_sql
                new_turn["explanation"] = explanation
            st.session_state.setdefault("chat", []).append(new_turn)
            st.experimental_rerun()


def render_scenario_nl(
    session: Session,
    qtable: str,
    system_scn_sql: str,
    system_scn_summary: str,
    model_name: str,
    run_guard: bool,
) -> None:
    """
    What-If（自然言語シナリオ）用の入力・SQL生成・要約・実行・Plotly を提供。
    """
    st.subheader("5) What-If（仮想シナリオ）— 自然言語で係数を変えて再計算")
    st.caption(
        "例: 「原材料が米の行を対象に単価を0.5倍にし、最新期の製品別ROICを再計算。ベースとの差分（Δ, %）も返して。」"
    )
    scenario_prompt = st.text_area("シナリオ指示（日本語）", height=100)
    col_scn1, col_scn2 = st.columns([1, 1])
    with col_scn1:
        gen_scn_sql = st.button("シナリオSQL生成（Cortex）")
    with col_scn2:
        clear_scn = st.button("シナリオ内容クリア")

    if clear_scn:
        for k in ["scenario_sql", "scenario_explain", "scenario_df"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

    if gen_scn_sql and scenario_prompt.strip():
        messages = [{"role": "system", "content": system_scn_sql}]
        base_sqls = [t["sql"] for t in st.session_state.get("chat", []) if "sql" in t]
        if base_sqls:
            messages.append(
                {
                    "role": "user",
                    "content": f"参考ベースSQL:\n```sql\n{base_sqls[-1]}\n```",
                }
            )
        messages.append({"role": "user", "content": f"シナリオ指示: {scenario_prompt}"})
        with st.spinner("シナリオSQLを生成中…"):
            raw = cortex_complete_messages(
                session, messages, model=model_name, guard=run_guard
            )
        scn_sql = extract_sql_from_text(raw)
        if not scn_sql:
            st.error(f"シナリオSQLが抽出できませんでした。\n出力:\n{raw}")
        else:
            st.session_state["scenario_sql"] = scn_sql
            sum_msg = [
                {"role": "system", "content": system_scn_summary},
                {
                    "role": "user",
                    "content": f"シナリオ指示: {scenario_prompt}\n\nSQL:\n```sql\n{scn_sql}\n```",
                },
            ]
            with st.spinner("シナリオの要約を生成中…"):
                explain = cortex_complete_messages(
                    session, sum_msg, model=model_name, guard=run_guard
                )
            st.session_state["scenario_explain"] = explain

    if "scenario_sql" in st.session_state:
        st.markdown("**シナリオSQL**")
        st.code(st.session_state["scenario_sql"], language="sql")
        if "scenario_explain" in st.session_state:
            st.markdown("**シナリオの要約**")
            st.write(st.session_state["scenario_explain"])

        run_scn = st.button("シナリオSQLを実行")
        if run_scn:
            if not enforce_select_single_table(
                st.session_state["scenario_sql"], qtable
            ):
                st.error("安全ガード（SELECT限定・対象テーブルのみ）。")
            else:
                try:
                    with st.spinner("シナリオ結果を取得中…"):
                        scn_df = run_sql_df(session, st.session_state["scenario_sql"])
                    st.success(f"Rows: {len(scn_df):,}")
                    render_dataframe(scn_df)
                    st.session_state["scenario_df"] = scn_df

                    st.markdown("### シナリオ結果の可視化（Plotly）")
                    cols = list(scn_df.columns)
                    if len(cols) >= 2:
                        x_col = st.selectbox("X軸", cols, index=0, key="scn_x")
                        y_col = st.selectbox("Y軸（数値）", cols, index=1, key="scn_y")
                        color_col = st.selectbox(
                            "色（任意）", ["(なし)"] + cols, index=0, key="scn_color"
                        )
                        kind = st.radio(
                            "種類", ["折れ線", "棒"], horizontal=True, key="scn_kind"
                        )
                        if x_col and y_col:
                            render_plot(
                                scn_df,
                                x_col,
                                y_col,
                                None if color_col == "(なし)" else color_col,
                                kind,
                            )
                except Exception as e:
                    st.error(f"シナリオ実行に失敗: {e}")


def render_comparison_and_validation(
    session: Session,
    product_col: str,
    bu_col: str,
    period_col: str,
    city_col: Optional[str],
) -> None:
    """
    ベース vs シナリオの左右比較・差分表・比較サマリー・係数検証 UI を表示。
    """
    st.subheader("6) ベース vs シナリオ比較（左右表示・差分・日本語サマリ・係数検証）")
    base_df = st.session_state.get("base_df")
    scn_df = st.session_state.get("scenario_df")

    if base_df is None or scn_df is None:
        st.info(
            "ベース（セクション3）とシナリオ（セクション5）の両方を実行すると比較できます。"
        )
        return

    prefer_keys = [product_col, bu_col, period_col] + ([city_col] if city_col else [])
    base_keys = pick_keys(base_df, prefer_keys)
    scn_keys = pick_keys(scn_df, prefer_keys)
    base_val = pick_value_col(base_df)
    scn_val = pick_value_col(scn_df)

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**左：ベース（元データ）**")
        render_dataframe(base_df)
        if base_val:
            x = base_keys[0] if base_keys else base_df.columns[0]
            try:
                fig = px.bar(base_df, x=x, y=base_val)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
    with rcol:
        st.markdown("**右：シナリオ（What-If）**")
        render_dataframe(scn_df)
        if scn_val:
            x = scn_keys[0] if scn_keys else scn_df.columns[0]
            try:
                fig = px.bar(scn_df, x=x, y=scn_val)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    st.markdown("### 差分テーブル（ベース vs シナリオ）")
    if base_keys and scn_keys and base_val and scn_val:
        join_keys = [k for k in base_keys if k in scn_keys]
        diff_df = make_diff_table(base_df, scn_df, join_keys, base_val, scn_val)
        if not diff_df.empty:
            render_dataframe(diff_df)

            # 簡易比較サマリー（Cortex）
            try:
                head_text = diff_df.head(20).to_csv(index=False)
                compare_prompt = f"キー: {join_keys}\n主指標: base_value vs scenario_value\n差分サンプル:\n{head_text}"
                system_compare = textwrap.dedent(
                    """
                あなたはデータ比較の説明者です。ベースとシナリオの結果テーブルの概要から、
                **日本語**で差分の読み方を箇条書きで簡潔にまとめてください。
                含める要素:
                - 比較軸（例：製品 / 期間 / 事業部 / 都市）
                - ベース vs シナリオの傾向（上昇/下落、上位項目）
                - 読み方のヒント（abs_delta 負=下落寄与、%変化の注意点 等）
                ※ 短い箇条書きのみ
                """
                ).strip()
                msgs = [
                    {"role": "system", "content": system_compare},
                    {"role": "user", "content": compare_prompt},
                ]
                summary = cortex_complete_messages(
                    session,
                    msgs,
                    model=st.session_state.get("model_name", "mistral-large"),
                    guard=st.session_state.get("run_guard", True),
                )
                st.markdown("### 比較サマリ（日本語）")
                st.write(summary)
            except Exception as e:
                st.warning(f"比較サマリ生成に失敗: {e}")
        else:
            st.info(
                "キーが一致せず差分を作れませんでした。キー列（製品/期間/事業部/都市など）をSQL側で揃えてください。"
            )
    else:
        st.info("キーまたは数値列が見つからず差分を作成できません。")

    st.markdown("---")
    st.markdown("#### 係数検証（任意：本当に0.5倍/1.2倍になっているか）")
    validator_col = st.text_input(
        "検証対象の数値列（例：PRICE / COST_PER_UNIT など）", value=""
    )
    validator_filter = st.text_input(
        "検証対象の絞り込み条件（pandas.query式, 例：RAW_MATERIAL=='rice'）", value=""
    )
    validator_expected_factor = st.text_input("想定係数（例：0.5, 1.2 など）", value="")
    if st.button("係数チェックを実行"):
        if validator_col:
            try:
                expect = (
                    float(validator_expected_factor)
                    if validator_expected_factor.strip()
                    else None
                )
            except Exception:
                expect = None
            res = validate_factor_via_means(
                st.session_state["base_df"],
                st.session_state["scenario_df"],
                validator_col,
                validator_filter,
                expect,
            )
            if res["base_mean"] is None:
                st.info(
                    "検証列が結果に存在しません。シナリオSQLでその列を含めてください。"
                )
            else:
                st.write(f"- ベース平均: {res['base_mean']}")
                st.write(f"- シナリオ平均: {res['scn_mean']}")
                if res["ratio"] is not None:
                    st.write(f"- 比率（シナリオ/ベース）: **{res['ratio']:.4f}**")
                if res["match"] is True:
                    st.success("想定係数に概ね一致しています。")
                elif res["match"] is False:
                    st.warning(
                        "想定係数からズレています。SQLや列の参照を確認してください。"
                    )
        else:
            st.info("検証対象の数値列を入力してください。")


def render_lever_panel_and_exec(
    session: Session,
    qtable: str,
    product_col: str,
    bu_col: str,
    period_col: str,
    roic_col: str,
    city_col: Optional[str],
    model_name: str,
    run_guard: bool,
) -> None:
    """
    レバーパネル（価格/原価/人件費/物流費）を UI で設定 → Cortex で SQL 生成 → 実行・表示。
    """
    st.subheader("7) レバーパネル — 係数をUI指定して即再計算（What-Ifの簡易操作）")

    with st.expander("レバーの設定（任意で使うものだけON）", expanded=True):
        colA, colB, colC, colD = st.columns(4)
        with colA:
            use_rice = st.checkbox("原材料：米 単価", value=False)
            rice_factor = st.number_input(
                "米の単価 ×",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.05,
                disabled=not use_rice,
            )
        with colB:
            use_price = st.checkbox("販売価格", value=False)
            price_factor = st.number_input(
                "販売価格 ×",
                min_value=0.0,
                max_value=10.0,
                value=1.00,
                step=0.01,
                disabled=not use_price,
            )
        with colC:
            use_labor = st.checkbox("人件費", value=False)
            labor_factor = st.number_input(
                "人件費 ×",
                min_value=0.0,
                max_value=10.0,
                value=1.00,
                step=0.05,
                disabled=not use_labor,
            )
        with colD:
            use_logi = st.checkbox("物流費", value=False)
            logi_factor = st.number_input(
                "物流費 ×",
                min_value=0.0,
                max_value=10.0,
                value=1.00,
                step=0.05,
                disabled=not use_logi,
            )

        lever_notes = st.text_area(
            "補足（任意）: 列や条件が特殊ならここに書いてください（例：RAW_MATERIAL='rice' の行にだけ適用、など）",
            placeholder="例：RAW_MATERIAL='rice' の行の COST_PER_UNIT にだけ rice_factor を掛ける。価格は PRICE に price_factor 掛け。",
        )

    lever_go = st.button("レバーパネルを適用してSQL生成 → 実行")
    if lever_go:
        levers = []
        if use_rice:
            levers.append({"name": "rice", "factor": rice_factor})
        if use_price:
            levers.append({"name": "price", "factor": price_factor})
        if use_labor:
            levers.append({"name": "labor", "factor": labor_factor})
        if use_logi:
            levers.append({"name": "logi", "factor": logi_factor})

        if not levers:
            st.warning(
                "レバーが1つも有効化されていません。少なくとも1つONにしてください。"
            )
        else:
            lever_json = json.dumps(levers, ensure_ascii=False)
            system_scn_sql = textwrap.dedent(
                f"""
            あなたは Snowflake 上のROIC分析者です。以下の情報で、係数を適用した「仮想シナリオ込みのSELECT SQL」を1本だけ返してください。

            条件:
            - 解析対象テーブル: {qtable}
            - 主要列: {product_col}, {bu_col}, {period_col}, {roic_col}{(" , "+city_col) if city_col else ""}
            - レバー（JSON）: {lever_json}
            - レバー適用の補足: {lever_notes}
            - 期待するSQLの構成:
              1) WITH scenario AS (SELECT <各factor> AS factor ... ) で係数を定義
              2) ベース集計（元の数式で ROIC を算出）
              3) シナリオ集計（係数がかかるコスト/価格/項目に factor を適用したバージョンで ROIC を再算出）
              4) キー（{product_col}/{bu_col}/{period_col}{("/"+city_col) if city_col else ""}）ごとに base_value, scenario_value, abs_delta, pct_change を返す
            - 返答は ```sql ... ``` の1つのブロックのみ。SELECT限定。
            注意:
            - 列名が不明な場合は、INFORMATION_SCHEMAの傾向から最善推定で記述。
            """
            ).strip()

            with st.spinner("レバー適用SQLを生成中…"):
                raw = cortex_complete_messages(
                    session,
                    [
                        {"role": "system", "content": system_scn_sql},
                        {"role": "user", "content": system_scn_sql},
                    ],
                    model=model_name,
                    guard=run_guard,
                )
            lever_sql = extract_sql_from_text(raw)
            if not lever_sql:
                st.error(f"SQLが抽出できませんでした。\n出力:\n{raw}")
            else:
                st.code(lever_sql, language="sql")
                if not enforce_select_single_table(lever_sql, qtable):
                    st.error(
                        "安全ガード（SELECT限定・対象テーブルのみ）に抵触しました。必要に応じてSQLを修正してください。"
                    )
                else:
                    try:
                        with st.spinner("実行中…"):
                            df = run_sql_df(session, lever_sql)
                        st.success(f"Rows: {len(df):,}")
                        render_dataframe(df)
                        st.session_state["scenario_df"] = df
                        st.session_state["scenario_sql"] = lever_sql
                        cols = list(df.columns)
                        if len(cols) >= 2 and pd.api.types.is_numeric_dtype(
                            df[cols[1]]
                        ):
                            fig = px.bar(df, x=cols[0], y=cols[1])
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"実行失敗: {e}")


def render_heatmap(
    session: Session,
    qtable: str,
    target_level: str,
    metric_to_show: str,
    lv1: str,
    lv2: str,
    x_min: float,
    x_max: float,
    x_step: float,
    y_min: float,
    y_max: float,
    y_step: float,
    model_name: str,
    run_guard: bool,
) -> None:
    """
    2レバー感度分析の SQL 生成 → 実行 → ヒートマップ描画。
    """
    prompt = f"""
    2つのレバー {lv1}（X軸）と {lv2}（Y軸）を感度分析します。
    {qtable} を対象に、以下の格子（factor_x と factor_y）を CROSS JOIN した1本のSQLで、
    {target_level} 粒度の 指標（ROICまたは abs_delta/pct_change）を算出し、ヒートマップ用に返してください。

    範囲:
    - {lv1}: {x_min} から {x_max} を {x_step} 刻み
    - {lv2}: {y_min} から {y_max} を {y_step} 刻み

    期待するSQLの構成（例）:
    - WITH x AS (SELECT * FROM VALUES ({x_min}), ... ), y AS (...),
      scenario(factor_x, factor_y) AS (SELECT x.v, y.v FROM x CROSS JOIN y)
    - 出力列: {target_level}, factor_x, factor_y, roic_value, abs_delta, pct_change など
    - SELECTのみ。```sql ... ``` 1ブロックで返答。
    """
    with st.spinner("感度分析SQLを生成中…"):
        raw = cortex_complete_messages(
            session,
            [
                {"role": "system", "content": "You are a SQL generator."},
                {"role": "user", "content": prompt},
            ],
            model=model_name,
            guard=run_guard,
        )
    grid_sql = extract_sql_from_text(raw)
    if not grid_sql:
        st.error(f"SQLが抽出できませんでした。\n出力:\n{raw}")
        return

    st.code(grid_sql, language="sql")
    if not enforce_select_single_table(grid_sql, qtable):
        st.error("安全ガード（SELECT限定・対象テーブルのみ）に抵触しました。")
        return

    try:
        with st.spinner("実行中…"):
            grid_df = run_sql_df(session, grid_sql)
        st.success(f"Rows: {len(grid_df):,}")
        render_dataframe(grid_df.head(200))

        # 列名推定
        metric_col = None
        if metric_to_show.lower() == "roic" and "roic" in "".join(
            [c.lower() for c in grid_df.columns]
        ):
            for c in grid_df.columns:
                if "roic" in c.lower():
                    metric_col = c
                    break
        elif metric_to_show == "abs_delta":
            metric_col = next(
                (c for c in grid_df.columns if "abs_delta" in c.lower()), None
            )
        elif metric_to_show == "pct_change":
            metric_col = next((c for c in grid_df.columns if "pct" in c.lower()), None)

        fx = next((c for c in grid_df.columns if "factor_x" in c.lower()), None)
        fy = next((c for c in grid_df.columns if "factor_y" in c.lower()), None)

        if fx and fy and metric_col:
            heat_df = grid_df[[fx, fy, metric_col]].copy()
            fig = px.density_heatmap(
                heat_df, x=fx, y=fy, z=metric_col, nbinsx=None, nbinsy=None
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "ヒートマップ用の列（factor_x/factor_y/指標）が見つかりませんでした。"
            )
    except Exception as e:
        st.error(f"実行失敗: {e}")


def render_heatmap_section(
    product_col: str,
    bu_col: str,
    city_col: Optional[str],
    roic_col: str,
    qtable: str,
    model_name: str,
    run_guard: bool,
    session: Session,
) -> None:
    """
    ヒートマップ（感度分析）の UI 全体（レバー選択／範囲設定／実行ボタン）を表示。
    """
    st.subheader("8) ヒートマップ（感度分析）— 2レバーの格子一括評価")

    with st.expander("レバーと範囲を指定", expanded=True):
        lever_names = ["rice", "price", "labor", "logi"]
        lv1 = st.selectbox("レバー1（X軸）", lever_names, index=0)
        lv2 = st.selectbox("レバー2（Y軸）", lever_names, index=1)
        c1, c2 = st.columns(2)
        with c1:
            x_min = st.number_input("X 最小", 0.1, 10.0, 0.6, 0.05)
            x_max = st.number_input("X 最大", 0.1, 10.0, 1.4, 0.05)
            x_step = st.number_input("X 刻み", 0.01, 1.0, 0.05, 0.01)
        with c2:
            y_min = st.number_input("Y 最小", 0.1, 10.0, 0.6, 0.05)
            y_max = st.number_input("Y 最大", 0.1, 10.0, 1.4, 0.05)
            y_step = st.number_input("Y 刻み", 0.01, 1.0, 0.05, 0.01)

        target_level = st.selectbox(
            "集計粒度",
            [product_col, bu_col] + ([city_col] if city_col else []),
            index=0,
        )
        metric_to_show = st.selectbox(
            "表示する指標", ["ROIC", "abs_delta", "pct_change"], index=0
        )

    if st.button("ヒートマップを生成（SQL→一括実行）"):
        render_heatmap(
            session,
            qtable,
            target_level,
            metric_to_show,
            lv1,
            lv2,
            x_min,
            x_max,
            x_step,
            y_min,
            y_max,
            y_step,
            model_name,
            run_guard,
        )


def render_roic_tree_and_waterfall(
    session: Session,
    qtable: str,
    bu_col: str,
    product_col: str,
    city_col: Optional[str],
    model_name: str,
    run_guard: bool,
) -> None:
    """
    ROICツリー（サンバースト／ツリーマップ）と、滝グラフ（寄与分解）を生成・表示。
    """
    st.subheader("9) ROICツリー & 寄与分解（サンバースト/ツリーマップ + 滝グラフ）")

    with st.expander("ツリー構成の選択", expanded=True):
        level1 = st.selectbox(
            "上位階層",
            [bu_col, product_col] + ([city_col] if city_col else []),
            index=0,
        )
        level2 = st.selectbox(
            "下位階層",
            [product_col, bu_col] + ([city_col] if city_col else []),
            index=1,
        )
        level3 = st.selectbox(
            "さらに下位（任意）",
            ["(なし)"]
            + ([city_col] if (city_col and city_col not in [level1, level2]) else []),
            index=0,
        )
        tree_kind = st.radio(
            "ツリーの種類", ["サンバースト", "ツリーマップ"], horizontal=True
        )

    if st.button("ROICツリーを生成（SQL→実行→プロット）"):
        prompt = f"""
        {qtable} を対象に、ROIC を {level1} → {level2}{(" → "+level3) if level3!="(なし)" else ""} の階層で集計し、
        階層別の ROIC と売上/投下資本など主要分解（NOPATマージン、回転率）を返す SELECT SQL を作成してください。
        - 出力列例: {level1}, {level2}{(", "+level3) if level3!="(なし)" else ""}, roic_value, margin, turnover, revenue, invested_capital
        - SELECTのみ、```sql ... ``` 1ブロック
        - 列名が不明なら最善推定
        """
        with st.spinner("ツリー用SQLを生成中…"):
            raw = cortex_complete_messages(
                session,
                [
                    {"role": "system", "content": "You are a SQL generator."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                guard=run_guard,
            )
        tree_sql = extract_sql_from_text(raw)
        if not tree_sql:
            st.error(f"SQLが抽出できませんでした。\n出力:\n{raw}")
        else:
            st.code(tree_sql, language="sql")
            if not enforce_select_single_table(tree_sql, qtable):
                st.error("安全ガード（SELECT限定・対象テーブルのみ）に抵触しました。")
            else:
                try:
                    with st.spinner("実行中…"):
                        treedf = run_sql_df(session, tree_sql)
                    st.success(f"Rows: {len(treedf):,}")
                    render_dataframe(treedf.head(200))

                    value_col = next(
                        (c for c in treedf.columns if "roic" in c.lower()), None
                    )
                    size_col = next(
                        (
                            c
                            for c in treedf.columns
                            if c.lower() in ["revenue", "sales", "amount", "value"]
                        ),
                        None,
                    )
                    path_cols = [level1, level2] + (
                        [level3] if level3 != "(なし)" else []
                    )
                    if tree_kind == "サンバースト":
                        fig = px.sunburst(
                            treedf,
                            path=path_cols,
                            values=size_col,
                            color=value_col,
                            color_continuous_scale="RdBu",
                        )
                    else:
                        fig = px.treemap(
                            treedf,
                            path=path_cols,
                            values=size_col,
                            color=value_col,
                            color_continuous_scale="RdBu",
                        )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"実行失敗: {e}")

    st.markdown("### 寄与分解（滝グラフ）— どの要因がROIC変化に効いたか")
    st.caption(
        "ベースとシナリオの差分を、主要ドライバごとの寄与に分解（One-At-A-Time法などの近似）。"
    )
    if st.button("滝グラフ（寄与分解）を生成"):
        if "base_df" not in st.session_state or "scenario_df" not in st.session_state:
            st.info(
                "先にベース（セクション3）とシナリオ（セクション5/7）を実行してください。"
            )
        else:
            prompt = f"""
            ROICの変化（ベース→シナリオ）を、主要ドライバ（例：原材料単価、販売価格、人件費、物流費 等）の寄与に分解してください。
            One-At-A-Time（OAT）近似で構いません。結果は滝グラフに使える形（driver, contribution）で返してください。
            - 対象テーブル: {qtable}
            - 出力列: driver, contribution_value（ROICポイントまたはΔ）
            - SELECTのみ、```sql ... ``` 1ブロック
            """
            with st.spinner("寄与分解SQLを生成中…"):
                raw = cortex_complete_messages(
                    session,
                    [
                        {"role": "system", "content": "You are a SQL generator."},
                        {"role": "user", "content": prompt},
                    ],
                    model=model_name,
                    guard=run_guard,
                )
            wf_sql = extract_sql_from_text(raw)
            if not wf_sql:
                st.error(f"SQLが抽出できませんでした。\n出力:\n{raw}")
            else:
                st.code(wf_sql, language="sql")
                if not enforce_select_single_table(wf_sql, qtable):
                    st.error(
                        "安全ガード（SELECT限定・対象テーブルのみ）に抵触しました。"
                    )
                else:
                    try:
                        with st.spinner("実行中…"):
                            wdf = run_sql_df(session, wf_sql)
                        st.success(f"Rows: {len(wdf):,}")
                        render_dataframe(wdf)

                        driver_col = next(
                            (c for c in wdf.columns if "driver" in c.lower()),
                            wdf.columns[0],
                        )
                        contrib_col = next(
                            (
                                c
                                for c in wdf.columns
                                if "contrib" in c.lower() or "value" in c.lower()
                            ),
                            wdf.columns[-1],
                        )
                        fig = px.waterfall(
                            wdf,
                            x=driver_col,
                            y=contrib_col,
                            title="ROIC変化の寄与（滝グラフ）",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"実行失敗: {e}")


def render_about_section() -> None:
    """
    アプリ内に簡易設計ノート（運用時の前提/拡張ポイント）を同梱する。
    """
    with st.expander("📘 About / 設計ノート", expanded=False):
        st.markdown(
            """
**信頼境界**: 生成AIは SNOWFLAKE.CORTEX.COMPLETE() により **Snowflake 内実行**。外部通信なし。  
**ガード**: SELECT限定 + 単一ビュー参照（CTE許容、完全修飾のみチェック）。  
**ROIC近似**: NOPAT≈(PRICE−COST_PER_UNIT)×QUANTITY×(1−0.21) / 投下資本≈(PRICE×QUANTITY)×0.8  
**期間取り扱い**: FY_Q は 'Q1'〜'Q4' を原文のまま使用。FYQ_LABEL = CONCAT('FY', FY, '-', FY_Q)。  
**UI動線**: 初期は「最新期の事業部別ROIC」を自動表示→会議の起点を提供。  
**拡張点**: ROIC厳密化はビュー差し替え / プロンプトは build_system_prompts() を編集 / 安全ポリシーは enforce_* を編集。
            """
        )


# =========================
# 8) メインフロー
# =========================
def main() -> None:
    """全体の画面フローを段階的に描画する。"""
    configure_page()
    roic_table, model_name, run_guard, sample_rows, session = (
        render_connection_sidebar()
    )
    st.session_state["model_name"] = model_name
    st.session_state["run_guard"] = run_guard
    st.session_state["sample_rows"] = int(sample_rows)

    # テーブル名を安全化
    qtable = sanitize_identifier(roic_table)

    # 初期表示（最新期の事業部別ROIC）
    render_default_division_roic(session, qtable)

    # スキーマ表示・ロール推測・列選択
    roles, _schema_df = render_schema_profile_and_role_guess(session, qtable)
    product_col = roles["product_col"]
    bu_col = roles["bu_col"]
    city_col = roles["city_col"]
    period_col = roles["period_col"]
    roic_col = roles["roic_col"]

    # SYSTEM プロンプト生成
    sys_sql, sys_summary, sys_scn_sql, sys_scn_summary = build_system_prompts(
        qtable,
        product_col,
        bu_col,
        period_col,
        roic_col,
        city_col if city_col else None,
    )

    # チャット→SQL生成→要約
    render_chat_to_sql(session, sys_sql, sys_summary, model_name, run_guard)

    st.divider()
    # 実行 & 可視化（ベース）
    render_execute_base(session, qtable)

    st.divider()
    # ワンクリック定型プロンプト
    render_preset_prompts(
        bu_col,
        product_col,
        roic_col,
        city_col if city_col else None,
        sys_sql,
        sys_summary,
        model_name,
        run_guard,
        session,
    )

    st.divider()
    # What-If（自然言語シナリオ）
    render_scenario_nl(
        session, qtable, sys_scn_sql, sys_scn_summary, model_name, run_guard
    )

    st.divider()
    # ベース vs シナリオ比較・検証
    render_comparison_and_validation(
        session, product_col, bu_col, period_col, city_col if city_col else None
    )

    st.divider()
    # ヒートマップ（感度分析）
    render_heatmap_section(
        product_col,
        bu_col,
        city_col if city_col else None,
        roic_col,
        qtable,
        model_name,
        run_guard,
        session,
    )

    st.divider()
    # ROICツリー & 滝グラフ
    render_roic_tree_and_waterfall(
        session,
        qtable,
        bu_col,
        product_col,
        city_col if city_col else None,
        model_name,
        run_guard,
    )

    # 設計ノート
    render_about_section()


# =========================
# 9) Script entry
# =========================
if __name__ == "__main__":
    main()

```


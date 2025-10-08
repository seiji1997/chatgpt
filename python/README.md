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




```python
# app_roic_min_snowflake_hardened.py
# ------------------------------------------------------------
# ROICミニ分析アプリ（Snowflake内限定 / Secrets不要 / エラーに強化）
# 目的:
#  - 日本語で質問 → Cortexが SELECT SQL を生成 → 構文/列/クォートを自動補正→ 検証→（必要なら）自動リペア再生成 → 実行
#  - 結果を表と簡易Plotlyで確認
# 前提:
#  - 実行は Snowsight の Streamlit（ローカル不可）
#  - FY_Q は 'Q1'〜'Q4' の**大文字文字列**をそのまま使用（未クォートやスマートダッシュは自動補正）
# 依存:
#  - streamlit, pandas, plotly, snowflake-snowpark-python
# ------------------------------------------------------------

import re
import json
import textwrap
import pandas as pd
import plotly.express as px
import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException

# =========================
# Page
# =========================
st.set_page_config(page_title="ROIC Mini — Snowflake (Hardened)", layout="wide")
st.title("ROIC Mini — Snowflake (Hardened)")
st.caption(
    "日本語→SQL→自動補正/検証/再生成→実行→表/Plotly（FY_Qは 'Q1'〜'Q4' を文字列のまま）"
)


# =========================
# Session（Snowflake内のアクティブセッションを使用）
# =========================
@st.cache_resource(show_spinner=False)
def get_session() -> Session:
    """
    Snowsight の既存アクティブセッションをそのまま利用（Secrets不要）。
    ローカル実行は不可。
    """
    from snowflake.snowpark.context import get_active_session

    return get_active_session()


def rerun():
    """Streamlitの再描画（API差異にポリフィル対応）"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


try:
    session = get_session()
    session.sql("select 1").collect()
except Exception as e:
    st.error(f"Snowflakeセッション取得エラー: {e}")
    st.stop()


# 現在のコンテキスト（短名補完にも使用）
def current_context(session: Session):
    row = session.sql(
        "select current_database(), current_schema(), current_warehouse(), current_role()"
    ).collect()[0]
    return row[0], row[1], row[2], row[3]  # db, schema, wh, role


cur_db, cur_schema, cur_wh, cur_role = current_context(session)


# =========================
# Helpers
# =========================
def qualify_name(name: str, db: str, sch: str) -> str:
    """
    短名 ('TBL' or 'SCH.TBL') を現在の DB/SCHEMA で完全修飾名に補完。
    既に完全修飾ならそのまま返す。
    """
    n = (name or "").strip().replace('"', "")
    if not n:
        return n
    parts = n.split(".")
    if len(parts) == 1:
        return f"{db}.{sch}.{parts[0]}"
    if len(parts) == 2:
        return f"{db}.{parts[0]}.{parts[1]}"
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def run_sql_df(sql: str) -> pd.DataFrame:
    """SnowflakeでSQLを実行して pandas DataFrame を返す。"""
    return session.sql(sql).to_pandas()


def build_schema_hint(session: Session, qualified_table: str) -> str:
    """
    INFORMATION_SCHEMA から列名と型を取得し、LLMへのヒント文字列を作る。
    列名の取り違えを防ぐために毎回渡す。
    """
    try:
        db, sch, tbl = [p.replace('"', "") for p in qualified_table.split(".")]
        df = session.sql(
            f"""
            SELECT column_name, data_type
            FROM {db}.information_schema.columns
            WHERE table_schema = '{sch.upper()}' AND table_name = '{tbl.upper()}'
            ORDER BY ordinal_position
        """
        ).to_pandas()
        cols = ", ".join([f"{r.COLUMN_NAME}({r.DATA_TYPE})" for _, r in df.iterrows()])
        return f"列一覧: {cols}"
    except Exception:
        return "列一覧: （取得失敗。既知の列を想定して回答）"


def system_sql_prompt(session: Session, qualified_table: str) -> str:
    """
    SQL生成用Systemプロンプト（列ヒント/厳格フォーマット指示込み）。
    FY_Q は 'Q1'〜'Q4' の文字列、必ずクォート。ダッシュは ASCII '-'。
    """
    cols_hint = build_schema_hint(session, qualified_table)
    return textwrap.dedent(
        f"""
    あなたは Snowflake 上の分析者です。ユーザーの日本語質問に対して
    **実行可能な Snowflake SQL（SELECTのみ）** を1本だけ作成します。

    対象: {qualified_table}
    {cols_hint}

    厳守:
    - 出力は **```sql** で始まり **```** で終わる **1ブロックのみ**
    - **SELECT 限定**（単一テーブル参照に限定。DDL/DMLは不可）
    - **FY_Q は 'Q1'〜'Q4' の大文字文字列**。WHERE/IN では **必ずシングルクォート**で囲む（例: FY_Q IN ('Q1','Q2')）
    - 文字列連結やラベルは **ASCII の '-'** を使う（`–` や `—` は禁止）
    - 列名は上記の列一覧から選ぶ（曖昧なら最も近い列を採用）
    - 読みやすい ORDER BY / LIMIT を付与してよい
    """
    ).strip()


def system_summary_prompt() -> str:
    """生成SQLの簡易サマリ（日本語、箇条書き）"""
    return textwrap.dedent(
        """
    あなたは SQL の説明者です。
    ユーザーの質問と生成されたSQLを受け取り、日本語で**箇条書きの短い要約**を返してください。
    - どの列で絞り込み（WHERE）
    - どの軸で集計/比較（GROUP BY/窓関数）
    - 並べ替え/上位抽出（ORDER BY, LIMIT）
    """
    ).strip()


def extract_sql_from_text(text: str) -> str | None:
    """LLM応答からSQLを頑丈に抽出。```sql```優先、なければ最初の SELECT から末尾。"""
    if not text:
        return None
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";")
    m = re.search(r"(?is)\bselect\b.+", text)
    if m:
        sql = m.group(0).strip()
        sql = re.sub(r"`{3,}\s*$", "", sql).strip()
        return sql.rstrip(";")
    return None


def normalize_sql(sql: str) -> str:
    """
    生成SQLの簡易補正：
      - Unicodeダッシュ（–—− 等）→ ASCII '-'
      - FY_Q = Q1 → FY_Q = 'Q1'
      - FY_Q IN (Q1, Q2) → FY_Q IN ('Q1','Q2')
    """
    if not sql:
        return sql
    # 1) ダッシュ正規化
    sql = re.sub(r"[\u2010-\u2015\u2212]", "-", sql)
    # 2) FY_Q の単一比較をクォート
    sql = re.sub(
        r"(\bFY_Q\s*(?:=|<>|!=)\s*)(Q[1-4])\b", r"\1'\2'", sql, flags=re.IGNORECASE
    )

    # 3) FY_Q IN (...) の中身をクォート
    def _quote_in(m):
        head, inside = m.group(1), m.group(2)
        fixed = re.sub(r"\b(Q[1-4])\b", r"'\1'", inside, flags=re.IGNORECASE)
        return f"{head}{fixed})"

    sql = re.sub(r"(\bFY_Q\s+IN\s*\()\s*([^)]+)\)", _quote_in, sql, flags=re.IGNORECASE)
    return sql


def validate_sql(session: Session, sql: str) -> tuple[bool, str | None]:
    """
    SQLを軽く検証。構文/列解決のみ確認（データ取得はしない）。
    WITH q AS (<sql>) SELECT * FROM q LIMIT 0 でチェック。
    """
    try:
        wrap = f"WITH q AS ({sql}) SELECT * FROM q LIMIT 0"
        session.sql(wrap).collect()
        return True, None
    except Exception as e:
        return False, str(e)


def enforce_readonly_single_table(sql: str, allowed_table: str) -> bool:
    """簡易チェック：SELECTのみ & FROM が対象の単一に限定。"""
    if not re.match(r"(?is)^\s*select\b", sql or ""):
        return False
    froms = re.findall(r"(?is)\bfrom\b\s+([A-Z0-9_.$\"]+)", (sql or "").upper())
    return all(f == allowed_table.upper() for f in froms) if froms else True


def cortex_complete_messages(messages, model: str, guard: bool = True) -> str:
    """
    Cortex COMPLETE を呼び出し、messages（Chat形式）に対する応答テキストを取得。
    Guardは環境に応じてON/OFF可。AI_COMPLETEフォールバックも用意。
    """
    options = {"temperature": 0.1}
    if guard:
        options["guard"] = "ON"
    jopt = json.dumps(options, ensure_ascii=False)
    jmsg = json.dumps(messages, ensure_ascii=False)

    # 1st: SNOWFLAKE.CORTEX.COMPLETE
    sql = f"""
      SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{model}',
        parse_json('{jmsg}'),
        parse_json('{jopt}')
      ) AS response
    """
    try:
        df = run_sql_df(sql)
        return df.iloc[0, 0]
    except Exception as e1:
        # 2nd: AI_COMPLETE フォールバック（環境差異対策）
        try:
            sql2 = f"SELECT AI_COMPLETE('{model}', parse_json('{jmsg}'), parse_json('{jopt}')) AS response"
            df2 = run_sql_df(sql2)
            return df2.iloc[0, 0]
        except Exception as e2:
            return f"[CORTEX ERROR] {e1} / [AI_COMPLETE ERROR] {e2}"


def generate_sql_with_repair(
    qualified_table: str, user_prompt: str, model: str, guard: bool = True
) -> tuple[str | None, str]:
    """
    1) 列ヒント入りSystemでSQL生成 → 抽出 → 正規化 → 単一テーブルガード
    2) 検証失敗なら、エラーメッセージをLLMに渡し“修正版 SQL”を再生成（最大1回）
    戻り値: (最終SQL or None, ログ文字列)
    """
    logs = []

    # 1st pass: 生成
    messages = [
        {"role": "system", "content": system_sql_prompt(session, qualified_table)},
        {"role": "user", "content": user_prompt},
    ]
    raw = cortex_complete_messages(messages, model=model, guard=guard)
    logs.append(f"[pass1 raw]\n{str(raw)[:1200]}")
    sql = extract_sql_from_text(raw)
    if not sql:
        return None, "\n".join(logs + ["[extract] failed"])
    sql = normalize_sql(sql)

    # 単一テーブル/SELECTガード
    if not enforce_readonly_single_table(sql, qualified_table):
        logs.append("[guard] single-table or SELECT-only failed → 修正指示")
        fix_msg = [
            {"role": "system", "content": system_sql_prompt(session, qualified_table)},
            {
                "role": "user",
                "content": f"{qualified_table} だけを参照する形で修正してください。\n```sql\n{sql}\n```",
            },
        ]
        raw2 = cortex_complete_messages(fix_msg, model=model, guard=guard)
        logs.append(f"[guard fix raw]\n{str(raw2)[:1200]}")
        sql = extract_sql_from_text(raw2) or sql
        sql = normalize_sql(sql)

    ok, err = validate_sql(session, sql)
    if ok:
        return sql, "\n".join(logs + ["[validate] OK"])

    # 2nd pass: エラーを渡してリペア
    repair_sys = textwrap.dedent(
        """
    あなたはSQL修復者です。与えられたSQLとエラーを元に、
    構文・列名・クォート・ダッシュを修正した **SELECTのみ** のSQLを1本返してください。
    出力は ```sql ... ``` の1ブロックのみ。
    FY_Q は 'Q1'〜'Q4' の文字列で必ずクォート、ダッシュはASCII '-'
    """
    ).strip()
    repair_msg = [
        {"role": "system", "content": repair_sys},
        {
            "role": "user",
            "content": f"対象: {qualified_table}\nエラー: {err}\n元SQL:\n```sql\n{sql}\n```",
        },
    ]
    raw3 = cortex_complete_messages(repair_msg, model=model, guard=guard)
    logs.append(f"[repair raw]\n{str(raw3)[:1200]}")
    sql2 = extract_sql_from_text(raw3)
    if sql2:
        sql2 = normalize_sql(sql2)
        ok2, err2 = validate_sql(session, sql2)
        if ok2:
            return sql2, "\n".join(logs + ["[repair] OK"])
        logs.append(f"[repair validate NG] {err2}")

    return None, "\n".join(logs + [f"[final NG] {err}"])


# =========================
# App State
# =========================
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # [{user, sql?, explanation?, error?, log?}]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("接続コンテキスト")
    st.caption(
        f"DB: **{cur_db}** / SCHEMA: **{cur_schema}** / WH: **{cur_wh}** / ROLE: **{cur_role}**"
    )

    st.header("対象 & モデル")
    target_table_raw = st.text_input(
        "対象 (DB.SCHEMA.TABLE / VIEW)",
        value="",  # 短名OK（現在DB/SCHEMAで補完）
        placeholder=f"{cur_db}.{cur_schema}.ROIC_BASE  または  ROIC_BASE",
        help="短名でも可。現在の DB/SCHEMA を使って完全修飾名に補完します。",
    )
    model_name = st.text_input("Cortexモデル", value="mistral-large")
    run_guard = st.checkbox("Cortex Guard（推奨ON）", value=True)
    st.markdown("---")
    st.caption(
        "FY_Q は 'Q1'〜'Q4' を**文字列**で扱います。未クォートやスマートダッシュはアプリ側で自動補正します。"
    )

qualified_table = (
    qualify_name(target_table_raw, cur_db, cur_schema) if target_table_raw else ""
)

# =========================
# Chat UI → SQL生成 → 要約
# =========================
st.subheader("チャットで質問 → SQL生成（自動補正/検証/再生成）→ 要約 → 実行")

with st.expander("直近の履歴", expanded=False):
    if not st.session_state["chat"]:
        st.write("（まだ履歴はありません）")
    else:
        for i, turn in enumerate(st.session_state["chat"], 1):
            st.markdown(f"**Q{i}（あなた）:** {turn['user']}")
            if "sql" in turn:
                st.markdown("**生成SQL（最終）**")
                st.code(turn["sql"], language="sql")
            if "explanation" in turn:
                st.markdown("**要約**")
                st.write(turn["explanation"])
            if "error" in turn:
                st.error(turn["error"])
            if "log" in turn:
                with st.expander("デバッグログ", expanded=False):
                    st.code(turn["log"])

user_prompt = st.text_area(
    "日本語で質問（例：FY=2024 の 'Q1'～'Q4'で、事業部別ROICを平均し、上位10件）",
    height=100,
)

c1, c2 = st.columns([1, 1])
with c1:
    do_generate = st.button("SQL生成（Cortex）")
with c2:
    if st.button("履歴クリア"):
        st.session_state["chat"] = []
        rerun()

if do_generate:
    if not qualified_table:
        st.warning("まずサイドバーの『対象』にテーブル/ビューを指定してください。")
    elif not user_prompt.strip():
        st.warning("質問を入力してください。")
    else:
        with st.spinner("CortexがSQLを作成・検証中…"):
            gen_sql, gen_log = generate_sql_with_repair(
                qualified_table, user_prompt, model_name, guard=run_guard
            )

        new_turn = {"user": user_prompt}
        if not gen_sql:
            new_turn["error"] = (
                "SQLの生成/修復に失敗しました。ログを開いて内容を確認してください。"
            )
            new_turn["log"] = gen_log
        else:
            # 要約
            messages_sum = [
                {"role": "system", "content": system_summary_prompt()},
                {
                    "role": "user",
                    "content": f"質問: {user_prompt}\n\nSQL:\n```sql\n{gen_sql}\n```",
                },
            ]
            with st.spinner("SQLの要約を作成中…"):
                summary = cortex_complete_messages(
                    messages_sum, model=model_name, guard=run_guard
                )
            new_turn["sql"] = gen_sql
            new_turn["explanation"] = summary
            new_turn["log"] = gen_log

        st.session_state["chat"].append(new_turn)
        rerun()

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
    elif not qualified_table:
        st.warning("『対象』を指定してください。")
    elif not enforce_readonly_single_table(selected_sql, qualified_table):
        st.error("安全ガード：SELECT限定・対象テーブルのみ参照で実行してください。")
    else:
        try:
            with st.spinner("SQL実行中…"):
                df = run_sql_df(selected_sql)
            st.success(f"Rows: {len(df):,}")
            st.dataframe(df, use_container_width=True)

            if not df.empty:
                cols = list(df.columns)
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
                    st.info("数値列が見当たらないため、グラフは省略します。")
        except Exception as e:
            st.error(f"実行失敗: {e}")

# =========================
# Footer
# =========================
st.caption(
    "Snowflake内限定・Secrets不要。FY_Qは 'Q1'〜'Q4' を文字列で扱い、未クォート/スマートダッシュを自動補正。構文エラー時はLLMに渡して自動リペア再生成。"
)

```

import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
import io
import unicodedata  


def normalize_issue(x):
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKC", str(x).strip())   # Unicode 正規化 + 去頭尾空白
    s = re.sub(r"\s+", " ", s)                          # 內文多空白壓成一個
    s = s.replace("－", "-").replace("–", "-").replace("—", "-")  # 破折號統一
    return s

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="台灣科技政策策略分析儀表板", layout="wide")

DATA_PATH = "Goriginal.csv"

# --------- 欄位同義詞 ---------
ALIASES = {
    "策略類型分類": ["策略類型", "策略類別"],
    "主題類別": ["主題分類", "主題"],
    "期數": ["期別", "期次"]
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    rename_map = {}
    for canon, alts in ALIASES.items():
        if canon in df.columns:
            continue
        for a in alts:
            if a in df.columns:
                rename_map[a] = canon
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# --------- 讀檔 ---------
@st.cache_data(show_spinner=False)
def load_data():
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950", "latin1"]:
        try:
            return normalize_columns(pd.read_csv(DATA_PATH, encoding=enc))
        except Exception:
            continue
    st.error(f"讀檔失敗：{DATA_PATH}")
    st.stop()

def parse_period(v):
    if pd.isna(v):
        return np.nan
    nums = re.findall(r"\d+", str(v))
    return int(nums[0]) if nums else np.nan

def sort_period_values(values):
    return sorted(values, key=lambda x: (parse_period(x), str(x)))

def group_counts_by_period(df, period_col="期數"):
    if period_col not in df.columns:
        return pd.DataFrame(columns=[period_col, "策略數量", "_sort"])
    g = df.groupby(period_col).size().rename("策略數量").reset_index()
    g["_sort"] = g[period_col].apply(parse_period)
    g = g.sort_values("_sort")
    return g

def group_counts_by_period_and_type(df, period_col="期數", type_col="策略類型分類"):
    if period_col not in df.columns or type_col not in df.columns:
        return pd.DataFrame(columns=[period_col, type_col, "筆數", "_sort"])
    g = df.groupby([period_col, type_col]).size().rename("筆數").reset_index()
    g["_sort"] = g[period_col].apply(parse_period)
    g = g.sort_values(["_sort", type_col])
    return g
    
    # 轉為DataFrame並排序
    df_words = pd.DataFrame(list(word_counts.items()), columns=['關鍵詞', '頻次'])
    df_words = df_words.sort_values('頻次', ascending=True)
    
    # 創建橫向條形圖
    fig = px.bar(df_words, x='頻次', y='關鍵詞', orientation='h',
                 title="策略關鍵詞頻次分析",
                 color='頻次',
                 color_continuous_scale='Blues')
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# --------- 網絡分析功能 ---------
def create_network_diagram(df):
    """創建網絡關係圖（使用散點圖模擬）"""
    if len(df) == 0:
        return None
    
    # 模擬網絡節點
    nodes_data = []
    
    # 議題節點
    for idx, row in df.iterrows():
        nodes_data.append({
            'x': np.random.uniform(0.1, 1.12),
            'y': np.random.uniform(0.6, 1),
            'size': int(row.get('策略數量', 3)) * 6,
            'type': '議題',
            'name': row.get('議題', f'議題{idx+1}')[1:24] + '...',
            'color': 'lightblue'
        })
    
    # 策略節點
    strategy_nodes = [
        {'x': 0.3, 'y': 0.3, 'size': 30, 'type': '策略', 'name': '發展', 'color': 'red'},
        {'x': 0.6, 'y': 0.3, 'size': 30, 'type': '策略', 'name': '強化', 'color': 'red'},
        {'x': 0.9, 'y': 0.3, 'size': 30, 'type': '策略', 'name': '創新', 'color': 'red'}
    ]
    nodes_data.extend(strategy_nodes)
    
    # 主題節點
    theme_positions = [(0.1, 0.05), (0.3, 0.05), (0.5, 0.05), (0.7, 0.05), (0.9, 0.05),(1.1, 0.05)]
    themes = ['科技應用', '綠色轉型', '其他跨域', '數位治理', '人力轉型','社會公平']
    
    for i, theme in enumerate(themes):
        if i < len(theme_positions):
            nodes_data.append({
                'x': theme_positions[i][0],
                'y': theme_positions[i][1],
                'size': 25,
                'type': '主題',
                'name': theme,
                'color': 'lightgreen'
            })
    
    df_nodes = pd.DataFrame(nodes_data)
    
    # 創建散點圖
    fig = px.scatter(df_nodes, x='x', y='y', size='size', color='type',
                     hover_name='name', title="台灣科技政策三層網絡結構圖",
                     color_discrete_map={'議題': 'lightblue', '策略': 'red', '主題': 'lightgreen'})
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    
    return fig

# ====== 載入資料 ======
df = load_data()

# ====== Header & KPI ======
st.title("🇹🇼 台灣科技政策策略分析儀表板")
st.caption(f"資料來源：{DATA_PATH}")

# 篩選器（僅資料欄位）
with st.sidebar:
    st.header("篩選器")
    period_options = sort_period_values(df["期數"].dropna().unique().tolist()) if "期數" in df.columns else []
    sel_period = st.multiselect("選擇期數", period_options, default=[])
    
    sel_type = st.multiselect("選擇策略類型",
                              sorted(df["策略類型分類"].dropna().unique().tolist()) if "策略類型分類" in df.columns else [])
    
    sel_theme = st.multiselect("選擇主題",
                              sorted(df["主題類別"].dropna().unique().tolist()) if "主題類別" in df.columns else [])

def apply_filters(df):
    out = df.copy()
    if "期數" in out.columns and sel_period:
        out = out[out["期數"].isin(sel_period)]
    if "策略類型分類" in out.columns and sel_type:
        out = out[out["策略類型分類"].isin(sel_type)]
    if "主題類別" in out.columns and sel_theme:
        out = out[out["主題類別"].isin(sel_theme)]
    return out

filtered = apply_filters(df)

# ---- KPI：依「期數」計算平均與最大 ----
g = group_counts_by_period(filtered)
# 以筆數計算，重複議題也各自計
total_issues = int(len(filtered))
avg_per_period = round(pd.to_numeric(filtered["策略數量"], errors="coerce").mean(), 1) if "策略數量" in filtered.columns and len(filtered) > 0 else 0
max_per_period = int(pd.to_numeric(filtered["策略數量"], errors="coerce").max()) if "策略數量" in filtered.columns and len(filtered) > 0 else 0
cover_periods = int(g["期數"].nunique()) if len(g) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("總議題數", total_issues)
with c2:
    st.metric("平均策略數量", avg_per_period)
with c3:
    st.metric("最大策略數量", max_per_period)
with c4:
    st.metric("涵蓋期數", cover_periods)

st.markdown("---")

# ====== 主頁互動圖 ======
col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 策略類型分布")
    if "策略類型分類" in filtered.columns and len(filtered)>0:
        counts = filtered["策略類型分類"].value_counts().rename_axis("策略類型").reset_index(name="筆數")
        fig = px.pie(counts, names="策略類型", values="筆數", hole=0.0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("資料缺少『策略類型分類』欄位。")

with col2:
    st.subheader("📊 各期議題數量趨勢")
    if len(g) > 0:
        fig = px.bar(g, x="期數", y="策略數量", category_orders={"期數": sort_period_values(g["期數"].tolist())})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("資料缺少『期數』欄位。")

st.markdown("---")

# ====== Notebook 互動圖（Tabs） ======
st.markdown("## 🔎 進階視覺化分析")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 堆疊長條：期數 × 類型", 
    "📈 折線趨勢：期數 × 類型", 
    "🔥 主題 × 類別：交叉表與熱力圖",
    "➰ 主題×策略類型交叉分析", 
    "🕸️ 三層網絡結構圖"
])

with tab1:
    st.subheader("🧩 堆疊長條：期數 × 類型")
    gpt = group_counts_by_period_and_type(filtered)
    if len(gpt) > 0:
        fig = px.bar(gpt, x="期數", y="筆數", color="策略類型分類", barmode="stack",
                     category_orders={"期數": sort_period_values(gpt["期數"].unique().tolist())})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("需要『期數』與『策略類型分類』欄位。")

with tab2:
    st.subheader("📈 折線趨勢：期數 × 類型")
    gpt = group_counts_by_period_and_type(filtered)
    if len(gpt) > 0:
        fig = px.line(gpt, x="期數", y="筆數", color="策略類型分類", markers=True,
                      category_orders={"期數": sort_period_values(gpt["期數"].unique().tolist())})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("需要『期數』與『策略類型分類』欄位。")

with tab3:
    st.subheader("🔥 主題 × 類型：交叉表與熱力圖")
    if {"主題類別","策略類型分類"}.issubset(filtered.columns):
        ctab = pd.crosstab(filtered["主題類別"], filtered["策略類型分類"])
        st.dataframe(ctab, use_container_width=True)
        fig = px.imshow(ctab, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="策略類型分類", yaxis_title="主題類別")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("需要『主題類別』與『策略類型分類』欄位。")

with tab4:
    st.subheader("➰ 主題×策略類型交叉分析圖")
    if {"主題類別","策略類型分類"}.issubset(filtered.columns) and len(filtered) > 0:
        # 創建交叉分析數據
        cross_data = []
        for _, row in filtered.iterrows():
            cross_data.append({
                '主題': row['主題類別'],
                '策略類型': row['策略類型分類'],
                '數量': 1
            })
        
        df_cross = pd.DataFrame(cross_data)
        cross_summary = df_cross.groupby(['主題', '策略類型'])['數量'].sum().reset_index()
        
        # 創建堆疊長條圖
        fig = px.bar(cross_summary, x='主題', y='數量', color='策略類型', 
                     title="各主題下的策略類型分布", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示詳細數據表
        st.write("**詳細交叉分析表:**")
        pivot_table = cross_summary.pivot(index='主題', columns='策略類型', values='數量').fillna(0)
        st.dataframe(pivot_table, use_container_width=True)
    else:
        st.info("需要『主題類別』與『策略類型分類』欄位。")



with tab5:
    st.subheader("🕸️ 台灣科技政策三層網絡結構圖")
    if len(filtered) > 0:
        network_fig = create_network_diagram(filtered)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
            
            st.write("**網絡結構說明:**")
            st.write("""
            - 🔵 **藍色圓點**: 政策議題（大小代表策略數量）
            - 🔴 **紅色圓點**: 核心策略（發展、強化、創新）
            - 🟢 **綠色圓點**: 主題分類（科技應用、綠色轉型等）
            
            此圖展示了議題、策略與主題之間的三層網絡關係，反映政策架構的整體布局。
            """)
        else:
            st.info("無法生成網絡圖")
    else:
        st.info("沒有數據可供分析")

st.markdown("---")

# ====== 策略詳情（依期數排序 + 次排序：策略數量↓） ======
st.markdown("## 📝 議題及策略詳情")

preferred_cols = ["期數", "議題", "策略類型分類", "策略數量", "主題類別", "主題分類", "策略"]
cols_to_show = [c for c in preferred_cols if c in filtered.columns]

_tmp = filtered.copy()
if "期數" in _tmp.columns:
    _tmp["_sort"] = _tmp["期數"].apply(parse_period)
    if "策略數量" in _tmp.columns:
        _tmp = _tmp.sort_values(by=["_sort", "策略數量"], ascending=[True, False])
    else:
        _tmp = _tmp.sort_values(by=["_sort"], ascending=True)
    _tmp = _tmp.drop(columns=["_sort"])

_tmp = _tmp.reset_index(drop=True)
st.dataframe(_tmp[cols_to_show] if cols_to_show else _tmp, use_container_width=True)

st.markdown("### 📥 匯出功能")
col1, col2 = st.columns(2)

with col1:
    st.download_button("📄 下載篩選後資料 (CSV)", 
                      data=filtered.to_csv(index=False).encode("utf-8-sig"),
                      file_name="filtered_policy_data.csv", 
                      mime="text/csv")

with col2:
    # 提供儀表板程式碼下載
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            code_content = f.read()
        st.download_button("💻 下載儀表板程式碼",
                          data=code_content.encode("utf-8"),
                          file_name="enhanced_policy_dashboard.py",
                          mime="text/plain")
    except:
        st.info("程式碼下載功能暫時無法使用")

# ====== 說明與使用指南 ======
with st.expander("📖 使用說明"):
    st.markdown("""
    ### 🎯 儀表板功能說明
    
    **基本功能:**
    - 左側篩選器：可依期數、策略類型、主題進行資料篩選
    - KPI指標：顯示總策略數、平均策略數量等關鍵指標
    
    **視覺化分析:**
    1. **策略類型分布**: 圓餅圖顯示延續型、創新、破壞式創新策略比例
    2. **各期趨勢**: 長條圖展示不同期數的策略數量變化
    3. **主題×策略交叉分析**: 深入分析各主題下的策略類型分布
    4. **關鍵詞文字雲**: 視覺化策略文本中的高頻關鍵詞
    5. **三層網絡圖**: 展示議題、策略、主題間的網絡關係
    
    **資料匯出:**
    - 可下載篩選後的資料為CSV格式
    - 可下載完整的儀表板程式碼
    
    ### 🔧 技術需求
    - Python 3.7+
    - Streamlit, Pandas, Plotly, Matplotlib
    - 安裝指令: pip install streamlit pandas plotly matplotlib numpy
    
    ### 📊 執行方式
    ```bash
    streamlit run enhanced_policy_dashboard.py
    ```
    
    ### 📁 資料路徑設定
    請將 DATA_PATH 變數修改為您的 CSV 檔案實際路徑
    """)

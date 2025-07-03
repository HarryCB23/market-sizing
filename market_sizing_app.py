import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# --- COLOR PALETTE ---
PRIMARY = "#3b5f8f"
SECONDARY = "#7cb8a4"
NEUTRAL = "#e7ecf3"
SURFACE = "#fff"
TEXT = "#232b32"
GRID = "#e7ecf3"

# --- MODERN UI CUSTOM CSS ---
st.markdown("""
<style>
html, body, [class*="stApp"] {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif!important;
    background-color: #f7f9fb;
}
h1, h2, h3, h4, h5, h6 {
    color: #3b5f8f;
    font-weight: 700;
    letter-spacing: -0.015em;
    margin-bottom: 0.5em;
}
.stTitle, .stHeader {
    margin-top: 0.5em;
    margin-bottom: 1em;
}
p, .stMarkdown, .stText, .stDataFrame, .stAlert {
    color: #232b32;
    font-size: 1.05rem;
}
.stTabs [role="tablist"] {
    background: #ffffff;
    border-radius: 14px 14px 0 0;
    padding: 0.5em 1.5em 0 1.5em;
    border-bottom: 1.5px solid #e7ecf3;
    margin-bottom: 0;
}
.stTabs [role="tab"] {
    color: #3b5f8f;
    font-weight: 600;
    font-size: 1.08em;
    border-radius: 8px 8px 0 0;
    padding: 0.5em 1.1em;
    margin-right: 0.3em;
}
.stTabs [aria-selected="true"] {
    background: #e7ecf3;
    box-shadow: 0 2px 8px 0 #dbe2ea44;
    border-bottom: 2.5px solid #3b5f8f;
}
section.main > div {
    padding: 2.5vw 2vw 2vw 2vw;
    max-width: 1200px;
    margin: auto;
}
@media (max-width: 900px) {
    section.main > div {
        padding: 1vw 2vw;
    }
    .stTabs [role="tab"] { font-size: 0.98em; }
}
/* File Uploader */
.stFileUploader {
    background: #e7ecf3;
    border: 2px dashed #bfc8d2;
    border-radius: 14px;
    padding: 2em 1.5em;
    margin-bottom: 1.5em;
    transition: border 0.2s;
}
.stFileUploader:hover {
    border: 2px solid #3b5f8f;
}
.stFileUploader label {
    font-weight: 600;
    color: #3b5f8f;
    font-size: 1.08em;
}
.stFileUploader > div > button {
    background: linear-gradient(90deg, #3b5f8f 0%, #7cb8a4 100%);
    color: #fff;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 0.45em 1.2em;
}
.stFileUploader > div > button:hover {
    background: #7cb8a4;
}
/* Sliders */
.stSlider > div[data-baseweb="slider"] {
    background: #e7ecf3;
    border-radius: 10px;
    padding: 0.7em 1.2em;
    margin-bottom: 1.2em;
}
.stSlider [role="slider"] {
    background: #7cb8a4;
    border: 3px solid #3b5f8f;
    height: 18px;
    width: 18px;
    box-shadow: 0 2px 8px #7cb8a42a;
}
.stSlider .rc-slider-track {
    background: #3b5f8f;
}
.stSlider .rc-slider-dot {
    border: 2px solid #7cb8a4;
}
.stSlider label, .stSlider span {
    color: #3b5f8f;
    font-weight: 500;
}
/* Multi-selects */
.stMultiSelect {
    background: #e7ecf3!important;
    border-radius: 10px;
    padding: 0.5em 0.9em;
}
.stMultiSelect label {
    color: #3b5f8f;
    font-weight: 500;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #dbe2ea;
    color: #3b5f8f;
    border-radius: 8px;
    padding: 0.2em 0.6em;
    font-weight: 500;
    font-size: 0.98em;
}
.stMultiSelect [data-baseweb="tag"]:hover, .stMultiSelect [data-baseweb="tag"]:focus {
    background: #7cb8a4;
    color: #fff;
}
.stMultiSelect input {
    font-size: 1em;
    color: #232b32;
}
/* Metric Cards */
.metric-card {
    background: #e7ecf3;
    border-radius: 16px;
    box-shadow: 0 2px 12px 0 #dbe2ea55;
    padding: 1.3em 1.1em 1.2em 1.1em;
    margin-bottom: 1.5em;
    text-align: center;
    min-height: 142px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.metric-card h4 {
    font-size: 1.12em;
    color: #3b5f8f;
    margin-bottom: 0.2em;
    font-weight: 700;
}
.metric-card .metric-number {
    color: #3b5f8f;
    font-size: 2.1em;
    font-weight: 700;
    margin: 0.1em 0 0.18em 0;
}
.metric-card .metric-label {
    font-size: 1em;
    color: #536074;
    margin: 0.3em 0 0 0;
}
/* Info/Alert Banners */
.stAlert {
    border-radius: 10px;
    font-size: 1.02em;
    padding: 0.6em 1.2em;
}
.stAlert-success {
    background: #e7f7ef;
    color: #217a4c;
    border-left: 5px solid #67b99a;
}
.stAlert-warning {
    background: #fff7e6;
    color: #a3740b;
    border-left: 5px solid #ffe58f;
}
.stAlert-error {
    background: #fff0f0;
    color: #b3261e;
    border-left: 5px solid #ffb1b1;
}
/* DataFrames, Tables */
.stDataFrame, .stTable {
    background: #fff;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    margin-bottom: 1em;
    box-shadow: 0 2px 12px 0 #dbe2ea33;
    font-size: 1em;
}
.stDataFrame th, .stDataFrame td {
    padding: 0.45em 0.7em;
}
.stDataFrame th {
    background: #e7ecf3;
    color: #3b5f8f;
    font-weight: 600;
}
/* Plotly Charts & Chart Card Effect */
.stPlotlyChart, .stVegaLiteChart {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 4px 24px 0 #dbe2ea44;
    padding: 1.2em 1.2em 0.2em 1.2em;
    margin-bottom: 2em;
}
.stPlotlyChart .legend text {
    font-size: 1em !important;
    color: #3b5f8f !important;
}
/* Buttons (General) */
.stButton>button {
    background: linear-gradient(90deg, #3b5f8f 0%, #7cb8a4 100%);
    color: #fff;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.08em;
    padding: 0.55em 1.6em;
    margin-top: 0.5em;
    margin-bottom: 1.2em;
    box-shadow: 0 1px 4px #bfc8d2aa;
    transition: background 0.15s, box-shadow 0.2s;
}
.stButton>button:hover {
    background: #7cb8a4;
    box-shadow: 0 3px 18px #bfc8d2cc;
}
.stButton>button:active {
    background: #3b5f8f;
}
/* Expanders */
.stExpanderHeader {
    background: #e7ecf3!important;
    color: #3b5f8f!important;
    font-weight: 600;
    border-radius: 10px 10px 0 0;
}
.stExpander {
    border-radius: 10px;
    box-shadow: 0 2px 10px #dbe2ea22;
    margin-bottom: 1.5em;
}
/* Misc */
hr, .stDivider {
    border: none;
    height: 1.5px;
    background: #e7ecf3;
    margin: 1.2em 0;
}
::-webkit-scrollbar-thumb {
    background: #dbe2ea;
    border-radius: 8px;
}
::-webkit-scrollbar {
    background: #f7f9fb;
}
/* Responsive */
@media (max-width:600px) {
    .stPlotlyChart, .stDataFrame, .stTable, .metric-card {
        padding: 0.5em !important;
        border-radius: 10px !important;
    }
    .stTabs [role="tab"] {
        font-size: 0.88em;
        padding: 0.3em 0.6em;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (unchanged) ---
def classify_search_intent_custom(keyword):
    keyword_lower = keyword.lower()
    transactional_keywords = [
        r'\bbuy\b', r'\bshop\b', r'\border\b', r'\bprice\b', r'\bcost\b', r'\bdeal\b',
        r'\bdiscount\b', r'\bcoupon\b', r'\bpurchase\b', r'\bfor sale\b', r'\bsale\b',
        r'\bshipping\b', r'\bcheckout\b', r'\bget\s+(\w+)\b', r'\bdownload\b',
        r'\bsign\s+up\b', r'\bsignup\b', r'\bsubscribe\b', r'\btrial\b', r'\bdemo\b',
        r'\bapply\b', r'\bcourse\b', r'\bwebinar\b', r'\bticket\b', r'\bbook\b',
        r'\bhire\b', r'\bfree\s+trial\b', r'\b(buy|order|get|purchase)\s+now\b'
    ]
    if any(re.search(k, keyword_lower) for k in transactional_keywords):
        return "Transactional"
    commercial_keywords = [
        r'\bbest\b', r'\breview\b', r'\bvs\b', r'\bcomparison\b', r'\btop\s+\d+\b', r'\bcompare\b',
        r'\balternatives\b', r'\bproduct\b', r'\bservice\b', r'\bbrand\b', r'\bfeatures\b',
        r'\bspecifications\b', r'\bratings\b', r'\bpros\s+and\s+cons\b', r'\bwhich\s+(\w+)\b',
        r'\bhow\s+to\s+choose\b', r'\bx\s+for\s+y\b', r'\bpricing\b', r'\bcost\s+of\b',
        r'\bmodel\b', r'\b(product)\s+list\b', r'\b(service)\s+provider\b', r'\bcritique\b',
        r'\bexamples\s+of\b'
    ]
    if any(re.search(k, keyword_lower) for k in commercial_keywords):
        return "Commercial Investigation"
    informational_keywords = [
        r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bguide\b', r'\btutorial\b', r'\bwhy\b', r'\bwhen\b',
        r'\bwhere\b', r'\bwho\b', r'\bdefinition\b', r'\bideas\b', r'\blearn\b', r'\bfacts\b',
        r'\bhistory\b', r'\bmeaning\b', r'\bsymptoms\b', r'\bcauses\b', r'\bexplain\b',
        r'\binformation\b', r'\btips\b', r'\badvice\b', r'\bexample\s+of\b', r'\btypes\s+of\b',
        r'\b(verb)\s+process\b', r'\bconcept\b', r'\btheory\b', r'\bwiki\b', r'\bmanual\b',
        r'\bfaq\b', r'\bproblems\b', r'\bsolutions\b', r'\b(what|how|why)\s+do\b'
    ]
    if any(re.search(k, keyword_lower) for k in informational_keywords):
        return "Informational"
    navigational_keywords = [
        r'\blogin\b', r'\bmy\s+account\b', r'\bcontact\s+us\b',
        r'\bhomepage\b', r'\bofficial\s+site\b', r'\bwebsite\b', r'\bdashboard\b'
    ]
    if len(keyword.split()) <= 2:
        if any(char.isupper() for char in keyword):
            return "Navigational"
    if any(re.search(k, keyword_lower) for k in navigational_keywords):
        return "Navigational"
    return "Unknown"

def get_keyword_type(keyword, head_term_word_limit=3):
    return "Head Term" if len(keyword.split()) <= head_term_word_limit else "Long Tail"

if 'df_keywords' not in st.session_state:
    st.session_state['df_keywords'] = None
if 'sidebar_params' not in st.session_state:
    st.session_state['sidebar_params'] = {}

tabs = st.tabs([
    "Setup",
    "Market Overview",
    "Keyword Word Count",
    "Search Intent",
    "Parent Categories",
    "Topic Trends"
])

# --- TAB 1: SETUP (Upload and Filters) ---
with tabs[0]:
    st.title("📊 SEO Market Sizing Tool")
    st.markdown("""
    Welcome to your SEO Market Sizing Tool! This app helps you analyze your Ahrefs keyword data to understand the Total Addressable Market (TAM), Serviceable Available Market (SAM), and Serviceable Obtainable Market (SOM).
    """)
    st.header("1. Upload Ahrefs Keyword Data")
    st.info("""
        Please export your keyword data from Ahrefs as a CSV file. Ensure the export includes at least the following columns:
        `Keyword`, `Volume`, `Difficulty`, `SERP Features`, `CPC`.
        Highly recommended: Include the `Intents` column for more accurate search intent classification.
        For trend analysis, ensure `Growth (3mo)`, `Growth (6mo)`, or `Growth (12mo)` columns are also included.
    """)

    uploaded_file = st.file_uploader("Choose an Ahrefs CSV file", type="csv")
    df_keywords = None
    growth_col_map_internal = {'3mo': None, '6mo': None, '12mo': None}

    if uploaded_file:
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'utf-16']
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)
                df_keywords = pd.read_csv(uploaded_file, encoding=encoding)
                st.success(f"Ahrefs data loaded successfully with {encoding} encoding!")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error loading CSV ({encoding}): {e}")
                df_keywords = None
                break
        if df_keywords is not None:
            df_keywords.columns = [col.strip().replace(' ', '_').replace('.', '').lower() for col in df_keywords.columns]
            if 'difficulty' in df_keywords.columns and 'kd' not in df_keywords.columns:
                df_keywords.rename(columns={'difficulty': 'kd'}, inplace=True)
            if 'parent_keyword' in df_keywords.columns and 'parent_topic' not in df_keywords.columns:
                df_keywords.rename(columns={'parent_keyword': 'parent_topic'}, inplace=True)
            required_cols = ['keyword', 'volume', 'kd', 'serp_features', 'cpc']
            missing_cols = [col for col in required_cols if col not in df_keywords.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}.")
                df_keywords = None
            else:
                df_keywords['volume'] = pd.to_numeric(df_keywords['volume'], errors='coerce').fillna(0).astype(int)
                df_keywords['kd'] = pd.to_numeric(df_keywords['kd'], errors='coerce').fillna(0).astype(int)
                df_keywords['cpc'] = pd.to_numeric(df_keywords['cpc'], errors='coerce').fillna(0.0)
                df_keywords['serp_features'] = df_keywords['serp_features'].fillna('')
                if 'parent_topic' not in df_keywords.columns:
                    st.warning("`Parent Topic` column not found. Using `Keyword` as fallback.")
                    df_keywords['parent_topic'] = df_keywords['keyword']
                df_keywords['parent_topic'] = df_keywords['parent_topic'].fillna(df_keywords['keyword'])
                if 'intents' in df_keywords.columns:
                    def parse_ahrefs_intents_with_modifiers(intent_str):
                        if pd.isna(intent_str) or not intent_str:
                            return "Unknown"
                        intents_list = [i.strip() for i in intent_str.split(',')]
                        primary_intent = "Unknown"
                        if 'Transactional' in intents_list:
                            primary_intent = "Transactional"
                        elif 'Commercial' in intents_list:
                            primary_intent = "Commercial Investigation"
                        elif 'Informational' in intents_list:
                            primary_intent = "Informational"
                        elif 'Navigational' in intents_list:
                            primary_intent = "Navigational"
                        modifiers = []
                        for x in ['Branded', 'Non-branded', 'Local', 'Non-local']:
                            if x in intents_list:
                                modifiers.append(x)
                        return f"{primary_intent} ({', '.join(modifiers)})" if modifiers else primary_intent
                    df_keywords['search_intent'] = df_keywords['intents'].apply(parse_ahrefs_intents_with_modifiers)
                else:
                    st.warning("No `Intents` column found. Using fallback intent classification.")
                    df_keywords['search_intent'] = df_keywords['keyword'].apply(classify_search_intent_custom)
                df_keywords['keyword_type'] = df_keywords['keyword'].apply(get_keyword_type)
                df_keywords['word_count'] = df_keywords['keyword'].apply(lambda x: len(str(x).split()))
                for col_suffix in ['3mo', '6mo', '12mo']:
                    for pc_name in [f'growth_({col_suffix})', f'global_growth_({col_suffix})']:
                        if pc_name in df_keywords.columns:
                            growth_col_map_internal[col_suffix] = pc_name
                            break
                df_keywords['growth_pct'] = 0
                df_keywords['trend'] = 'N/A'
                if growth_col_map_internal['12mo']:
                    df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_map_internal['12mo']], errors='coerce').fillna(0)
                elif growth_col_map_internal['6mo']:
                    df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_map_internal['6mo']], errors='coerce').fillna(0)
                elif growth_col_map_internal['3mo']:
                    df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_map_internal['3mo']], errors='coerce').fillna(0)
                df_keywords.loc[df_keywords['growth_pct'] > 5, 'trend'] = 'Trending Up'
                df_keywords.loc[df_keywords['growth_pct'] < -5, 'trend'] = 'Trending Down'
                st.session_state['df_keywords'] = df_keywords
                st.session_state['growth_col_map_internal'] = growth_col_map_internal
    else:
        st.info("Upload your Ahrefs data to begin.")

    df_keywords = st.session_state['df_keywords']
    if df_keywords is not None:
        st.subheader("Market Filtering (SAM)")
        min_volume = st.slider("Minimum Monthly Search Volume", min_value=0, max_value=int(df_keywords['volume'].max()), value=0, step=10)
        fixed_max_kd_for_sam = 100
        all_ahrefs_intents = sorted(df_keywords['search_intent'].unique().tolist())
        selected_intents = st.multiselect(
            "Include Search Intents",
            options=all_ahrefs_intents,
            default=[intent for intent in all_ahrefs_intents if 'Branded' not in intent]
        )
        selected_keyword_types = st.multiselect(
            "Include Keyword Types",
            options=df_keywords['keyword_type'].unique(),
            default=df_keywords['keyword_type'].unique()
        )
        st.subheader("Monetization & Obtainable Market (SOM)")
        average_rpm = st.number_input("Average RPM for 1000 Clicks ($)", min_value=0.0, value=20.0, step=1.0, format="%.2f")
        average_ctr_percentage = st.slider("Average Organic CTR (%)", min_value=1.0, max_value=100.0, value=35.0, step=0.5, format="%.1f")
        som_percentage = st.slider("Serviceable Obtainable Market (SOM) %", min_value=0, max_value=100, value=10, step=1)
        st.session_state['sidebar_params'] = dict(
            min_volume=min_volume,
            fixed_max_kd_for_sam=fixed_max_kd_for_sam,
            selected_intents=selected_intents,
            selected_keyword_types=selected_keyword_types,
            average_rpm=average_rpm,
            average_ctr_percentage=average_ctr_percentage,
            som_percentage=som_percentage
        )

df_keywords = st.session_state.get('df_keywords', None)
params = st.session_state.get('sidebar_params', None)
growth_col_map_internal = st.session_state.get('growth_col_map_internal', {'3mo': None, '6mo': None, '12mo': None})

if df_keywords is not None and params:

    min_volume = params['min_volume']
    fixed_max_kd_for_sam = params['fixed_max_kd_for_sam']
    selected_intents = params['selected_intents']
    selected_keyword_types = params['selected_keyword_types']
    average_rpm = params['average_rpm']
    average_ctr_percentage = params['average_ctr_percentage']
    som_percentage = params['som_percentage']

    average_ctr_decimal = average_ctr_percentage / 100.0

    df_sam = df_keywords[
        (df_keywords['volume'] >= min_volume) &
        (df_keywords['kd'] <= fixed_max_kd_for_sam) &
        (df_keywords['search_intent'].isin(selected_intents)) &
        (df_keywords['keyword_type'].isin(selected_keyword_types))
    ].copy()

    total_market_volume_tam = df_keywords['volume'].sum()
    total_market_clicks_tam = total_market_volume_tam * average_ctr_decimal
    total_market_revenue_tam = (total_market_clicks_tam / 1000) * average_rpm

    serviceable_market_volume_sam = df_sam['volume'].sum()
    serviceable_market_clicks_sam = serviceable_market_volume_sam * average_ctr_decimal
    serviceable_market_revenue_sam = (serviceable_market_clicks_sam / 1000) * average_rpm

    obtainable_market_volume_som = serviceable_market_volume_sam * (som_percentage / 100)
    obtainable_market_clicks_som = obtainable_market_volume_som * average_ctr_decimal
    obtainable_market_revenue_som = (obtainable_market_clicks_som / 1000) * average_rpm

    # --- TAB 2: Market Overview ---
    with tabs[1]:
        st.header("Market Sizing Overview")
        col_tam, col_sam, col_som = st.columns(3)
        with col_tam:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Addressable Market (TAM)</h4>
                    <div class="metric-number">{total_market_volume_tam:,} Searches</div>
                    <div class="metric-label">
                        Est. Clicks: {int(total_market_clicks_tam):,}<br>
                        Pot. Revenue: ${total_market_revenue_tam:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_sam:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Serviceable Available Market (SAM)</h4>
                    <div class="metric-number">{serviceable_market_volume_sam:,} Searches</div>
                    <div class="metric-label">
                        Est. Clicks: {int(serviceable_market_clicks_sam):,}<br>
                        Pot. Revenue: ${serviceable_market_revenue_sam:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_som:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Serviceable Obtainable Market (SOM)</h4>
                    <div class="metric-number">{int(obtainable_market_volume_som):,} Searches</div>
                    <div class="metric-label">
                        Est. Clicks: {int(obtainable_market_clicks_som):,}<br>
                        Pot. Revenue: ${obtainable_market_revenue_som:,.2f} ({som_percentage}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
        # (SERP Features section unchanged for brevity)

    # --- TAB 3: Keyword Word Count ---
    with tabs[2]:
        st.header("Keyword Word Count & Volume Distribution (SAM)")
        if not df_sam.empty:
            word_count_data = df_sam.groupby('word_count').agg(
                keyword_count=('keyword', 'count'),
                total_volume=('volume', 'sum')
            ).reset_index().sort_values(by='word_count')
            X = word_count_data['word_count']
            bar_y = word_count_data['keyword_count']
            line_y = word_count_data['total_volume']
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=X,
                y=bar_y,
                name='Number of Keywords',
                marker=dict(color=PRIMARY, line=dict(color=PRIMARY, width=0)),
                hovertemplate='<b>%{x} words</b><br>Keywords: %{y}<extra></extra>',
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=X,
                y=line_y,
                name='Total Search Volume',
                mode='lines+markers',
                line=dict(color=SECONDARY, width=3, shape='spline'),
                marker=dict(size=9, color=SECONDARY, line=dict(width=2, color=SURFACE)),
                hovertemplate='<b>%{x} words</b><br>Total Volume: %{y:,}<extra></extra>',
            ), secondary_y=True)
            fig.update_layout(
                plot_bgcolor=SURFACE,
                paper_bgcolor=SURFACE,
                font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color=TEXT),
                title=dict(
                    text='Keyword Word Count & Volume Distribution (SAM)',
                    font=dict(size=20, color=PRIMARY, family="Inter, Segoe UI, Arial, sans-serif"),
                    x=0.5
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    bgcolor=NEUTRAL,
                    bordercolor=NEUTRAL,
                    borderwidth=1,
                    font=dict(size=14, color=TEXT),
                    itemclick='toggleothers'
                ),
                margin=dict(l=40, r=40, b=60, t=60),
                showlegend=True,
                hovermode="x unified",
            )
            fig.update_xaxes(
                title_text="Number of Words in Keyword",
                showgrid=False,
                tickangle=-10,
                linecolor=GRID,
                tickfont=dict(size=14, color=PRIMARY),
            )
            fig.update_yaxes(
                title_text="Number of Keywords",
                showgrid=True,
                gridcolor=GRID,
                zeroline=False,
                linecolor=GRID,
                tickfont=dict(size=13, color=TEXT),
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text="Total Search Volume",
                showgrid=False,
                zeroline=False,
                linecolor=GRID,
                tickfont=dict(size=13, color=TEXT),
                secondary_y=True,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("This chart shows the distribution of keywords by their word count and the corresponding total search volume for each.")
            with st.expander("Word Count Table (Dropdown)"):
                st.dataframe(word_count_data, hide_index=True, use_container_width=True)
        else:
            st.warning("No keywords in SAM for word count breakdown. Adjust your filters in Setup.")

    # --- TAB 4: Search Intent ---
    with tabs[3]:
        st.header("Search Intent Breakdown (SAM) by Volume")
        if not df_sam.empty:
            intent_data = df_sam.groupby('search_intent').agg(
                keyword_count=('keyword', 'count'),
                total_volume=('volume', 'sum')
            ).reset_index().sort_values(by='total_volume', ascending=False)
            if not intent_data.empty:
                X = intent_data['search_intent']
                bar_y = intent_data['keyword_count']
                line_y = intent_data['total_volume']
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=X,
                    y=bar_y,
                    name='Keyword Count',
                    marker=dict(color=PRIMARY, line=dict(color=PRIMARY, width=0)),
                    hovertemplate='<b>%{x}</b><br>Keyword Count: %{y}<extra></extra>',
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=X,
                    y=line_y,
                    name='Total Volume',
                    mode='lines+markers',
                    line=dict(color=SECONDARY, width=3, shape='spline'),
                    marker=dict(size=9, color=SECONDARY, line=dict(width=2, color=SURFACE)),
                    hovertemplate='<b>%{x}</b><br>Total Volume: %{y:,}<extra></extra>',
                ), secondary_y=True)
                fig.update_layout(
                    plot_bgcolor=SURFACE,
                    paper_bgcolor=SURFACE,
                    font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color=TEXT),
                    title=dict(
                        text='Search Intent Breakdown (SAM) by Volume',
                        font=dict(size=20, color=PRIMARY, family="Inter, Segoe UI, Arial, sans-serif"),
                        x=0.5
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.05,
                        xanchor="center",
                        x=0.5,
                        bgcolor=NEUTRAL,
                        bordercolor=NEUTRAL,
                        borderwidth=1,
                        font=dict(size=14, color=TEXT),
                        itemclick='toggleothers'
                    ),
                    margin=dict(l=40, r=40, b=60, t=60),
                    showlegend=True,
                    hovermode="x unified",
                )
                fig.update_xaxes(
                    title_text="Search Intent",
                    showgrid=False,
                    tickangle=-10,
                    linecolor=GRID,
                    tickfont=dict(size=14, color=PRIMARY),
                )
                fig.update_yaxes(
                    title_text="Keyword Count",
                    showgrid=True,
                    gridcolor=GRID,
                    zeroline=False,
                    linecolor=GRID,
                    tickfont=dict(size=13, color=TEXT),
                    secondary_y=False,
                )
                fig.update_yaxes(
                    title_text="Total Search Volume",
                    showgrid=False,
                    zeroline=False,
                    linecolor=GRID,
                    tickfont=dict(size=13, color=TEXT),
                    secondary_y=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("This chart breaks down keywords by their primary search intent, showing both the count of keywords and their aggregated search volume.")
                with st.expander("Intent Breakdown Table (Dropdown)"):
                    table_data = intent_data.copy()
                    table_data['Example Keyword'] = table_data['search_intent'].apply(
                        lambda x: df_sam[df_sam['search_intent'] == x]['keyword'].iloc[0] if not df_sam[df_sam['search_intent'] == x].empty else "N/A"
                    )
                    st.dataframe(table_data, hide_index=True, use_container_width=True)
            else:
                st.warning("No search intent data found in SAM for this chart.")
        else:
            st.warning("No keywords in SAM for breakdown. Adjust your filters in Setup.")

    # --- TAB 5: Parent Categories ---
    with tabs[4]:
        st.header("Top Parent Categories by Volume & Average KD")
        if not df_keywords.empty:
            parent_topic_data = df_keywords.groupby('parent_topic').agg(
                total_volume=('volume', 'sum'),
                average_kd=('kd', 'mean'),
                average_cpc=('cpc', 'mean'),
                weighted_growth_pct=('growth_pct', lambda x: (x * df_keywords.loc[x.index, 'volume']).sum() / df_keywords.loc[x.index, 'volume'].sum() if df_keywords.loc[x.index, 'volume'].sum() > 0 else 0)
            ).reset_index()
            top_10_parent_topics = parent_topic_data.sort_values(by='total_volume', ascending=False).head(10)
            absolute_top_parent_topic = parent_topic_data.sort_values(by='total_volume', ascending=False).iloc[0] if not parent_topic_data.empty else None
            if absolute_top_parent_topic is not None:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Absolute Top Parent Category: <b>{absolute_top_parent_topic['parent_topic']}</b></h4>
                        <div class="metric-number">Total Volume: <b>{absolute_top_parent_topic['total_volume']:,}</b> Searches</div>
                        <div class="metric-label">
                            Avg. KD: <b>{absolute_top_parent_topic['average_kd']:.2f}</b> | 
                            Avg. CPC: <b>${absolute_top_parent_topic['average_cpc']:.2f}</b><br>
                            <span style="color:{'green' if absolute_top_parent_topic['weighted_growth_pct'] > 0 else 'red' if absolute_top_parent_topic['weighted_growth_pct'] < 0 else 'gray'};">
                                Avg. Growth (Weighted): <b>{absolute_top_parent_topic['weighted_growth_pct']:+.2f}%</b>
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            if not top_10_parent_topics.empty:
                X = top_10_parent_topics['parent_topic']
                bar_y = top_10_parent_topics['total_volume']
                line_y = top_10_parent_topics['average_kd']
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=X,
                    y=bar_y,
                    name='Total Search Volume',
                    marker=dict(color=PRIMARY, line=dict(color=PRIMARY, width=0)),
                    hovertemplate='<b>%{x}</b><br>Total Volume: %{y:,}<extra></extra>',
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=X,
                    y=line_y,
                    name='Average KD',
                    mode='lines+markers',
                    line=dict(color=SECONDARY, width=3, shape='spline'),
                    marker=dict(size=9, color=SECONDARY, line=dict(width=2, color=SURFACE)),
                    hovertemplate='<b>%{x}</b><br>Avg KD: %{y:.2f}<extra></extra>',
                ), secondary_y=True)
                fig.update_layout(
                    plot_bgcolor=SURFACE,
                    paper_bgcolor=SURFACE,
                    font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color=TEXT),
                    title=dict(
                        text='Top 10 Parent Categories: Volume vs. Average KD',
                        font=dict(size=20, color=PRIMARY, family="Inter, Segoe UI, Arial, sans-serif"),
                        x=0.5
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.05,
                        xanchor="center",
                        x=0.5,
                        bgcolor=NEUTRAL,
                        bordercolor=NEUTRAL,
                        borderwidth=1,
                        font=dict(size=14, color=TEXT),
                        itemclick='toggleothers'
                    ),
                    margin=dict(l=40, r=40, b=60, t=60),
                    showlegend=True,
                    hovermode="x unified",
                )
                fig.update_xaxes(
                    title_text="Parent Topic",
                    showgrid=False,
                    tickangle=-45,
                    linecolor=GRID,
                    tickfont=dict(size=14, color=PRIMARY),
                )
                fig.update_yaxes(
                    title_text="Total Search Volume",
                    showgrid=True,
                    gridcolor=GRID,
                    zeroline=False,
                    linecolor=GRID,
                    tickfont=dict(size=13, color=TEXT),
                    secondary_y=False,
                )
                fig.update_yaxes(
                    title_text="Average Keyword Difficulty (KD)",
                    showgrid=False,
                    zeroline=False,
                    linecolor=GRID,
                    tickfont=dict(size=13, color=TEXT),
                    secondary_y=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("This chart displays the top 10 parent topics by total search volume, with their average Keyword Difficulty (KD) overlaid.")
                with st.expander("Top Parent Categories Table (Dropdown)"):
                    df_display = top_10_parent_topics[['parent_topic', 'total_volume', 'average_kd', 'average_cpc', 'weighted_growth_pct']].copy()
                    df_display.rename(columns={
                        'parent_topic': 'Parent Topic',
                        'total_volume': 'Total Volume',
                        'average_kd': 'Avg. KD',
                        'average_cpc': 'Avg. CPC',
                        'weighted_growth_pct': 'Avg. Growth (%)'
                    }, inplace=True)
                    st.dataframe(df_display.style.format({
                        'Total Volume': '{:,.0f}',
                        'Avg. KD': '{:.2f}',
                        'Avg. CPC': '${:.2f}',
                        'Avg. Growth (%)': '{:+.2f}%'
                    }), hide_index=True, use_container_width=True)
            else:
                st.warning("No parent topics found with sufficient data for this chart.")
        else:
            st.warning("Please upload Ahrefs data to view parent category analysis.")

    # --- TAB 6: Topic Trends (unchanged for brevity, but you can similarly update any charts here) ---
    with tabs[5]:
        st.header("Topic Trend (Overall Data)")
        col_trend_3mo, col_trend_6mo, col_trend_12mo = st.columns(3)
    
        def display_trend_card(col_obj, period_label, growth_col_name, df_data):
            if growth_col_name and not df_data.empty:
                overall_growth_sum_numerator = df_data[growth_col_name] * df_data['volume']
                overall_growth_sum_denominator = df_data['volume']
                if overall_growth_sum_denominator.sum() > 0:
                    overall_growth_pct = overall_growth_sum_numerator.sum() / overall_growth_sum_denominator.sum()
                else:
                    overall_growth_pct = 0
                trend_indicator_color = "green" if overall_growth_pct > 0 else "red" if overall_growth_pct < 0 else "gray"
                col_obj.markdown(f"""
    <div class="metric-card">
        <h4>Overall {period_label} Growth</h4>
        <div class="metric-number" style="color:{trend_indicator_color};">
            {overall_growth_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
            else:
                col_obj.info(f"'{period_label}' data not found.")
    
        display_trend_card(col_trend_3mo, "3-Month", growth_col_map_internal['3mo'], df_keywords)
        display_trend_card(col_trend_6mo, "6-Month", growth_col_map_internal['6mo'], df_keywords)
        display_trend_card(col_trend_12mo, "12-Month", growth_col_map_internal['12mo'], df_keywords)
    
        if any(growth_col_map_internal.values()):
            if df_keywords['trend'].nunique() > 1:
                trend_counts = df_keywords['trend'].value_counts().reset_index()
                trend_counts.columns = ['Trend', 'Count']
                # Modern Plotly bar chart for trend distribution
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=trend_counts['Trend'],
                    y=trend_counts['Count'],
                    marker=dict(color=PRIMARY),
                    hovertemplate='<b>%{x}</b><br>Keywords: %{y}<extra></extra>',
                ))
                fig_trend.update_layout(
                    plot_bgcolor=SURFACE,
                    paper_bgcolor=SURFACE,
                    font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color=TEXT),
                    title=dict(
                        text='Keyword Trend Distribution',
                        font=dict(size=20, color=PRIMARY, family="Inter, Segoe UI, Arial, sans-serif"),
                        x=0.5
                    ),
                    margin=dict(l=40, r=40, b=60, t=60),
                    showlegend=False,
                    hovermode="x unified",
                )
                fig_trend.update_xaxes(
                    title_text="Trend",
                    showgrid=False,
                    linecolor=GRID,
                    tickfont=dict(size=14, color=PRIMARY),
                )
                fig_trend.update_yaxes(
                    title_text="Number of Keywords",
                    showgrid=True,
                    gridcolor=GRID,
                    zeroline=False,
                    linecolor=GRID,
                    tickfont=dict(size=13, color=TEXT),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                st.info("Trends are categorized based on the primary Ahrefs 'Growth (Xmo)' column (>5% increase/decrease).")
            else:
                st.info("Most keywords show a 'Stable' trend or trend data is limited.")
    
            col_up, col_down = st.columns(2)
            with col_up:
                trending_up_keywords = df_keywords[df_keywords['trend'] == 'Trending Up'].sort_values(by=['growth_pct', 'volume'], ascending=[False, False])
                if not trending_up_keywords.empty:
                    with st.expander("📈 Top Trending Up Keywords"):
                        st.dataframe(trending_up_keywords[['keyword', 'volume', 'kd', 'growth_pct']].head(10).rename(columns={'growth_pct': 'Growth (%)'}), use_container_width=True)
                else:
                    st.info("No keywords identified as 'Trending Up'.")
            with col_down:
                trending_down_keywords = df_keywords[df_keywords['trend'] == 'Trending Down'].sort_values(by=['growth_pct', 'volume'], ascending=[True, False])
                if not trending_down_keywords.empty:
                    with st.expander("📉 Top Trending Down Keywords"):
                        st.dataframe(trending_down_keywords[['keyword', 'volume', 'kd', 'growth_pct']].head(10).rename(columns={'growth_pct': 'Growth (%)'}), use_container_width=True)
                else:
                    st.info("No keywords identified as 'Trending Down'.")
        else:
            st.info("No Growth columns found in your uploaded data. Trend analysis is limited.")

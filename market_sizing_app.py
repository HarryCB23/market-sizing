import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re # For regex-based search intent classification

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SEO Market Sizing Tool")

# --- Helper Functions ---

def classify_search_intent_custom(keyword):
    """
    Classifies keyword intent based on common patterns. This is a fallback
    if Ahrefs 'Intents' column is not available.
    Args:
        keyword (str): The keyword to classify.
    Returns:
        str: The classified intent (Informational, Navigational, Commercial Investigation, Transactional, Unknown).
    """
    keyword_lower = keyword.lower()

    # --- Transactional Intent ---
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

    # --- Commercial Investigation Intent ---
    commercial_keywords = [
        r'\bbest\b', r'\breview\b', r'\bvs\b', r'\bcomparison\b', r'\btop\s+\d+\b', r'\bcompare\b',
        r'\balternatives\b', r'\bproduct\b', r'\bservice\b', r'\bbrand\b', r'\bfeatures\b',
        r'\bspecifications\b', r'\bratings\b', r'\bpros\s+and\s+cons\b', r'\bwhich\s+(\w+)\b',
        r'\bhow\s+to\s+choose\b', r'\bx\s+for\s+y\b', r'\bpricing\b', r'\bcost\s+of\b',
        r'\bmodel\b', r'\b(product)\s+list\b', r'\b(service)\s+provider\b', r'\bcritique\b',
        r'\bexamples\s+of\b' # Can be informational, but often precedes commercial
    ]
    if any(re.search(k, keyword_lower) for k in commercial_keywords):
        return "Commercial Investigation"

    # --- Informational Intent ---
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

    # --- Navigational Intent (still simplified without a specific brand list) ---
    navigational_keywords = [
        r'\blogin\b', r'\bmy\s+account\b', r'\bcontact\s+us\b',
        r'\bhomepage\b', r'\bofficial\s+site\b', r'\bwebsite\b', r'\bdashboard\b'
    ]
    # For navigational, we need to be careful not to catch informational phrases.
    # A simple heuristic for direct navigational searches is very short queries often including a proper noun.
    if len(keyword.split()) <= 2:
        if any(char.isupper() for char in keyword): # Check for capitalization indicating a proper noun/brand
            return "Navigational"
    if any(re.search(k, keyword_lower) for k in navigational_keywords):
        return "Navigational"


    return "Unknown"

def get_keyword_type(keyword, head_term_word_limit=3):
    """
    Classifies a keyword as 'Head Term' or 'Long Tail' based on word count.
    Args:
        keyword (str): The keyword to classify.
    Returns:
        str: 'Head Term' or 'Long Tail'.
    """
    return "Head Term" if len(keyword.split()) <= head_term_word_limit else "Long Tail"


# --- Streamlit App Layout ---

st.title("📊 SEO Market Sizing Tool")

st.markdown("""
Welcome to your SEO Market Sizing Tool! This app helps you analyze your Ahrefs keyword data to understand the Total Addressable Market (TAM), Serviceable Available Market (SAM), and Serviceable Obtainable Market (SOM) for your niche.
You can also assess competitive landscape, search intent, trends, and monetization potential.
""")

st.divider()

# --- 1. Ahrefs Data Upload ---
st.header("1. Upload Ahrefs Keyword Data")
st.info("""
    Please export your keyword data from Ahrefs (e.g., from Keywords Explorer or Site Explorer's Organic Keywords report)
    as a CSV file. Ensure the export includes at least the following columns:
    `Keyword`, `Volume`, `Difficulty`, `SERP Features`.
    **Highly recommended:** Include the `Intents` column for more accurate search intent classification.
    For improved trend analysis, ensure `Growth (3mo)`, `Growth (6mo)`, or `Growth (12mo)` columns are also included.
""")
uploaded_file = st.file_uploader("Choose an Ahrefs CSV file", type="csv")

df_keywords = None
if uploaded_file is not None:
    # Try reading CSV with multiple encodings
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'utf-16'] 
    
    for encoding in encodings_to_try:
        try:
            # Reset file pointer for each attempt
            uploaded_file.seek(0) 
            df_keywords = pd.read_csv(uploaded_file, encoding=encoding)
            st.success(f"Ahrefs data loaded successfully with {encoding} encoding!")
            break # Exit loop if successful
        except UnicodeDecodeError:
            continue # Try next encoding
        except Exception as e:
            st.error(f"An unexpected error occurred while reading with {encoding} encoding: {e}")
            df_keywords = None
            break # Stop trying if another type of error occurs
    
    if df_keywords is None:
        st.error("Could not read the CSV file with common encodings (UTF-8, Latin-1, CP1252, UTF-16). Please check your file's encoding and format.")
    else:
        # Standardize column names for easier access (case-insensitive and handle common variations)
        df_keywords.columns = [col.strip().replace(' ', '_').replace('.', '').lower() for col in df_keywords.columns]

        # Explicitly rename 'difficulty' to 'kd' if present
        if 'difficulty' in df_keywords.columns and 'kd' not in df_keywords.columns:
            df_keywords.rename(columns={'difficulty': 'kd'}, inplace=True)

        # Check for essential columns (now including 'kd' after potential rename)
        required_cols = ['keyword', 'volume', 'kd', 'serp_features'] 
        missing_cols = [col for col in required_cols if col not in df_keywords.columns]

        if missing_cols:
            st.error(f"Missing required columns in your CSV: {', '.join(missing_cols)}. Please ensure `Keyword`, `Volume`, `Difficulty`, `SERP Features` are present in your Ahrefs export.")
            df_keywords = None # Invalidate dataframe if essential columns are missing
        else:
            st.write("First 5 rows of your data:")
            st.dataframe(df_keywords.head())
            # Data Cleaning and Preparation
            df_keywords['volume'] = pd.to_numeric(df_keywords['volume'], errors='coerce').fillna(0).astype(int)
            df_keywords['kd'] = pd.to_numeric(df_keywords['kd'], errors='coerce').fillna(0).astype(int)
            df_keywords['serp_features'] = df_keywords['serp_features'].fillna('')
            df_keywords['parent_topic'] = df_keywords.get('parent_topic', df_keywords['keyword']).fillna(df_keywords['keyword']) # Use keyword if parent_topic is missing

            # --- Search Intent Classification ---
            if 'intents' in df_keywords.columns:
                st.info("Using Ahrefs 'Intents' column for search intent classification.")
                def parse_ahrefs_intent(intent_str):
                    if pd.isna(intent_str) or not intent_str:
                        return "Unknown"
                    intents = [i.strip().lower() for i in intent_str.split(',')]
                    
                    # Prioritize core intents
                    if 'transactional' in intents:
                        return "Transactional"
                    if 'commercial' in intents:
                        return "Commercial Investigation"
                    if 'informational' in intents:
                        return "Informational"
                    if 'navigational' in intents:
                        return "Navigational"
                    return "Unknown"
                df_keywords['search_intent'] = df_keywords['intents'].apply(parse_ahrefs_intent)
            else:
                st.warning("Ahrefs 'Intents' column not found. Falling back to custom keyword-based intent classification (may be less accurate).")
                df_keywords['search_intent'] = df_keywords['keyword'].apply(classify_search_intent_custom)

            # Apply custom keyword type classification
            df_keywords['keyword_type'] = df_keywords['keyword'].apply(get_keyword_type)

            # --- Trend Analysis (using Ahrefs Growth columns) ---
            # Prioritize Growth (12mo), then (6mo), then (3mo)
            growth_col_found = None
            for col_suffix in ['12mo', '6mo', '3mo']:
                # Handle variations in column names like 'global_growth_(12mo)' or just 'growth_(12mo)'
                potential_col_names = [f'growth_({col_suffix})', f'global_growth_({col_suffix})']
                for pc_name in potential_col_names:
                    if pc_name in df_keywords.columns:
                        growth_col_found = pc_name
                        break
                if growth_col_found:
                    break
            
            if growth_col_found:
                st.info(f"Using `{growth_col_found}` column for trend analysis.")
                df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_found], errors='coerce').fillna(0)

                df_keywords['trend'] = 'Stable'
                df_keywords.loc[df_keywords['growth_pct'] > 0, 'trend'] = 'Trending Up'
                df_keywords.loc[df_keywords['growth_pct'] < 0, 'trend'] = 'Trending Down'
            else:
                st.warning("No Ahrefs `Growth (3mo)`, `Growth (6mo)`, or `Growth (12mo)` columns found. Trend analysis will be limited.")
                df_keywords['trend'] = 'N/A'


if df_keywords is not None:
    st.divider()

    # --- 2. Market Definition and Sizing Parameters ---
    st.header("2. Define Your Market")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Market Filtering")
        st.write("Filter keywords to define your specific Serviceable Available Market (SAM).")

        min_volume = st.slider("Minimum Monthly Search Volume for SAM", min_value=0, max_value=int(df_keywords['volume'].max()), value=0, step=10)
        max_kd = st.slider("Maximum Keyword Difficulty (KD) for SAM", min_value=0, max_value=100, value=100, step=5)
        
        selected_intents = st.multiselect(
            "Include Search Intents for SAM",
            options=df_keywords['search_intent'].unique(),
            default=df_keywords['search_intent'].unique()
        )

        selected_keyword_types = st.multiselect(
            "Include Keyword Types for SAM",
            options=df_keywords['keyword_type'].unique(),
            default=df_keywords['keyword_type'].unique()
        )

        # Apply filters to create SAM dataset
        df_sam = df_keywords[
            (df_keywords['volume'] >= min_volume) &
            (df_keywords['kd'] <= max_kd) &
            (df_keywords['search_intent'].isin(selected_intents)) &
            (df_keywords['keyword_type'].isin(selected_keyword_types))
        ].copy() # Use .copy() to avoid SettingWithCopyWarning


    with col2:
        st.subheader("Monetization & Obtainable Market (SOM)")

        average_rpm = st.number_input(
            "Average Revenue Per Mille (RPM) for 1000 Clicks ($)",
            min_value=0.0,
            value=20.0, # Default RPM
            step=1.0,
            format="%.2f"
        )
        st.info("RPM represents the estimated revenue you generate per 1000 organic clicks.")

        som_percentage = st.slider(
            "Serviceable Obtainable Market (SOM) Percentage of SAM (%)",
            min_value=0,
            max_value=100,
            value=10, # Default SOM capture
            step=1
        )
        st.info(f"This is the percentage of the Serviceable Available Market (SAM) you realistically expect to capture.")

    st.divider()

    # --- 3. Market Sizing Calculations ---
    st.header("3. Market Sizing Overview")

    # Calculate TAM, SAM, SOM
    total_market_volume_tam = df_keywords['volume'].sum()
    total_market_clicks_tam = total_market_volume_tam * 0.35 # Assuming an average 35% organic CTR for top results
    total_market_revenue_tam = (total_market_clicks_tam / 1000) * average_rpm

    serviceable_market_volume_sam = df_sam['volume'].sum()
    serviceable_market_clicks_sam = serviceable_market_volume_sam * 0.35 # Assuming an average 35% organic CTR for top results
    serviceable_market_revenue_sam = (serviceable_market_clicks_sam / 1000) * average_rpm

    obtainable_market_volume_som = serviceable_market_volume_sam * (som_percentage / 100)
    obtainable_market_clicks_som = obtainable_market_volume_som * 0.35 # Assuming an average 35% organic CTR
    obtainable_market_revenue_som = (obtainable_market_clicks_som / 1000) * average_rpm


    st.subheader("Estimated Market Potential")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Addressable Market (TAM)", f"{total_market_volume_tam:,} Monthly Searches")
        st.caption(f"Estimated Clicks: {int(total_market_clicks_tam):,} | Potential Revenue: ${total_market_revenue_tam:,.2f}")

    with col2:
        st.metric("Serviceable Available Market (SAM)", f"{serviceable_market_volume_sam:,} Monthly Searches")
        st.caption(f"Estimated Clicks: {int(serviceable_market_clicks_sam):,} | Potential Revenue: ${serviceable_market_revenue_sam:,.2f}")


    with col3:
        st.metric(f"Serviceable Obtainable Market (SOM) ({som_percentage}%)", f"{int(obtainable_market_volume_som):,} Monthly Searches")
        st.caption(f"Estimated Clicks: {int(obtainable_market_clicks_som):,} | Potential Revenue: ${obtainable_market_revenue_som:,.2f}")

    st.markdown("---")

    # --- Market Breakdown Visualizations ---
    st.header("4. Market Breakdown & Insights")

    # Keyword Type Breakdown
    st.subheader("Keyword Type Breakdown (SAM)")
    if not df_sam.empty:
        type_counts = df_sam['keyword_type'].value_counts().reset_index()
        type_counts.columns = ['Keyword Type', 'Count']
        fig_type = px.pie(type_counts, values='Count', names='Keyword Type', title='Distribution of Keyword Types in SAM')
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.warning("No keywords in SAM for breakdown. Adjust your filters.")

    # Search Intent Breakdown
    st.subheader("Search Intent Breakdown (SAM)")
    if not df_sam.empty:
        intent_counts = df_sam['search_intent'].value_counts().reset_index()
        intent_counts.columns = ['Search Intent', 'Count']
        fig_intent = px.pie(intent_counts, values='Count', names='Search Intent', title='Distribution of Search Intents in SAM')
        st.plotly_chart(fig_intent, use_container_width=True)
    else:
        st.warning("No keywords in SAM for breakdown. Adjust your filters.")

    # Average Keyword Difficulty
    st.subheader("Average Keyword Difficulty (SAM)")
    if not df_sam.empty:
        avg_kd_sam = df_sam['kd'].mean()
        st.metric("Average KD in SAM", f"{avg_kd_sam:.2f}")
    else:
        st.warning("No keywords in SAM to calculate average KD.")

    # SERP Features Breakdown
    st.subheader("SERP Features Present (SAM)")
    if not df_sam.empty:
        all_features = {}
        for features_str in df_sam['serp_features']:
            if features_str: # Ensure not empty string
                # Regex to extract features, handling variations like "Featured Snippet, Sitelinks"
                current_features = [f.strip() for f in re.split(r',|;', features_str) if f.strip()]
                for feature in current_features:
                    all_features[feature] = all_features.get(feature, 0) + 1
        
        if all_features:
            features_df = pd.DataFrame(all_features.items(), columns=['SERP Feature', 'Count'])
            fig_features = px.bar(features_df, x='SERP Feature', y='Count', title='Count of SERP Features in SAM Keywords')
            st.plotly_chart(fig_features, use_container_width=True)
        else:
            st.info("No specific SERP features detected in the selected keywords (or 'N/A' for all).")
    else:
        st.warning("No keywords in SAM for SERP features breakdown.")

    # Topic Trending Up or Down
    if 'trend' in df_keywords.columns and df_keywords['trend'].nunique() > 1: # Check if trend analysis was performed and there's variation
        st.subheader("Topic Trend (Overall Data)")
        trend_counts = df_keywords['trend'].value_counts().reset_index()
        trend_counts.columns = ['Trend', 'Count']
        fig_trend = px.bar(trend_counts, x='Trend', y='Count', title='Keyword Trend Distribution')
        st.plotly_chart(fig_trend, use_container_width=True)
        st.info("Trends are calculated based on Ahrefs 'Growth (Xmo)' columns from your export.")
    else:
        st.info("Ahrefs Growth columns (e.g., 'Growth (12mo)') not found or show no significant variation. Trend analysis limited.")

    st.divider()

    # --- 6. Google Search Console Connection (Placeholder) ---
    st.header("5. Your Market Position (via Google Search Console)")
    st.info("""
        This section would connect to the Google Search Console API to show your actual performance (impressions, clicks, average position)
        over time for the keywords identified in your market. This allows for a comparison of your obtainable market against your current reality.
        
        **Note:** For a live integration, you would typically need to set up OAuth2 authentication with the Google Search Console API.
        This often involves user consent and handling client IDs/secrets, which is beyond the scope of a direct, single-file Streamlit app
        in this environment.
        
        For demonstration purposes, imagine this section would display:
        - Your total organic clicks and impressions over time.
        - Your average ranking position for target keywords.
        - Your estimated market share based on GSC clicks vs. total market clicks.
    """)
    # Placeholder for GSC data
    st.write("*(GSC data trends and market share comparison would appear here after API integration)*")
    st.button("Connect to Google Search Console (Feature under development)") # Placeholder button


    st.divider()

    st.header("What's Next?")
    st.markdown("""
    This app provides a foundational market sizing analysis. Here are some ideas for future enhancements:
    * **Advanced GSC Integration:** Fully implement the GSC API connection to pull live data for your actual performance and market share.
    * **Interactive CTR Model:** Allow users to define a custom CTR curve based on ranking positions for more accurate click estimations.
    * **Keyword Grouping:** Enable manual or AI-driven grouping of keywords into content clusters for more refined market analysis.
    * **Downloadable Reports:** Export summary data and visualizations to PDF/CSV.
    * **Historical Trends:** Deeper analysis of keyword trends using all available historical volume data if the 'Growth' columns aren't sufficient.
    """)

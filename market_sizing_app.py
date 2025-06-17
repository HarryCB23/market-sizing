import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re # For regex-based search intent classification

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SEO Market Sizing Tool")

# --- Helper Functions ---

def classify_search_intent(keyword):
    """
    Classifies keyword intent based on common patterns.
    Args:
        keyword (str): The keyword to classify.
    Returns:
        str: The classified intent (Informational, Navigational, Commercial Investigation, Transactional, Unknown).
    """
    keyword_lower = keyword.lower()

    # Transactional intent
    transactional_keywords = ['buy', 'shop', 'order', 'price', 'cost', 'deal', 'discount', 'coupon', 'purchase', 'for sale']
    if any(k in keyword_lower for k in transactional_keywords):
        return "Transactional"

    # Commercial investigation intent
    commercial_keywords = ['best', 'review', 'vs', 'comparison', 'top', 'compare', 'alternatives', 'product', 'service']
    if any(k in keyword_lower for k in commercial_keywords):
        return "Commercial Investigation"

    # Informational intent
    informational_keywords = ['how to', 'what is', 'guide', 'tutorial', 'examples', 'why', 'when', 'where', 'who', 'definition', 'ideas', 'learn']
    if any(k in keyword_lower for k in informational_keywords):
        return "Informational"

    # Navigational intent (harder without specific brand list, but can look for very short, specific queries)
    # This is a simplified approach; real-world navigational often involves specific brand names
    if len(keyword.split()) <= 2 and any(char.isupper() for char in keyword): # Simple heuristic for potential brand
        return "Navigational"

    return "Unknown"

def get_keyword_type(keyword, head_term_word_limit=3):
    """
    Classifies a keyword as 'Head Term' or 'Long Tail' based on word count.
    Args:
        keyword (str): The keyword to classify.
        head_term_word_limit (int): Maximum number of words for a head term.
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
    `Keyword`, `Volume`, `KD`, `SERP Features`, `Parent Topic`.
    If you want trend analysis, ensure historical volume columns are also included (e.g., `Volume Jan 2024`, `Volume Feb 2024`, etc.).
""")
uploaded_file = st.file_uploader("Choose an Ahrefs CSV file", type="csv")

df_keywords = None
if uploaded_file is not None:
    try:
        df_keywords = pd.read_csv(uploaded_file)
        st.success("Ahrefs data loaded successfully!")
        st.write("First 5 rows of your data:")
        st.dataframe(df_keywords.head())

        # Standardize column names for easier access (case-insensitive and handle common variations)
        df_keywords.columns = [col.strip().replace(' ', '_').replace('.', '').lower() for col in df_keywords.columns]

        # Check for essential columns
        required_cols = ['keyword', 'volume', 'kd', 'serp_features'] # Parent topic might not always be there
        missing_cols = [col for col in required_cols if col not in df_keywords.columns]

        if missing_cols:
            st.error(f"Missing required columns in your CSV: {', '.join(missing_cols)}. Please check your Ahrefs export settings.")
            df_keywords = None # Invalidate dataframe if essential columns are missing
        else:
            # Data Cleaning and Preparation
            df_keywords['volume'] = pd.to_numeric(df_keywords['volume'], errors='coerce').fillna(0).astype(int)
            df_keywords['kd'] = pd.to_numeric(df_keywords['kd'], errors='coerce').fillna(0).astype(int)
            df_keywords['serp_features'] = df_keywords['serp_features'].fillna('')
            df_keywords['parent_topic'] = df_keywords.get('parent_topic', df_keywords['keyword']).fillna(df_keywords['keyword']) # Use keyword if parent_topic is missing

            # Apply custom classification functions
            df_keywords['search_intent'] = df_keywords['keyword'].apply(classify_search_intent)
            df_keywords['keyword_type'] = df_keywords['keyword'].apply(get_keyword_type)

            # --- Trend Analysis (looking for historical volume columns) ---
            # Identify columns that look like monthly volumes (e.g., 'volume_jan_2024')
            volume_cols = [col for col in df_keywords.columns if re.match(r'volume_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)_\d{4}', col)]
            
            if len(volume_cols) >= 2: # Need at least 2 months to determine a trend
                st.info(f"Found {len(volume_cols)} historical volume columns for trend analysis.")
                # Sort columns by year and month
                volume_cols_sorted = sorted(volume_cols, key=lambda x: (int(x.split('_')[-1]), pd.to_datetime(x.split('_')[1], format='%b').month))

                df_keywords['recent_volume'] = pd.to_numeric(df_keywords[volume_cols_sorted[-1]], errors='coerce').fillna(0)
                df_keywords['previous_volume'] = pd.to_numeric(df_keywords[volume_cols_sorted[-2]], errors='coerce').fillna(0) # Compare last two available months for a simple trend

                # Calculate monthly change
                df_keywords['monthly_volume_change'] = df_keywords['recent_volume'] - df_keywords['previous_volume']
                df_keywords['monthly_volume_change_pct'] = ((df_keywords['recent_volume'] - df_keywords['previous_volume']) / df_keywords['previous_volume'].replace(0, 1)) * 100 # Avoid division by zero

                # Simple trend classification
                df_keywords['trend'] = 'Stable'
                df_keywords.loc[df_keywords['monthly_volume_change_pct'] > 10, 'trend'] = 'Trending Up' # Define threshold for 'up'
                df_keywords.loc[df_keywords['monthly_volume_change_pct'] < -10, 'trend'] = 'Trending Down' # Define threshold for 'down'
            else:
                st.warning("Less than two historical volume columns found. Trend analysis will be limited.")
                df_keywords['trend'] = 'N/A' # Set default if no trend data


    except Exception as e:
        st.error(f"Error reading CSV: {e}. Please ensure it's a valid CSV file.")

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
        st.info("Trends are calculated based on the last two available historical volume columns in your Ahrefs export (if present).")
    else:
        st.info("Historical volume data for trend analysis is not sufficiently available in your CSV or shows no significant variation.")

    st.divider()

    # --- 5. Competitor Analysis (Placeholder) ---
    st.header("5. Competitor Analysis")
    st.info("""
        To analyze competitors, you would typically need their estimated organic traffic and keyword overlaps.
        For a full competitor bubble graph, you would either upload separate competitor data or connect to Ahrefs API.
        For this prototype, we'll provide a placeholder for where you'd input competitor data or where a more
        advanced integration would pull it.
    """)

    st.subheader("Top Competitors (Manual Input Placeholder)")
    comp_data = []
    num_competitors = st.number_input("Number of Top Competitors to Analyze", min_value=1, max_value=5, value=3)

    for i in range(num_competitors):
        st.markdown(f"**Competitor {i+1}**")
        name = st.text_input(f"Name of Competitor {i+1}", key=f"comp_name_{i}")
        est_traffic = st.number_input(f"Estimated Monthly Organic Traffic for {name} (from Ahrefs Site Explorer)", min_value=0, value=0, step=1000, key=f"comp_traffic_{i}")
        keywords_overlap = st.number_input(f"Number of Overlapping Keywords with Your Site for {name}", min_value=0, value=0, step=100, key=f"comp_keywords_{i}")
        if name and est_traffic > 0:
            comp_data.append({'Competitor': name, 'Estimated Traffic': est_traffic, 'Keyword Overlap': keywords_overlap})
    
    if comp_data:
        df_comp = pd.DataFrame(comp_data)
        st.write("Competitor Data (Manual Input):")
        st.dataframe(df_comp)

        # Bubble chart (Traffic vs Keyword Overlap, size by Traffic)
        if not df_comp.empty:
            fig_bubble = px.scatter(df_comp, x='Keyword Overlap', y='Estimated Traffic', size='Estimated Traffic',
                                    text='Competitor', hover_name='Competitor',
                                    title='Top Competitors: Traffic vs. Keyword Overlap',
                                    size_max=60)
            fig_bubble.update_traces(textposition='top center')
            fig_bubble.update_layout(xaxis_title="Keyword Overlap (with Your Site)",
                                     yaxis_title="Estimated Monthly Organic Traffic")
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("Enter competitor data to visualize.")


    st.divider()

    # --- 6. Google Search Console Connection (Placeholder) ---
    st.header("6. Your Market Position (via Google Search Console)")
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
    * **Historical Trends:** Deeper analysis of keyword trends beyond just two months.
    """)

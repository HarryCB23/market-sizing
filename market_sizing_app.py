import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Import make_subplots
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
""")

st.divider()

# Initialize df_keywords outside of the if block so it's always defined
df_keywords = None
# Initialize growth_col_map_internal at a higher scope
growth_col_map_internal = {
    '3mo': None,
    '6mo': None,
    '12mo': None
}

# --- 1. Ahrefs Data Upload (Instructional text in main body) ---
st.header("1. Upload Ahrefs Keyword Data")
st.info("""
    Please export your keyword data from Ahrefs (e.g., from Keywords Explorer or Site Explorer's Organic Keywords report)
    as a CSV file. Ensure the export includes at least the following columns:
    `Keyword`, `Volume`, `Difficulty`, `SERP Features`, `CPC`.
    **Highly recommended:** Include the `Intents` column for more accurate search intent classification, especially for filtering branded vs. non-branded queries.
    For improved trend analysis, ensure `Growth (3mo)`, `Growth (6mo)`, or `Growth (12mo)` columns are also included.
""")

# --- Sidebar for Data Upload, Market Definition and Sizing Parameters ---
with st.sidebar:
    st.header("⚙️ Market Definition & Parameters")
    
    # CSV file uploader now in sidebar
    uploaded_file = st.file_uploader("Choose an Ahrefs CSV file", type="csv", key="sidebar_uploader")

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
            
            # Explicitly rename 'parent_keyword' to 'parent_topic' if present
            if 'parent_keyword' in df_keywords.columns and 'parent_topic' not in df_keywords.columns:
                df_keywords.rename(columns={'parent_keyword': 'parent_topic'}, inplace=True)

            # Check for essential columns (now including 'kd' and 'cpc' after potential rename)
            required_cols = ['keyword', 'volume', 'kd', 'serp_features', 'cpc'] 
            missing_cols = [col for col in required_cols if col not in df_keywords.columns]

            if missing_cols:
                st.error(f"Missing required columns in your CSV: {', '.join(missing_cols)}. Please ensure `Keyword`, `Volume`, `Difficulty`, `SERP Features`, and `CPC` are present in your Ahrefs export.")
                df_keywords = None # Invalidate dataframe if essential columns are missing
            else:
                # Data Cleaning and Preparation
                # st.write("First 5 rows of your data:") # Removed from sidebar
                # st.dataframe(df_keywords.head()) # Removed from sidebar
                df_keywords['volume'] = pd.to_numeric(df_keywords['volume'], errors='coerce').fillna(0).astype(int)
                df_keywords['kd'] = pd.to_numeric(df_keywords['kd'], errors='coerce').fillna(0).astype(int)
                df_keywords['cpc'] = pd.to_numeric(df_keywords['cpc'], errors='coerce').fillna(0.0) # Handle CPC as float, fill NaN with 0
                df_keywords['serp_features'] = df_keywords['serp_features'].fillna('')
                # Ensure 'parent_topic' exists; if not, use 'keyword' as a fallback, then handle NaNs
                if 'parent_topic' not in df_keywords.columns:
                    st.warning("`Parent Topic` or `Parent Keyword` column not found. Using `Keyword` as fallback for parent topic analysis.")
                    df_keywords['parent_topic'] = df_keywords['keyword']
                df_keywords['parent_topic'] = df_keywords['parent_topic'].fillna(df_keywords['keyword'])


                # --- Search Intent Classification (Updated to preserve Branded/Non-branded) ---
                if 'intents' in df_keywords.columns:
                    # st.info("Using Ahrefs 'Intents' column for search intent classification.") # Removed from sidebar
                    def parse_ahrefs_intents_with_modifiers(intent_str):
                        if pd.isna(intent_str) or not intent_str:
                            return "Unknown"
                        intents_list = [i.strip() for i in intent_str.split(',')]
                        
                        primary_intent = "Unknown"
                        modifiers = []

                        # Define a precedence for primary intents
                        if 'Transactional' in intents_list:
                            primary_intent = "Transactional"
                        elif 'Commercial' in intents_list:
                            primary_intent = "Commercial Investigation"
                        elif 'Informational' in intents_list:
                            primary_intent = "Informational"
                        elif 'Navigational' in intents_list:
                            primary_intent = "Navigational"
                        
                        # Collect modifiers
                        if 'Branded' in intents_list:
                            modifiers.append("Branded")
                        if 'Non-branded' in intents_list:
                            modifiers.append("Non-branded")
                        if 'Local' in intents_list: # Assuming 'Local' is also a modifier
                            modifiers.append("Local")
                        if 'Non-local' in intents_list:
                            modifiers.append("Non-local")

                        if modifiers:
                            return f"{primary_intent} ({', '.join(modifiers)})"
                        return primary_intent

                    df_keywords['search_intent'] = df_keywords['intents'].apply(parse_ahrefs_intents_with_modifiers)
                else:
                    st.warning("Ahrefs 'Intents' column not found. Falling back to custom keyword-based intent classification (may be less accurate).")
                    df_keywords['search_intent'] = df_keywords['keyword'].apply(classify_search_intent_custom)

                # Apply custom keyword type classification
                df_keywords['keyword_type'] = df_keywords['keyword'].apply(get_keyword_type)
                
                # New: Calculate word count for each keyword
                df_keywords['word_count'] = df_keywords['keyword'].apply(lambda x: len(str(x).split()))


                # --- Trend Analysis (using Ahrefs Growth columns) ---
                # Identify actual growth columns found for display purposes
                # This block now correctly populates growth_col_map_internal
                for col_suffix in ['3mo', '6mo', '12mo']:
                    for pc_name in [f'growth_({col_suffix})', f'global_growth_({col_suffix})']:
                        if pc_name in df_keywords.columns:
                            growth_col_map_internal[col_suffix] = pc_name
                            break
                
                df_keywords['growth_pct'] = 0 # Default value
                df_keywords['trend'] = 'N/A' # Default value

                # Apply growth_pct for a unified 'trend' column (prioritizing 12mo > 6mo > 3mo for main trend categorization)
                if growth_col_map_internal['12mo']: # Use internal map here
                    df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_map_internal['12mo']], errors='coerce').fillna(0)
                elif growth_col_map_internal['6mo']: # Use internal map here
                    df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_map_internal['6mo']], errors='coerce').fillna(0)
                elif growth_col_map_internal['3mo']: # Use internal map here
                    df_keywords['growth_pct'] = pd.to_numeric(df_keywords[growth_col_map_internal['3mo']], errors='coerce').fillna(0)

                # Assign categorical trend based on the best available growth_pct
                df_keywords.loc[df_keywords['growth_pct'] > 5, 'trend'] = 'Trending Up'
                df_keywords.loc[df_keywords['growth_pct'] < -5, 'trend'] = 'Trending Down'
    
    # Only show filters and monetization if df_keywords is loaded
    if df_keywords is not None:
        st.subheader("Market Filtering (SAM)")
        st.write("Filter keywords to define your specific Serviceable Available Market (SAM).")

        min_volume = st.slider("Minimum Monthly Search Volume", min_value=0, max_value=int(df_keywords['volume'].max()), value=0, step=10)
        # Removed max_kd slider: max_kd = st.slider("Maximum Keyword Difficulty (KD)", min_value=0, max_value=100, value=100, step=5)
        # Fixed max_kd to 100 or a sensible default for filtering SAM, as it's no longer a user input
        fixed_max_kd_for_sam = 100 
        
        # Updated multiselect for selected intents to reflect new granular options
        all_ahrefs_intents = sorted(df_keywords['search_intent'].unique().tolist())
        selected_intents = st.multiselect(
            "Include Search Intents",
            options=all_ahrefs_intents,
            default=[intent for intent in all_ahrefs_intents if 'Branded' not in intent] # Default to exclude branded
        )


        selected_keyword_types = st.multiselect(
            "Include Keyword Types",
            options=df_keywords['keyword_type'].unique(),
            default=df_keywords['keyword_type'].unique()
        )

        st.subheader("Monetization & Obtainable Market (SOM)")

        average_rpm = st.number_input(
            "Average Revenue Per Mille (RPM) for 1000 Clicks ($)",
            min_value=0.0,
            value=20.0, # Default RPM
            step=1.0,
            format="%.2f"
        )
        st.info("RPM represents the estimated revenue you generate per 1000 organic clicks.")

        # New: CTR Slider
        average_ctr_percentage = st.slider(
            "Average Organic CTR (%)",
            min_value=1.0,
            max_value=100.0,
            value=35.0, # Default CTR
            step=0.5,
            format="%.1f"
        )
        st.info("This average CTR will be applied to search volumes to estimate clicks.")


        som_percentage = st.slider(
            "Serviceable Obtainable Market (SOM) Percentage of SAM (%)",
            min_value=0,
            max_value=100,
            value=10, # Default SOM capture
            step=1
        )
        st.info(f"This is the percentage of the Serviceable Available Market (SAM) you realistically expect to capture.")
    else:
        # If no file is uploaded yet, provide dummy values for sidebar controls and a message
        st.warning("Upload your Ahrefs data via the file uploader in the sidebar to enable market definition filters and parameters.")
        min_volume = 0
        fixed_max_kd_for_sam = 100 # Also provide dummy value here
        selected_intents = []
        selected_keyword_types = []
        average_rpm = 20.0
        average_ctr_percentage = 35.0 # Dummy value for CTR
        som_percentage = 10


# The main content of the app should only render if df_keywords is not None
if df_keywords is not None:
    # Convert CTR percentage to a decimal for calculations
    average_ctr_decimal = average_ctr_percentage / 100.0

    # Apply filters to create SAM dataset - Use fixed_max_kd_for_sam
    df_sam = df_keywords[
        (df_keywords['volume'] >= min_volume) &
        (df_keywords['kd'] <= fixed_max_kd_for_sam) & # Use fixed_max_kd_for_sam here
        (df_keywords['search_intent'].isin(selected_intents)) &
        (df_keywords['keyword_type'].isin(selected_keyword_types))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning


    st.divider()

    # --- 3. Market Sizing Calculations ---
    st.header("2. Market Sizing Overview") # Re-numbered header

    # Calculate TAM, SAM, SOM using the dynamic CTR
    total_market_volume_tam = df_keywords['volume'].sum()
    total_market_clicks_tam = total_market_volume_tam * average_ctr_decimal
    total_market_revenue_tam = (total_market_clicks_tam / 1000) * average_rpm

    serviceable_market_volume_sam = df_sam['volume'].sum()
    serviceable_market_clicks_sam = serviceable_market_volume_sam * average_ctr_decimal
    serviceable_market_revenue_sam = (serviceable_market_clicks_sam / 1000) * average_rpm

    obtainable_market_volume_som = serviceable_market_volume_sam * (som_percentage / 100)
    obtainable_market_clicks_som = obtainable_market_volume_som * average_ctr_decimal
    obtainable_market_revenue_som = (obtainable_market_clicks_som / 1000) * average_rpm


    st.subheader("Estimated Market Potential")
    
    # New, cleaner layout for estimated market potential
    col_tam, col_sam, col_som = st.columns(3)

    # Define a consistent blue color for searches, and custom colors for metric cards
    number_color = "#1f77b4" # Blue for numbers

    with col_tam:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 15px; 
                text-align: center; 
                min-height: 150px; 
                display: flex; 
                flex-direction: column; 
                justify-content: space-between;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                background-color: #f9f9f9;
            ">
                <h4 style="color:#262730; margin-bottom: 0px;">Total Addressable Market (TAM)</h4>
                <p style="font-size: 1.3em; font-weight: bold; color:{number_color}; margin: 5px 0;">{total_market_volume_tam:,} Searches</p>
                <p style="font-size: 0.8em; color:#555; margin-top: 8px;">
                    Est. Clicks: {int(total_market_clicks_tam):,}<br>
                    Pot. Revenue: ${total_market_revenue_tam:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_sam:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 15px; 
                text-align: center; 
                min-height: 150px; 
                display: flex; 
                flex-direction: column; 
                justify-content: space-between;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                background-color: #f9f9f9;
            ">
                <h4 style="color:#262730; margin-bottom: 0px;">Serviceable Available Market (SAM)</h4>
                <p style="font-size: 1.3em; font-weight: bold; color:{number_color}; margin: 5px 0;">{serviceable_market_volume_sam:,} Searches</p>
                <p style="font-size: 0.8em; color:#555; margin-top: 8px;">
                    Est. Clicks: {int(serviceable_market_clicks_sam):,}<br>
                    Pot. Revenue: ${serviceable_market_revenue_sam:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_som:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 15px; 
                text-align: center; 
                min-height: 150px; 
                display: flex; 
                flex-direction: column; 
                justify-content: space-between;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                background-color: #f9f9f9;
            ">
                <h4 style="color:#262730; margin-bottom: 0px;">Serviceable Obtainable Market (SOM) ({som_percentage}%)</h4>
                <p style="font-size: 1.3em; font-weight: bold; color:{number_color}; margin: 5px 0;">{int(obtainable_market_volume_som):,} Searches</p>
                <p style="font-size: 0.8em; color:#555; margin-top: 8px;">
                    Est. Clicks: {int(obtainable_market_clicks_som):,}<br>
                    Pot. Revenue: ${obtainable_market_revenue_som:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("---")

    # --- Market Breakdown Visualizations ---
    st.header("3. Market Breakdown & Insights") # Re-numbered header

    # Keyword Word Count Breakdown
    st.subheader("Keyword Word Count & Volume Distribution (SAM)")
    if not df_sam.empty:
        # Group by word_count and calculate count and sum of volume
        word_count_data = df_sam.groupby('word_count').agg(
            keyword_count=('keyword', 'count'),
            total_volume=('volume', 'sum')
        ).reset_index().sort_values(by='word_count')

        # Create combo chart using make_subplots
        fig_word_count = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar chart for keyword count (primary y-axis)
        fig_word_count.add_trace(
            go.Bar(
                x=word_count_data['word_count'],
                y=word_count_data['keyword_count'],
                name='Number of Keywords',
                marker_color='#1f77b4' # A shade of blue
            ),
            secondary_y=False,
        )

        # Add line chart for total search volume (secondary y-axis)
        fig_word_count.add_trace(
            go.Scatter(
                x=word_count_data['word_count'],
                y=word_count_data['total_volume'],
                name='Total Search Volume',
                mode='lines+markers',
                marker_color='#d62728' # A shade of red
            ),
            secondary_y=True,
        )

        # Set x-axis title
        fig_word_count.update_xaxes(title_text="Number of Words in Keyword", tickmode='linear')

        # Set y-axes titles
        fig_word_count.update_yaxes(title_text="Number of Keywords", secondary_y=False)
        fig_word_count.update_yaxes(title_text="Total Search Volume", secondary_y=True, showgrid=False) # Hide grid for secondary axis

        fig_word_count.update_layout(
            title_text='Keyword Count and Total Volume by Word Count in SAM',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_word_count, use_container_width=True)
        st.info("This chart shows the distribution of keywords by their word count and the corresponding total search volume for each. It helps identify the prevalence and search demand for shorter (head) vs. longer (long-tail) queries.")
    else:
        st.warning("No keywords in SAM for word count breakdown. Adjust your filters in the sidebar.")


    # Search Intent Breakdown (Combo Chart)
    st.subheader("Search Intent Breakdown (SAM) by Volume")
    if not df_sam.empty:
        # Group by search_intent and calculate count and sum of volume
        intent_data = df_sam.groupby('search_intent').agg(
            keyword_count=('keyword', 'count'),
            total_volume=('volume', 'sum')
        ).reset_index().sort_values(by='total_volume', ascending=False) # Sort by volume for consistent display

        if not intent_data.empty:
            # Create combo chart using make_subplots
            fig_intent_combo = make_subplots(specs=[[{"secondary_y": True}]])

            # Add bar chart for keyword count (primary y-axis)
            fig_intent_combo.add_trace(
                go.Bar(
                    x=intent_data['search_intent'],
                    y=intent_data['keyword_count'],
                    name='Number of Keywords',
                    marker_color='#1f77b4' # A shade of blue
                ),
                secondary_y=False,
            )

            # Add line chart for total search volume (secondary y-axis)
            fig_intent_combo.add_trace(
                go.Scatter(
                    x=intent_data['search_intent'],
                    y=intent_data['total_volume'],
                    name='Total Search Volume',
                    mode='lines+markers',
                    marker_color='#d62728' # A shade of red
                ),
                secondary_y=True,
            )

            # Set x-axis title and tick angle
            fig_intent_combo.update_xaxes(title_text="Search Intent")

            # Set y-axes titles
            fig_intent_combo.update_yaxes(title_text="Number of Keywords", secondary_y=False)
            fig_intent_combo.update_yaxes(title_text="Total Search Volume", secondary_y=True, showgrid=False) # Hide grid for secondary axis

            fig_intent_combo.update_layout(
                title_text='Search Intent Distribution and Total Volume in SAM',
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_intent_combo, use_container_width=True)
            st.info("This chart breaks down keywords by their primary search intent, showing both the count of keywords and their aggregated search volume. This helps prioritize intents with higher market potential.")
        else:
            st.warning("No search intent data found in SAM for this chart. Adjust your filters in the sidebar.")
    else:
        st.warning("No keywords in SAM for breakdown. Adjust your filters in the sidebar.")


    # Top Parent Categories by Volume & Average KD (now with a summary box and table)
    st.subheader("Top Parent Categories by Volume & Average KD")
    if not df_keywords.empty:
        # Group by parent_topic and calculate total volume and average KD, CPC, and weighted growth
        parent_topic_data = df_keywords.groupby('parent_topic').agg(
            total_volume=('volume', 'sum'),
            average_kd=('kd', 'mean'),
            average_cpc=('cpc', 'mean'),
            # Calculate weighted average growth for each parent topic
            weighted_growth_pct=('growth_pct', lambda x: (x * df_keywords.loc[x.index, 'volume']).sum() / df_keywords.loc[x.index, 'volume'].sum() if df_keywords.loc[x.index, 'volume'].sum() > 0 else 0)
        ).reset_index()

        # Sort by total volume and get the top 10 for chart display
        top_10_parent_topics = parent_topic_data.sort_values(by='total_volume', ascending=False).head(10)
        
        # Get the absolute top parent topic for the summary box
        absolute_top_parent_topic = parent_topic_data.sort_values(by='total_volume', ascending=False).iloc[0] if not parent_topic_data.empty else None

        if absolute_top_parent_topic is not None:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    min-height: 180px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                    background-color: #f9f9f9;
                ">
                    <h4 style="color:#262730; margin-bottom: 5px;">Absolute Top Parent Category: **{absolute_top_parent_topic['parent_topic']}**</h4>
                    <p style="font-size: 1.1em; color:{number_color}; margin: 0;">
                        Total Volume: **{absolute_top_parent_topic['total_volume']:,}** Searches
                    </p>
                    <p style="font-size: 0.9em; color:#555; margin-top: 5px;">
                        Avg. KD: **{absolute_top_parent_topic['average_kd']:.2f}** |
                        Avg. CPC: **${absolute_top_parent_topic['average_cpc']:.2f}**
                    </p>
                    <p style="font-size: 0.9em; color:{'green' if absolute_top_parent_topic['weighted_growth_pct'] > 0 else 'red' if absolute_top_parent_topic['weighted_growth_pct'] < 0 else 'gray'}; margin: 0;">
                        Avg. Growth (Weighted): **{absolute_top_parent_topic['weighted_growth_pct']:+.2f}%**
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")


        if not top_10_parent_topics.empty:
            # Create combo chart using make_subplots
            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

            # Add bar chart for total volume (primary y-axis)
            fig_combo.add_trace(
                go.Bar(
                    x=top_10_parent_topics['parent_topic'],
                    y=top_10_parent_topics['total_volume'],
                    name='Total Search Volume',
                    marker_color='#1f77b4' # A shade of blue
                ),
                secondary_y=False,
            )

            # Add line chart for average KD (secondary y-axis)
            fig_combo.add_trace(
                go.Scatter(
                    x=top_10_parent_topics['parent_topic'],
                    y=top_10_parent_topics['average_kd'],
                    name='Average Keyword Difficulty (KD)',
                    mode='lines+markers',
                    marker_color='#d62728' # A shade of red
                ),
                secondary_y=True,
            )

            # Set x-axis title and tick angle
            fig_combo.update_xaxes(title_text="Parent Topic", tickangle=-45)

            # Set y-axes titles
            fig_combo.update_yaxes(title_text="Total Search Volume", secondary_y=False)
            fig_combo.update_yaxes(title_text="Average Keyword Difficulty (KD)", secondary_y=True, showgrid=False) # Hide grid for secondary axis

            fig_combo.update_layout(
                title_text='Top 10 Parent Categories: Volume vs. Average KD',
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_combo, use_container_width=True)
            st.info("This chart displays the top 10 parent topics by total search volume, with their average Keyword Difficulty (KD) overlaid. Use the metrics to identify high-volume, potentially lower-difficulty opportunities.")

            # Display table for Top 10 Parent Categories as collapsible
            st.subheader("Top 10 Parent Categories Details")
            with st.expander("📊 View Top 10 Parent Categories Table"):
                # Ensure the columns for display are correct and formatted
                df_display = top_10_parent_topics[['parent_topic', 'total_volume', 'average_kd', 'average_cpc', 'weighted_growth_pct']].copy()
                df_display.rename(columns={
                    'parent_topic': 'Parent Topic',
                    'total_volume': 'Total Volume',
                    'average_kd': 'Avg. KD',
                    'average_cpc': 'Avg. CPC',
                    'weighted_growth_pct': 'Avg. Growth (%)'
                }, inplace=True)

                st.dataframe(
                    df_display.style.format({
                        'Total Volume': '{:,.0f}',
                        'Avg. KD': '{:.2f}',
                        'Avg. CPC': '${:.2f}',
                        'Avg. Growth (%)': '{:+.2f}%'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("No parent topics found with sufficient data for this chart. Ensure 'parent_topic' and 'kd' columns are present and data is not empty.")
    else:
        st.warning("Please upload Ahrefs data to view parent category analysis.")


    # SERP Features Breakdown
    st.subheader("SERP Features Present (SAM)")
    if not df_sam.empty:
        # Calculate percentage of keywords with AI Overviews
        # Updated to check for 'AI overview' (singular, case-insensitive)
        ai_overview_keywords_count = df_sam['serp_features'].apply(
            lambda x: bool(re.search(r'\b(?:ai overview|ai overviews)\b', str(x).lower()))
        ).sum()
        total_sam_keywords = len(df_sam)
        ai_overview_percentage = (ai_overview_keywords_count / total_sam_keywords) * 100 if total_sam_keywords > 0 else 0

        # Display AI Overviews metric card
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                background-color: #f9f9f9;
            ">
                <p style="font-size: 1.1em; margin-bottom: 5px;">Keywords with AI Overviews</p>
                <p style="font-size: 1.8em; font-weight: bold; color:{number_color}; margin: 0;">
                    {ai_overview_percentage:.2f}%
                </p>
                <p style="font-size: 0.8em; color:#555; margin-top: 5px;">
                    ({ai_overview_keywords_count} of {total_sam_keywords} keywords)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---") # Separator for the card

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
            st.info("No specific SERP features detected in the selected keywords (or 'N/A' for all). Adjust your filters in the sidebar.")
    else:
        st.warning("No keywords in SAM for SERP features breakdown. Adjust your filters in the sidebar.")

    # Topic Trending Up or Down (New enhanced section)
    st.subheader("Topic Trend (Overall Data)")
    
    # Define the consistent blue color for numbers
    number_color = "#1f77b4"

    # Identify actual growth columns found for display purposes
    growth_col_map_internal = {
        '3mo': None,
        '6mo': None,
        '12mo': None
    }
    for col_suffix in ['3mo', '6mo', '12mo']:
        for pc_name in [f'growth_({col_suffix})', f'global_growth_({col_suffix})']:
            if pc_name in df_keywords.columns:
                growth_col_map_internal[col_suffix] = pc_name
                break
                
    col_trend_3mo, col_trend_6mo, col_trend_12mo = st.columns(3)

    # Function to display a single trend metric card
    def display_trend_card(col_obj, period_label, growth_col_name, df_data, num_color):
        if growth_col_name and not df_data.empty:
            # Calculate weighted average growth using the identified growth column
            overall_growth_sum_numerator = df_data[growth_col_name] * df_data['volume']
            overall_growth_sum_denominator = df_data['volume']
            
            if overall_growth_sum_denominator.sum() > 0:
                overall_growth_pct = overall_growth_sum_numerator.sum() / overall_growth_sum_denominator.sum()
            else:
                overall_growth_pct = 0

            trend_indicator_color = "green" if overall_growth_pct > 0 else "red" if overall_growth_pct < 0 else "gray"

            col_obj.markdown(
                f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    min-height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                    background-color: #f9f9f9;
                ">
                    <p style="font-size: 1.1em; margin-bottom: 5px;">Overall {period_label} Growth</p>
                    <p style="font-size: 1.8em; font-weight: bold; color:{trend_indicator_color}; margin: 0;">
                        {overall_growth_pct:+.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            col_obj.info(f"'{period_label}' data not found.")

    display_trend_card(col_trend_3mo, "3-Month", growth_col_map_internal['3mo'], df_keywords, number_color)
    display_trend_card(col_trend_6mo, "6-Month", growth_col_map_internal['6mo'], df_keywords, number_color)
    display_trend_card(col_trend_12mo, "12-Month", growth_col_map_internal['12mo'], df_keywords, number_color)

    if any(growth_col_map_internal.values()): # Only show distribution if at least one growth column was found
        # Distribution of trend categories (still relevant for individual keyword distribution)
        if df_keywords['trend'].nunique() > 1:
            trend_counts = df_keywords['trend'].value_counts().reset_index()
            trend_counts.columns = ['Trend', 'Count']
            fig_trend = px.bar(trend_counts, x='Trend', y='Count', title='Keyword Trend Distribution (Based on Primary Growth Column)')
            st.plotly_chart(fig_trend, use_container_width=True)
            st.info("Trends (Trending Up/Down/Stable) are categorized based on the primary Ahrefs 'Growth (Xmo)' column found (>5% increase/decrease thresholds).")
        else:
            st.info("Most keywords in your dataset show a 'Stable' trend or trend data is limited for categorical breakdown.")

        # Side-by-side expanders for top trending keywords
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
        st.info("No Ahrefs `Growth (3mo)`, `Growth (6mo)`, or `Growth (12mo)` columns found in your uploaded data. Trend analysis is limited.")

    st.divider()

    # --- 5. Google Search Console Connection (Placeholder) ---
    st.header("4. Your Market Position (via Google Search Console)") # Re-numbered header
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

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SentinAL | Risk Command Center",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM ENTERPRISE STYLING ---
st.markdown("""
<style>
    /* Core Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Glass Morphism Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    /* Enhanced Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d3a 0%, #0a0e27 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    /* Custom Alert Boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
        border-left: 4px solid #667eea;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 24px 0 16px 0;
    }
    
    /* Data Cards */
    .data-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .data-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.7) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
    
    /* Info/Warning/Error Boxes */
    .element-container .stMarkdown .stAlert > div {
        padding: 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ADVANCED DATA ENGINE ---
DATA_FILE = 'outputs/final_risk_analysis.json'

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, None
    
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        processed_rows = []
        for item in data:
            row = {
                'Entity ID': str(item['entity_id']),
                'Risk Score': item['composite_risk_score'],
                'Risk Level': item.get('risk_level', 'Unknown'),
                'Last Updated': item.get('timestamp'),
                'ROA': 0.0,
                'Debt Ratio': 0.0,
                'Op Margin': 0.0
            }
            
            signals = item.get('contributing_signals', [])
            credit_sig = next((s for s in signals if s['risk_type'] == 'credit'), None)
            
            if credit_sig:
                inputs = credit_sig.get('metadata', {}).get('input_used', {})
                row['ROA'] = inputs.get('roa', 0.0)
                row['Debt Ratio'] = inputs.get('debt_ratio', 0.0)
                row['Op Margin'] = inputs.get('operating_margin', 0.0)
            
            processed_rows.append(row)

        df = pd.DataFrame(processed_rows)
        raw_data = {str(item['entity_id']): item for item in data}
        
        return df, raw_data
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return None, None

df, raw_data_dict = load_data()

# --- 4. PREMIUM SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>🛡️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 1.5rem; margin-bottom: 0.25rem;'>SentinAL</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.5); font-size: 0.875rem;'>Financial Intelligence Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if df is not None:
        st.markdown("### 🎯 Filters")
        min_score = st.slider("Risk Score Threshold", 0, 100, 0, help="Filter entities by minimum risk score")
        
        all_levels = ['Critical', 'High', 'Medium', 'Low']
        selected_levels = st.multiselect(
            "Risk Categories", 
            all_levels, 
            default=['Critical', 'High'],
            help="Select risk levels to display"
        )
        
        filtered_df = df[
            (df['Risk Score'] >= min_score) & 
            (df['Risk Level'].isin(selected_levels))
        ]
        
        st.markdown("---")
        
        # Enhanced Stats Box
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", f"{len(df):,}", help="Total entities in system")
        with col2:
            st.metric("In View", f"{len(filtered_df):,}", help="Entities matching filters")
        
        if len(filtered_df) > 0:
            pct = (len(filtered_df) / len(df)) * 100
            st.progress(pct / 100)
            st.caption(f"{pct:.1f}% of portfolio")
        
        st.markdown("---")
        st.markdown("### 📥 Export")
        st.download_button(
            "Download Dataset",
            filtered_df.to_csv(index=False),
            "sentinal_risk_export.csv",
            "text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.75rem; margin-top: 2rem;'>© 2026 SentinAL Intelligence</p>", unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD ---
if df is None:
    st.error(f"❌ Data not found at `{DATA_FILE}`. Run `python run_full_analysis.py` first.")
    st.stop()

# Header
st.markdown("<h1>📊 Risk Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: rgba(255,255,255,0.6); font-size: 1.1rem; margin-bottom: 2rem;'>Real-time financial risk monitoring and portfolio analytics</p>", unsafe_allow_html=True)

# TABS
tab_overview, tab_finance, tab_inspector, tab_raw = st.tabs([
    "🎯 Overview", 
    "📈 Financial Analytics", 
    "🔍 Entity Deep Dive", 
    "📋 Raw Data"
])

# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
with tab_overview:
    if filtered_df.empty:
        st.warning("⚠️ No data matches current filters. Adjust your filter settings.")
    else:
        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric("Portfolio Entities", f"{len(df):,}")
        with kpi2:
            st.metric("Active Alerts", len(filtered_df), delta=f"{len(filtered_df) - len(df)}")
        with kpi3:
            avg_debt = filtered_df['Debt Ratio'].mean()
            st.metric("Avg Debt Ratio", f"{avg_debt:.2%}")
        with kpi4:
            avg_roa = filtered_df['ROA'].mean()
            st.metric("Avg ROA", f"{avg_roa:.2%}")

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        
        # Charts Row
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("<div class='section-header'><h3>Risk Distribution</h3></div>", unsafe_allow_html=True)
            
            fig_pie = px.pie(
                filtered_df, 
                names='Risk Level', 
                color='Risk Level',
                color_discrete_map={
                    'Low':'#10b981', 
                    'Medium':'#f59e0b', 
                    'High':'#ef4444', 
                    'Critical':'#7f1d1d'
                },
                hole=0.6
            )
            
            fig_pie.update_traces(
                textposition='outside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#0a0e27', width=2))
            )
            
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with c2:
            st.markdown("<div class='section-header'><h3>Risk Score Distribution</h3></div>", unsafe_allow_html=True)
            
            fig_hist = px.histogram(
                filtered_df, 
                x="Risk Score", 
                nbins=40, 
                color="Risk Level",
                color_discrete_map={
                    'Low':'#10b981', 
                    'Medium':'#f59e0b', 
                    'High':'#ef4444', 
                    'Critical':'#7f1d1d'
                }
            )
            
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                showlegend=True,
                legend=dict(bgcolor='rgba(255,255,255,0.05)'),
                margin=dict(t=20, b=40, l=40, r=20),
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 2: FINANCIAL FORENSICS
# ==========================================
with tab_finance:
    if filtered_df.empty:
        st.warning("⚠️ No data available for financial analysis.")
    else:
        st.markdown("<div class='section-header'><h3>🎯 Financial Health Matrix</h3><p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>Solvency analysis: Debt leverage vs. Profitability (ROA)</p></div>", unsafe_allow_html=True)
        
        fig_matrix = px.scatter(
            filtered_df,
            x="Debt Ratio",
            y="ROA",
            color="Risk Level",
            size="Risk Score",
            hover_data=['Entity ID', 'Op Margin'],
            color_discrete_map={
                'Low':'#10b981', 
                'Medium':'#f59e0b', 
                'High':'#ef4444', 
                'Critical':'#7f1d1d'
            },
            height=600
        )
        
        avg_debt = filtered_df['Debt Ratio'].mean()
        avg_roa = filtered_df['ROA'].mean()
        
        fig_matrix.add_vline(
            x=avg_debt, 
            line_dash="dash", 
            line_color="rgba(102, 126, 234, 0.5)", 
            annotation_text="Avg Debt",
            annotation_position="top"
        )
        fig_matrix.add_hline(
            y=avg_roa, 
            line_dash="dash", 
            line_color="rgba(102, 126, 234, 0.5)", 
            annotation_text="Avg ROA",
            annotation_position="right"
        )
        
        fig_matrix.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,39,0.5)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title_font=dict(size=14)),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title_font=dict(size=14)),
            legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'><h3>🌊 Risk Flow Analysis</h3><p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>Visualization of how financial metrics cascade into risk categories</p></div>", unsafe_allow_html=True)
        
        viz_df = filtered_df.copy()
        viz_df['Debt Level'] = pd.qcut(viz_df['Debt Ratio'], q=3, labels=["Low Debt", "Med Debt", "High Debt"], duplicates='drop')
        viz_df['Profitability'] = pd.qcut(viz_df['ROA'], q=3, labels=["Low ROA", "Med ROA", "High ROA"], duplicates='drop')
        
        fig_flow = px.parallel_categories(
            viz_df,
            dimensions=['Debt Level', 'Profitability', 'Risk Level'],
            color="Risk Score",
            color_continuous_scale='Inferno'
        )
        
        fig_flow.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            margin=dict(t=40, b=40, l=40, r=40),
            height=500
        )
        
        st.plotly_chart(fig_flow, use_container_width=True)

# ==========================================
# TAB 3: ENTITY INSPECTOR
# ==========================================
with tab_inspector:
    if filtered_df.empty:
        st.warning("⚠️ No entities available for inspection.")
    else:
        # Entity Selection with Search and Sort Options
        col_sel, col_sort, col_search = st.columns([2, 1, 1])
        
        with col_sort:
            sort_option = st.selectbox(
                "Sort By",
                ["Risk Score (High to Low)", "Risk Score (Low to High)", "Entity ID"],
                help="Choose how to sort entities"
            )
        
        # Apply sorting
        if sort_option == "Risk Score (High to Low)":
            sorted_df = filtered_df.sort_values(by='Risk Score', ascending=False)
        elif sort_option == "Risk Score (Low to High)":
            sorted_df = filtered_df.sort_values(by='Risk Score', ascending=True)
        else:
            sorted_df = filtered_df.sort_values(by='Entity ID')
        
        sorted_ids = sorted_df['Entity ID'].unique()
        
        with col_sel:
            selected_id = st.selectbox(
                "🎯 Select Entity for Deep Analysis", 
                sorted_ids, 
                help="Choose an entity to perform comprehensive analysis",
                format_func=lambda x: f"Entity {x} - Risk: {filtered_df[filtered_df['Entity ID']==x]['Risk Score'].iloc[0]:.1f}"
            )
        
        with col_search:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("📋 View All Entities", use_container_width=True):
                st.session_state.show_all_entities = True

        if selected_id:
            # Initialize session state for view toggle
            if 'show_all_entities' not in st.session_state:
                st.session_state.show_all_entities = False
            
            # Toggle between individual and all entities view
            if st.session_state.show_all_entities:
                if st.button("⬅️ Back to Individual Analysis"):
                    st.session_state.show_all_entities = False
                    st.rerun()
                
                # ===== ALL ENTITIES OVERVIEW =====
                st.markdown("<div class='section-header'><h3>📊 All Entities Portfolio View</h3><p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>Comprehensive analysis of all entities in current filter</p></div>", unsafe_allow_html=True)
                
                # Portfolio Summary Metrics
                met1, met2, met3, met4, met5 = st.columns(5)
                with met1:
                    st.metric("Total Entities", len(filtered_df))
                with met2:
                    critical_count = len(filtered_df[filtered_df['Risk Level'] == 'Critical'])
                    st.metric("Critical Risk", critical_count)
                with met3:
                    high_count = len(filtered_df[filtered_df['Risk Level'] == 'High'])
                    st.metric("High Risk", high_count)
                with met4:
                    avg_risk = filtered_df['Risk Score'].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.1f}")
                with met5:
                    max_risk = filtered_df['Risk Score'].max()
                    st.metric("Max Risk Score", f"{max_risk:.1f}")
                
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                
                # All Entities Comparison Charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("#### Risk Score Comparison")
                    fig_bar = px.bar(
                        sorted_df,
                        x='Entity ID',
                        y='Risk Score',
                        color='Risk Level',
                        color_discrete_map={
                            'Low':'#10b981', 
                            'Medium':'#f59e0b', 
                            'High':'#ef4444', 
                            'Critical':'#7f1d1d'
                        },
                        hover_data=['Debt Ratio', 'ROA', 'Op Margin']
                    )
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Entity ID'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Risk Score'),
                        showlegend=True,
                        legend=dict(bgcolor='rgba(255,255,255,0.05)'),
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col_chart2:
                    st.markdown("#### Financial Metrics Heatmap")
                    # Create normalized heatmap data
                    heatmap_data = sorted_df[['Entity ID', 'Risk Score', 'Debt Ratio', 'ROA', 'Op Margin']].copy()
                    heatmap_data = heatmap_data.set_index('Entity ID')
                    
                    fig_heat = px.imshow(
                        heatmap_data.T,
                        labels=dict(x="Entity ID", y="Metric", color="Value"),
                        color_continuous_scale='RdYlGn_r',
                        aspect='auto'
                    )
                    fig_heat.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                
                # Detailed Entity Table
                st.markdown("#### 📋 Detailed Entity Breakdown")
                display_df = sorted_df[['Entity ID', 'Risk Score', 'Risk Level', 'Debt Ratio', 'ROA', 'Op Margin']].copy()
                display_df['Risk Score'] = display_df['Risk Score'].round(2)
                display_df['Debt Ratio'] = (display_df['Debt Ratio'] * 100).round(2).astype(str) + '%'
                display_df['ROA'] = (display_df['ROA'] * 100).round(2).astype(str) + '%'
                display_df['Op Margin'] = (display_df['Op Margin'] * 100).round(2).astype(str) + '%'
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Entity ID": st.column_config.TextColumn("Entity ID", width="small"),
                        "Risk Score": st.column_config.NumberColumn("Risk Score", width="small"),
                        "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
                        "Debt Ratio": st.column_config.TextColumn("Debt Ratio", width="small"),
                        "ROA": st.column_config.TextColumn("ROA", width="small"),
                        "Op Margin": st.column_config.TextColumn("Op Margin", width="small")
                    }
                )
                
                # Multi-entity Radar Comparison (Top 5 vs Bottom 5)
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### 🎯 Top vs Bottom Performers")
                
                col_radar1, col_radar2 = st.columns(2)
                
                def create_comparison_radar(entities_df, title, color):
                    avg_metrics = {
                        'Risk Score': 100 - entities_df['Risk Score'].mean(),
                        'Debt Safety': 100 - (entities_df['Debt Ratio'].mean() * 100),
                        'ROA': entities_df['ROA'].mean() * 100,
                        'Op Margin': entities_df['Op Margin'].mean() * 100
                    }
                    
                    categories = list(avg_metrics.keys())
                    values = list(avg_metrics.values())
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=title,
                        line_color=color,
                        fillcolor=f'{color}33'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=10),
                        showlegend=True,
                        title=dict(text=title, x=0.5, xanchor='center'),
                        margin=dict(t=60, b=40, l=40, r=40),
                        height=350
                    )
                    return fig
                
                with col_radar1:
                    top_5 = sorted_df.nsmallest(5, 'Risk Score')
                    fig_top = create_comparison_radar(top_5, "Top 5 (Lowest Risk)", "#10b981")
                    st.plotly_chart(fig_top, use_container_width=True)
                
                with col_radar2:
                    bottom_5 = sorted_df.nlargest(5, 'Risk Score')
                    fig_bottom = create_comparison_radar(bottom_5, "Top 5 (Highest Risk)", "#ef4444")
                    st.plotly_chart(fig_bottom, use_container_width=True)
            
            else:
                # ===== INDIVIDUAL ENTITY ANALYSIS =====
                entity_record = raw_data_dict.get(str(selected_id))
                entity_row = df[df['Entity ID'] == str(selected_id)].iloc[0]
                
                # Header Card with Navigation
                col_header, col_nav = st.columns([3, 1])
                
                with col_header:
                    risk_color = {
                        'Critical': '#7f1d1d',
                        'High': '#ef4444',
                        'Medium': '#f59e0b',
                        'Low': '#10b981'
                    }.get(entity_row['Risk Level'], '#667eea')
                    
                    st.markdown(f"""
                    <div class='data-card' style='border-left: 4px solid {risk_color};'>
                        <h2 style='margin: 0; color: #667eea;'>Entity {selected_id}</h2>
                        <p style='margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.6);'>
                            Risk Level: <span style='color: {risk_color}; font-weight: 600;'>{entity_row['Risk Level']}</span> | 
                            Risk Score: <span style='color: {risk_color}; font-weight: 600;'>{entity_row['Risk Score']:.2f}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_nav:
                    # Navigation buttons
                    current_idx = list(sorted_ids).index(selected_id)
                    col_prev, col_next = st.columns(2)
                    
                    with col_prev:
                        if current_idx > 0:
                            if st.button("⬅️ Prev", use_container_width=True):
                                # Update selection - note: Streamlit resets selection on rerun without session state tricks,
                                # but usually simple rerun works if we don't force index in selectbox. 
                                # A better way is using session_state to drive selectbox index.
                                pass # Simplification for now
                    
                    with col_next:
                        if current_idx < len(sorted_ids) - 1:
                            if st.button("Next ➡️", use_container_width=True):
                                pass 
                
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                
                # Quick Stats Row
                quick1, quick2, quick3, quick4 = st.columns(4)
                with quick1:
                    st.metric("Risk Score", f"{entity_row['Risk Score']:.2f}")
                with quick2:
                    st.metric("Debt Ratio", f"{entity_row['Debt Ratio']:.2%}")
                with quick3:
                    st.metric("ROA", f"{entity_row['ROA']:.2%}")
                with quick4:
                    st.metric("Op Margin", f"{entity_row['Op Margin']:.2%}")
                
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            
                # Peer Benchmarking
                st.markdown("<div class='section-header'><h3>📊 Peer Benchmarking</h3></div>", unsafe_allow_html=True)
                
                categories = ['Risk Score', 'Debt Ratio', 'ROA', 'Op Margin']
                
                # --- CRITICAL NUMERICAL FIX: Inverse logic for Risk & Debt ---
                def normalize(val, col_name, inverse=False):
                    max_val = df[col_name].max()
                    if max_val == 0: return 0
                    pct = (val / max_val) * 100
                    return 100 - pct if inverse else pct
                
                e_vals = [
                    normalize(entity_row['Risk Score'], 'Risk Score', inverse=True),
                    normalize(entity_row['Debt Ratio'], 'Debt Ratio', inverse=True),
                    normalize(entity_row['ROA'], 'ROA'),
                    normalize(entity_row['Op Margin'], 'Op Margin')
                ]
                
                avg_vals = [
                    normalize(df['Risk Score'].mean(), 'Risk Score', inverse=True),
                    normalize(df['Debt Ratio'].mean(), 'Debt Ratio', inverse=True),
                    normalize(df['ROA'].mean(), 'ROA'),
                    normalize(df['Op Margin'].mean(), 'Op Margin')
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=e_vals,
                    theta=categories,
                    fill='toself',
                    name='Selected Entity',
                    line_color='#667eea',
                    fillcolor='rgba(102, 126, 234, 0.3)'
                ))
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=avg_vals,
                    theta=categories,
                    fill='toself',
                    name='Market Average',
                    line_color='#10b981',
                    fillcolor='rgba(16, 185, 129, 0.2)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True,
                    legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
                    margin=dict(t=40, b=40, l=40, r=40),
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                
                # Deep Dive Cards
                st.markdown("<div class='section-header'><h3>🔬 Fundamental Analysis</h3></div>", unsafe_allow_html=True)
                
                signals = entity_record.get('contributing_signals', [])
                sig_credit = next((s for s in signals if s['risk_type'] == 'credit'), None)
                sig_sent = next((s for s in signals if s['risk_type'] == 'sentiment'), None)
                sig_sys = next((s for s in signals if s['risk_type'] == 'systemic'), None)

                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown("""
                    <div class='data-card' style='border-left: 4px solid #10b981;'>
                        <h4 style='color: #10b981; margin-top: 0;'>💰 Credit Metrics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Debt Ratio", f"{entity_row['Debt Ratio']:.4f}")
                    st.metric("ROA", f"{entity_row['ROA']:.4f}")
                    st.metric("Op Margin", f"{entity_row['Op Margin']:.4f}")
                    if sig_credit:
                        st.caption(f"📊 Raw PD: {sig_credit['metadata'].get('raw_pd_probability',0):.4f}")
                        st.caption(f"📊 Credit Score: {sig_credit.get('normalized_score', 0):.2f}")

                with m2:
                    st.markdown("""
                    <div class='data-card' style='border-left: 4px solid #f59e0b;'>
                        <h4 style='color: #f59e0b; margin-top: 0;'>📰 Sentiment</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    score = sig_sent['normalized_score'] if sig_sent else 0
                    st.metric("Sentiment Risk", f"{score:.2f}")
                    if sig_sent:
                        headline = sig_sent['metadata'].get('top_headline', 'No News')
                        st.caption(f"📌 {headline[:80]}...")
                        sentiment_val = sig_sent['metadata'].get('sentiment_score', 0)
                        st.caption(f"📊 Raw Sentiment: {sentiment_val:.2f}")

                with m3:
                    st.markdown("""
                    <div class='data-card' style='border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-top: 0;'>🕸️ Systemic</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    score = sig_sys['normalized_score'] if sig_sys else 0
                    st.metric("Systemic Score", f"{score:.2f}")
                    if sig_sys:
                        st.caption("🔗 Network contagion analyzed")
                        centrality = sig_sys['metadata'].get('centrality_score', 0)
                        st.caption(f"📊 Centrality: {centrality:.4f}")

# ==========================================
# TAB 4: RAW DATA
# ==========================================
with tab_raw:
    st.markdown("<div class='section-header'><h3>📋 Raw Dataset</h3><p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>Complete data table with all metrics</p></div>", unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True, height=600)
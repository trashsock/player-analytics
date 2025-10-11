import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os

# Brisbane Broncos color scheme
BRONCOS_MAROON = '#7C2529'
BRONCOS_GOLD = '#FFB81C'
BRONCOS_WHITE = '#FFFFFF'
BRONCOS_DARK = '#3D1214'

def apply_broncos_theme():
    """Apply Brisbane Broncos theme to the app."""
    st.markdown(f"""
        <style>
        /* Main app styling */
        .stApp {{
            background: linear-gradient(135deg, {BRONCOS_MAROON} 0%, {BRONCOS_DARK} 100%);
        }}
        
        /* Header styling */
        h1, h2, h3 {{
            color: {BRONCOS_GOLD};
            font-weight: 600;
        }}
        
        /* Dataframe styling */
        .stDataFrame {{
            background-color: {BRONCOS_WHITE};
            border: 2px solid {BRONCOS_GOLD};
            border-radius: 8px;
        }}
        
        /* Selectbox styling */
        .stSelectbox label {{
            color: {BRONCOS_GOLD};
            font-weight: 600;
        }}
        
        /* Selectbox dropdown text */
        .stSelectbox [data-testid="stSelectbox"] div, .stSelectbox [data-testid="stSelectbox"] span {{
            color: {BRONCOS_DARK};
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {BRONCOS_GOLD};
            color: {BRONCOS_MAROON};
            font-weight: 600;
            border: 2px solid {BRONCOS_MAROON};
            border-radius: 6px;
        }}
        
        .stButton > button:hover {{
            background-color: {BRONCOS_MAROON};
            color: {BRONCOS_GOLD};
            border: 2px solid {BRONCOS_GOLD};
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {BRONCOS_DARK} 0%, {BRONCOS_MAROON} 100%);
        }}
        
        section[data-testid="stSidebar"] * {{
            color: {BRONCOS_WHITE};
        }}
        
        /* Info/Warning boxes */
        .stAlert {{
            background-color: rgba(124, 37, 41, 0.2);
            border: 2px solid {BRONCOS_GOLD};
            color: {BRONCOS_WHITE};
        }}
        
        /* Metric styling */
        [data-testid="stMetricValue"] {{
            color: {BRONCOS_GOLD};
        }}
        
        /* Text color for various elements */
        p, span, div {{
            color: {BRONCOS_DARK};
        }}
        
        /* Caption styling */
        .caption {{
            color: {BRONCOS_GOLD};
            font-style: italic;
        }}
        </style>
    """, unsafe_allow_html=True)

def run_query_app(data_file, gps_file):
    """Run Streamlit app for querying Broncos insights."""
    
    # Apply Broncos theme
    # apply_broncos_theme()
    
    # Custom header with Broncos branding
    st.markdown(f"""
        <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, {BRONCOS_MAROON} 0%, {BRONCOS_GOLD} 50%, {BRONCOS_MAROON} 100%); border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: {BRONCOS_WHITE}; margin: 0; font-size: 2em;'>
                Brisbane Broncos NRL Players
            </h1>
            <p style='color: {BRONCOS_WHITE}; font-size: 1em; margin: 5px 0 0 0; font-weight: 600;'>
                Performance Analytics Dashboard
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Add debug info to sidebar
    st.sidebar.markdown("### System Diagnostics")
    st.sidebar.write(f"Data file: {data_file}")
    st.sidebar.write(f"GPS file: {gps_file}")
    st.sidebar.write(f"Data exists: {os.path.exists(data_file)}")
    st.sidebar.write(f"GPS exists: {os.path.exists(gps_file)}")
    st.sidebar.write(f"Lineup exists: {os.path.exists('ga_lineup.pkl')}")
    st.sidebar.write(f"Predictions exists: {os.path.exists('lstm_predictions.pkl')}")

    # Load precomputed data with error handling
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv(data_file)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].astype('float32')
            return df
        except Exception as e:
            st.error(f"Error loading data file: {e}")
            return None

    @st.cache_data
    def load_lineup():
        try:
            if os.path.exists('ga_lineup.pkl'):
                with open('ga_lineup.pkl', 'rb') as f:
                    return pickle.load(f)
            else:
                st.warning("ga_lineup.pkl not found. Using mock data.")
                return ['Player A', 'Player B', 'Player C']
        except Exception as e:
            st.error(f"Error loading lineup: {e}")
            return []

    @st.cache_data
    def load_predictions():
        try:
            if os.path.exists('lstm_predictions.pkl'):
                with open('lstm_predictions.pkl', 'rb') as f:
                    return pickle.load(f)
            else:
                st.warning("lstm_predictions.pkl not found. Using mock data.")
                return {
                    'players': ['Player A', 'Player B'],
                    'injury_risk': [0.65, 0.72]
                }
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            return {'players': [], 'injury_risk': []}

    @st.cache_data
    def load_gps():
        try:
            gps = pd.read_csv(gps_file)
            numeric_cols = gps.select_dtypes(include=['float64', 'int64']).columns
            gps[numeric_cols] = gps[numeric_cols].astype('float32')
            return gps[gps['player'].isin(data['player'].unique())]
        except Exception as e:
            st.error(f"Error loading GPS data: {e}")
            return pd.DataFrame()

    # Load data
    data = load_data()
    
    if data is None or data.empty:
        st.error("Failed to load main data file. Please check the file path and format.")
        return
    
    best_lineup = load_lineup()
    future_predictions = load_predictions()
    gps_data = load_gps()

    # Show data info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"Shape: {data.shape}")
    st.sidebar.write(f"Players: {data['player'].nunique() if 'player' in data.columns else 'N/A'}")

    # Query Interface with Broncos styling
    st.markdown(f"<h2 style='color: {BRONCOS_GOLD};'>Select Query</h2>", unsafe_allow_html=True)
    
    question = st.selectbox("", [
        "Who's at high injury risk in 2026?",
        "What's the optimal lineup for 2026?",
        "Who drives comebacks?",
        "Who has high GPS fatigue risk?"  
    ])

    try:
        if "injury risk" in question:
            st.markdown(f"<h3 style='color: {BRONCOS_GOLD};'>Injury Risk Analysis - 2026 Season</h3>", unsafe_allow_html=True)
            
            if 'injury_risk' in data.columns:
                high_risk = data[data['injury_risk'] > 0.7][['player', 'injury_risk']].drop_duplicates()
                if not high_risk.empty:
                    st.dataframe(high_risk.sort_values('injury_risk', ascending=False), use_container_width=True)
                else:
                    st.info("No players with injury risk > 0.7. Showing top 10 highest risk players:")
                    top_risk = data[['player', 'injury_risk']].drop_duplicates().nlargest(10, 'injury_risk')
                    st.dataframe(top_risk, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Risk", f"{data['injury_risk'].max():.1%}")
                    with col2:
                        st.metric("Avg Risk", f"{data['injury_risk'].mean():.1%}")
                    with col3:
                        st.metric("Min Risk", f"{data['injury_risk'].min():.1%}")
            else:
                st.error("Column 'injury_risk' not found in data")
            
            st.markdown(f"<h3 style='color: {BRONCOS_GOLD};'>2026 LSTM Forecast</h3>", unsafe_allow_html=True)
            pred_df = pd.DataFrame({
                'Player': future_predictions['players'],
                '2026 Injury Risk': future_predictions['injury_risk']
            })
            if not pred_df.empty:
                st.dataframe(pred_df.sort_values('2026 Injury Risk', ascending=False), use_container_width=True)
            else:
                st.warning("No LSTM predictions available")

        elif "optimal lineup" in question:
            st.markdown(f"<h3 style='color: {BRONCOS_GOLD};'>Optimal 17-Man Squad for 2026</h3>", unsafe_allow_html=True)
            
            TEAM_SIZE = 17
            unique_lineup = list(dict.fromkeys(best_lineup))[:TEAM_SIZE]
            lineup_df = pd.DataFrame({'Player': unique_lineup})
            
            if len(unique_lineup) < TEAM_SIZE:
                st.warning(f"Only {len(unique_lineup)} unique players available (need {TEAM_SIZE})")
            
            if 'position' in data.columns:
                lineup_df = lineup_df.merge(
                    data[['player', 'position']].drop_duplicates(), 
                    left_on='Player', 
                    right_on='player',
                    how='left'
                )
                st.dataframe(lineup_df[['Player', 'position']], use_container_width=True)
                st.caption(f"Starting 13 + 4 Interchange | Total: {len(lineup_df)} unique players")
            else:
                st.dataframe(lineup_df, use_container_width=True)
                st.warning("Position data not available")

        elif "comebacks" in question:
            st.markdown(f"<h3 style='color: {BRONCOS_GOLD};'>Clutch Performers Analysis</h3>", unsafe_allow_html=True)
            
            fig = px.scatter(data, x='meters_run', y='tackles', color='performance_cluster',
                            size='try_assists', text='player', 
                            title="Player Performance Clusters: Clutch vs. Anchors")
            
            fig.update_traces(textposition='top center', marker=dict(line=dict(width=2)))
            fig.update_layout(
                plot_bgcolor=BRONCOS_WHITE,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=BRONCOS_DARK, size=12),
                title_font=dict(size=18, color=BRONCOS_GOLD)
            )
            st.plotly_chart(fig, use_container_width=True)

        elif "GPS fatigue" in question:
            st.markdown(f"<h3 style='color: {BRONCOS_GOLD};'>GPS Fatigue Risk Analysis</h3>", unsafe_allow_html=True)
            
            high_gps_risk = data[data['high_speed_runs'] > 5][['player', 'high_speed_runs', 'total_distance', 'max_acceleration']].drop_duplicates()
            st.dataframe(high_gps_risk.sort_values('high_speed_runs', ascending=False), use_container_width=True)
            
            st.markdown(f"<h3 style='color: {BRONCOS_GOLD};'>Player Movement Heatmap</h3>", unsafe_allow_html=True)
            selected_player = st.selectbox("Select Player:", data['player'].unique())
            player_gps = gps_data[gps_data['player'] == selected_player]
            
            fig_heatmap = px.density_heatmap(player_gps, x='x_pos', y='y_pos', z='speed',
                                            title=f"{selected_player}'s Movement Heatmap (Speed Intensity)")
            
            fig_heatmap.update_layout(
                plot_bgcolor=BRONCOS_WHITE,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=BRONCOS_WHITE, size=12),
                title_font=dict(size=16, color=BRONCOS_GOLD)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing query: {e}")
        st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: {BRONCOS_GOLD}; padding: 15px;'>
            <p>Find me on <a href="https://www.linkedin.com/in/ritikagiridhar">LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Page config with Broncos branding
    st.set_page_config(
        page_title="Brisbane Broncos NRL Analytics",
        page_icon=":horse:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    data_file = "broncos_analytics_output.csv"
    gps_file = "broncos_gps_raw.csv"
    run_query_app(data_file, gps_file)

if __name__ == "__main__":
    main()

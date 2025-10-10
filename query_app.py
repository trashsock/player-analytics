import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

def run_query_app(data_file, gps_file):
    """Run Streamlit app for querying Broncos insights."""
    st.title("AI Insights for Brisbane Broncos")

    # Load precomputed data
    @st.cache_data
    def load_data():
        return pd.read_csv(data_file, dtype='float32')

    @st.cache_data
    def load_lineup():
        with open('ga_lineup.pkl', 'rb') as f:
            return pickle.load(f)

    @st.cache_data
    def load_predictions():
        with open('lstm_predictions.pkl', 'rb') as f:
            return pickle.load(f)

    data = load_data()
    best_lineup = load_lineup()
    future_predictions = load_predictions()

    # Load raw GPS for visualization (lightweight subset)
    @st.cache_data
    def load_gps():
        gps = pd.read_csv(gps_file, dtype='float32')
        return gps[gps['player'].isin(data['player'].unique())]  # Subset for demo

    gps_data = load_gps()

    # Query Interface
    question = st.selectbox("Ask AI a Question:", [
        "Who’s at high injury risk in 2026?",
        "What’s the optimal lineup for 2026?",
        "Who drives comebacks?",
        "Who has high GPS fatigue risk?"  
    ])

    if question == "Who’s at high injury risk in 2026?":
        st.subheader("Predicted Injury Risks for 2026")
        high_risk = data[data['injury_risk'] > 0.7][['player', 'injury_risk']].drop_duplicates()
        st.dataframe(high_risk)
        st.subheader("2026 Forecast for Key Players")
        pred_df = pd.DataFrame({
            'Player': future_predictions['players'],
            '2026 Injury Risk': future_predictions['injury_risk']
        })
        st.dataframe(pred_df)

    elif question == "What’s the optimal lineup for 2026?":
        st.subheader("Optimal Lineup for 2026")
        lineup_df = pd.DataFrame({'Player': best_lineup})
        lineup_df = lineup_df.merge(data[['player', 'position']].drop_duplicates(), left_on='Player', right_on='player')
        st.dataframe(lineup_df[['Player', 'position']])

    elif question == "Who drives comebacks?":
        st.subheader("Clutch Performers (2025 Grand Final Style)")
        fig = px.scatter(data, x='meters_run', y='tackles', color='performance_cluster',
                        size='try_assists', text='player', title="Player Clusters: Clutch vs. Anchors",
                        color_continuous_scale=['#F1AB00', '#C8305D', '#0066CC'])
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    elif question == "Who has high GPS fatigue risk?":
        st.subheader("Players with High GPS Fatigue Risk (High-Speed Runs >5)")
        high_gps_risk = data[data['high_speed_runs'] > 5][['player', 'high_speed_runs', 'total_distance', 'max_acceleration']].drop_duplicates()
        st.dataframe(high_gps_risk.sort_values('high_speed_runs', ascending=False))
        # Heatmap visualization of sample player movements
        selected_player = st.selectbox("View Heatmap for Player:", data['player'].unique())
        player_gps = gps_data[gps_data['player'] == selected_player]
        fig_heatmap = px.density_heatmap(player_gps, x='x_pos', y='y_pos', z='speed',
                                        title=f"{selected_player}'s Movement Heatmap (Speed Intensity)",
                                        color_continuous_scale='Viridis', width=600, height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

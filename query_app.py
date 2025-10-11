import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import sys

# Parse args
sport = sys.argv[2] if len(sys.argv) > 1 else 'NRL'
user_type = sys.argv[4] if len(sys.argv) > 3 else 'pro'

def run_query_app(data_file, gps_file, sport=sport, user_type=user_type):
    """Run Streamlit app for AFL/NRL analytics."""
    st.title(f"AI Insights for {sport} Teams")
    with open(f'sport_config_{sport}.json', 'r') as f:
        config = json.load(f)[sport]

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv(data_file)

    @st.cache_data
    def load_lineup():
        with open(f'ga_lineup_{sport}.pkl', 'rb') as f:
            return pickle.load(f)

    @st.cache_data
    def load_predictions():
        with open(f'lstm_predictions{sport}.pkl', 'rb') as f:
            return pickle.load(f)

    @st.cache_data
    def load_gps():
        gps = pd.read_csv(gps_file)
        return gps[gps['player'].isin(data['player'].unique())]

    data = load_data()
    best_lineup = load_lineup()
    future_predictions = load_predictions()
    gps_data = load_gps()

    # Query Interface
    questions = [
        "Who's at high injury risk in 2026?",
        "What's the optimal lineup for 2026?",
        # "Who drives comebacks?",
        "Who has high GPS fatigue risk?",
        "Track young players' development" if user_type == 'pro' else "Who's the next rising star?",
        "Compare players across teams"
    ]
    question = st.selectbox("Ask AI a Question:", questions)

    if question == "Who's at high injury risk in 2026?":
        st.subheader("Predicted Injury Risks for 2026")
        high_risk = data[data['injury_risk'] > 0.7][['team', 'player', 'injury_risk']].drop_duplicates()
        st.dataframe(high_risk)
        if user_type == 'pro' and future_predictions['players']:
            st.subheader("2026 Forecast for Key Players")
            pred_df = pd.DataFrame({'Player': future_predictions['players'], '2026 Injury Risk': future_predictions['injury_risk']})
            st.dataframe(pred_df)

    elif question == "What's the optimal lineup for 2026?":
        st.subheader(f"Optimal Lineup for {sport} 2026")
        lineup_df = pd.DataFrame({'Player': best_lineup})
        lineup_df = lineup_df.merge(data[['team', 'player', 'position']].drop_duplicates(), left_on='Player', right_on='player')
        st.dataframe(lineup_df[['team', 'Player', 'position']])

    # elif question == "Who drives comebacks?":
    #     st.subheader("Clutch Performers")
    #     fig = px.scatter(data, x=config['metrics'][1], y=config['metrics'][0], color='performance_cluster',
    #                      size=config['metrics'][2], text='player', title="Player Clusters: Clutch vs. Anchors",
    #                      color_continuous_scale=['#F1AB00', '#C8305D', '#0066CC'])
    #     fig.update_traces(textposition='top center')
    #     st.plotly_chart(fig, use_container_width=True)

    elif question == "Who has high GPS fatigue risk?":
        st.subheader("Players with High GPS Fatigue Risk")
        high_gps_risk = data[data['high_speed_runs'] > config['league_avg']['high_speed_runs'] * 1.2][
            ['team', 'player', 'high_speed_runs', 'total_distance', 'max_acceleration']].drop_duplicates()
        st.dataframe(high_gps_risk.sort_values('high_speed_runs', ascending=False))
        if user_type == 'pro':
            selected_player = st.selectbox("View Heatmap for Player:", data['player'].unique())
            player_gps = gps_data[gps_data['player'] == selected_player]
            fig_heatmap = px.density_heatmap(player_gps, x='x_pos', y='y_pos', z='speed',
                                             title=f"{selected_player}'s Movement Heatmap",
                                             color_continuous_scale='Viridis', width=600, height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

    elif question in ["Track young players' development", "Who's the next rising star?"]:
        st.subheader("Player Development Tracker" if user_type == 'pro' else "Rising Star Predictor")
        young_players = data[data['age'] < 23][['team', 'player', 'age', 'game_date'] + config['metrics'] + 
                                              ['breakout_prob', 'training_recommendations']].drop_duplicates('player')
        if user_type == 'pro':
            selected_player = st.selectbox("Select Player:", young_players['player'].unique())
            player_data = data[data['player'] == selected_player]
            fig = go.Figure()
            for metric in config['metrics']:
                fig.add_trace(go.Scatter(x=player_data['game_date'], y=player_data[metric], mode='lines+markers', name=metric))
            fig.update_layout(title=f"{selected_player}'s Performance Trends", xaxis_title="Game Date", yaxis_title="Metric Value")
            st.plotly_chart(fig, use_container_width=True)
            breakout_prob = young_players[young_players['player'] == selected_player]['breakout_prob'].iloc[0]
            st.write(f"Breakout Probability for 2026: {breakout_prob:.2%}")
            st.write("Training Recommendations:", young_players[young_players['player'] == selected_player]['training_recommendations'].iloc[0])
            if st.button("Download Development Report"):
                pdf_file = f"{selected_player}_development_report.pdf"
                c = canvas.Canvas(pdf_file, pagesize=letter)
                c.drawString(100, 750, f"Development Report: {selected_player} ({sport})")
                c.drawString(100, 730, f"Team: {player_data['team'].iloc[0]}")
                c.drawString(100, 710, f"Breakout Probability: {breakout_prob:.2%}")
                c.drawString(100, 690, "Training Recommendations:")
                for i, rec in enumerate(young_players[young_players['player'] == selected_player]['training_recommendations'].iloc[0]):
                    c.drawString(120, 670 - i * 20, f"- {rec}")
                c.save()
                with open(pdf_file, 'rb') as f:
                    st.download_button("Download PDF", f, pdf_file)
        else:
            st.subheader("Top Rising Stars")
            top_stars = young_players.sort_values('breakout_prob', ascending=False)[['team', 'player', 'age', 'breakout_prob']]
            st.dataframe(top_stars.rename(columns={'breakout_prob': 'Breakout Probability'}))

    elif question == "Compare players across teams":
        st.subheader("Player Comparison Across Teams")
        teams = data['team'].unique()
        selected_players = []
        for i in range(2):
            team = st.selectbox(f"Select Team {i+1}:", teams, key=f"team_{i}")
            player = st.selectbox(f"Select Player from {team}:", data[data['team'] == team]['player'].unique(), key=f"player_{i}")
            selected_players.append((team, player))
        comparison = data[data[['team', 'player']].apply(tuple, axis=1).isin(selected_players)][
            ['team', 'player'] + config['metrics'] + ['total_distance', 'high_speed_runs', 'injury_risk']].drop_duplicates('player')
        fig = go.Figure()
        for _, row in comparison.iterrows():
            fig.add_trace(go.Bar(x=config['metrics'] + ['total_distance', 'high_speed_runs', 'injury_risk'],
                                 y=[row[m] for m in config['metrics']] + [row['total_distance'], row['high_speed_runs'], row['injury_risk']],
                                 name=f"{row['player']} ({row['team']})"))
        fig.update_layout(title="Player Comparison", xaxis_title="Metrics", yaxis_title="Values", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
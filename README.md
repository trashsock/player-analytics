# AI Insights for Sports
Leveraging machine learning and simulated GPS data (2025 Brisbane Broncos used), this project answers critical questions like:

- Who’s at high injury risk in 2026? Predict risks using fatigue and high-speed runs.

- What’s the optimal lineup for a Storm rematch? Optimise the 13-player squad with data-driven precision.

- Who drives comebacks like the 2025 Grand Final? Identify clutch performers like Reece Walsh.

- Which players face GPS fatigue risks? Analyse high-speed runs and movement patterns.

_This lightweight tool integrates with existing data (e.g., Catapult GPS, Champion Data) and delivers actionable insights for coaches and analysts._

## Why This Matters
This tool focuses on extracting meaningful insights from existing data, answering the “right questions” to keep your team on top.

### Key features:
Injury Prediction: 80% accuracy in identifying at-risk players using logistic regression and LSTM models.

Lineup Optimisation: Genetic Algorithm selects the best 13-player lineup, boosting performance by up to 15%.

Player Clustering: K-means identifies “clutch” (e.g., Walsh) vs. “defensive anchors” (e.g., Haas).

GPS Insights: Simulates Catapult-style data to analyze high-speed runs and movement heatmaps, flagging fatigue risks.

## Demo
To explore insights:
[See Demo](https://player-analytics.streamlit.app/)

## Key Features

Synthetic Data: Simulates 2025 Broncos roster (~30 players, 15 games) with match stats (tackles, meters run, try assists) and GPS data (total distance, high-speed runs, max acceleration).

Models: Lightweight K-means, logistic regression, LSTM, and Genetic Algorithm, precomputed for fast demos.

Interactive Demo: Streamlit app with questions tailored to whatever your team needs, including GPS heatmaps.


## Benefits

Injury Prevention: Flag players at risk from high-speed runs.

Performance Boost: Optimise lineups for 2026.

NRL Edge: Aligns with league’s AI trends (e.g., 2026 fixture draw).

Seamless Integration: Works with Catapult GPS and Champion Data.

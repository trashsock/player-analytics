import pandas as pd
import numpy as np

# Full 2025 Broncos roster (~30 players)
players = [
    'Reece Walsh', 'Payne Haas', 'Patrick Carrigan', 'Ezra Mam', 'Adam Reynolds',
    'Selwyn Cobbo', 'Kotoni Staggs', 'Brendan Piakura', 'Jordan Riki', 'Corey Jensen',
    'Billy Walters', 'Deine Mariner', 'Herbie Farnworth', 'Xavier Willison', 'Corey Oates',
    'Jesse Arthars', 'Blake Mozer', 'Tyson Smoothy', 'Martin Taupau', 'Ben Te Kura',
    'Fletcher Baker', 'Kobe Hetherington', 'Jack Gosiewski', 'Josiah Karapani', 'Coby Black',
    'Jaiyden Hunt', 'Tristan Sailor', 'Delouise Hoeter', 'Israel Leota', 'Josh Rogers'
]
positions = [
    'Fullback', 'Prop', 'Lock', 'Five-eighth', 'Halfback', 'Winger', 'Centre', 
    'Second-row', 'Second-row', 'Prop', 'Hooker', 'Winger', 'Centre', 'Prop', 
    'Winger', 'Winger', 'Hooker', 'Hooker', 'Prop', 'Prop', 'Prop', 'Second-row', 
    'Second-row', 'Centre', 'Halfback', 'Prop', 'Fullback', 'Centre', 'Winger', 'Halfback'
]

# Synthetic main dataset (300 rows: ~30 players, 10 games each)
np.random.seed(42)
n_games = 10
n_players = len(players)
data = {
    'player': np.repeat(players, n_games),
    'position': np.repeat(positions, n_games),
    'game_date': pd.date_range('2025-03-01', '2025-10-05', periods=n_games).repeat(n_players),
    'tackles': np.random.randint(5, 50, n_players * n_games).astype('float32'),
    'meters_run': np.random.randint(20, 200, n_players * n_games).astype('float32'),
    'try_assists': np.random.randint(0, 5, n_players * n_games).astype('float32'),
    'fatigue_score': np.random.uniform(0, 100, n_players * n_games).astype('float32'),
    'injury_status': np.random.choice([0, 1], n_players * n_games, p=[0.9, 0.1]).astype('float32')
}
df_main = pd.DataFrame(data)

# Adjust stats for key players (e.g., Walsh: high meters, try assists; Haas: high tackles)
for player in ['Reece Walsh', 'Payne Haas', 'Patrick Carrigan', 'Ezra Mam', 'Adam Reynolds']:
    mask = df_main['player'] == player
    if player == 'Reece Walsh':
        df_main.loc[mask, 'meters_run'] *= 1.5
        df_main.loc[mask, 'try_assists'] *= 1.5
    elif player == 'Payne Haas':
        df_main.loc[mask, 'tackles'] *= 1.5
    elif player == 'Patrick Carrigan':
        df_main.loc[mask, 'tackles'] *= 1.2
        df_main.loc[mask, 'meters_run'] *= 1.2
    elif player in ['Ezra Mam', 'Adam Reynolds']:
        df_main.loc[mask, 'try_assists'] *= 1.3

# Simulate dummy GPS data (time-series: 10 time points per player per game, ~3000 rows total)
n_time_points = 10  # Lightweight for Colab
gps_data = {
    'player': np.repeat(df_main['player'], n_time_points),
    'game_date': np.repeat(df_main['game_date'], n_time_points),
    'timestamp': np.tile(np.arange(n_time_points), len(df_main)),
    'x_pos': np.random.uniform(0, 100, len(df_main) * n_time_points).astype('float32'),  # Field length: 100m
    'y_pos': np.random.uniform(0, 68, len(df_main) * n_time_points).astype('float32'),   # Field width: 68m
    'speed': np.random.uniform(0, 10, len(df_main) * n_time_points).astype('float32'),   # m/s (max ~10 m/s in rugby)
    'acceleration': np.random.uniform(-5, 5, len(df_main) * n_time_points).astype('float32')  # m/s²
}
df_gps = pd.DataFrame(gps_data)

# Aggregate GPS metrics per player per game (add to main dataset)
gps_agg = df_gps.groupby(['player', 'game_date']).agg(
    total_distance=('speed', lambda x: np.sum(x) * 8),  # Simulate distance (speed * time interval ~8s)
    high_speed_runs=('speed', lambda x: np.sum(x > 5)),  # Count runs >5 m/s
    max_acceleration=('acceleration', 'max')
).reset_index()

df_main = df_main.merge(gps_agg, on=['player', 'game_date'])

# Adjust GPS for key players (e.g., Walsh: high speed/distance; Haas: high acceleration in tackles)
for player in ['Reece Walsh', 'Payne Haas']:
    mask = df_main['player'] == player
    if player == 'Reece Walsh':
        df_main.loc[mask, 'total_distance'] *= 1.5
        df_main.loc[mask, 'high_speed_runs'] *= 1.5
    elif player == 'Payne Haas':
        df_main.loc[mask, 'max_acceleration'] *= 1.3

df_main.to_csv('broncos_2025_stats.csv', index=False)
df_gps.to_csv('broncos_gps_raw.csv', index=False)  
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib
import pickle
from deap import base, creator, tools, algorithms
import json
import gc

def train_models(data_file, sport):
    """Train models on league-wide data for injury, lineup, and breakout prediction."""
    data = pd.read_csv(data_file, dtype='float32')
    data.fillna(0, inplace=True)
    with open(f'sport_config_{sport}.json', 'r') as f:
        config = json.load(f)[sport]

    # 1. K-means: Player types league-wide
    features = config['metrics'] + ['total_distance', 'high_speed_runs', 'max_acceleration']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=50, random_state=42)
    data['performance_cluster'] = kmeans.fit_predict(X_scaled)
    joblib.dump(scaler, f'scaler_{sport}.pkl')
    joblib.dump(kmeans, f'kmeans_{sport}.pkl')

    # 2. Logistic Regression: Injury risk
    X_injury = data[['fatigue_score'] + features]
    y_injury = data['injury_status']
    lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
    lr.fit(X_injury, y_injury)
    data['injury_risk'] = lr.predict_proba(X_injury)[:, 1]
    joblib.dump(lr, f'lr_{sport}.pkl')

    # 3. LSTM: 2026 injury risk
    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(2, 32, 1, batch_first=True)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    players = data['player'].unique()
    X_seq = []
    y_seq = []
    for player in players:
        player_data = data[data['player'] == player][['fatigue_score', 'high_speed_runs', 'injury_risk']].values
        if len(player_data) >= 10:
            X_seq.append(player_data[:10, :2].reshape(1, 10, 2))
            y_seq.append(player_data[9, 2])
    if X_seq:
        X_seq = np.concatenate(X_seq)
        y_seq = np.array(y_seq)
        scaler_seq = StandardScaler()
        X_seq_scaled = scaler_seq.fit_transform(X_seq.reshape(-1, 2)).reshape(X_seq.shape)
        joblib.dump(scaler_seq, f'scaler_seq_{sport}.pkl')

        model = LSTM().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(50):
            inputs = torch.tensor(X_seq_scaled, dtype=torch.float32).cuda()
            targets = torch.tensor(y_seq, dtype=torch.float32).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f'lstm_{sport}.pth')

        future_X = X_seq_scaled[-5:] if len(X_seq_scaled) >= 5 else X_seq_scaled
        pred = torch.sigmoid(model(torch.tensor(future_X, dtype=torch.float32).cuda())).cpu().detach().numpy()
        future_predictions = {'players': players[-len(pred):], 'injury_risk': pred.flatten()}
        with open(f'lstm_predictions_{sport}.pkl', 'wb') as f:
            pickle.dump(future_predictions, f)
    else:
        with open(f'lstm_predictions{sport}.pkl', 'wb') as f:
            pickle.dump({'players': [], 'injury_risk': []}, f)

    # 4. Genetic Algorithm: Optimal lineup (team-specific, but trained league-wide)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def evaluate_lineup(individual):
        lineup = [data.iloc[i] for i in individual]
        score = sum(p[config['metrics'][0]] + p['total_distance'] / 1000 - p['high_speed_runs'] / 10 for p in lineup)
        positions = [p['position'] for p in lineup]
        required = {pos: 1 if pos in config['positions'][:5] else 2 for pos in config['positions']}
        if sport == 'AFL':
            required.update({'Midfielder': 3, 'Forward': 3, 'Defender': 3})
        if len(lineup) != config['lineup_size'] or any(positions.count(pos) != required.get(pos, 0) for pos in set(positions)):
            return -np.inf,
        return score,

    toolbox = base.Toolbox()
    toolbox.register("indices", np.random.permutation, len(data))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_lineup)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, halloffame=hof)

    best_lineup = [data.iloc[i]['player'] for i in hof[0]]
    with open(f'ga_lineup_{sport}.pkl', 'wb') as f:
        pickle.dump(best_lineup, f)

    # 5. Breakout Prediction: Young players league-wide
    young_players = data[data['age'] < 23]
    X_breakout = young_players[features]
    breakout_scores = young_players[config['metrics']].apply(lambda x: (x - pd.Series(config['league_avg'])).sum(), axis=1)
    y_breakout = (breakout_scores > breakout_scores.quantile(0.75)).astype(int) if not breakout_scores.empty else np.zeros(len(young_players))
    lr_breakout = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
    lr_breakout.fit(X_breakout, y_breakout)
    young_players['breakout_prob'] = lr_breakout.predict_proba(X_breakout)[:, 1]
    joblib.dump(lr_breakout, f'lr_breakout{sport}.pkl')

    # Training recommendations
    recommendations = []
    for _, row in young_players.iterrows():
        rec = []
        for metric in config['metrics']:
            if row[metric] < config['league_avg'][metric] * 0.8:
                rec.append(f"Increase training on {metric.replace('_', ' ')}")
        if row['high_speed_runs'] > config['league_avg']['high_speed_runs'] * 1.2:
            rec.append("Reduce sprint load to lower injury risk")
        recommendations.append(rec if rec else ["Maintain current training"])
    young_players['training_recommendations'] = recommendations

    # Merge and save
    data = data.merge(young_players[['team', 'player', 'game_date', 'breakout_prob', 'training_recommendations']], 
                      on=['team', 'player', 'game_date'], how='left')
    data.to_csv(f'league_analytics_output_{sport}.csv', index=False)
    return True
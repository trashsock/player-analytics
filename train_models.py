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
import gc
import os  # Ensure os is imported

def train_models(data_file):
    """
    Train K-means, logistic regression, LSTM, and Genetic Algorithm models.

    Args:
        data_file (str): Path to the input CSV file containing player data.

    Returns:
        bool: True if training is successful, False otherwise.

    Raises:
        FileNotFoundError: If the data file does not exist.
        KeyError: If required columns are missing in the data.
        ValueError: If data types are incompatible with expected formats.
    """
    # Validate input file
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")

    # Define expected columns and their data types
    required_columns = [
        'tackles', 'meters_run', 'try_assists', 'total_distance', 'high_speed_runs',
        'max_acceleration', 'fatigue_score', 'injury_status', 'player', 'position'
    ]
    numeric_columns = [
        'tackles', 'meters_run', 'try_assists', 'total_distance', 'high_speed_runs',
        'max_acceleration', 'fatigue_score', 'injury_status'
    ]
    categorical_columns = ['player', 'position']

    try:
        # Load data without enforcing dtypes initially to inspect
        data = pd.read_csv(data_file)
        
        # Validate required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Check and clean numeric columns
        for col in numeric_columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().any():
                    print(f"Warning: Column '{col}' contains non-numeric values or NaNs. Filling NaNs with 0.")
                    data[col].fillna(0, inplace=True)
            except Exception as e:
                raise ValueError(f"Cannot convert column '{col}' to numeric: {str(e)}")
            
            # Ensure float32 type
            data[col] = data[col].astype('float32')

        # Ensure categorical columns are strings
        for col in categorical_columns:
            data[col] = data[col].astype(str)

        # Ensure injury_status is binary (0 or 1)
        if not data['injury_status'].isin([0, 1]).all():
            print("Warning: 'injury_status' contains non-binary values. Converting to binary (0 or 1).")
            data['injury_status'] = data['injury_status'].apply(lambda x: 1 if x > 0 else 0).astype('float32')

        # Debugging: Print data types and sample data
        print("Data types after cleaning:")
        print(data.dtypes)
        print("Sample data:")
        print(data.head())

        # 1. K-means Clustering
        features = ['tackles', 'meters_run', 'try_assists', 'total_distance', 'high_speed_runs', 'max_acceleration']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[features])
        kmeans = MiniBatchKMeans(n_clusters=3, batch_size=50, random_state=42)
        data['performance_cluster'] = kmeans.fit_predict(X_scaled)
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(kmeans, 'kmeans.pkl')

        # 2. Logistic Regression: Injury risk (create injury_risk before LSTM)
        X_injury = data[['fatigue_score', 'tackles', 'meters_run', 'high_speed_runs', 'max_acceleration']]
        y_injury = data['injury_status']
        lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
        lr.fit(X_injury, y_injury)
        data['injury_risk'] = lr.predict_proba(X_injury)[:, 1].astype('float32')
        joblib.dump(lr, 'lr.pkl')

        # 3. LSTM: Predict 2026 injury risk
        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(2, 32, 1, batch_first=True)  # fatigue + high_speed_runs
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
        
        if not X_seq:
            print("Warning: No players with sufficient data for LSTM training.")
            return False

        X_seq = np.concatenate(X_seq)
        y_seq = np.array(y_seq)

        scaler_seq = StandardScaler()
        X_seq_scaled = scaler_seq.fit_transform(X_seq.reshape(-1, 2)).reshape(X_seq.shape)
        joblib.dump(scaler_seq, 'scaler_seq.pkl')

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(50):
            inputs = torch.tensor(X_seq_scaled, dtype=torch.float32).to(device)
            targets = torch.tensor(y_seq, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'lstm.pth')

        future_X = X_seq_scaled[-5:]
        pred = torch.sigmoid(model(torch.tensor(future_X, dtype=torch.float32).to(device))).cpu().detach().numpy()
        future_predictions = {'players': players[-5:], 'injury_risk': pred.flatten()}
        with open('lstm_predictions.pkl', 'wb') as f:
            pickle.dump(future_predictions, f)

        # 4. Genetic Algorithm: Optimal lineup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,), module=__name__)
        creator.create("Individual", list, fitness=creator.FitnessMax, module=__name__)

        def evaluate_lineup(individual):
            lineup = [data.iloc[i] for i in individual]
            score = sum(p['try_assists'] + p['total_distance'] / 1000 - p['high_speed_runs'] / 10 for p in lineup)
            positions = [p['position'] for p in lineup]
            required = {'Fullback': 1, 'Winger': 2, 'Centre': 2, 'Five-eighth': 1, 'Halfback': 1,
                        'Hooker': 1, 'Prop': 2, 'Second-row': 2, 'Lock': 1}
            if len(lineup) != 13 or any(positions.count(pos) != required.get(pos, 0) for pos in set(positions)):
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
        with open('ga_lineup.pkl', 'wb') as f:
            pickle.dump(best_lineup, f)

        data.to_csv('broncos_analytics_output.csv', index=False)
        
        # Clean up memory
        del data, X_scaled, X_seq, pop
        gc.collect()
        return True

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False
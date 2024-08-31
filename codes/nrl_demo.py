import cv2
import numpy as np
import torch
from sort import Sort  # DeepSORT for tracking players and ball
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize YOLOv8 model (detects both players and the ball)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Switch to yolov8 if installed

# Initialize DeepSORT tracker for both players and ball
tracker = Sort()

# Dictionary to store player trajectories (player_id -> list of positions)
frame_count = 0
ball_trajectory = []
trajectories = defaultdict(list)  # To store player trajectories

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_ball_in_goal(ball_pos, goal_coords):
    x1, y1, x2, y2 = goal_coords
    return x1 <= ball_pos[0] <= x2 and y1 <= ball_pos[1] <= y2

def detect_goal(frame, results):
    goal_coords = None
    for *xyxy, conf, cls in results.xyxy[0]:  
        if cls == 'goal':  # Update this class ID based on your model
            x1, y1, x2, y2 = map(int, xyxy)
            goal_coords = (x1, y1, x2, y2)
            break  
    if goal_coords:
        cv2.rectangle(frame, (goal_coords[0], goal_coords[1]), (goal_coords[2], goal_coords[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Goal", (goal_coords[0], goal_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return goal_coords

def generate_heatmap(positions, field_dim=(105, 68), bins=50, show=True):
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # Create heatmap data
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=bins, range=[[0, field_dim[0]], [0, field_dim[1]]])
    heatmap_normalized = heatmap / np.sum(heatmap)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    if show:
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_normalized.T, extent=extent, origin='lower', cmap='coolwarm', cbar=True)
        plt.title('Player Movement Heatmap')
        plt.xlabel('Field Length (meters)')
        plt.ylabel('Field Width (meters)')
        plt.show()

    return heatmap_normalized, extent

def analyze_heatmap(heatmap_data, extent, field_dim=(105, 68)):
    metrics = {}

    max_density = np.max(heatmap_data)
    high_activity_threshold = 0.05
    hot_spots = np.argwhere(heatmap_data > high_activity_threshold)

    xedges = np.linspace(0, field_dim[0], heatmap_data.shape[0])
    yedges = np.linspace(0, field_dim[1], heatmap_data.shape[1])

    hot_spot_positions = [(xedges[x], yedges[y]) for x, y in hot_spots]

    metrics['hot_spot_positions'] = hot_spot_positions
    metrics['hot_spot_count'] = len(hot_spots)

    x_coords, y_coords = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    center_x = np.sum(x_coords * heatmap_data) / np.sum(heatmap_data)
    center_y = np.sum(y_coords * heatmap_data) / np.sum(heatmap_data)
    metrics['center_of_activity'] = (center_x, center_y)

    field_coverage = np.sum(heatmap_data > 0) / np.product(heatmap_data.shape) * 100
    metrics['field_coverage_percentage'] = field_coverage

    var_x = np.var(x_coords * heatmap_data)
    var_y = np.var(y_coords * heatmap_data)
    metrics['movement_spread_variance'] = (var_x, var_y)

    thirds = np.array_split(xedges, 3)
    defensive_zone = (heatmap_data[xedges < thirds[0][-1], :]).sum()
    midfield_zone = (heatmap_data[(xedges >= thirds[0][-1]) & (xedges < thirds[2][0]), :]).sum()
    offensive_zone = (heatmap_data[xedges >= thirds[2][0], :]).sum()

    metrics['time_in_defensive_zone'] = defensive_zone / np.sum(heatmap_data) * 100
    metrics['time_in_midfield_zone'] = midfield_zone / np.sum(heatmap_data) * 100
    metrics['time_in_offensive_zone'] = offensive_zone / np.sum(heatmap_data) * 100

    return metrics

# Load video
cap = cv2.VideoCapture('path_to_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detect players, ball, and goal using YOLO
    results = model(frame)

    # Detect goal dynamically
    goal_coords = detect_goal(frame, results)

    # Extract detected objects (players, ball)
    dets = []
    ball_detected = None
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.5:
            if cls == 'ball':
                ball_detected = [(int(xyxy[0] + xyxy[2]) // 2, int(xyxy[1] + xyxy[3]) // 2)]
            elif cls == 'player':  
                dets.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf.item()])

    dets = np.array(dets)

    # Track players and the ball
    track_bbs_ids = tracker.update(dets)

    # Track ball
    if ball_detected:
        ball_trajectory.append(ball_detected[0])

    # Process tracked objects
    for track in track_bbs_ids:
        x1, y1, x2, y2, track_id = map(int, track)

        # Draw bounding box and track ID on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Calculate player center position and update trajectory
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        trajectories[track_id].append(center)

    # Check for events (Pass, Goal, Tackle)
    if ball_detected:
        ball_pos = ball_detected[0]

        # Check for passes (ball moves from one player to another)
        for player_id, positions in trajectories.items():
            if len(positions) > 1 and calculate_distance(ball_pos, positions[-1]) < 50:
                print(f"Pass detected at frame {frame_count}, Player {player_id}")

        # Check for goals (ball entering dynamically detected goal area)
        if goal_coords and is_ball_in_goal(ball_pos, goal_coords):
            print(f"Goal detected at frame {frame_count}, Ball position: {ball_pos}")

        # Check for tackles (players in close proximity)
        for player1_id, pos1 in trajectories.items():
            for player2_id, pos2 in trajectories.items():
                if player1_id != player2_id and calculate_distance(pos1[-1], pos2[-1]) < 30:
                    print(f"Tackle detected between Player {player1_id} and Player {player2_id} at frame {frame_count}")

    # Display frame
    cv2.imshow('Rugby Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After processing the video, generate and analyze the heatmap for each player
for player_id, positions in trajectories.items():
    print(f"Analyzing heatmap for Player {player_id}...")
    heatmap_data, extent = generate_heatmap(positions, field_dim=(105, 68), show=False)
    metrics = analyze_heatmap(heatmap_data, extent)
    for key, value in metrics.items():
        print(f'{key}: {value}')

# Close video stream
cap.release()
cv2.destroyAllWindows()

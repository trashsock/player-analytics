# player-analytics
Give player analytics for footy (AFL/W) and rugby (NRL/W).

## Key features:
1. Highly accurate player and ball detection (YOLOv8)
2. Tracks players over time and assigns unique IDs (DeepSORT)
3. Track player movement frame by frame to calculate speed and total distance
4. Visualise player trajectories or heatmaps over time
5. Extend the same approach to track the ball and generate ball possession statistics
6. Event detection:
    6.1. A pass can be inferred when the ball moves from one player to another
    6.2. Detect when the ball enters the goal area
    6.3. Detect when two players get very close to each other, implying a possible tackle.


## HOW TO USE
### Run the Code in Conda

1. **Create a new Conda environment**
   ```
   conda create -n tracking_env python=3.8
   conda activate tracking_env
   ```

2. **Install the necessary packages**
   ```
   conda install -c conda-forge opencv
   conda install numpy matplotlib seaborn
   conda install pytorch torchvision torchaudio -c pytorch
   ```

3. **Install YOLO (either v5 or v8) and DeepSORT**
   ```
   pip install ultralytics 
   ```
   ```
   pip install sort
   ```

4. **Check your environment**
   ```
   conda list
   ```

5. **Run the code**
   ```
   python fileName.py
   ```

### Video usage
- Code has a temporary placeholder for inputting video; please replace with your actual video.
- Currently only accepts video from disc, but can modify to accept videos from online (e.g., YouTube).
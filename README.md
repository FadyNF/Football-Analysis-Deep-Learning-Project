# ⚽ Football Player Tracking and Team Assignment

This project leverages object detection, tracking, and clustering techniques to detect, track, and assign football players to their respective teams based on jersey color in a video.

## 📌  Features
- Detect players, referees, goalkeepers, and the ball using YOLOv8.
- Track objects using ByteTrack.
- Assign players to teams by clustering their jersey colors using KMeans.
- Annotate frames with ellipses, triangle markers, and team information.
- Export annotated video.

## 📂  Project Structure
```├── main.py # Entry point
├── src/
│ ├── tracker/ # Tracking and annotation
│ └── team_assigner/ # Clustering and team assignment 
│ └── training/ # Jupyter script to fine tune YOLO models
| 
├── utils/ # Utilities (e.g., read/save video) 
|
├── models/ 
│ └── best_yolo12x.pt # Trained YOLO models 
│ └── etc... 
|
├── data/ 
│ └── footage/ # Input videos 
|
├── output/ 
│ └── footage/ # Output directory for videos 
|
└── stubs/ # (Optional) Cached detections
```

## 🛠️ Installation
```bash
pip install -r requirements.txt
```

Dependencies include:

- OpenCV
- NumPy
- scikit-learn
- Ultralytics (YOLOv8)
- Supervision (for ByteTrack)


## 🚀 Usage
```bash
python main.py
```

The script will:

1) Read video from `data/footage/`
2) Detect and track players and ball
3) Assign players to teams based on jersey color clustering
4) Draw annotated visualizations
5) Save the output video to `output/footage/`


## 📒 Notes
- YOLO model should be trained with custom football classes: `player`, `goalkeeper`, `referee`, and `ball`.

## 📦 Dataset
This project uses a custom dataset from RoboFlow for fine-tuning the YOLO models. The dataset includes annotated football video frames labeled with:

- Players
- GoalKeeper
- Ball
- Referees

You can access the dataset here:  
🔗 [Download Dataset from RoboFlow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

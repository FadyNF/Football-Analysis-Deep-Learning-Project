import cv2
from utils import save_video, read_video
from src import Tracker, TeamAssigner


def main():
    vid_frames = read_video("data/footage/5.mp4")

    tracker = Tracker("models/best_yolo12x.pt")
    tracks = tracker.get_object_tracks(vid_frames, read_from_stub=False, stub_path=None)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(vid_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                vid_frames[frame_num], track["bbox"], player_id
            )

            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    output_video_frames = tracker.draw_annotations(vid_frames, tracks)

    save_video(output_video_frames, "output/footage/5_yolo12x.avi")


if __name__ == "__main__":
    main()

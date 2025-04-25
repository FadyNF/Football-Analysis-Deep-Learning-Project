from sklearn.cluster import KMeans  # Import for color clustering
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations


class TeamAssigner:
    """
    Class responsible for assigning players to teams based on jersey colors
    using K-means clustering on player bounding boxes.
    """

    def __init__(self):
        """
        Initialize the TeamAssigner with empty dictionaries for team colors
        and player-team assignments.
        """
        self.team_colors = {}  # Dictionary to store team colors: {team_id: color_array}
        self.player_team_dict = (
            {}
        )  # Dictionary to store player to team mappings: {player_id: team_id}

    def get_clustering_model(self, image):
        """
        Create and fit a K-means clustering model with 2 clusters to an image.

        Args:
            image (ndarray): Input image to cluster

        Returns:
            KMeans: Fitted K-means clustering model
        """
        # Reshape the image to a 2D array where each row is a pixel (r,g,b)
        image_2d = image.reshape(-1, 3)

        # Create and fit K-means model with 2 clusters (assuming 2 teams)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant jersey color of a player from their bounding box.
        Uses a technique to distinguish between player jersey and background pixels.

        Args:
            frame (ndarray): The video frame containing the player
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            ndarray: RGB color array representing the player's jersey color
        """
        # Extract player image from frame using bounding box
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Use only the top half of the player (likely to contain more jersey pixels)
        top_half = image[0 : int(image.shape[0] / 2), :]

        # Apply K-means clustering to separate jersey from background
        kmeans = self.get_clustering_model(top_half)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to match the image dimensions
        clustered_img = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Sample the corners of the image to determine which cluster is background
        # Assumption: corners are more likely to contain background pixels
        corner_clusters = [
            clustered_img[0, 0],  # Top left
            clustered_img[0, -1],  # Top right
            clustered_img[-1, 0],  # Bottom left
            clustered_img[-1, -1],  # Bottom right
        ]

        # The most common cluster in the corners is likely the background
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # The other cluster is assumed to be the player's jersey
        player_cluster = 1 - non_player_cluster

        # Get the RGB color of the player's jersey cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Determine team colors by clustering all player jersey colors into two teams.

        Args:
            frame (ndarray): Video frame containing players
            player_detections (dict): Dictionary of player detections {player_id: {"bbox": [x1,y1,x2,y2], ...}}
        """
        player_colors = []

        # Extract colors for all players
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        # Store the clustering model for later use
        self.kmeans = kmeans

        # Assign team colors - team IDs are 1-based
        self.team_colors[1] = kmeans.cluster_centers_[0]  # First team color
        self.team_colors[2] = kmeans.cluster_centers_[1]  # Second team color

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine which team a player belongs to based on their jersey color.
        Reuses previous assignments for consistency.

        Args:
            frame (ndarray): Video frame containing the player
            player_bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            player_id (int): Unique identifier for the player

        Returns:
            int: Team ID (1 or 2)
        """
        # If player already assigned to a team, return that assignment
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player's jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict which team cluster the player belongs to
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Convert to 1-based indexing

        # Store the player's team assignment for future frames
        self.player_team_dict[player_id] = team_id
        return team_id

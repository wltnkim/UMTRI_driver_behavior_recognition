This project recognizes driver behavior by fusing features from three key modalities. We use the following state-of-the-art models to analyze posture, gaze, and facial movements.

1. Pose Estimation: YOLOv8-Pose

We use [YOLOv8-Pose](https://github.com/ultralytics/ultralytics) to extract the 2D coordinates of the driver's body keypoints (e.g., shoulders, hips, wrists). This postural information is crucial for identifying physical activities like operating the infotainment system, interacting with a mobile phone, or reaching for objects.

2. Gaze Estimation: UniGaze

We use [UniGaze](https://github.com/ut-vision/UniGaze) to estimate the driver's 2D gaze direction, providing a direct measure of visual attention. This allows us to classify where the driver is looking (e.g., on-road, center console, mirrors, lap) to detect visual distraction.

3. Facial Movement Analysis: FMAE-IAT

We use [FMAE-IAT](https://github.com/forever208/FMAE-IAT) to analyze fine-grained facial movements and head orientation (yaw, pitch, roll). This complements the gaze data by providing insights into the driver's cognitive state and helps differentiate between activities like talking, yawning, or singing.
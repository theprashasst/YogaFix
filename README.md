# YogaFix: AI-Powered Yoga Pose Correction and Guidance

YogaFix is an application designed to help users practice yoga poses correctly by providing real-time feedback and guidance. It uses computer vision to detect body landmarks and a combination of an expert system and a  deep learning model [CNN] to analyze and correct yoga postures.

## Features

*   **Real-time Pose Detection:** Utilizes MediaPipe Pose to detect 33 key body landmarks in real-time from a webcam feed.
*   **Expert System for Pose Correction:**
    *   Defines correct joint angles and acceptable ranges for various yoga asanas in a flexible JSON configuration file (`exercise_config.json`).
    *   Provides specific corrective feedback if joint angles are outside the defined ranges.
    *   Visual feedback: Draws lines on the user's body, colored green for correct angles and red for incorrect ones.
*   **Guided Asana Sequence:**
    *   Presents the name and description of the current asana.
    *   Shows a reference image of the target pose.
    *   Monitors the duration for which a pose is held correctly.
*   **Text-to-Speech (TTS) Feedback:**
    *   Provides audio cues for pose names, descriptions, and corrective feedback.
    *   Runs in a separate thread to avoid UI freezes.
*   **CNN based Pose Classification:**
    *   Aims to automatically identify the yoga pose being performed by the user using a Deep Learning model trained on landmark data.
    *   Once classified, the expert system takes over for detailed correction of that specific pose.

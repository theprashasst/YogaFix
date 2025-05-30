document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const canvasCtx = overlayCanvas.getContext('2d');

    const poseNameDisplay = document.getElementById('poseName');
    const poseImageDisplay = document.getElementById('poseImage');
    const poseDescriptionDisplay = document.getElementById('poseDescription');
    const feedbackMessagesDisplay = document.getElementById('feedbackMessages');
    const startStopButton = document.getElementById('startStopButton');

    let isWebcamStarted = false;
    let currentExerciseConfig = null;
    let currentPoseFromSequence = null; // Stores the name of the current pose in the sequence
    let currentPoseData = null; // Stores the full data for the current pose from config.poses
    // let currentPoseNameFromAPI = null; // Not directly used for now, but could be for comparison
    let currentSequenceName = 'Default Sequence'; // Or dynamically load available sequences
    let currentPoseIndexInSequence = 0;
    let animationFrameId = null;
    let allCorrectStartTime = null; // To track duration of correct pose
    const CORRECT_POSE_DURATION_THRESHOLD = 2000; // 2 seconds

    // --- Initialization ---
    async function initializeApp() {
        await fetchExerciseConfig();
        // updatePoseDisplay will be called by fetchExerciseConfig if successful
    }

    // --- Core Functions ---
    async function startStopWebcam() {
        if (!isWebcamStarted) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = stream;
                video.play(); // Ensure video plays on some browsers
                isWebcamStarted = true;
                startStopButton.textContent = 'Stop Webcam';

                // Ensure config is loaded before starting processing
                if (!currentExerciseConfig) {
                    await fetchExerciseConfig();
                }
                if (currentExerciseConfig) { // Check if config loaded successfully
                   requestAnimationFrame(processVideoFrame);
                } else {
                    feedbackMessagesDisplay.innerHTML = '<p style="color: red;">Error: Exercise configuration could not be loaded. Cannot start.</p>';
                    stopWebcam(); // Stop webcam if config failed
                }

            } catch (err) {
                console.error('Error accessing webcam:', err);
                feedbackMessagesDisplay.innerHTML = `<p style="color: red;">Error accessing webcam: ${err.message}</p>`;
            }
        } else {
            stopWebcam();
        }
    }

    function stopWebcam() {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        video.srcObject = null;
        isWebcamStarted = false;
        startStopButton.textContent = 'Start Webcam';
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        canvasCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Clear canvas
        allCorrectStartTime = null; // Reset timer
    }

    async function fetchExerciseConfig() {
        try {
            const response = await fetch('/get_exercise_config');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            currentExerciseConfig = await response.json();
            if (currentExerciseConfig && currentExerciseConfig.sequences && currentExerciseConfig.poses) {
                // For now, let's assume one sequence or pick the first one.
                // A more robust app might let the user choose a sequence.
                const sequenceKeys = Object.keys(currentExerciseConfig.sequences);
                if (sequenceKeys.length > 0) {
                    currentSequenceName = sequenceKeys[0]; // Take the first sequence by name
                    currentPoseIndexInSequence = 0; // Start from the first pose
                    updatePoseDisplay();
                } else {
                    feedbackMessagesDisplay.innerHTML = '<p style="color: red;">No sequences found in configuration.</p>';
                    currentExerciseConfig = null; // Invalidate config
                }
            } else {
                feedbackMessagesDisplay.innerHTML = '<p style="color: red;">Exercise configuration is invalid.</p>';
                currentExerciseConfig = null; // Invalidate config
            }
        } catch (error) {
            console.error('Error fetching exercise config:', error);
            feedbackMessagesDisplay.innerHTML = `<p style="color: red;">Error fetching config: ${error.message}</p>`;
            currentExerciseConfig = null; // Invalidate config
        }
    }

    function updatePoseDisplay() {
        if (!currentExerciseConfig || !currentExerciseConfig.sequences[currentSequenceName]) {
            poseNameDisplay.textContent = 'N/A';
            poseDescriptionDisplay.textContent = 'Load an exercise sequence.';
            poseImageDisplay.src = '';
            currentPoseFromSequence = null;
            currentPoseData = null;
            return;
        }

        const sequence = currentExerciseConfig.sequences[currentSequenceName].poses;
        if (currentPoseIndexInSequence >= sequence.length) {
            // Sequence complete
            poseNameDisplay.textContent = "Sequence Complete!";
            poseDescriptionDisplay.textContent = "Congratulations! You've completed the sequence.";
            poseImageDisplay.src = ''; // Clear image or show a completion image
            feedbackMessagesDisplay.innerHTML = '<p style="color: green;">Sequence Complete!</p>';
            stopWebcam();
            return;
        }

        currentPoseFromSequence = sequence[currentPoseIndexInSequence];
        currentPoseData = currentExerciseConfig.poses[currentPoseFromSequence];

        if (currentPoseData) {
            poseNameDisplay.textContent = currentPoseData.name || 'Unknown Pose';
            poseDescriptionDisplay.textContent = currentPoseData.instructions ? currentPoseData.instructions.join(' ') : 'No description available.';
            if (currentPoseData.image_url) {
                poseImageDisplay.src = `/${currentPoseData.image_url}`; // Assuming image_url is relative to static/ or handled by a route
            } else {
                poseImageDisplay.src = ''; // Clear image if none
            }
        } else {
            poseNameDisplay.textContent = 'Error';
            poseDescriptionDisplay.textContent = `Pose data for '${currentPoseFromSequence}' not found in config.`;
            poseImageDisplay.src = '';
            currentPoseFromSequence = null; // Invalidate current pose
        }
        allCorrectStartTime = null; // Reset timer for new pose
    }

    async function processVideoFrame() {
        if (!isWebcamStarted || !currentPoseFromSequence) {
            if (isWebcamStarted) requestAnimationFrame(processVideoFrame); // Keep trying if webcam is on but pose not ready
            return;
        }

        // Match canvas dimensions to video
        if (overlayCanvas.width !== video.videoWidth || overlayCanvas.height !== video.videoHeight) {
            overlayCanvas.width = video.videoWidth;
            overlayCanvas.height = video.videoHeight;
        }

        // Draw video frame to a temporary canvas to get image data (or use video directly if possible with API)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        const base64ImageData = tempCanvas.toDataURL('image/jpeg', 0.8); // Quality 0.8

        const payload = {
            image: base64ImageData.split(',')[1], // Remove 'data:image/jpeg;base64,' part
            current_pose_name: currentPoseFromSequence
        };

        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`API Error: ${errorData.error || response.status}`);
            }

            const result = await response.json();

            // currentPoseNameFromAPI = result.predicted_pose_name_cnn; // Store if needed

            canvasCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Clear previous drawings

            if (result.landmarks_mp) {
                drawLandmarks(result.landmarks_mp, canvasCtx);
            }

            if (result.angle_details_expert && result.angle_details_expert.angles) {
                drawAngles(result.landmarks_mp, result.angle_details_expert.angles, canvasCtx);
                displayFeedback(result.angle_details_expert.feedback, result.all_joints_correct_expert);

                if (result.all_joints_correct_expert) {
                    if (!allCorrectStartTime) {
                        allCorrectStartTime = Date.now();
                    } else if (Date.now() - allCorrectStartTime >= CORRECT_POSE_DURATION_THRESHOLD) {
                        currentPoseIndexInSequence++;
                        updatePoseDisplay();
                        allCorrectStartTime = null; // Reset for the next pose
                    }
                } else {
                    allCorrectStartTime = null; // Reset if pose becomes incorrect
                }

            } else {
                feedbackMessagesDisplay.innerHTML = '<p>No angle details received.</p>';
                 allCorrectStartTime = null; // Reset if no details
            }
             if (result.error_message) {
                feedbackMessagesDisplay.innerHTML += `<p style="color: orange;">Server Message: ${result.error_message}</p>`;
            }


        } catch (error) {
            console.error('Error processing frame:', error);
            feedbackMessagesDisplay.innerHTML = `<p style="color: red;">Error processing frame: ${error.message}</p>`;
            allCorrectStartTime = null; // Reset on error
        }

        if (isWebcamStarted) { // Check again in case webcam was stopped during async operations
            animationFrameId = requestAnimationFrame(processVideoFrame);
        }
    }

    // --- Drawing Functions ---
    function drawLandmarks(landmarks, ctx) {
        ctx.fillStyle = 'rgb(0, 255, 0)'; // Green
        for (const lm of landmarks) {
            if (lm.visibility === undefined || lm.visibility > 0.5) { // Check visibility if available
                ctx.beginPath();
                ctx.arc(lm.x * ctx.canvas.width, lm.y * ctx.canvas.height, 5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }

    // Simple mediapipe connections for drawing skeleton for context
    const POSE_CONNECTIONS = [
        // Torso
        [11, 12], [11, 23], [12, 24], [23, 24],
        // Arms
        [11, 13], [13, 15], [12, 14], [14, 16],
        // Legs
        [23, 25], [25, 27], [24, 26], [26, 28]
    ];

    function drawAngles(landmarks_mp, angleDetails, ctx) {
        if (!landmarks_mp) return;

        // First, draw a basic skeleton for context
        ctx.strokeStyle = 'rgba(200, 200, 200, 0.5)'; // Light grey for skeleton
        ctx.lineWidth = 2;
        POSE_CONNECTIONS.forEach(conn => {
            const lm1 = landmarks_mp[conn[0]];
            const lm2 = landmarks_mp[conn[1]];
            if (lm1 && lm2 && (lm1.visibility === undefined || lm1.visibility > 0.3) && (lm2.visibility === undefined || lm2.visibility > 0.3)) {
                ctx.beginPath();
                ctx.moveTo(lm1.x * ctx.canvas.width, lm1.y * ctx.canvas.height);
                ctx.lineTo(lm2.x * ctx.canvas.width, lm2.y * ctx.canvas.height);
                ctx.stroke();
            }
        });


        // Then draw angle lines based on correctness
        ctx.lineWidth = 4;
        for (const angleName in angleDetails) {
            const angle = angleDetails[angleName];
            const points = angle.landmarks_indices; // e.g., [11, 13, 15] for left elbow

            if (points && points.length === 3) {
                const p1 = landmarks_mp[points[0]];
                const p2 = landmarks_mp[points[1]]; // Joint
                const p3 = landmarks_mp[points[2]];

                if (p1 && p2 && p3 &&
                    (p1.visibility === undefined || p1.visibility > 0.3) &&
                    (p2.visibility === undefined || p2.visibility > 0.3) &&
                    (p3.visibility === undefined || p3.visibility > 0.3)) {

                    ctx.strokeStyle = angle.correct ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)'; // Green if correct, Red if incorrect

                    // Line from p1 to p2
                    ctx.beginPath();
                    ctx.moveTo(p1.x * ctx.canvas.width, p1.y * ctx.canvas.height);
                    ctx.lineTo(p2.x * ctx.canvas.width, p2.y * ctx.canvas.height);
                    ctx.stroke();

                    // Line from p2 to p3
                    ctx.beginPath();
                    ctx.moveTo(p2.x * ctx.canvas.width, p2.y * ctx.canvas.height);
                    ctx.lineTo(p3.x * ctx.canvas.width, p3.y * ctx.canvas.height);
                    ctx.stroke();
                }
            }
        }
    }


    function displayFeedback(feedback, allCorrect) {
        let html = '';
        if (Object.keys(feedback).length === 0 && allCorrect) {
            html = '<p style="color: green;">All joints are correct!</p>';
        } else {
            for (const joint in feedback) {
                const message = feedback[joint];
                // Assuming feedback message already implies correctness or incorrectness color-wise
                // For now, just display. Could add color based on content if needed.
                html += `<p>${joint}: ${message}</p>`;
            }
        }
        feedbackMessagesDisplay.innerHTML = html;
    }

    // --- Event Listeners ---
    startStopButton.addEventListener('click', startStopWebcam);

    // --- Initial Load ---
    initializeApp(); // Fetch config on page load to show first pose info
});

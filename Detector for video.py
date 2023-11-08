import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the trained model
model = load_model("D:/Project DD/Deepfake Detector/model_version 2.h5")

# Open the video
video_path = "D:/Project DD/videos/1.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize variables to aggregate predictions
frame_predictions = []

# Loop through video frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = resized_frame / 255.0

    # Make a prediction for the frame
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))

    # Append the prediction to the list
    frame_predictions.append(prediction)

# Aggregate predictions (you can implement your own aggregation logic)
# For example, you might calculate an average confidence score
aggregated_prediction = np.mean(frame_predictions)

# Decide based on the threshold
threshold = 0.9
if aggregated_prediction > threshold:
    result = "Fake"
else:
    result = "Real"

print("Video Prediction:", result)

# Release the video capture
cap.release()
cv2.destroyAllWindows()

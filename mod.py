import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
try:
    model = tf.keras.models.load_model('pothole_model.keras', compile=False)  # Load model without optimizer
    print("Model loaded successfully without compilation.")
    
    # Recompile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model recompiled successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Initialize video capture with your video file (replace with your actual video file path)
video_path = 'E:/pothole anto code/potooo/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Preprocess the frame to match the model input size
    input_frame = cv2.resize(frame, (224, 224))  # Adjust size based on your model's input
    input_frame = input_frame / 255.0  # Normalize to [0, 1]
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Run the model to detect potholes
    predictions = model.predict(input_frame)

    # Assume the model outputs probabilities for two classes: [non-pothole, pothole]
    pothole_prob = predictions[0][1]  # Probability of pothole
    threshold = 0.5  # Confidence threshold

    # Check if the frame contains a pothole
    pothole_detected = pothole_prob > threshold

    # Annotate the frame
    if pothole_detected:
        cv2.putText(frame, "Pothole Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Pothole", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the probability on the frame
    cv2.putText(frame, f"Confidence: {pothole_prob:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Pothole Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()

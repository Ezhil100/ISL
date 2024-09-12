import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp

# Load the trained model and label encoder
model = tf.keras.models.load_model('isl_gesture_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Preprocess the landmarks
def preprocess_landmarks(landmarks):
    # Flatten the list of landmarks and ensure the shape is (1, 126)
    return np.expand_dims(np.array(landmarks), axis=0)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and extract hand landmarks
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        # Initialize combined landmarks for both hands
        combined_landmarks = [0] * (21 * 3 * 2)  # 42 landmarks for both hands (x, y, z)
        index = 0
        
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                if index < len(combined_landmarks):
                    combined_landmarks[index] = landmark.x
                    combined_landmarks[index + 1] = landmark.y
                    combined_landmarks[index + 2] = landmark.z
                    index += 3
        
        # Handle cases where less than 2 hands are detected
        if len(result.multi_hand_landmarks) == 1:
            combined_landmarks[63:] = [0] * (63)  # Zero out landmarks for the second hand if only one hand is detected
        
        # Preprocess the landmarks and make a prediction
        processed_landmarks = preprocess_landmarks(combined_landmarks)
        prediction = model.predict(processed_landmarks)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        
        # Display the predicted gesture on the frame
        cv2.putText(frame, f"Gesture: {predicted_class[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw hand landmarks on the frame
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame
    cv2.imshow('Indian Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
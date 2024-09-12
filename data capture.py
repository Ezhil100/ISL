import cv2
import mediapipe as mp
import csv
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# List to store captured landmarks and labels
landmark_data = []
labels = []

# Capture hand landmarks for both hands
def capture_hand_landmarks(frame, label):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        # Initialize a list to store landmarks of both hands
        combined_landmarks = [0] * (21 * 3 * 2)  # 42 landmarks for both hands (x, y, z)
        
        # Process landmarks for each detected hand
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            for i, landmark in enumerate(hand_landmarks.landmark):
                if idx == 0:  # First hand (left or right)
                    combined_landmarks[i*3:i*3+3] = [landmark.x, landmark.y, landmark.z]
                elif idx == 1:  # Second hand (left or right)
                    combined_landmarks[63+i*3:63+i*3+3] = [landmark.x, landmark.y, landmark.z]
        
        # Append landmarks and label to the list
        landmark_data.append(combined_landmarks)
        labels.append(label)
        return True

    return False  # No hands were detected

# Save dataset as CSV
def save_dataset(filename='isl_gesture_dataset.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(landmark_data)):
            writer.writerow(landmark_data[i] + [labels[i]])

# Capture data for a given gesture with a start key
def capture_data_for_gesture(gesture_label, num_samples=100):
    cap = cv2.VideoCapture(0)
    collected_samples = 0
    start_capture = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if the 's' key is pressed to start capturing
        if not start_capture:
            cv2.putText(frame, "Press 's' to start capturing data", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Capture Hand Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                start_capture = True
            continue  # Skip the rest of the loop until 's' is pressed

        # Detect hand landmarks and capture them
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            if capture_hand_landmarks(frame, gesture_label):
                collected_samples += 1

        # Display the frame with feedback
        cv2.putText(frame, f"Collecting {gesture_label}: {collected_samples}/{num_samples}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not result.multi_hand_landmarks:
            cv2.putText(frame, "No hands detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw hand landmarks on the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Capture Hand Gestures", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Stop once enough samples have been collected
        if collected_samples >= num_samples:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to collect data for multiple gestures
def collect_gesture_data(gesture_list, samples_per_gesture=250):
    for gesture in gesture_list:
        print(f"Start collecting for gesture: {gesture}")
        time.sleep(2)  # Wait for 2 seconds before starting
        capture_data_for_gesture(gesture, samples_per_gesture)
        print(f"Finished collecting for gesture: {gesture}")
    
    # Save the dataset
    save_dataset()

# Example usage: Collect data for multiple gestures
gestures_to_collect = ['A', 'B', 'C', 'D', 'E']  # Add more gestures as needed
collect_gesture_data(gestures_to_collect, samples_per_gesture=250)
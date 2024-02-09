import pickle

import cv2
import mediapipe as mp
import numpy as np
import random

model_datas = pickle.load(open('./trained_model.p', 'rb'))
model = model_datas['model']

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
               24: 'Y', 25: 'Z'}
scaling_factor = 2.4
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1] * scaling_factor) + 200, (int(frame.shape[0] * scaling_factor) +200 )))

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        
        hand_num = random.randint(0, len(results.multi_hand_landmarks) - 1)
        hand_landmarks = results.multi_hand_landmarks[hand_num]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,  
            mp_hands.HAND_CONNECTIONS,  
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0.5, 0.5, 0.5), 2)
        cv2.putText(frame,predicted_character, (x1, y1 -15 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("Letter Detected:", predicted_character)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

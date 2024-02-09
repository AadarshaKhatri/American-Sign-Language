import os
import pickle
import mediapipe as mp
import cv2

def initialize_hand_tracker():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def process_image(img_path, hand_tracker):
    dataset_list = []
    x_ = []
    y_ = []

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hand_tracker.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                dataset_list.append(x - min(x_))
                dataset_list.append(y - min(y_))

    return dataset_list

def main():
    hand_tracker = initialize_hand_tracker()
    data_folder = './data'
    data = []
    labels = []

    for dir_ in os.listdir(data_folder):
        for img_path in os.listdir(os.path.join(data_folder, dir_)):
            data_aux = process_image(os.path.join(data_folder, dir_, img_path), hand_tracker)
            if data_aux:
                data.append(data_aux)
                labels.append(dir_)

    f = open('Images_serialized.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

if __name__ == "__main__":
    main()

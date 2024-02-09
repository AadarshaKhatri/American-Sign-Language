import cv2
from cvzone.HandTrackingModule import HandDetector



import os

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)




datas_folder = './data'


if not os.path.exists(datas_folder):
    os.makedirs(datas_folder)

number_total_alphabhets = 26
number_of_Images_stored = 100

for j in range(number_total_alphabhets):
    if not os.path.exists(os.path.join(datas_folder, str(j))):
        os.makedirs(os.path.join(datas_folder, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        success, img = capture.read()
        hands, img = detector.findHands(img)
        cv2.putText(img, 'Press "C" for collecting the images ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Hand_detetctor', img)

        if cv2.waitKey(1) == ord('c'):
                
                break
        

    counter = 0
    while counter <  number_of_Images_stored and not done:
        success, frame = capture.read()
        
        cv2.waitKey(1)

        cv2.imwrite(os.path.join(datas_folder, str(j), '{}.jpg'.format(counter)), frame)
        counter = counter + 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
         break
capture.release()
cv2.destroyAllWindows()
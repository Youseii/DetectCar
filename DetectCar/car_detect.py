# ----------------------------Simple Cars Detection Code-----------------------
import cv2

video_src = 'video2.avi'

video = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier("haarcascade_cars.xml")

while True:
    ret, img = video.read()

    if type(img) == type(None):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)

    cv2.imshow('video', img)

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()

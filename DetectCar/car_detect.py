import cv2
import time

class Detect:
    def __init__(self):
        
        self.video_src = 'video2.avi'
        self.video = cv2.VideoCapture(self.video_src)
        self.car_cascade = cv2.CascadeClassifier("haarcascade_cars.xml")
        self.ret, self.img = self.video.read()

    def draw(self, ret, img):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 2)

        for (x, y, w, h) in cars:
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 255), 3)

        cv2.imshow('video', self.img)


    def read(self):
        while True:
            self.ret, self.img = self.video.read()

            if type(self.img) == type(None):
                break
            
            Detect.draw(self, self.ret, self.img)

            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
                break

        return self.ret, self.img


if __name__ == '__main__':
    
    obj = Detect()
    obj.read()
    
    obj.video.release()
    cv2.destroyAllWindows()

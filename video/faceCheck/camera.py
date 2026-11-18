import cv2

class CameraReader:
    def __init__(self, path:str, interval:int=10):
        self.cap = cv2.VideoCapture(path)

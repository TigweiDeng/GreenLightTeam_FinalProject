
import cv2
import sys
import gc
from DataModel import Model
from DataProcess import per_dictionary

if __name__ == '__main__':

    model = Model()
    model.load_model(file_path='./model/me.face.model.h5')

    color = (0, 255, 0)

    video = ''

    cap = cv2.VideoCapture(video)

    cascade_path = "haarcascade_frontalface_default.xml"

    while True:
        _, frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)

                for name, num in per_dictionary.items():
                    if num == faceID:

                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                        # 文字提示
                        cv2.putText(frame, name,
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                else:
                    pass

        cv2.imshow("FaceDetect", frame)


        k = cv2.waitKey(10)

        if k & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
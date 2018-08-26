import cv2
import os

def openvideo(window_name ,video_id ,save_pic_num, picture_path):
    cv2.namedWindow(window_name)

    classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    color = (0, 255, 0)
    num = 0

    cap = cv2.VideoCapture(video_id)
    while cap.isOpened():
        is_opened, frame = cap.read()
        if not is_opened:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测，标记
        face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(200, 200))
        if len(face_rects) > 0:
            for faceRect in face_rects:
                x, y, w, h = faceRect

                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imencode('.jpg', image)[1].tofile(picture_path+'\\'+str(num)+'.jpg')
                num += 1
                if num > (save_pic_num):
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        cv2.imshow(window_name, frame)
        c=cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    print('open camera...')
    path = 'C:\\Users\\13737\\Pictures\\Saved Pictures\\save_pic'
    name = '邓虎威'
    pic_path = path + name
    os.makedirs(pic_path)
    openvideo('openvideo', 0, 500, pic_path)


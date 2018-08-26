from DataProcess import *

path = 'C:\\Users\\13737\\Pictures\\Saved Pictures'

def SavePicFromVideo(window_name, video_path, save_pic_num, picture_path):
    # collected_images = []
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(video_path)

    classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    color = (0, 255, 0)
    num = 0
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
                # image = resize_image(image)
                # 上一句保存resize后的灰度图片
                # cv2.imwrite(, )
                cv2.imencode('.jpg', image)[1].tofile(picture_path+'\\'+str(num)+'.jpg')
                # collected_images.append(image)

                num += 1
                if num > (save_pic_num):
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
        if num > (save_pic_num):
            break
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return

# 试验程序
if __name__ == '__main__':
    source_path = path + '\\source'
    save_path = path + '\\save_pic'
    for dir_item in os.listdir(source_path):
        full_path = os.path.abspath(os.path.join(source_path, dir_item))
        pic_path = os.path.abspath(os.path.join(save_path, dir_item))
        (name_path, mp) = os.path.splitext(pic_path)
        try:
            os.makedirs(name_path)
        except FileExistsError:
            pass
        else:
            SavePicFromVideo("保存人脸", video_path=full_path, save_pic_num=300, picture_path=name_path)



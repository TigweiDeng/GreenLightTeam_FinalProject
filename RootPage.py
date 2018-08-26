
from tkinter import *
from tkinter.filedialog import *

root = Tk()
root.title('人脸识别')

FilePath = ''
filePath = StringVar()
Format = [("格式", "gif jpg jpeg png bmp rmvb avi wmv mpg mpeg mp4")]

homePage = Frame(root, width = 750, height = 570)
faceRecognizePage = Frame(root)

faceModelingBg = PhotoImage(file = 'faceModelingBg.gif')
faceRecognizeBg = PhotoImage(file = 'faceRecognizeBg.gif')





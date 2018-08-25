
from RootPage import *

def createHomePage(faceModelingBg, faceRecognizeBg):
    pageFrame()
    homePage.pack()

    faceModeling = Label(homePage, image = faceModelingBg, width = 302, height = 396)
    faceModeling.place(x = 50, y = 30)

    faceRecognize = Label(homePage, image = faceRecognizeBg, width = 302, height = 396)
    faceRecognize.place(x = 400, y = 30)

    faceModelingButton = Button(homePage, text = '人脸建模\n(未开放)', bg = 'grey', font = ('微软雅黑',20), activebackground = 'grey')
    faceModelingButton.place(x = 101, y = 430, width = 200, height = 100)

    faceRecognizeButton = Button(homePage,  text = '人脸识别', bg = 'skyblue', font = ('微软雅黑',20), activebackground = 'blue', command = createRecognizePage)
    faceRecognizeButton.place(x = 451, y = 430, width = 200, height = 100)

def createRecognizePage():
    homePage.destroy()
    faceRecognizePage.pack()

    path = Entry(faceRecognizePage, textvariable = filePath)
    path.grid(row=0, stick=W, pady = 10, padx = 10)

    choosePath = Button(faceRecognizePage, text = "路径选择", font = ('微软雅黑',10), command = lambda : selectPath(Format))
    choosePath.grid(row=0, column=1, stick=E, padx = 10) 

    confirmButton = Button(faceRecognizePage, text = '确定', font = ('微软雅黑',10), command = confirm)
    confirmButton.grid(row=1, stick=W, pady=10, padx = 10)

    backButton = Button(faceRecognizePage, text = '返回', font = ('微软雅黑',10), command = back)
    backButton.grid(row=1, column=1, stick=E, padx = 10) 

def pageFrame():
    global homePage
    global faceRecognizePage

    homePage = Frame(root, width = 750, height = 570)
    faceRecognizePage = Frame(root)

def selectPath(format):
    global FilePath
    FilePath = askopenfilename(filetypes = format)
    filePath.set(FilePath)

def confirm():
    print(FilePath)

def back():
    faceRecognizePage.destroy()
    createHomePage(faceModelingBg, faceRecognizeBg)
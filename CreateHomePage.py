
#!/usr/bin/env python
# -*- coding:utf-8 -*-

from RootPage import *
#from ButtonFunction import *

def createHomePage(faceModelingBg, faceRecognizeBg):
    homePage.pack()

    faceModeling = Label(homePage, image = faceModelingBg, width = 302, height = 396)
    faceModeling.place(x = 50, y = 30)

    faceRecognize = Label(homePage, image = faceRecognizeBg, width = 302, height = 396)
    faceRecognize.place(x = 400, y = 30)

    faceModelingButton = Button(homePage, text = '人脸建模', bg = 'skyblue', font = ('微软雅黑',20), activebackground = 'blue')
    faceModelingButton.place(x = 101, y = 430, width = 200, height = 100)

    faceRecognizeButton = Button(homePage,  text = '人脸识别', bg = 'pink', font = ('微软雅黑',20), activebackground = 'red')
    faceRecognizeButton.place(x = 451, y = 430, width = 200, height = 100)


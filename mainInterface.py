
#!/usr/bin/env python
# -*- coding:utf-8 -*-


from Root import *

bg = PhotoImage(file = 'bg.gif')
backGround = Label(root, image = bg)
backGround.pack(expand = YES, fill = 'both')

uploadButton = Button(root, width = 20, height = 10, text = 'Upload', bg = 'skyblue')
uploadButton.pack(side = 'left')

recognizeButton = Button(root, width = 20, height = 10, text = 'Recognize', bg = 'white')
recognizeButton.pack(side= 'right')

root.mainloop()

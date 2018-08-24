import tkinter
import os


class Choice:
    def __init__(self, root):
        self.root = root
        self.start()

    def start(self):
        self.frame = tkinter.Frame(height=20, width=20, bg='white')
        botton1 = tkinter.Button(frame, text='储存图片', command=self.save).pack()
        # botton2 = tkinter.Button(self.root, text='训练模型', command=train).pack()
        # botton3 = tkinter.Button(self.root, text='识别图片', command=detect).pack()

    def save(self):

        label_name = tkinter.StringVar()
        entry = tkinter.Entry(self.top1, textvariable=label_name).pack()
        label_name.set('请输入人物标签')
        entry.pack()
        botton4 = tkinter.Button(self.top1, text='确定', command=lambda: self.path_back(label_name)).pack()

        self.top1.mainloop()

    def path_back(self, name):
        self.top1.destory()
        return name


if __name__ == '__main__':

    root = tkinter.Tk()
    start1 = Choice(root)

    root.mainloop()


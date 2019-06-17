import tkinter as tk
import numpy as np
import tkinter.messagebox
import is_end
import AI_run
"""一些全局参数"""
per_size=50
win = tk.Tk()
win.geometry("900x900")
win.title("my first tkinter")
chess_on = -1
"""创建指示牌"""
var = tk.StringVar()
board = np.zeros([15, 15])    #正棋盘
anti_board=np.zeros([15,15])    #负棋盘，用于判断对方棋局得分情况
status = np.zeros([15, 15])
steps = 0
[x1,y1]=[1,1]


"""以下为实际操作时所需使用函数"""
"""标识此时下棋方"""
def GUI():
    def exchange(i):
        if(i==0):
            s.create_oval(10, 10, 50, 50, fill="white")
            var.set("现在白方执棋")
        elif(i==1):
            s.create_oval(10, 10, 50, 50, fill="black")
            var.set("现在黑方执棋")
        else:
            pass

    def game_start():
        global chess_on
        chess_on = 0
        exchange(chess_on)
        s.bind("<Button-1>", fallin)
        
    def game_start1():
        global chess_on
        chess_on = 1
        exchange(chess_on)
        s.bind("<Button-1>", fallin)
            
    """绘制棋子"""
    def draw(c, x0, y0):
        if(c==0):
            s.create_oval(70 + per_size * x0, 70 + per_size * y0, 70 + per_size * (x0 + 1), 70 + per_size * (y0 + 1), fill="white")
        elif (c == 1):
            s.create_oval(70 + per_size * x0, 70 + per_size * y0, 70 + per_size * (x0 + 1), 70 + per_size * (y0 + 1), fill="black")

    """下棋落子状态"""
    def fallin(event):
        global chess_on
        global anti_board
        global board
        global steps
        if (chess_on == 0):
            x1 = (event.x - 70) // per_size
            y1 = (event.y - 70) // per_size
            if (board[x1][y1] != 0):
                tkinter.messagebox.showwarning(title="我的五子棋", message="这个位置已经有人下过啦")
            elif ((x1 < 0) or (x1 > 14) or (y1 <0) or (y1 > 14)):
                tkinter.messagebox.showwarning(title="我的五子棋", message="这个地方超过棋盘界限啦")
            else:
                draw(chess_on, x1, y1)
                board[x1][y1] = 1
                anti_board[x1][y1] = -1   #保留一个逆棋盘，以便随后进行对敌方的得分估计
                steps=steps+1
                chess_on = 1 - chess_on
                exchange(chess_on)
        else:
            [x1,y1]=AI_run.AI2(board,anti_board)
            draw(chess_on, x1, y1)
            board[x1][y1] = -1
            anti_board[x1][y1] = 1
            chess_on = 1 - chess_on
            exchange(chess_on)
        t = is_end.check(board)
        if (t == -1):
            tkinter.messagebox.showinfo(title="我的五子棋", message="黑棋赢了")
            retry = tkinter.messagebox.askokcancel(title='你输了', message="想再玩一把吗")
            if (retry == 1):
                s.delete("all")
                drawout()
                board = np.zeros([15, 15])
                anti_board=np.zeros([15,15])
            else:
                win.quit()
            # var.set("黑棋win")
        elif (t == 1):
            tkinter.messagebox.showinfo(title="我的五子棋", message="白棋赢了")
            retry = tkinter.messagebox.askokcancel(title='你赢了', message="想再玩一把吗")
            if (retry == 1):
                s.delete("all")
                drawout()
                board = np.zeros([15, 15])
                chess_on = 0
                exchange(chess_on)
            else:
                win.quit()
            # var.set("白棋win")
        else:
            pass
        
    """状态表示框"""
    def showstatus():
        game_on = tk.Label(win, textvariable=var)
        game_on.place(x=805,y=20,anchor='nw')
        var.set("游戏还未开始")

    """创建画布"""
    s=tk.Canvas(win,bg="saddlebrown",width=830,height=900)
    s.pack(side='left')

    """创建位置"""
    start_x = [3, 3, 7, 11, 11]
    start_y = [3, 11, 7, 3, 11]

    """创建框架放置按钮"""
    fr = tk.Frame(win, bg="green")
    fr.pack(side="right")

    """创建按钮"""
    button1=  tk.Button(fr,text="退出游戏",command=win.quit)
    button1.pack()
    button2 = tk.Button(fr, text="玩家先手开始游戏",command=game_start)
    button2.pack()
    button3 = tk.Button(fr, text="玩家后手开始游戏",command=game_start1)
    button3.pack()

    def drawout():
        """绘制棋盘线条"""
        for i in range(16):
            s.create_line(70 + per_size * i, 70, 70 + per_size * i, 70+per_size*15, fill="black", width=2)
            s.create_line(70, 70 + per_size * i, 70 + per_size * 15, per_size * i + 70, fill="black", width=2)

        """绘制棋盘字母与数字编号"""
        for i in range(16):
            label = tk.Label(s, text = str(i + 1), fg = "black", bg = "saddlebrown",
                            width = 2, anchor = tk.E)
            label.place(x=20, y=per_size * i + 70)

        count = 0
        for i in range(65, 81):
            label = tk.Label(s, text = chr(i), fg = "black", bg = "saddlebrown")
            label.place(x = per_size * count + 70, y = 20)
            count += 1
            showstatus()
    
    drawout()
    win.mainloop()



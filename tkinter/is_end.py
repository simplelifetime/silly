import numpy as np
a = [[0, 0], [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
def dfs(board, s, x, y, count, dire):
    if (dire == 0):
        all = 0
        for i in range(1,9):
            all += dfs(board, s, x + a[i][0], y + a[i][1], 1, i)
        return all
    else:
        if ((x < 0) or (x > 13) or (y < 0) or (y > 13)):
            return 0
        elif (board[x][y] != s):
            return 0
        elif (count == 4):
            return s
        else:
            return dfs(board, s, x + a[dire][0], y + a[dire][1], count + 1, dire)
        

def check(board):
    for i in range(15):
        for j in range(15):
            if (board[i][j] != 0):
                sum = dfs(board, board[i][j], i, j, 0, 0)
                if (sum > 0):
                    return 1
                elif (sum < 0):
                    return - 1
                else:
                    pass
                

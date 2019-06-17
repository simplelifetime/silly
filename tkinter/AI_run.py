import numpy as np
maxn = 100000
dire = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0],
        [-1, -1], [0, -1], [1, -1]]  # 获取一条线的单一棋形
chess_shape = [['11111', 100000],  # 长连棋形，最高分    #所有的棋形列举（为了保证对称性从而减少某些方向取值的重运算，故对数据进行了对称处理，牺牲了少量空间减少了将近一半的运算时间）
               ['011110', 10000],  # 活四棋形，次高分
               ['011112', 5000],
               ['211110', 5000],  # 冲四棋形，3
               ['0101110', 5000],
               ['0111010', 5000],
               ['0110110', 5000],
               ['01110', 2000],  # 活三棋形，4
               ['010110', 2000],
               ['011010', 2000],
               ['211100', 1000],
               ['001112', 1000],  # 眠三棋形，5
               ['010112', 1000],
               ['211010', 1000],
               ['011012', 1000],
               ['210110', 1000],
               ['11001', 800],
               ['10011', 800],
               ['10101', 800],
               ['2011102', 600],
               ['00110', 200],  # 活2棋形，6
               ['01100', 200],
               ['01010', 200],
               ['010010', 200],
               ['000112', 200],  # 眠2棋形 ，7
               ['211000', 200],
               ['210100', 200],
               ['001012', 100],
               ['210010', 100],
               ['010012', 100],
               ['10001', 100],
               ['2010102', 100],
               ['2011002', 100],
               ['2001102', 100],
               ['211112', 50],  # 死4棋形 ，9
               ['21112', 0],  # 死3棋形 ，8
               ['2112', -20],  # 死2棋形 ，9
               ['212', -50]
               ]  # 死1棋形 ，10
# 一共有38个棋形分布

status = np.zeros([15, 15])  # status保留每个点棋局的权值


def locate_loss(x, y):
    return -5*(abs(x-7)+abs(y-7))  # 保存位置失分，保证在棋局初始时AI行棋主要集中在中路


def get_score(a, x, y):  # 获得某一点的分数
    global dire
    sum = 0
    for i in range(4):
        s = ""
        for j in range(9):
            xnow = x + (j-4)*dire[i][0]
            ynow = y + (j-4)*dire[i][1]
            if (xnow < 0 or xnow > 14 or ynow < 0 or ynow > 14):
                pass
            else:
                if(a[xnow][ynow] == -1):
                    s += '1'
                elif(a[xnow][ynow] == 0):
                    s += '0'
                else:
                    s += '2'  # 获得对应方向字符串
        for i in range(38):  # 判断字符串
            t = s.find(chess_shape[i][0])
            if (t != -1):
                sum += chess_shape[i][1]
                break
    return sum


def total_score(board, anti_board, x, y):
    global dire
    sum = 0
    board[x][y] = -1
    sum += get_score(board, x, y)  # 进攻得分
    board[x][y] = 0
    anti_board[x][y] = -1
    sum += get_score(anti_board, x, y)  # 防守得分
    anti_board[x][y] = 0
    sum += locate_loss(x, y)
    return sum  # 得到该位置总得分


def AI(board, anti_board):
    for i in range(15):
        for j in range(15):
            if (board[i][j] != 0):
                status[i][j] = -1000000  # 用一个足够小的数保留原棋盘不可下位置，保证AI不会将该点作为最忧点
            else:
                status[i][j] = total_score(board, anti_board, i, j)
    index = np.unravel_index(status.argmax(), status.shape)
    (x, y) = index
    np.set_printoptions(suppress=True)  # 获取status中最大值索引
    return index


def boundary(board):  # 用以获得搜索边界，从而大幅度降低搜索复杂度
    s = np.nonzero(board)
    xmin = s[0][0]
    xmax = s[0][-1]
    ymin = s[1][0]
    ymax = s[1][-1]
    # (xmin,xmax,ymin,ymax)
    return [max(xmin - 3, 0), min(xmax + 3, 14), max(ymin - 3, 0), min(ymax + 3, 14)]

# def total_score(board, anti_board, x, y):
#     global dire
#     sum = 0
#     board[x][y] = -1
#     sum += get_score(board, x, y)    #进攻得分
#     board[x][y] = 0
#     anti_board[x][y] = -1
#     sum += get_score(anti_board, x, y)     #防守得分
#     anti_board[x][y] = 0
#     sum += locate_loss(x, y)
#     return sum  #得到该位置总得分


def get_min_node(board, anti_board):
    status = 0  # 储存局部棋局得分（可删除）
    max = -5000000  # 设一个足够大的数来储存最小值
    t = boundary(board)
    for i in range(t[0], t[1]+1):
        for j in range(t[2], t[3]+1):
            if (board[i][j] != 0):  # 若此处无法行棋则直接抛弃
                pass
            else:
                status = 0
                anti_board[i][j] = -1
                status += get_score(anti_board, i, j)  # 防守得分
                anti_board[i][j] = 0  # 进行每个位置行棋的可能得分计算
                # print([status, i, j])
                if (status > max):
                    max = status
    return max  # 得到相应位置最小值


def get_max_node(board, anti_board):
    maxx = -1
    maxy = -1  # 储存坐标
    max = -5000000
    status = 0
    t = boundary(board)
    for i in range(t[0], t[1]+1):
        for j in range(t[2], t[3]+1):
            if (board[i][j] != 0):
                pass
            else:
                board[i][j] = 1
                status += -get_min_node(board, anti_board)    #减去对方得分的最大值（即是负得分的最小值）
                # board[i][j] = -1
                # status += get_score(board, i, j)  # 加上己方得分
                # board[i][j] = 0
                # status += locate_loss(i, j)    #加上位置损失，得到该位置的总得分
                # # print([i, j, status])
                status += total_score(board, anti_board, i, j)
                print([status,i,j])
                if (status > max):
                    max = status
                    maxx = i
                    maxy = j
                status=0
    return [max, maxx, maxy]  # 储存值与坐标，值用于可能的多层搜索，坐标用于第一层AI参数的还原


def AI2(board, anti_board):
    s = get_max_node(board, anti_board)
    return [s[1], s[2]]

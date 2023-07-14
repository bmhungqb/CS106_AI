import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState): #Hàm nhận vào tham số gameState là trạng thái hiện tại của game bao gồm bản đồ và vị trí người chơi
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #Trả về vị trí ban đầu của các Box trong map
    beginPlayer = PosOfPlayer(gameState) #Trả về vị trí ban đầu của Player 
    startState = (beginPlayer, beginBox) #startState: trạng thái bắt đầu bao gồm beginBox và beginPlayer
    frontier = collections.deque([[startState]]) #Hàng đợi double-ending queue(các phần tử có thể xem và xóa từ hai đầu).Trạng thái bắt đầu có 1 phần tử startState
    exploredSet = set() #Một tập rỗng exploredSet được tạo để lưu trữ các trạng thái đã được thăm
    actions = [[0]] #Danh sách các hành động của Player
    temp = [] #Lưu kết quả
    while frontier: #Kiểm tra frontier rỗng hay không ? Nếu còn state thì vào trong vòng lặp để xử lý
        node = frontier.pop() #Lấy phần tử bên trái nhất là trạng thái bao gồm (Player và Box) của deque frontier
        node_action = actions.pop() #Lấy ra hàng động nhân vật trong hàng đợi
        if isEndState(node[-1][-1]):  #Kiểm tra đây có phải là trạng thái đích không ?
            temp += node_action[1:]  #Lưu lại kết quả
            break # Kết thúc vòng lặp DFS
        if node[-1] not in exploredSet: #Kiểm tra trạng thái hiện tại đã có trong tập exploredSet chưa ?(đã được thăm chưa ?) 
            exploredSet.add(node[-1]) #Thêm trạng thái vào tập exploredSet để không xét lại
            for action in legalActions(node[-1][0], node[-1][1]): #Xét tất cả các hành động hợp pháp có thể thực hiện từ hành động hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Vị trí mới của Player và Boxes khi thực hiện hành động hợp pháp
                if isFailed(newPosBox): #Kiểm tra tính hợp lệ khi thực hiện hành động hợp pháp
                    continue # Nếu thất bại => bỏ qua không xét
                frontier.append(node + [(newPosPlayer, newPosBox)]) #Thêm trạng thái vào cuối queue frontier
                actions.append(node_action + [action[-1]]) #Thêm hành động mới vào cuối hàng đợi actions dựa vào vị trí trước đó
    return temp # Trả về kết quả đúng của DFS và thoát khỏi hàm

def breadthFirstSearch(gameState):#Hàm nhận vào tham số gameState là trạng thái hiện tại của game bao gồm bản đồ và vị trí người chơi
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #Trả về vị trí ban đầu của các Box trong map
    beginPlayer = PosOfPlayer(gameState) #Trả về vị trí ban đầu của Player 
    #startState: trạng thái bắt đầu bao gồm beginBox và beginPlayer
    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    #Hàng đợi double-ending queue(các phần tử có thể xem và xóa từ hai đầu).Trạng thái bắt đầu có 1 phần tử startState
    frontier = collections.deque([[startState]]) # store states
    #Danh sách các hành động của Player
    actions = collections.deque([[0]]) # store actions
    exploredSet = set() #Một tập rỗng exploredSet được tạo để lưu trữ các trạng thái đã được thăm
    temp = [] #Lưu kết quả
    ### Implement breadthFirstSearch here
    while frontier: #Kiểm tra frontier rỗng hay không ? Nếu còn state thì vào trong vòng lặp để xử lý
        node = frontier.pop() #Lấy phần tử bên trái nhất là trạng thái bao gồm (Player và Box) của deque frontier
        node_action = actions.pop() #Lấy ra hàng động nhân vật trong hàng đợi
        if isEndState(node[-1][-1]): #Kiểm tra đây có phải là trạng thái đích không ?
            temp += node_action[1:] #Lưu lại kết quả
            break # Kết thúc vòng lặp BFS
        if node[-1] not in exploredSet: #Kiểm tra trạng thái hiện tại đã có trong tập exploredSet chưa ?(đã được thăm chưa ?) 
            exploredSet.add(node[-1]) #Thêm trạng thái vào tập exploredSet để không xét lại
            for action in legalActions(node[-1][0], node[-1][1]): #Xét tất cả các hành động hợp pháp có thể thực hiện từ hành động hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Vị trí mới của Player và Boxes khi thực hiện hành động hợp pháp
                if isFailed(newPosBox): #Kiểm tra tính hợp lệ khi thực hiện hành động hợp pháp
                    continue # Nếu thất bại => bỏ qua không xét
                frontier.insert(0,node + [(newPosPlayer, newPosBox)]) #Thêm trạng thái vào đầu queue frontier
                actions.insert(0,node_action + [action[-1]]) #Thêm hành động mới vào đầu hàng đợi actions dựa vào vị trí trước đó
    return temp # Trả về kết quả đúng của BFS và thoát khỏi hàm
    
def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState): #Hàm nhận vào tham số gameState là trạng thái hiện tại của game bao gồm bản đồ và vị trí người chơi
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) #Trả về vị trí ban đầu của các Box trong map
    beginPlayer = PosOfPlayer(gameState) #Trả về vị trí ban đầu của Player 
    startState = (beginPlayer, beginBox) #startState: trạng thái bắt đầu bao gồm beginBox và beginPlayer
    frontier = PriorityQueue() #Hàng đợi Priority queue frontier(Sắp xếp các phần tử theo độ ưu tiên)
    frontier.push([startState], 0) #Trạng thái bắt đầu có 1 phần tử startState được lưu trữ với key = 0
    exploredSet = set() #Một tập rỗng exploredSet được tạo để lưu trữ các trạng thái đã được thăm
    actions = PriorityQueue() #Hàng đợi Priority queue actions(Sắp xếp các phần tử theo độ ưu tiên)
    actions.push([0], 0) #Trạng thái băt đầu có 1 phần tử được lưu trữ với key = 0
    temp = [] #Lưu kết quả
    ### Implement uniform cost search here
    while frontier: #Kiểm tra frontier rỗng hay không ? Nếu còn state thì vào trong vòng lặp để xử lý
        node = frontier.pop() #Lấy phần tử bên trái nhất là trạng thái bao gồm (Player và Box) của priority queue frontier
        node_action = actions.pop() #Lấy ra hàng động nhân vật trong hàng đợi Priority queue
        if isEndState(node[-1][-1]): #Kiểm tra đây có phải là trạng thái đích không ?
            temp += node_action[1:] #Lưu lại kết quả
            break # Kết thúc vòng lặp UCS
        if node[-1] not in exploredSet: #Kiểm tra trạng thái hiện tại đã có trong tập exploredSet chưa ?(đã được thăm chưa ?)
            exploredSet.add(node[-1]) #Thêm trạng thái vào tập exploredSet để không xét lại
            Cost = cost(node_action[1:]) #Xét tất cả các hành động hợp pháp có thể thực hiện từ hành động hiện tại
            for action in legalActions(node[-1][0], node[-1][1]): #Xét tất cả các hành động hợp pháp có thể thực hiện từ hành động hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Vị trí mới của Player và Boxes khi thực hiện hành động hợp pháp
                if isFailed(newPosBox): #Kiểm tra tính hợp lệ khi thực hiện hành động hợp pháp
                    continue # Nếu thất bại => bỏ qua không xét
                frontier.push(node + [(newPosPlayer, newPosBox)],Cost) #Thêm trạng thái mới vào frontier
                actions.push(node_action + [action[-1]],Cost) #Thêm hành động mới vào hàng đợi actions dựa vào vị trí trước đó
    return temp # Trả về kết quả đúng của UCS và thoát khỏi hàm

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return [result,method,time_end-time_start]

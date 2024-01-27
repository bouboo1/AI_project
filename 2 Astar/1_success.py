from queue import PriorityQueue

# 定义目标顺序
target_state = [1,3,5,7,0,2,6,8,4]

# 定义魔法魔方的状态类
class State:
    def __init__(self, puzzle, parent=None, move=None):
        self.puzzle = puzzle
        self.parent = parent
        self.move = move
        if self.puzzle:
            self.blank = self.puzzle.index(0)

    # 获取可行移动的邻居状态列表
    def get_neighbors(self):
        neighbors = []
        moves = {
            'up': -3,
            'down': 3,
            'left': -1,
            'right': 1
        }
        for move, direction in moves.items():
            if (self.blank % 3 == 0 and move == 'left') or \
                (self.blank % 3 == 2 and move == 'right') or \
                (self.blank < 3 and move == 'up') or \
                (self.blank > 5 and move == 'down'):
                continue
            new_puzzle = list(self.puzzle)
            new_puzzle[self.blank], new_puzzle[self.blank + direction] = new_puzzle[self.blank + direction], new_puzzle[self.blank]
            neighbors.append(State(new_puzzle, self, move))
        return neighbors

    # 计算当前状态与目标状态错误的位置
    def misplaced_tiles(self):
        count = 0
        for i in range(9):
            if self.puzzle[i] != target_state[i]:
                count += 1
        return count

    # 定义状态的优先级，用于A*算法中节点的排序
    def __lt__(self, other):
        return (self.misplaced_tiles() + self.get_cost()) < (other.misplaced_tiles() + other.get_cost())
    
    def get_cost(self):
        cost = 0
        parent = self.parent
        while parent:
            cost += 1
            parent = parent.parent
        return cost

    # 判断当前状态是否为目标状态
    def is_goal(self):
        return self.puzzle == target_state
    def steps_to_goal(self):
        return self.get_cost()

    # 输出从初始状态到目标状态的移动序列
    def print_solution(self):
        if self.parent:
            self.parent.print_solution()
        if self.move:
            print(f"Move {self.move}")
        print(self.puzzle)
        


# 使用A*算法解决问题
def solve(puzzle):
    start_state = State(puzzle)
    queue = PriorityQueue()
    queue.put(start_state)

    while not queue.empty():
        current_state = queue.get()

        if current_state.is_goal():
            #current_state.print_solution()
            print(current_state.steps_to_goal())
            break

        for neighbor in current_state.get_neighbors():
            queue.put(neighbor)

# 测试
with open('1_test.txt', 'r') as f:
    tests = f.readlines()
for test in tests:
    puzzle_ = test.strip()#直接使用有换行符，这里做一步处理
    puzzle = list(puzzle_)
    puzzle = [int(x) for x in puzzle]
    solve(puzzle)

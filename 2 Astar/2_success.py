import heapq
from queue import PriorityQueue


def shortest_k_paths(n, m, k, edges):
    # 构建图的邻接表表示
    graph = [[] for _ in range(n)]
    graph_ = [[] for _ in range(n)]
    for x, y, d in edges:
        graph[x-1].append((y-1, d))
        graph_[x-1].append((y-1, d))
        graph_[y-1].append((x-1, d))
   

    # 初始化优先队列和闭集
    start_node = (0, 0, 0, [0])  # (f_score, g_score, node, path)
    open_set = [start_node]
    k_shortest_paths = []
    dis = heuristic(n-1,graph_)

    while open_set and len(k_shortest_paths) < k:
        # 从优先队列中弹出具有最低 f_score 的节点
        _, g_score, current_node, path = heapq.heappop(open_set)
       
        # 检查是否到达目标节点
        if current_node == n - 1:
            k_shortest_paths.append(g_score)
            
        else:
            # 检查当前节点的所有邻居
            for neighbor, distance in graph[current_node]:
                # 如果邻居节点已经在，则跳过
                if neighbor in path:
                    continue

                # 计算邻居节点的 tentative_g_score
                tentative_g_score = g_score + distance

                # 将邻居节点加入优先队列，并记录路径
                f_score = tentative_g_score + dis[neighbor]
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, [neighbor]))
           
    while(len(k_shortest_paths) < k):
        k_shortest_paths.extend([-1] * (k - len(k_shortest_paths)))        
    return k_shortest_paths


def heuristic(goal, graph):
    # 使用 Dijkstra 算法找到从节点到目标节点的最短路径
    dis = {}  # 节点到图中每个节点的距离
    visited = set()  # 记录已访问过的节点
    pq = PriorityQueue()  # Dijkstra 算法的优先队列

    # 初始化距离数组，除了起始节点外都设置为无穷大
    for i in range(len(graph)):
        dis[i] = float('inf')
    dis[goal] = 0

    # 将目标节点加入优先队列
    pq.put((0, goal))

    while not pq.empty():

        # 从优先队列中获取距离最小的节点
        
        dist, curr_node = pq.get()
       
        # 如果节点已经访问过，则跳过
        if curr_node in visited:
            continue

        # 将当前节点标记为已访问
        visited.add(curr_node)

        # 更新邻居节点的距离
        for neighbor, cost in graph[curr_node]:
            new_dist = dis[curr_node] + cost
            if new_dist < dis[neighbor]:
                dis[neighbor] = new_dist
                pq.put((new_dist, neighbor))
           

    # 返回节点到目标节点的最短距离
    return dis



#测试
with open('2_test.txt', 'r') as file:
    test_cases = file.read().split('\n\n')

for i, test_case in enumerate(test_cases):
    print(test_case)
    lines = test_case.split('\n')
    N, M, K = map(int, lines[0].split())
    edges = []
    for line in lines[1:]:
        X, Y, D = map(int, line.split())
        edges.append((X, Y, D))
    paths = shortest_k_paths(N, M, K, edges)
    for k in paths:
        print(k)
    print()

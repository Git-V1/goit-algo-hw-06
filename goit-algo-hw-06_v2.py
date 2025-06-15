import timeit
import chardet
import heapq
import matplotlib.pyplot as plt
from collections import deque

# Функція для автоматичного визначення кодування
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
    return chardet.detect(raw_data)["encoding"]

# Читання тексту з правильним кодуванням
def read_text(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, encoding=encoding) as f:
        return f.read(), encoding

# Побудова графа метро Києва
def create_kyiv_metro_graph():
    G = {}

    connections = [
        ("Академмістечко", "Житомирська", 3),
        ("Житомирська", "Святошин", 2),
        ("Святошин", "Нивки", 2),
        ("Нивки", "Берестейська", 2),
        ("Берестейська", "Шулявська", 2),
        ("Шулявська", "Політехнічний інститут", 2),
        ("Політехнічний інститут", "Вокзальна", 2),
        ("Вокзальна", "Університет", 2),
        ("Університет", "Театральна", 1),
        ("Театральна", "Хрещатик", 1),
        ("Хрещатик", "Арсенальна", 2),
        ("Арсенальна", "Дніпро", 2),
        ("Дніпро", "Гідропарк", 3),
        ("Гідропарк", "Лівобережна", 2)
    ]

    for start, end, weight in connections:
        if start not in G:
            G[start] = []
        if end not in G:
            G[end] = []
        G[start].append((end, weight))
        G[end].append((start, weight))

    return G

# Візуалізація графа
def visualize_graph(graph):
    plt.figure(figsize=(10, 6))
    pos = {station: (i, 0) for i, station in enumerate(graph.keys())}
    
    for station, edges in graph.items():
        for neighbor, weight in edges:
            plt.plot(
                [pos[station][0], pos[neighbor][0]], 
                [pos[station][1], pos[neighbor][1]], 
                "ro-"
            )
            plt.text(
                (pos[station][0] + pos[neighbor][0]) / 2,
                (pos[station][1] + pos[neighbor][1]) / 2 + 0.1,
                str(weight),
                fontsize=10,
                color="blue"
            )
    
    for station, coord in pos.items():
        plt.text(coord[0], coord[1] - 0.1, station, fontsize=10, ha="center", color="black")
    
    plt.title("Київський метрополітен — Червона лінія")
    plt.show()

# BFS пошук
def bfs_path(graph, start, goal):
    visited = set()
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph[node]:
                new_path = path + [neighbor]
                queue.append(new_path)

# DFS пошук
def dfs_path(graph, start, goal):
    visited = set()
    stack = [[start]]
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph[node]:
                new_path = path + [neighbor]
                stack.append(new_path)

# Реалізація алгоритму Дейкстри 
def dijkstra(graph, start):
    """Алгоритм Дейкстри для знаходження найкоротших шляхів"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    priority_queue = [(0, start)]  # (відстань, вершина)
    shortest_paths = {start: [start]}  # Збереження маршрутів

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]

    return distances, shortest_paths

# Основна логіка виконання
if __name__ == "__main__":
    kyiv_metro_graph = create_kyiv_metro_graph()
    visualize_graph(kyiv_metro_graph)

    print("🔹 Кількість станцій:", len(kyiv_metro_graph))
    print("🔹 Кількість сполучень:", sum(len(edges) for edges in kyiv_metro_graph.values()) // 2)

    print("\n🔍 BFS шлях від Академмістечко до Лівобережна:")
    print(bfs_path(kyiv_metro_graph, "Академмістечко", "Лівобережна"))

    print("\n🔍 DFS шлях від Академмістечко до Лівобережна:")
    print(dfs_path(kyiv_metro_graph, "Академмістечко", "Лівобережна"))

    print("\n📍 Найкоротші шляхи (Дейкстра):")
    distances, paths = dijkstra(kyiv_metro_graph, "Академмістечко")
    for station, distance in distances.items():
        print(f"Академмістечко → {station}: {distance} хв, шлях: {' → '.join(paths[station])}")

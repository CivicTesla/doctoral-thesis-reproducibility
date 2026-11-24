import math

def simulate_laser_rangefinder(gridMap, agent_position, max_range, num_beams):
    angles = [2 * math.pi * i / num_beams for i in range(num_beams)]
    distances = []

    for angle in angles:
        x, y = agent_position
        dx = math.cos(angle)
        dy = math.sin(angle)

        for step in range(1, max_range + 1):
            nx, ny = int(x + step * dx), int(y + step * dy)

            if 0 <= nx < len(gridMap) and 0 <= ny < len(gridMap[0]):
                if gridMap[nx][ny] == 1:  # 遇到障碍物
                    distances.append(step)
                    break
            else:
                break  # 超出地图范围

    return distances

# # 举例
# gridMap = [
#     [0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0]
# ]

# agent_position = (1, 2)
# max_range = 3
# num_beams = 8

# laser_distances = simulate_laser_rangefinder(gridMap, agent_position, max_range, num_beams)
# print("Laser Distances:", laser_distances)

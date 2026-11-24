import math
import numpy as np

def move_along_angle(x, y, angle, step_length):
    # 将角度转换为弧度
    angle_rad = math.radians(angle+90)
    # print('angle_rad',angle_rad)
    
    # 计算新的坐标
    new_x = round(x + step_length * math.cos(angle_rad))
    new_y = round(y + step_length * math.sin(angle_rad))
    
    return np.array([new_x, new_y])

# # TestCode##########################33
# current_position = (4, 4)
# new_position = move_along_angle(*current_position, angle=45, step_length=1)
# # new_position = tuple(round(coord) for coord in move_along_angle(*current_position, angle=225, step_length=2))


# print(new_position)

#TODO: angle not correct
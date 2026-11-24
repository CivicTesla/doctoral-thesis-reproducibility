def divide_grid_map(gridMap):
    rows = len(gridMap)
    cols = len(gridMap[0])

    # 计算每个区域的大小
    region_rows = rows // 2
    region_cols = cols // 4

    region_id = 1
    divided_map = [[0] * cols for _ in range(rows)]

    for i in range(0, rows, region_rows):
        for j in range(0, cols, region_cols):
            # 给当前区域赋予编号
            for x in range(i, i + region_rows):
                for y in range(j, j + region_cols):
                    divided_map[x][y] = region_id
            region_id += 1

    return divided_map




# #Test Code
# from Image2gridMap import create_grid_map
# # gridMap = create_grid_map('grid_map.png')
# gridMap = create_grid_map('org_map.png')


# result = divide_grid_map(gridMap)

# # 打印分区后的地图
# for row in result:
#     print(row)
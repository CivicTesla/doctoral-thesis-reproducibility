from cv2 import cv2
def create_grid_map(image_path):
	# 读取图像
	image = cv2.imread(image_path)

	# 将图像转换为灰度
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 设置阈值，将图像二值化，黑色区域标记为1，白色区域标记为0
	# _, gridMap = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV) # for gridMapValue
	_, gridMap = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY) # for write an image

	# 显示或保存 gridMap
	cv2.imshow('Grid Map', gridMap)
	cv2.imwrite('org_map.png', gridMap * 255)  # 保存时将0/1转换为0/255以显示黑白颜色

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print('gridMap',gridMap)


orgMap = create_grid_map('map.png')
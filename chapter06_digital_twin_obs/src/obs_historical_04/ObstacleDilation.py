# import cv2
# import numpy as np

# # 读取输入图像
# image = cv2.imread('grid_map.png', cv2.IMREAD_GRAYSCALE)

# # 阈值化图像（将图像转为二值图像）
# _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # 定义膨胀核大小（调整核大小以实现更大的膨胀效果）
# kernel_size = 15
# kernel = np.ones((kernel_size, kernel_size), np.uint8)

# # 进行膨胀操作
# dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# # 显示原始图像、二值图像和膨胀后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Binary Image', binary_image)
# cv2.imshow('Dilated Image', dilated_image)

# # 等待按键退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # 读取输入图像
# image = cv2.imread('grid_map.png', cv2.IMREAD_GRAYSCALE)

# # 阈值化图像（将图像转为二值图像）
# _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)  # 注意这里的阈值化反转

# # 定义横向和纵向膨胀核
# kernel_horizontal = np.ones((1, 15), np.uint8)  # 1行15列，横向膨胀
# kernel_vertical = np.ones((15, 1), np.uint8)  # 15行1列，纵向膨胀

# # 分别进行横向和纵向膨胀操作
# dilated_image_horizontal = cv2.dilate(binary_image, kernel_horizontal, iterations=1)
# dilated_image_vertical = cv2.dilate(binary_image, kernel_vertical, iterations=1)

# # 将横向和纵向膨胀的结果合并
# dilated_image = cv2.bitwise_or(dilated_image_horizontal, dilated_image_vertical)

# # 显示原始图像、二值图像和膨胀后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Binary Image', binary_image)
# cv2.imshow('Dilated Image', dilated_image)

# # 等待按键退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 读取输入图像
image = cv2.imread('org_map.png', cv2.IMREAD_GRAYSCALE)

# 阈值化图像（将图像转为二值图像）
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)  # 注意这里的阈值化反转

# 定义横向和纵向膨胀核
kernel_horizontal = np.ones((1, 15), np.uint8)  # 1行15列，横向膨胀
kernel_vertical = np.ones((15, 1), np.uint8)  # 15行1列，纵向膨胀

# 分别进行横向和纵向膨胀操作
dilated_image_horizontal = cv2.dilate(binary_image, kernel_horizontal, iterations=1)
dilated_image_vertical = cv2.dilate(binary_image, kernel_vertical, iterations=1)

# 将横向和纵向膨胀的结果合并
dilated_image = cv2.bitwise_or(dilated_image_horizontal, dilated_image_vertical)

# 取反操作
binary_image_inv = cv2.bitwise_not(binary_image)
dilated_image_inv = cv2.bitwise_not(dilated_image)

# # 显示原始图像、二值图像和膨胀后的图像（黑白颜色相反）
# cv2.imshow('Original Image', image)
# cv2.imshow('Binary Image (Inverted)', binary_image_inv)
# cv2.imshow('Dilated Image (Inverted)', dilated_image_inv)

# # 等待按键退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存膨胀后的图像
cv2.imwrite('dilatedImg4orgMap.png', dilated_image_inv)
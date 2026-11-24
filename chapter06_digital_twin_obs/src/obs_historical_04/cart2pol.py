import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

# 示例
x = 3.0
y = 4.0

rho, phi = cart2pol(x, y)

print("Cartesian Coordinates:")
print("x =", x)
print("y =", y)

print("\nPolar Coordinates:")
print("rho =", rho)
print("phi =", phi)

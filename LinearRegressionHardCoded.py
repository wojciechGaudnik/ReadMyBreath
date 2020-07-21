import pandas as pd
import matplotlib.pyplot as plt

def get_y(m, b, x):
	return m * x + b


def calculate_error(m, b, p):
	(x_point, y_point) = p
	y_value = get_y(m, b, x_point)
	return abs(y_value - y_point)


datapoints = [(1, 2), (2, 0), (3, 4), (4, 4), (5, 3)]

def calculate_all_error(m, b, points):
	error = 0
	for point in points:
		error += calculate_error(m, b, point)
	return error


possible_ms = [n * 0.1 for n in range(-100, 101, 1)]
possible_bs = [n * 0.1 for n in range(-200, 201, 1)]

smallest_error = float("inf")
best_m = 0.0
best_b = 0.0
for m in possible_ms:
	for b in possible_bs:
		if calculate_all_error(m, b, datapoints) < smallest_error:
			smallest_error = calculate_all_error(m, b, datapoints)
			best_m = m
			best_b = b

print(best_b)
print(best_m)
print(smallest_error)
print("m = 0.3")
print("b = 1.7")

plt.scatter([p[0] for p in datapoints], [p[1] for p in datapoints])
# plt.plot([x for x in range (6)], [get_y(best_m, best_b, x) for x in range(6)])
best_m = 0.3
best_b = 1.7
plt.plot([x for x in range (6)], [get_y(best_m, best_b, x) for x in range(6)])
plt.show()

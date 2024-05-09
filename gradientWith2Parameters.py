import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления среднеквадратичной ошибки между наблюдаемыми данными y и линейной моделью a*z + b.
def calculate_error(y, a, b):
    predicted_values = np.array([a * z + b for z in range(num_experiments)])
    return np.dot((y - predicted_values).T, (y - predicted_values))

# Функция для вычисления производной ошибки по параметру 'a'.
def calculate_derivative_a(y, a, b):
    predicted_values = np.array([a * z + b for z in range(num_experiments)])
    return -2 * np.dot((y - predicted_values).T, range(num_experiments))

# Функция для вычисления производной ошибки по параметру 'b'.
def calculate_derivative_b(y, a, b):
    predicted_values = np.array([a * z + b for z in range(num_experiments)])
    return -2 * (y - predicted_values).sum()

# Количество экспериментов и количество итераций для оптимизации.
num_experiments = 100
num_iterations = 25

# Стандартное отклонение наблюдаемых значений и теоретические значения параметров.
observation_noise = 3
true_a = 0.5
true_b = 2

# Начальные значения параметров для градиентного спуска.
a = 0
b = 0
learning_rate_a = 0.000001
learning_rate_b = 0.0005

# Генерация исходных данных и шумных наблюдений.
true_values = np.array([true_a * z + true_b for z in range(num_experiments)])
observed_values = np.array(true_values + np.random.normal(0, observation_noise, num_experiments))

# Диапазоны параметров 'a' и 'b' для визуализации.
a_range = np.arange(-1, 2, 0.1)
b_range = np.arange(0, 3, 0.1)

# Вычисление ошибки для каждой комбинации параметров 'a' и 'b' для визуализации.
error_surface = np.array([[calculate_error(observed_values, a1, b1) for a1 in a_range] for b1 in b_range])

# Создание 3D-графика поверхности ошибки.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a1, b1 = np.meshgrid(a_range, b_range)
ax.plot_surface(a1, b1, error_surface, color='y', alpha=0.5)

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Error')

# Инициализация начальной точки на поверхности ошибки.
point = ax.scatter(a, b, calculate_error(observed_values, a, b), c='red')
plt.ion()  # Включение интерактивного режима для обновлений в реальном времени.

# Выполнение градиентного спуска для оптимизации параметров 'a' и 'b'.
for iteration in range(num_iterations):
    a = a - learning_rate_a * calculate_derivative_a(observed_values, a, b)
    b = b - learning_rate_b * calculate_derivative_b(observed_values, a, b)
    ax.scatter(a, b, calculate_error(observed_values, a, b), c='red')  # Обновление точки на поверхности ошибки.
    plt.pause(1)  # Пауза для обновления визуализации.
    print(f"Iteration {iteration+1}: a = {a:.4f}, b = {b:.4f}, Error = {calculate_error(observed_values, a, b):.4f}")

plt.ioff()  # Отключение интерактивного режима после окончания оптимизации.
plt.show()  # Отображение окончательного 3D-графика.

# Визуализация исходной функции, шумных данных и оптимизированной линейной модели.
optimized_values = np.array([a * z + b for z in range(num_experiments)])  # Вычисление оптимизированной линейной модели.
plt.scatter(range(num_experiments), observed_values, s=2, c='red')  # Визуализация наблюдаемых данных.
plt.plot(true_values)  # Визуализация исходной функции.
plt.plot(optimized_values, c='red')  # Визуализация оптимизированной линейной модели.
plt.title('Approximation')
plt.xlabel('Experiment Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()  # Отображение окончательного графика аппроксимации.

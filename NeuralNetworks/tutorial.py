import numpy as np


# Активаційна функція - sigmoid
def f(x):
    return 1 / (1 + np.exp(-x))


# Процес прямого поширення використовуючи цикли Python
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        # Формується вхідний масив - перемноження ваг у кожному шарі
        # Якщо перший шар, то вхідний масив дорівнює вектору х
        # Якщо шар не перший, вхід для поточного шару дорівнює виходу попереднього
        if l == 0:
            node_in = x
        else:
            node_in = h

        # Формує вихідний масив для вузлів у шарі l + 1
        h = np.zeros((w[l].shape[0],))

        # проходить по рядкам масиву ваг
        for i in range(w[l].shape[0]):

            # рахує суму всередині активаційної функції
            f_sum = 0

            # проходить по стовпцям масиву ваг
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]

            # додає зміщення
            f_sum += b[l][i]
            # використовує активаційну функцію для розрахунку
            # i-того виходу, у даному випадку h1, h2, h3
            h[i] = f(f_sum)

    return h

# Перше рандомне призначення ваг кожному зв'язку в з першого рівня ШНМ
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

# Перше рандомне призначення ваг кожному зв'язку в з другого рівня ШНМ
w2 = np.array([[0.5, 0.5, 0.5]])

# Значення ваги зміщеня для Ш1
b1 = np.array([0.8, 0.8, 0.8])

# Значення ваги зміщеня для Ш2
b2 = np.array([0.2])

w = [w1, w2]
b = [b1, b2]

# рандомізований вхідний вектор x
x = [1.5, 2.0, 3.0]


# Процес прямого поширення використовуючи матричний добуток numpy
def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h


x_old = 0  # Немає різниці, яке значення, головне abs(x_new - x_old) > точність
x_new = 6  # Алгоритм починається з x=6
gamma = 0.01  # Розмір кроку
precision = 0.00001  # Точність


# Похідна функції f(x) = x**4 – 3*(x**3) + 2
def df(x):
    y = 4 * x**3 - 9 * x**2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * df(x_old)

# print("Локальний мінімум знаходиться на %f" % x_new)


# Імплементація нейронної мережі мовою Пайтон
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Записування даних -> цифори написані від руки (MNIST)
digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

# Розділення даних на тренувальний і тестувальний сети
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# Конвертація числа у векторне представлення
# 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

# Структура НМ
# Вхідний рівень - 64 вузли (ксть пікселів)
# Вихідний рівень - 10 вузлів (ксть класів(цифр))
nn_structure = [64, 30, 10]


# Похідна сигмоїдної функції
def f_deriv(x):
    return f(x) * (1 - f(x))


# Рандомна ініціалізація ваг для кожного шару нм
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = np.random.random_sample((nn_structure[l],))
    return W, b


# Рандомна ініціалізація значень для дельта W та дельта b
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


# Процес прямого поширення
def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # якщо перший шар, то вагами є x, в іншому випадку,
        # це є виходом з останнього шару
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1])  # h^(l) = f(z^(l))
    return h, z


# Знаходження вихідного шару δ(nl) та значення δ(l) у прихованих шарах для запуску зворотного поширення
def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


# Тренування моделі
def train_nn(nn_structure, X, y, iter_num=10000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Початок градієнтного спуску для {} ітерацій'.format(iter_num))
    while cnt < iter_num:
        if cnt % 500 == 0:
            print('Ітерація {} від {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # запускає процес прямого поширення та повертає отримані значення h та z, щоб використати у
            # етапі з градієнтним спуском
            h, z = feed_forward(X[i, :], W, b)
            # цикл від nl-1 до 1 зворотного поширення похибок
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # запускає градієнтний спуск для ваг у кожному шарі
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # завершує розрахунки загальної оцінки
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)
plt.plot(avg_cost_func)
plt.ylabel('Середня J')
plt.xlabel('Кількість ітерацій')
plt.show()


# Передбачаємо точності моделі
def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y


# Оцінка точності результату
from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 3)
accuracy = accuracy_score(y_test, y_pred) * 100
print(accuracy)
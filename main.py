import numpy
import heapq
import matplotlib.pyplot as plt


def generate_map(height: int = None, width: int = None, wall_chance: float = 0.3) -> numpy.ndarray:
    """
    Функция создаёт поле, по которому перемещается робот.
    Важно: начальная позиция робота всегда равна (0, 0), а координаты его цели (height - 1, width - 1).

    :param height: int - "Высота" поля перемещения, если None - выбирается случайным образом.
    :param width: int - "Ширина" поля перемещения, если None - выбирается случайным образом.
    :param wall_chance: float - Вероятность появления стены для каждой ячейки поля перемещения, если None - 0.3.
    :return: numpy.ndarray - Матрица, представляющая собой поле перемещения. 0 - свободная ячейка, 1 - стена.
    """

    # Определение высоты и ширины, в случае случайной генерации
    if height is None:
        height = numpy.random.randint(2, 100)
    if width is None:
        width = numpy.random.randint(2, 100)

    # Валидация параметров функции
    assert height > 1
    assert width > 1
    assert 1 >= wall_chance >= 0

    return numpy.random.choice(
        [0, 1],
        size=(height, width),
        p=[1 - wall_chance, wall_chance],
    )


def heuristic(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    """
    Функция для оценки расстояния между точками p1 и p2.

    :param p1: tuple[int, int] - Координаты первой точки.
    :param p2: tuple[int, int] - Координаты второй точки.
    :return: int - Манхэтэнское расстояния aka эвристика для точек p1 и p2.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def find_path(
        robo_map: numpy.ndarray,
        start: tuple[int, int] = None,
        target: tuple[int, int] = None
) -> list[tuple[int, int]] | None:
    """
    Функция вычисляет путь, применяя алгоритм A*.
    Если путь не найден, возвращается None.

    :param robo_map: numpy.ndarray - Матрица, представляющая собой поле перемещения. 0 - свободная ячейка, 1 - стена.
    :param start: tuple[int, int] - Координаты первого участка пути, если None - (0, 0).
    :param target: tuple[int, int] - Координаты конечного участка пути, если None - последний элемент матрицы robo_map.
    :return: list[tuple[int, int]] - Список координат, принадлежащих найденному пути.
    """
    # Определение координат, в случае случайной генерации
    if start is None:
        start = (0, 0)
    if target is None:
        target = (robo_map.shape[0] - 1, robo_map.shape[1] - 1)

    # Валидация параметров функции
    assert robo_map.shape[0] > 1
    assert robo_map.shape[1] > 1
    assert 0 <= start[0] <= robo_map.shape[0] - 1
    assert 0 <= start[1] <= robo_map.shape[1] - 1
    assert 0 <= target[0] <= robo_map.shape[0] - 1
    assert 0 <= target[1] <= robo_map.shape[1] - 1

    # Определяем необходимые переменные для работы алгоритма
    to_be_processed = []  # Куча с узлами, ожидающими обработки
    came_from = {}  # Путь до текущей расссматриваемой точки

    # Словарь, где ключ - координаты точки, а значение - длина пути до точки от начала пути
    g_len = {start: 0}

    # Слоаврь, где ключ - координаты точки, а значение - длина пути от начала координат, до конца пути ЧЕРЕЗ эту точку
    f_len = {start: heuristic(start, target)}

    # Начинаем обработку, предварительно добавив начало пути, с минимальным приоритетом очереди
    heapq.heappush(to_be_processed, (0, start))
    while to_be_processed:
        current = heapq.heappop(to_be_processed)[1]  # Извлекаем точку с наименьшим приоритетом

        # Если текущий элемент - цель, тогда просто восстанавливаем маршрут и возвращаем искомый путь
        if current == target:
            found_path = []
            while current in came_from:
                found_path.append(current)
                current = came_from[current]
            found_path.append(start)
            return found_path[::-1]

        # Обрабатываем соседние координаты
        for cell in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + cell[0], current[1] + cell[1])

            # Проверяем не вышли ли мы за границы карты
            if 0 <= neighbor[0] < robo_map.shape[0] and 0 <= neighbor[1] < robo_map.shape[1]:

                # Проверяем, является ли ячейка пустой
                if robo_map[neighbor[0]][neighbor[1]] == 0:

                    tmp_g_len = g_len[current] + 1  # Рассчитываем временную стоимость пути до текущей точки

                    # Самое интересное: Если соседняя точка не была посещена
                    # или временный путь текущего элемента меньше чем у соседней точки:
                    if neighbor not in g_len or tmp_g_len < g_len[neighbor]:
                        # Переписываем координаты точки (или же прост записываем новые, если точки не было)
                        came_from[neighbor] = current

                        # Обновляем словари длин путей данными текущей точки
                        g_len[neighbor] = tmp_g_len
                        f_len[neighbor] = heuristic(neighbor, target) + tmp_g_len

                        # Добавлеям соседа в кучу, что бы потом ео обработать, согласно его приоритету в ней
                        heapq.heappush(to_be_processed, (f_len[neighbor], neighbor))


mappy = generate_map()
path = find_path(mappy)

plt.imshow(mappy, cmap='binary')
if path:
    path = numpy.array(path)
    plt.plot(path[:, 1], path[:, 0], color='red')
    plt.title('Маршрут найден!')
else:
    plt.title("Маршрут не найден!")
plt.show()

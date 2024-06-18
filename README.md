# ROB-IN
Небольшой скрипт, демонстрирующий работу алгоритма A-Star для поиска кратчайшего пути.

## Установка
*Для корректной установки требуется python3.12*
```shell
git clone https://github.com/tomasrock18/rob-in.git
cd rob-in
pip install -r requirements.txt
```

## Запуск
```shell
python3 main.py
```
Данная команда выполнит построение случайной карты и попробует применить алгоритм поиска пути. В случае отсутствия пути, программа отобразит карту, но без маршрута. Для повторного запуска, закройте окно qt и повторить вышеуказанную команду. (Или же просто можно открыть скрипт в любой удобной IDE)

### Описание функций
В данном решении реализованы 4 функции:
1) generate_map - функция создает карту с препятствиями.
2) heuristic - функция вычисляет эвристику между двумя точками.
3) find_path - реализует алгоритм A-Star на заданной карте.
4) optimize_path - оптимизирует переданный маршрут.

### Примеры использования
```python3
# Создаст произвольного размера карту, с вероятностью появления препятствия 30%
map = generate_map() 

# Аналогично, только число строк в матрице карты будет равно заданному, в данном случае 100
map = generate_map(100)

# Карта с заданными размерами, но вероятность препятствия всё так же 30%
map = generate_map(100, 100)

# Карта с заданными размерами, но уже с вдвое большей вероятностью появления препятствия
map = generate_map(100, 100, 0.6)
```

```python3
# Изначально это была лямбда функция, но PyCharm не понравилось, поэтому вынес её отдельно
delta = heuristic((0,0), (5,5))
```

```python3
# Передаём карту, получаем путь. Начало и конец заданы по-умолчанию.
path = find_path(map)

# Тоже самое, но теперь мы указали откуда начать движение
path = find_path(map, (10, 10))

# А теперь ещё и конец
path = find_path(map, (10, 10), (0, 0))
```

```python3
opt_path = optimize_path(path)
```
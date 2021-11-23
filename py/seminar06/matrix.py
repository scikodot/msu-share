"""Модуль базовых алгоритмов линейной алгебры.
Задание состоит в том, чтобы имплементировать класс Matrix
(следует воспользоваться кодом из семинара ООП), учтя рекомендации pylint.
Для проверки кода следует использовать команду pylint matrix.py.
Pylint должен показывать 10 баллов.
Кроме того, следует добавить поддержку исключений в отмеченных местах.
Для проверки корректности алгоритмов следует сравнить результаты с соответствующими функциями numpy.
"""
import math
import random
import json
import copy
import numpy as np

class Matrix:
    def __init__(self, nrows, ncols, init="zeros"):
        """Конструктор класса Matrix.
        Создаёт матрицу резмера nrows x ncols и инициализирует её методом init.
        nrows - количество строк матрицы
        ncols - количество столбцов матрицы
        init - метод инициализации элементов матрицы:
            "zeros" - инициализация нулями
            "ones" - инициализация единицами
            "random" - случайная инициализация
            "eye" - матрица с единицами на главной диагонали
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("Matrix dimensions cannot be negative.")
        
        self.nrows = nrows
        self.ncols = ncols
        
        if init == 'zeros':
            self.data = [[0.] * ncols for _ in range(nrows)]
        elif init == 'ones':
            self.data = [[1.] * ncols for _ in range(nrows)]
        elif init == 'eye':
            self.data = [[0.] * ncols for _ in range(nrows)]
            for i in range(min(nrows, ncols)):
                self.data[i][i] = 1.
        elif init == 'random':
            self.data = [[random.random()] * ncols for _ in range(nrows)]
        else:
            raise ValueError("Inappropriate init method.")
    
    @staticmethod
    def from_dict(data):
        "Десериализация матрицы из словаря"
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols*nrows
        result = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                result[(row, col)] = items[ncols*row + col]
        return result
    
    @staticmethod
    def to_dict(matr):
        "Сериализация матрицы в словарь"
        assert isinstance(matr, Matrix)
        nrows, ncols = matr.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(matr[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}
    
    def __str__(self):
        res = ""
        max_len = 0

        # Get max length of element string
        for i in range(self.nrows):
            for j in range(self.ncols):
                txt = f"{float(self[(i, j)]):.4f}"
                l = len(txt)
                if l > max_len:
                    max_len = l

        # Get matrix string
        for i in range(self.nrows):
            if i > 0:
                res += '\n'
            res += '['
            for j in range(self.ncols):
                if j > 0:
                    res += ' '
                txt = "{0:.4f}".format(float(self[(i, j)]))
                l = len(txt)
                for _ in range(max_len - l):
                    res += ' '
                res += txt
            res += ']'
        
        return res
    
    def __repr__(self):
        return f"Matrix {self.nrows}x{self.ncols}"
    
    def shape(self):
        "Вернуть кортеж размера матрицы (nrows, ncols)"
        return self.nrows, self.ncols
    
    def __getitem__(self, index):
        """Получить элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        """
        if not isinstance(index, (tuple, list)):
            raise ValueError("Inappropriate index type.")
        
        if len(index) != 2:
            raise ValueError("Inappropriate index size.")
        
        row, col = index
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError("Index out of bounds.")
        
        return self.data[row][col]
    
    def __setitem__(self, index, value):
        """Задать элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        value - Устанавливаемое значение
        """
        if not isinstance(index, (tuple, list)):
            raise ValueError("Inappropriate index type.")
        
        if len(index) != 2:
            raise ValueError("Inappropriate index size.")
        
        row, col = index
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError("Index out of bounds.")
        
        self.data[row][col] = value
    
    def __sub__(self, rhs):
        "Вычесть матрицу rhs и вернуть результат"
        if rhs.nrows != self.nrows or rhs.ncols != self.ncols:
            raise ValueError("Inappropriate RHS matrix size.")

        lhs = copy.deepcopy(self)
        for i in range(lhs.nrows):
            for j in range(lhs.ncols):
                lhs[(i, j)] -= rhs[(i, j)]
        
        return lhs
    
    def __add__(self, rhs):
        "Сложить с матрицей rhs и вернуть результат"
        if rhs.nrows != self.nrows or rhs.ncols != self.ncols:
            raise ValueError("Inappropriate RHS matrix size.")

        lhs = copy.deepcopy(self)
        for i in range(lhs.nrows):
            for j in range(lhs.ncols):
                lhs[(i, j)] += rhs[(i, j)]
        
        return lhs
    
    def __mul__(self, rhs):
        "Умножить на матрицу rhs и вернуть результат"
        if rhs.nrows != self.ncols:
            raise ValueError("Inappropriate RHS matrix size.")

        lhs = Matrix(self.nrows, rhs.ncols)
        for i in range(lhs.nrows):
            for j in range(lhs.ncols):
                elem = 0
                for k in range(self.ncols):
                    elem += self[(i, k)] * rhs[(k, j)]
                lhs[(i, j)] = elem
        
        return lhs
    
    def __pow__(self, power):
        "Возвести все элементы в степень pow и вернуть результат"
        lhs = copy.deepcopy(self)
        for i in range(lhs.nrows):
            for j in range(lhs.ncols):
                lhs[(i, j)] **= power

        return lhs
    
    def sum(self):
        "Вернуть сумму всех элементов матрицы"
        sum_ = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                sum_ += self[(i, j)]

        return sum_
        
    def det(self):
        "Вычислить определитель матрицы"
        if self.nrows != self.ncols:
            raise ArithmeticError("Matrix is not square.")
        
        lhs = copy.deepcopy(self)
        det_ = 1.

        # Perform Gaussian elimination
        for i in range(self.nrows):
            scale = 1 / lhs[(i, i)]
            for j in range(i + 1, self.nrows):
                s = scale * lhs[(j, i)]
                for k in range(self. ncols):
                    lhs[(j, k)] -= s * lhs[(i, k)]

            # Update determinant
            det_ *= lhs[(i, i)]

        return det_
    
    def transpose(self):
        "Транспонировать матрицу и вернуть результат"
        lhs = Matrix(self.ncols, self.nrows)
        for i in range(lhs.nrows):
            for j in range(lhs.ncols):
                lhs[(i, j)] = self[(j, i)]

        return lhs
    
    def inv(self):
        "Вычислить обратную матрицу и вернуть результат"
        if self.nrows != self.ncols:
            raise ArithmeticError("Matrix is not square.")
        
        if self.det() == 0:
            raise ArithmeticError("Matrix is degenerate.")
        
        lhs = copy.deepcopy(self)
        rhs = Matrix(self.nrows, self.ncols, init='eye')

        # Perform Gaussian elimination
        for i in range(self.nrows):
            # Scale
            scale = 1 / lhs[(i, i)]
            for j in range(self.ncols):
                lhs[(i, j)] *= scale
                rhs[(i, j)] *= scale

            # Eliminate
            for j in range(self.nrows):
                if j == i:
                    continue

                s = lhs[(j, i)]
                for k in range(self. ncols):
                    lhs[(j, k)] -= s * lhs[(i, k)]
                    rhs[(j, k)] -= s * rhs[(i, k)]
        
        return rhs

    def tonumpy(self):
        "Приведение к массиву numpy"
        return np.array(self.data)

def test():
    eye = Matrix(3, 3, init='eye')
    eye.tonumpy()
    assert eye.det() == np.linalg.det(eye.tonumpy())

if __name__ == "__main__":
    test()
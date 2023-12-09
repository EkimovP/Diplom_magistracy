from gui import Ui_Dialog
from graph import Graph
from drawer import Drawer as drawer
#from FastFourierTransform import FFT
#from Processing_image import Processing_Image

#from PyQt5.QtWidgets import QFileDialog
#from tkinter import *
from PyQt5 import QtCore
import random
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


# ФМ-2 модуляция
def FM2phi(t):
    # t приводим от 0 до 1
    t -= int(t)
    bits = 9600
    t *= bits
    if (int(t) % 2):
        return np.pi
    return 0


# Формула синусоиды
def sinusoida(amplitude, frequency, phase, t):
    # Фаза задается в радианах (1 рад = 57°17′)
    # A * sin(2 * pi * w * t + (phi + FM2phi))
    return amplitude * np.sin(2. * np.pi * frequency * t + (phase))  #+ FM2phi(t)))


# Функция для шума (нормальное распределение по Гауссу)
def uniform_distribution():
    repeat = 12
    val = 0
    for i in range(repeat):
        val += random.random()  # значение от 0.0 до 1.0
    return val / repeat


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):

    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        # Дополнительные функции окна
        dialog.setWindowFlags(  # Передаем флаги создания окна
            QtCore.Qt.WindowCloseButtonHint |  # Закрытие
            QtCore.Qt.WindowMaximizeButtonHint |  # Во весь экран (развернуть)
            QtCore.Qt.WindowMinimizeButtonHint  # Свернуть
        )
        self.setupUi(dialog)  # Устанавливаем пользовательский интерфейс
        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Исходный сигнал
        self.graph_1 = Graph(
            layout=self.layout_plot,
            widget=self.widget_plot,
            name_graphics="График №1. Исходный сигнал"
        )
        # Параметры 2 графика - Исходный сигнал c шумом
        self.graph_2 = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="График №2. Исходный сигнал c шумом"
        )
        # Параметры 3 графика - Спектр сигнала (для проверки)
        self.graph_3 = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="График №3. Спектр сигнала"
        )
        # Параметры 4 графика - Спектрограмма
        self.graph_4 = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="График №4. Спектрограмма"
        )

        # Картинки этапов обработки
        self.original_signal = None
        self.original_signal_t = None
        self.signal_noise = None
        #self.module_spectrum = None

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ
        # Смена способа построения спектра
        # self.radioButton_FFT.clicked.connect(self.)
        # self.radioButton_AKF.clicked.connect(self.)
        # self.radioButton_AR.clicked.connect(self.)
        # self.radioButton_MMD.clicked.connect(self.)

        # Алгоритм обратки
        # Генерация сигнала
        self.pushButton_generate_signal.clicked.connect(self.construction_sinusoids)
        # Добавление шума
        self.pushButton_display_noise.clicked.connect(self.add_noise)
        # Построение спектрограммы
        #self.pushButton_building_spectrogram.clicked.connect(self.spectrum_numpy)
        # Построение вероятности обнаружения
        #self.pushButton_probability_of_detection.clicked.connect(self.search_zone)

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1 Построение сигнала)
    def construction_sinusoids(self):
        # Запрашиваем параметры синусоид
        # Первая синусоида
        amplitude_1 = float(self.lineEdit_amplitude_1.text())
        frequency_1 = float(self.lineEdit_frequency_1.text())
        phase_1 = float(self.lineEdit_phase_1.text())
        # Вторая синусоида
        amplitude_2 = float(self.lineEdit_amplitude_2.text())
        frequency_2 = float(self.lineEdit_frequency_2.text())
        phase_2 = float(self.lineEdit_phase_2.text())
        # Третья синусоида
        amplitude_3 = float(self.lineEdit_amplitude_3.text())
        frequency_3 = float(self.lineEdit_frequency_3.text())
        phase_3 = float(self.lineEdit_phase_3.text())
        # Запрашиваем частоту дискретизации (fd)
        sampling_rate = float(self.lineEdit_sampling_rate.text())
        # Запрашиваем максимальное время (t_max)
        maximum_time = float(self.lineEdit_maximum_time.text())
        # Считаем количество отсчетов (N = t_max * fd)
        number_counts = int(sampling_rate * maximum_time)
        # Создаем пустой сигнал размером N
        self.original_signal = np.zeros(number_counts)
        # Шаг по времени (step = 1 / fd)
        step_time = 1 / sampling_rate
        # Отсчеты для отображения оси времени
        self.original_signal_t = np.arange(0, maximum_time, step_time)
        # Для каждого отсчета сигнала считаем сумму синусоид
        for t in range(number_counts):
            self.original_signal[t] = sinusoida(amplitude_1, frequency_1, phase_1, t / sampling_rate) + \
                                        sinusoida(amplitude_2, frequency_2, phase_2, t / sampling_rate) + \
                                        sinusoida(amplitude_3, frequency_3, phase_3, t / sampling_rate)
        # Отображаем исходный сигнал
        drawer.graph_signal(self.graph_1, self.original_signal, self.original_signal_t)

    # (2 а) Добавление шума в процентах
    def noise(self):
        # Нет исходного сигнала - сброс
        if self.original_signal is None:
            return
        # Копируем исходный сигнал
        self.signal_noise = self.original_signal.copy()
        noise_counting = np.zeros(self.signal_noise.size)
        # Размер сигнала
        size_signal = self.signal_noise.size
        # Считаем энергию шума
        energy_noise = 0
        for j in range(size_signal):
            val = uniform_distribution()
            # Записываем отсчет шума
            noise_counting[j] = val
            # Копим энергию шума
            energy_noise += val * val
        # Запрашиваем процент шума
        noise_percentage = float(self.lineEdit_noise.text()) / 100
        # Считаем энергию исходного сигнала
        energy_signal = 0
        for i in range(size_signal):
            energy_signal += self.original_signal[i] * self.original_signal[i]
        # Считаем коэффициент/множитель шума
        noise_coefficient = math.sqrt(noise_percentage *
                                      (energy_signal / energy_noise))
        # К отсчетам исходного сигнала добавляем отсчеты шума
        for k in range(size_signal):
            self.signal_noise[k] += noise_coefficient * noise_counting[k]
        # Отображаем итог
        drawer.graph_signal(self.graph_2, self.signal_noise, self.original_signal_t)

    # (2 б) Добавление шума в дБ
    def add_noise(self):
        # Запрашиваем шум в дБ
        noise_dicibels = float(self.lineEdit_noise.text())
        amplitude_noise = np.sqrt(1 / pow(10, noise_dicibels / 10.))
        noise = np.random.normal(0, amplitude_noise, self.original_signal.size)
        self.signal_noise = self.original_signal + noise
        # Отображаем итог
        drawer.graph_signal(self.graph_2, self.signal_noise, self.original_signal_t)

    #
    # # (3) Спектр изображения с шумом. Спектр с диагональной перестановкой
    # def spectrum_numpy(self):
    #     if self.noise_image is None:
    #         return
    #     # Перевод изображения в комплексное
    #     complex_image = np.array(self.noise_image, dtype=complex)
    #     # Считаем спектр
    #     self.picture_spectrum = FFT.matrix_fft(complex_image)
    #     # Берем модуль, для отображения
    #     module_picture_spectrum = abs(self.picture_spectrum)
    #     module_picture_spectrum[0, 0] = 0
    #     # Матрица со спектром посередине
    #     height, width = module_picture_spectrum.shape
    #     middle_h = height // 2
    #     middle_w = width // 2
    #     self.module_spectrum_repositioned = np.zeros((height, width))
    #     # Меняем по главной диагонали
    #     self.module_spectrum_repositioned[0:middle_h, 0:middle_w] = \
    #         module_picture_spectrum[middle_h:height, middle_w:width]
    #     self.module_spectrum_repositioned[middle_h:height, middle_w:width] = \
    #         module_picture_spectrum[0:middle_h, 0:middle_w]
    #     # Меняем по главной диагонали
    #     self.module_spectrum_repositioned[middle_h:height, 0:middle_w] = \
    #         module_picture_spectrum[0:middle_h, middle_w:width]
    #     self.module_spectrum_repositioned[0:middle_h, middle_w:width] = \
    #         module_picture_spectrum[middle_h:height, 0:middle_w]
    #     # Отображаем спектр
    #     drawer.image_gray_2d(self.graph_3, self.module_spectrum_repositioned,
    #                          logarithmic_axis=self.radioButton_logarithmic_axis.isChecked())


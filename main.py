# This is a sample Python script.

import seaborn as sns
import functions as ft
sns.set()


TEXTO = input("Introduce el ticker del activo que quieras visualizar. ")
activo = TEXTO
grafica = ft.calculate_moving_average_crossovers(activo)

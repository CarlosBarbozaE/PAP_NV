# This is a sample Python script.

import seaborn as sns
import functions as ft
sns.set()


TEXTO = input("Introduce el ticker del activo que quieras visualizar. ")
activo = TEXTO
grafica = ft.calculate_moving_average_crossovers(activo)
estrategiasmi = ft.smivisual(activo)
estrategiasmi2 = ft.smiestrategia(activo)
estrategiamacd = ft.macdestrategia(activo)
estrategiarsi = ft.estrategia_rsi(activo)
resistencias = ft.soporteresistencia(activo)

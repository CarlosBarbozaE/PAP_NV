# This is a sample Python script.

import seaborn as sns
import functions as ft
sns.set()


TEXTO = input("Introduce el ticker del activo que quieras visualizar. ")
activo = TEXTO
TEXTO2 = input("Introduce la estrategia que quieras visualizar. ")
estrategia = TEXTO2

dur = ft.medida_duracion(activo, estrategia)
efe = ft.medida_efectividad(activo, estrategia)
print(dur, efe)
# mm_20_100 = ft.estrategia_promediosm(activo)
# grafica_mm = ft.plot(mm_20_100[2], mm_20_100[3], mm_20_100[4], mm_20_100[0])
# # rsi_14 = ft.rsi(activo)
# estrategiasmi = ft.smivisual(activo)
# estrategiasmi2 = ft.smiestrategia(activo)
# estrategiamacd = ft.estrategia_macd(activo)
# estrategiarsi = ft.estrategia_rsi(activo)
# resistencias = ft.soporteresistencia(activo)

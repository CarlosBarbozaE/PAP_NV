
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import data as dt


class ActionColorMapping(Enum):
    SELL = 'red'
    BUY = 'green'


class ActionPricePoint:
    def __init__(self, price, date, action):
        self.price = price
        self.date = date
        self.action = action


def sell():
    return lambda left, right: left < right


def buy():
    return lambda left, right: left >= right


def plot(price, ma_20, ma_100, action_price_points):
    ax = price.plot(figsize=(16, 10))

    ma_20.plot(label='Promedio móvil 20 dias', ax=ax)
    ma_100.plot(label='Promedio móvil 100 dias', ax=ax)

    ax.set_xlabel('Número de día', fontsize=18)
    ax.set_ylabel('Precio de cierre', fontsize=18)
    ax.set_title('Activo', fontsize=20)
    ax.legend(loc='upper left', fontsize=15)

    for position in action_price_points:
        plt.scatter(position.date, position.price, s=600, c=position.action.value)

    plt.show()


def retrieve_closing_price(symbol):
    df = dt.precios.get(symbol)
    return df['Close']


def data_not_available(price):
    return np.isnan(price)


def calculate_moving_average_crossovers(symbol):
    closing_price = retrieve_closing_price(symbol)

    rm_20 = closing_price.rolling(window=20).mean()
    rm_100 = closing_price.rolling(window=100).mean()

    action = ActionColorMapping.SELL
    signal_detected = sell()
    signals = []

    for index in range(closing_price.size):
        if data_not_available(rm_20[index]) or data_not_available(rm_100[index]):
            continue

        if signal_detected(rm_20[index], rm_100[index]):
            mean_price = (rm_20[index] + rm_100[index]) / 2
            action = ActionPricePoint(mean_price, index, action)
            signals.append(action)

        if rm_20[index] >= rm_100[index]:
            action = ActionColorMapping.SELL
            signal_detected = sell()
        else:
            action = ActionColorMapping.BUY
            signal_detected = buy()
    plot(closing_price, rm_20, rm_100, signals)
    
    
def rsi(symbol):
    df = dt.precios.get(symbol)
    n = 14  # len(df)
    i = 0
    upi = [0]
    doi = [0]
    while i + 1 <= df.index[-1]:
        upmove = df['High'][i + 1] - df['High'][i]
        domove = df['Low'][i] - df['Low'][i + 1]
        if upmove > domove and upmove > 0:
            upd = upmove
        else:
            upd = 0
        upi.append(upd)
        if domove > upmove and domove > 0:
            dod = domove
        else:
            dod = 0
        doi.append(dod)
        i = i + 1
    upi = pd.Series(upi)
    doi = pd.Series(doi)
    posdi = pd.Series(pd.Series.ewm(upi, span=n, min_periods=n - 1).mean())
    negdi = pd.Series(pd.Series.ewm(doi, span=n, min_periods=n - 1).mean())
    RSI = pd.Series(posdi / (posdi + negdi), name='RSI_' + str(n))
    df = df.join(RSI)
    df.set_index('Date', inplace=True)
    x = range(len(df.index))
    fig = plt.figure(figsize=(16, 8))
    gs = mt.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    plt.plot(x, df.Close)
    plt.grid(True)
    plt.title('RSI_' + str(symbol))
    ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.plot(x, df.RSI_14, color='r')
    plt.axhline(y=0.7, color='k', linestyle='--')
    plt.axhline(y=0.3, color='k', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()
    return df


def soporteresistencia(symbol):
    """
    :param symbol: Esta variable, al igual que las demas funciones, necesita el nombre del ticker que quiere visualizar.
    :return: Grafica con los soportes/resistencias creados bajo la condiciones mencionadas en el codigo.
    """
    df = dt.precios.get(symbol)
    data = df.set_index('Date')

    pivot = []  # Se inicializa la variable pivot, aqui se iran pegando los puntos pivote que existan a traves de la historia.
    dates = []  # Aqui al igual que en pivot, se pegaran las fechas donde fueron los pivotes.
    counter = 0  # Este es un contador que noayudara a saber si el puntos es el maximo
    lastPivot = 0  # Se inicia en pivot 0, se ira modificando a traves del tiempo.

    Range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Este rango es para verificar que el punto sea el mas alto.
    dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in data.index:
        currentMax = max(Range, default=0)
        value = round(data['High'][i], 2)

        Range = Range[1:9]
        Range.append(value)
        dateRange = dateRange[1:9]
        dateRange.append(i)

        if currentMax == max(Range, default=0):
            counter += 1

        else:
            counter = 0

        if counter == 5:
            lastPivot = currentMax
            dateloc = Range.index(lastPivot)
            lastDate = dateRange[dateloc]

            pivot.append(lastPivot)
            dates.append(lastDate)

    timeD = dat.timedelta(days=60)

    for index in range(len(pivot)):
        print(str(pivot[index]) + ": " + str(dates[index]))

        plt.plot_date([dates[index], dates[index] + timeD],
                      [pivot[index], pivot[index]], linestyle="-", linewidth=2, marker=",")

    data['High'].plot(label='High')
    plt.show()


def stochastic(symbol):
    """
    :param symbol:
    :return:
    """

    df = dt.precios.get(symbol)
    data = df.set_index('Date')

    # stoch3 = smi(input_data=data, period=5, smoothing_period=3, double_smoothing_period=3)
    # stoch12 = smi(input_data=data, period=5, smoothing_period=5, double_smoothing_period=5)
    stoch3 = smi(input_data=data, period=3, smoothing_period=3)
    stoch12 = smi(input_data=data, period=5, smoothing_period=5)
    stoch3 = stoch3.getTiData()
    stoch12 = stoch12.getTiData()
    stoch3.reset_index(inplace=True)
    stoch12.reset_index(inplace=True)
    return df, stoch3, stoch12


def smiventa(symbol):
    """
    :param symbol: Ticker que es el que el usuario quiere visualizar
    :return: Datos numericos, de las senales de venta.
    """
    df = stochastic(symbol=symbol)[0]
    stoch3 = stochastic(symbol=symbol)[1]
    stoch12 = stochastic(symbol=symbol)[2]

    numeros_max = []
    y = np.array(stoch3['smi'])
    x = np.linspace(1, len(y), y.size)

    peaks_indx = argrelextrema(y, np.greater)[0]

    for p in peaks_indx:
        numeros_max.append(p)

    fechas_max = []
    puntaje_smi_max = []
    precio_max = []

    for n in numeros_max:
        fechas_max.append(df['Date'][n])
        puntaje_smi_max.append(stoch3['smi'][n])
        precio_max.append(df['High'][n])

    fechas = []
    puntaje = []
    preciov = []

    for i in range(len(numeros_max) - 1):
        if puntaje_smi_max[i] > 40:
            fechas.append(fechas_max[i])
            puntaje.append(puntaje_smi_max[i])
            preciov.append(precio_max[i])
    return fechas, puntaje, preciov


def smicompra(symbol):
    """
    :param symbol: Ticker que el usuario quiere visualizar.
    :return: Datos numericos para las senales de compra.
    """
    df = stochastic(symbol=symbol)[0]
    stoch3 = stochastic(symbol=symbol)[1]
    stoch12 = stochastic(symbol=symbol)[2]

    numeros_min = []
    y = np.array(stoch3['smi'])
    # x = np.linspace(1, len(y), y.size)

    # get peaks
    peaks_indx = argrelextrema(y, np.less)[0]

    for p in peaks_indx:
        numeros_min.append(p)

    fechas_min = []
    puntaje_smi_min = []
    precio_min = []

    for n in numeros_min:
        fechas_min.append(df['Date'][n])
        puntaje_smi_min.append(stoch3['smi'][n])
        precio_min.append(df['Low'][n])

    fechasc = []
    puntajec = []
    precioc = []

    for i in range(len(numeros_min) - 1):
        if puntaje_smi_min[i] < -40:
            fechasc.append(fechas_min[i])
            puntajec.append(puntaje_smi_min[i])
            precioc.append(precio_min[i])
    return fechasc, puntajec, precioc


def smiconjunto(symbol):
    """
    :param symbol:
    :return:
    """
    grsventa = []
    grpventa = []
    grscompra = []
    grpcompra = []
    preciov = []
    precioc = []

    precioventa = smiventa(symbol)[2]
    pivot = smiventa(symbol)[1]
    senalventa = smiventa(symbol)[0]
    preciocompra = smicompra(symbol)[2]
    pivotcompra = smicompra(symbol)[1]
    senalcompra = smicompra(symbol)[0]

    for i in range(len(precioventa) - 1):
        difv1 = precioventa[i] - precioventa[i - 1]
        difv2 = pivot[i] - pivot[i - 1]
        if difv1 > 0 and difv2 < 0 or difv1 < 0 and difv2 > 0:
            grsventa.append(senalventa[i])
            grpventa.append(pivot[i])
            preciov.append(precioventa[i])

        else:
            pass

    for i in range(len(preciocompra) - 1):
        difv1 = preciocompra[i] - preciocompra[i - 1]
        difv2 = pivotcompra[i] - pivotcompra[i - 1]
        if difv1 > 0 and difv2 < 0 or difv1 < 0 and difv2 > 0:
            grscompra.append(senalcompra[i])
            grpcompra.append(pivotcompra[i])
            precioc.append(preciocompra[i])

        else:
            pass
    return grsventa, grpventa, preciov, grscompra, grpcompra, precioc


def smivisual(symbol):
    """
    :param symbol: El ticker que el ussuario decida.
    :return: Visualmente, te grafica mediante el indicador las senales.
    """
    stoch3 = stochastic(symbol)[1]
    stoch12 = stochastic(symbol)[2]
    grsventa = smiconjunto(symbol)[0]
    grpventa = smiconjunto(symbol)[1]
    grscompra = smiconjunto(symbol)[3]
    grpcompra = smiconjunto(symbol)[4]

    plt.figure(figsize=(16, 12))
    plt.plot(stoch3['Date'], stoch3['smi'], label='Smooth 3')
    plt.plot(stoch12['Date'], stoch12['smi'], label='Smooth 5')
    plt.scatter(grsventa, grpventa, s=120, c='red', label='Puntos de venta')
    plt.scatter(grscompra, grpcompra, s=120, c='green', label='Puntos de compra')
    plt.title('STOCHASTIC MOMENTUM INDEX STRATEGY', size=20)
    plt.axhline(40, 0, 1)
    plt.axhline(-40, 0, 1)
    plt.legend()
    plt.show()


def smiestrategia(symbol):
    """
    :param symbol: Es el ticker del activo que se quiere visualizar
    :return: Grafica en donde aparecen los precios historicos del activo y sus respectivas senales.
    """
    # Fijamos los datos que necesitamos para la grafica.
    # Estos provienen de otras funciones anteriores, como stochastic() y smiconjunto()
    df = stochastic(symbol=symbol)[0]
    grsventa = smiconjunto(symbol)[0]
    preciov = smiconjunto(symbol)[2]
    grscompra = smiconjunto(symbol)[3]
    precioc = smiconjunto(symbol)[5]

    plt.figure(figsize=(16, 12))
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.scatter(grsventa, preciov, s=120, c='red', label='Puntos de venta')
    plt.scatter(grscompra, precioc, s=120, c='green', label='Puntos de compra')
    plt.title('STOCHASTIC MOMENTUM INDEX STRATEGY', size=20)
    plt.legend()
    plt.show()


def macdestrategia(symbol):
    """
    :param symbol: El ticker que el usuario decida visualizar y analizar.
    :return: Grafica las senales de venta o compra que analice la estrategia
    """
    df = dt.precios.get(symbol)  # Dataframe de precios descargados de yahoo finance
    # data = df.set_index('Date')  # Pones la columna Date como index, esto se necesita para la libreria de indicadores
    # signal = macd.MovingAverageConvergenceDivergence(input_data=data)  # Este es el indicador de MACD
    # signal = signal.getTiData()  # Guardas variables numericas en signal
    shortema = df.Close.ewm(span=12, adjust=False).mean()
    longema = df.Close.ewm(span=26, adjust=False).mean()
    macd2 = shortema - longema
    signal_s = macd2.ewm(span=9, adjust=False).mean()

    signal = pd.DataFrame()
    signal['macd'] = macd2
    signal['signal_line'] = signal_s

    Buy = []
    buydate = []
    Sell = []
    selldate = []
    flag = -1

    for i in range(0, len(signal)):
        if signal['macd'][i] > signal['signal_line'][i]:
            # Sell.append(np.nan)
            if flag != 1:
                Buy.append(df['Close'][i])
                buydate.append(df['Date'][i])
                flag = 1
            else:
                pass
                # Buy.append(np.nan)
        elif signal['macd'][i] < signal['signal_line'][i]:
            # Buy.append(np.nan)
            if flag != 0:
                Sell.append(df['Close'][i])
                selldate.append(df['Date'][i])
                flag = 0
            else:
                pass
                # Sell.append(np.nan)
        else:
            pass
            # Buy.append(np.nan)
            # Sell.append(np.nan)

    plt.figure(figsize=(12, 8))
    plt.plot(df['Close'], label='Close Price', alpha=0.35)
    plt.scatter(selldate, Sell, s=120, c='red', label='Puntos de venta', marker='v', alpha=1)
    plt.scatter(buydate, Buy, s=120, c='green', label='Puntos de compra', marker='^', alpha=1)
    plt.title('MACD STRATEGY', size=20)
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend(loc='best')
    plt.show()


def estrategia_rsi(symbol):
    """
    :param symbol: Simbolo o ticker que el usuario decidira visualizar.
    :return: Grafica con senales de compra o venta, segun lo indique el indicador.
    """
    df = dt.precios.get(symbol)  # Dataframe de precios descargados de yahoo finance
    data = df.set_index('Date')  # Pones la columna Date como index, esto se necesita para la libreria de indicadores

    r14 = rsi(input_data=data, period=14)
    r14 = r14.getTiData()

    r7 = rsi(input_data=data, period=7)
    r7 = r7.getTiData()

    senalventa = []
    puntoventa = []
    precioventa = []
    senalcompra = []
    puntocompra = []
    preciocompra = []

    for i in range(1, len(df['Date']) - 1):

        if r7['rsi'][i] > 70 and r14['rsi'][i] > 70:
            dif1 = float(r7['rsi'][i - 1] - r14['rsi'][i - 1])
            dif2 = float(r7['rsi'][i] - r14['rsi'][i])

            if dif1 > 0 and dif2 < 0 or dif1 < 0 and dif2 > 0:
                senalventa.append(df['Date'][i])
                puntoventa.append(r7['rsi'][i])
                precioventa.append(df['close'][i])

        elif r7['rsi'][i] < 30 and r14['rsi'][i] < 30:
            dif1 = float(r7['rsi'][i - 1] - r14['rsi'][i - 1])
            dif2 = float(r7['rsi'][i] - r14['rsi'][i])

            if dif1 > 0 and dif2 < 0 or dif1 < 0 and dif2 > 0:
                senalcompra.append(df['Date'][i])
                puntocompra.append(r7['rsi'][i])
                preciocompra.append(df['close'][i])

    plt.figure(figsize=(16, 12))
    plt.plot(df['Date'], df['close'], label='Close')
    plt.scatter(senalventa, precioventa, s=200, c='red', label='Puntos de venta')
    plt.scatter(senalcompra, preciocompra, s=200, c='green', label='Puntos de compra')
    plt.title('RSI STRATEGY', size=20)
    plt.legend()
    plt.show()

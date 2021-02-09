
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
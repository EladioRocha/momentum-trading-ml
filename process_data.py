import numpy as np
import talib as ta
from sklearn.model_selection import train_test_split

def set_rsi_momentum_signals(df, column_prices='None', timeperiod=14, upper=70, lower=30, centerline=50, shorts=True):
    position = np.zeros(df.shape[0])
    df_copy = df.copy()

    column_name = f'xs_rsi_{timeperiod}'
    column_name_pos = f'x_rsi_pos_{timeperiod}'

    df_copy[column_name] = ta.RSI(df_copy[column_prices], timeperiod=timeperiod)

    for i, _row in enumerate(df_copy.iterrows()):
        row = _row[1]
        if np.isnan(row[column_name]):
            last_row = row.copy()
            continue
        if row[column_name] > centerline and last_row[column_name] < centerline:
            if position[i-1] != 1:
                position[i] = 1
        elif row[column_name] > centerline and position[i-1] == 1:
            if last_row[column_name] > upper and row[column_name] < upper:
                position[i] = 0
            else:
                position[i] = 1
        elif position[i-1] == 1 and row[column_name] < centerline:
            if shorts:
                position[i] = 0
            else:
                position[i] = -1

        elif shorts:
            if row[column_name] < centerline and last_row[column_name] > centerline:
                if position[i-1] != -1:
                    position[i] = -1
                elif row[column_name] < centerline and position[i-1] == -1:
                    if last_row[column_name] < lower and row[column_name] > lower:
                        position[i] = 0
                    else:
                        position[i] = -1
            elif position[i-1] == -1 and row[column_name] > centerline:
                position[i] = 1
        last_row = row.copy()
 
    df_copy[column_name_pos] = position

    df_copy.dropna(inplace=True)

    return df_copy

def set_macd_crossover_signals(df, column_prices=None, fastperiod=9, slowperiod=21, signalperiod=9):
    df_copy = df.copy()
    column_name_pos = f'x_macd_pos_{fastperiod}_{slowperiod}_{signalperiod}'
    column_name_macd = f'xs_macd_{fastperiod}_{slowperiod}_{signalperiod}'
    column_name_macd_signal = f'xs_macd_signal_{fastperiod}_{slowperiod}_{signalperiod}'

    macds = ta.MACD(df_copy[column_prices], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    macd, macd_signal, _ = macds
    df_copy[column_name_pos] = np.where(macd - macd_signal > 0, 1, -1)
    df_copy[column_name_macd] = macd
    df_copy[column_name_macd_signal] = macd_signal
    return df_copy

def set_technical_indicators_and_signals(df, column_prices=None, settings = {
    "rsi": {
        'timeperiod': 14,
        'upper': 70,
        'lower': 30,
        'centerline': 50,
        'shorts': True
    },
    "macd": {
        'fastperiod': 9,
        'slowperiod': 21,
        'signalperiod': 9
    }
}):
    df_copy = df.copy()

    # Set RSI momentum signals
    df_copy = set_rsi_momentum_signals(df_copy, column_prices=column_prices, timeperiod=settings['rsi']['timeperiod'], upper=settings['rsi']['upper'], lower=settings['rsi']['lower'], centerline=settings['rsi']['centerline'], shorts=settings['rsi']['shorts'])
    df_copy = set_macd_crossover_signals(df_copy, column_prices=column_prices, fastperiod=settings['macd']['fastperiod'], slowperiod=settings['macd']['slowperiod'], signalperiod=settings['macd']['signalperiod'])
    df_copy.dropna(inplace=True)

    return df_copy

def set_targeet(df, horizon=1, column_prices=None):
    df_copy = df.copy()

    df_copy['target'] = df_copy[column_prices].pct_change(horizon).shift(-horizon)
    df_copy.loc[df_copy['target'] >= 0, 'target'] = 1
    df_copy.loc[df_copy['target'] < 0, 'target'] = -1

    df_copy.dropna(inplace=True)

    return df_copy

def get_train_and_validation_df(
        df, 
        train_size=0.8, 
        start_train_date=None, 
        end_train_date=None, 
        start_validation_date=None, 
        end_validation_date=None
    ):
    df_copy = df.copy()

    if start_train_date is None and end_train_date is None and start_validation_date is None and end_validation_date is None:
        # Use train_test_split to split the data. Maintain the same order of the data
        train_df, validation_df = train_test_split(df_copy, train_size=train_size, shuffle=False)
        return train_df, validation_df
    else:
        train_df = df_copy.loc[start_train_date:end_train_date]
        validation_df = df_copy.loc[start_validation_date:end_validation_date]
        return train_df, validation_df
    
def get_X_y(df, features_prefix="x", column_target='target'):
    df_copy = df.copy()
    X = df_copy[[col for col in df_copy.columns if col.startswith(features_prefix)]]
    y = df_copy[[column_target]]
    return X, y
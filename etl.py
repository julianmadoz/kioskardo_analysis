import pandas as pd
from datetime import datetime, timedelta
from product import Product

def ETL(args):
    df = pd.read_csv(args['file'])
    df['subtotal'] = df.Cantidad * df['Coste de artículo']
    df['ex_date'] = df['Fecha del pedido'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
    df['date'] = df['ex_date'].apply(lambda x: to_fecha(x))
    df['day_id'] = df['date'].apply(lambda x: days_count(x))
    df['Año'] = df['date'].apply(lambda x: x.year)
    df['Mes'] = df['date'].apply(lambda x: x.month)
    df['Dia'] = df['date'].apply(lambda x: x.day)
    df['Dia_sem'] = df['ex_date'].apply(lambda x: convert_to_day(x))
    df['Hora'] = df['ex_date'].apply(lambda x: x.hour)
    df['Hora_float'] = df['ex_date'].apply(lambda x: convert_to_hour(x))
    df['art_subtot'] = df.Cantidad * df['Coste de artículo']
    carrito = df.groupby(by='Número de pedido').sum()
    df['cant_tot_items'] = df['Número de pedido'].apply(lambda x: carrito.loc[x, 'Cantidad'])
    df['Categoría'] = df['Categoría'].apply(lambda x: categories_classifier(x))
    return df

def creates_products(df):
    products = {}
    for p in df['Item Name'].unique():
            products[p] = Product(df, p)
    return products

def convert_to_day(date):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 0 <= date.hour < 6:
        if (date.weekday() - 1) == -1:
            # return days[6]
            return 6
        else:
            return date.weekday() - 1
    else:
        return date.weekday()

def convert_to_hour(date):
    if date.hour < 6:
        hour = date.hour + date.minute / 60 + 24
    else:
        hour = date.hour + date.minute / 60
    return hour

def to_fecha(x):
    rtn = x
    if (x.hour < 6):
        rtn = x - timedelta(1)
    return rtn

def days_count(x):
    rtn = x - datetime.strptime('2020-09-29', '%Y-%m-%d')
    return rtn.days

def categories_classifier(x):
    dict={
        'Cerveza': 'Cerveza',
        'BEBIDAS ALCOHÓLICAS': 'Otras bebidas alcohólicas',
        'Bebidas CDA':'Otras bebidas alcohólicas',
        'Chocolate':'Chocolate',
        'Gaseosas':'Bebidas sin alcohol',
        'Jugos':'Bebidas sin alcohol',
        'Alfajores':'Alfajores',
        'Caramelos y Chupaletas':'Caramelos y Chupaletas'
    }
    if x != x:
      x = 'Sin categorizar'
    for key in dict:
      if key in x:
        x = dict[key]
    return x



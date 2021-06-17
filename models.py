import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from handy_functions import to_weekday_name


class poly():
    def __init__(self,acum,lookback, deg,lookahead):
        self.lookback = lookback
        self.last = acum.tail(1).cum.iloc[0]
        self.deg = deg
        self.lookahead = lookahead
        self.train(acum)
        self.pr = pd.DataFrame()
        
    def train(self, acum):
        self.poly_pr = {}
        self.poly_models = acum.groupby(by='day_id').last().cum.reset_index()[['day_id','cum']]
        for i in self.lookback:
            self.poly_pr[i] = {}
            model =  np.polyfit(self.poly_models.day_id[-i:],self.poly_models.cum[-i:],self.deg)
            self.poly_pr[i] = {
                'model': model,
                'days': i,
                'pr': round(np.polyval(model, self.poly_models.day_id.max() + self.lookahead) - self.poly_models.cum.max()),
                'conf': 0}
    def predict(self):
        minval,maxval = self.poly_models.day_id.min() , self.poly_models.day_id.max()
        # self.poly_models['day_id'] = range(minval, maxval+lookahead)
        days_range = range(int(minval), int(maxval+self.lookahead))
        for i in self.poly_pr:
            dif = np.polyval(self.poly_pr[i]['model'], maxval) - self.last
            
            for d in days_range:
                ap_dict = {
                    'day_id': d,
                    'pr' : np.polyval(self.poly_pr[i]['model'], d) - dif,
                    'pr_type': str(i)
                        } 
                self.pr = self.pr.append(
                    ap_dict,
                    ignore_index = True)
        return self.pr
    
    
    
    
    def last_pr(self, stock):
        bars = pd.DataFrame()
        if stock == stock:
            bars = bars.append(
                {'days': 'Stock',
                 'pr': stock
                    },
                ignore_index=True
                )

        for i in self.pr.pr_type.unique():
            bars = bars.append(
                {'days': i,
                 'pr': round(self.pr[self.pr.pr_type == i].tail(1)['pr'].iloc[0] - self.last)
                    },
                ignore_index=True
                )

            buy = self.pr[self.pr.pr_type == i].tail(1)['pr'].iloc[0] - self.last - stock

            if buy < 0:
                buy=0
                
            bars = bars.append(
                {'days': i + 'C',
                 'pr': round(buy)
                },
                ignore_index=True
                )
        return bars

class model_cat():
  def __init__(self, data, key, epochs, periods):
    self.scl = MinMaxScaler()
    self.sclcum = MinMaxScaler()
    self.test_size = 0.01
    self.model = Sequential()
    self.epochs = epochs
    self.init_data = data
    self.pr_key = key
    print(self.init_data.tail(20))
    self.data = self.shifter(self.init_data, periods, ['derv'])
    # self.future = self.create_future(self.init_data, periods)
    self.future = self.data[-7:]
    self.data =self.data[:-7]
    print(self.data.tail(20))
    print('will predict:  ' + self.pr_key )
    print('woth the following data: ')
    print(self.data.drop(columns=self.pr_key).columns)
    
  
  def shifter(self,df,look_back, list):

    for col in list:
      df[col] = df[col].shift(periods=look_back,fill_value=0)
    
    df['t-' + str(look_back)]= self.init_data[self.pr_key].shift(periods=look_back,fill_value=0)
    return df

  def create_future(self, periods):
    self.future = self.data.iloc[:-14]


  def train(self):
    self.x = self.scl.fit_transform(self.data.drop(columns=[self.pr_key]))
    self.y = self.sclcum.fit_transform(self.data[self.pr_key].to_numpy().reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size= self.test_size)
    self.model.add(Dense(self.x.shape[1]))
    self.model.add(Dense(self.x.shape[1]/2))
    self.model.add(Dense(4))
    # self.model.add(Dense(10))
    # self.model.add(Dropout(0.1, noise_shape = None, seed = None))
    self.model.add(Dense(1))

    self.model.add(Dense(1))
    self.model.compile(loss='mean_squared_error', optimizer='adam')
    history = self.model.fit(X_train, y_train, epochs= self.epochs,batch_size=1, verbose=0, shuffle=False,validation_data=(X_test, y_test))
    print(history.history['loss'][-1])
    print(history.history['val_loss'][-1])
    if history.history['loss'][-1]>0.0005 or history.history['val_loss'][-1]>0.0005:
      self.epochs = self.epochs + 20
      print('running again' + str(self.epochs))
      if self.epochs < 61:
        self.run()
      else:
        return
    
    
  def predict(self):
    self.predictions = pd.DataFrame()

    self.predictions['day_id'] = self.future.day_id
    self.predictions['pr'] = self.model.predict(self.scl.transform(self.future.drop(columns=[self.pr_key])), batch_size=1)
    self.predictions['pr'] = self.sclcum.inverse_transform(self.predictions.pr.to_numpy().reshape(-1, 1))
    return self.predictions
  def run(self):
    self.train()
    return self.predict()

class LSTM_model():
  def __init__(self, data, epochs):
    self.scl = MinMaxScaler()
    self.sclcum = MinMaxScaler()
    self.test_size = 0.33
    self.model = Sequential()
    self.epochs = 1
    self.data = data
    self.data['t-1']= self.data.cum.shift(periods=1,fill_value=0)
    self.data['t-2']= self.data.cum.shift(periods=2,fill_value=0)
    self.data['t-3']= self.data.cum.shift(periods=3,fill_value=0)
    self.data['t-5']= self.data.cum.shift(periods=5,fill_value=0)
    self.data['t-7']= self.data.cum.shift(periods=6,fill_value=0)
    self.data['t-8']= self.data.cum.shift(periods=7,fill_value=0)
    # self.data['t-20']= self.data.cum.shift(periods=20,fill_value=0)
    # self.data['t-50']= self.data.cum.shift(periods=50,fill_value=0)
    # self.data['t-100']= self.data.cum.shift(periods=100,fill_value=0)
    self.future = self.data.iloc[-14:]
    self.data = self.data.iloc[:-14]

  def train(self):
    self.x = self.scl.fit_transform(self.data.drop(columns=['cum']))
    self.y = self.sclcum.fit_transform(self.data['cum'].to_numpy().reshape(-1, 1))
  
    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size= self.test_size)

    self.model.add(Dense(5))
    self.model.add(Dense(10))
    self.model.add(Dense(5))

    self.model.add(Dense(1))
    self.model.compile(loss='mean_squared_error', optimizer='adam')
    self.model.fit(X_train, y_train, epochs= self.epochs,batch_size=1, verbose=1, shuffle=False,validation_data=(X_test, y_test))
    
  def predict(self):
    self.predictions = pd.DataFrame()

    self.predictions['day_id'] = self.future.day_id
    self.predictions['pr'] = self.model.predict(self.scl.transform(self.future.drop(columns=['cum'])), batch_size=1)
    self.predictions['pr'] = self.sclcum.inverse_transform(self.predictions.pr.to_numpy().reshape(-1, 1))
    return self.predictions
  def run(self):
    self.train()
    return self.predict()

class kmeanclass():
    def __init__(self,df,clip=None, fillna=None):
        
        self.results= {}
        self.models = {}
        self.grouped = {}
        self.means = {}
        self.df = (df[['Número de pedido','Categoría','Cantidad']]
                   .pivot_table(index= 'Número de pedido', columns = 'Categoría', values='Cantidad', aggfunc='sum')
                   .fillna(0) 
                   .reset_index()
                   )

        if clip != None:
            self.df[self.df.drop(columns='Número de pedido')>=1] = clip
        if fillna != None:
            self.df[self.df.drop(columns='Número de pedido')==0] = fillna
            
    def train(self, clusters):
        self.models = KMeans(n_clusters=clusters)
        self.results = self.models.fit_predict(self.df.drop(columns='Número de pedido'))
        self.df['Cat-c' + str(clusters)] = self.results 
    
    def group(self, cluster):
        self.grouped = (self.df
                                 .groupby(by='Cat-c' + str(cluster))
                                 .agg(['min','max','mean','median','sum'])
                                 .transpose()
                                 .reset_index()
                                 )
        self.means = self.grouped[self.grouped['level_1']=='mean'].drop(columns='level_1')
        self.means = self.means[self.means['level_0']!= 'Número de pedido']
        return self.df 
        
    def plot(self,cluster):
        for group in self.means.drop(columns='level_0').columns:
            list = self.means[['level_0',group]].sort_values(by=group, ascending=False)
            print(list)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.color_palette("tab10")
        plt.suptitle(title)
        # sns.scatterplot(data=self.df,x=1,y=0, hue='Cat-c',palette='tab10')
        sns.kdeplot(data=self.df,x=1,y=0,fill=True)
        plt.show()
        plt.close()
                
class kmeanclass_tot():
    def __init__(self,df, clusters):
        
        # temp = (df[['Número de pedido','Importe de subtotal del pedido','Cantidad']]
        #            .reset_index()
        #            .groupby(by='Número de pedido')
        #            )
        # self.df = temp.sum().drop(columns='Importe de subtotal del pedido')
        # self.df['Importe de subtotal del pedido'] = temp.mean()['Importe de subtotal del pedido']
        # self.df = self.df.drop(columns='index')
        # if clip != None:
        #     self.df[self.df.drop(columns='Número de pedido')>=1] = clip
        # if fillna != None:
        #     self.df[self.df.drop(columns='Número de pedido')==0] = fillna
        
        self.scl = MinMaxScaler()        
        self.df = pd.DataFrame(self.scl.fit_transform(df))

        if type(clusters) == type(5):
            self.model = KMeans(n_clusters=clusters)        
            self.results = self.model.fit_predict(self.df[[0,1]])
            
            self.df = pd.DataFrame(self.scl.inverse_transform(self.df))
            self.df['Cat-c'] = self.results 
        
        if type(clusters) == type([0,1]):
            r = {}
            for i in range(clusters[0],clusters[1]):
                KMeans_model = KMeans(n_clusters=i, random_state=42)
                KMeans_model.fit(self.df)
                r[i]  = KMeans_model.inertia_
            fig, ax = plt.subplots(figsize=(10,10))
            sns.lineplot(data=pd.DataFrame.from_dict(r, orient='index'),markers='o')
            plt.show()
            plt.close()
            return

    
    def group(self):
        self.grouped = (self.df
                                 .groupby(by='Cat-c')
                                 .agg(['min','max','mean','median','count'])
                                 .transpose()
                                 .reset_index()
                                 )
        self.means = self.grouped[self.grouped['level_1']=='median'].drop(columns='level_1')
        self.means = self.means[self.means['level_0']!= 'Número de pedido']
        return self.grouped 
        
    def plot(self,title):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.color_palette("tab10")
        plt.suptitle(title)
        # sns.scatterplot(data=self.df,x=1,y=0, hue='Cat-c',palette='tab10')
        sns.kdeplot(data=self.df,x=1,y=0,fill=True)
        plt.show()
        plt.close()

def analyze_assoc(df):
    from mlxtend.frequent_patterns import apriori, association_rules
    cerveceros = df[df['Item Name'].str.lower().str.contains('cerveza')]['Número de pedido']

    def changename(x):
        if 'Alfajor' in x:
            return 'Alfajor'
        if 'Caramelo' in x:
            return 'Caramelo'
        if 'Cerveza' in x:
            return 'Cerveza'
        if 'Cigarrillos' in x:
            return 'Cigarrillos'
        if 'Saladix' in x:
            return 'Saladix'
        if 'Flynn' in x:
            return 'Caramelo'
        return x

    df = pd.pivot_table(
        df[df['Número de pedido'].isin(cerveceros)],
        # df,
        index='Número de pedido',
        columns='Item Name',
        values='Cantidad',
        aggfunc='count',
        fill_value=0
    )
    print('asdasdasdasdasdasdasdasdasdasdasda')
    df[df > 0] = 1
    freq_items = apriori(df, min_support=0.1, use_colnames=True, verbose=1)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.3)
    # rules = rules[rules.support > 0.007]
    sns.scatterplot(data=rules.reset_index(), x='support', y='lift')

    return rules

def analyze_products_poly(products, limit=1000):

        pp = PdfPages('Analisis productos.pdf')

        counter = 0

        for p in products:
            if counter > limit:
                break
            counter = counter + 1
            print('Analizando: ' + p)
            products[p].poly([7, 14, 30, 60], 1, 7)
            products[p].plot_poly(30, 7, pp)

        pp.close()

def cat_cumulatives(df):
    df_cat = {}
    cats = df.groupby(by='Categoría').count()['Número de pedido'].sort_values(ascending=False)

    for i in cats.index:
        df_cat[i] = df[df.Categoría == i]
        df_cat[i] = df_cat[i].reset_index().sort_values(by='index', ascending=False).reset_index()
        a = df_cat[i].groupby(by='day_id').sum().Cantidad
        df_cat[i]['cum'] = df_cat[i].Cantidad.cumsum()
        df_cat[i]['sales_cum'] = df_cat[i].art_subtot.cumsum()
        df_cat[i] = df_cat[i].groupby(by='day_id').last()
        df_cat[i]['Cantidad'] = a

        # df_cat[i]['derv'] = np.gradient(df_cat[i].cum)
        # df_cat[i]['derv2'] = np.gradient(df_cat[i].derv)
        df_cat[i].reset_index(inplace=True)

        df_cat[i] = df_cat[i][['day_id', 'Dia', 'Dia_sem', 'cum', 'sales_cum']]

    fig, axes = plt.subplots(figsize=(17, 10))
    for cat in df_cat:
        sns.lineplot(data=df_cat[cat], x='day_id', y='sales_cum', legend=True)

def sales_hist(df):
    def clusters(x):
        cl = floor(x / 100) * 100
        # if 1000<cl<1200:
        #     cl = 1000
        # if 1200<cl<1500:
        #     cl = 1000
        # if cl > 1500:
        #     cl = 1500
        return cl

    shist = df[
        ['Número de pedido', 'Teléfono (facturación)', 'Cantidad', 'Categoría', 'Item Name', 'art_subtot',
         'Importe de subtotal del pedido']].reset_index().drop(columns='index')

    shist = shist.groupby(by='Número de pedido').agg(
        telefono=pd.NamedAgg(column="Teléfono (facturación)", aggfunc="last"),
        items_tot=pd.NamedAgg(column="Cantidad", aggfunc="sum"),
        subtot=pd.NamedAgg(column='Importe de subtotal del pedido', aggfunc="mean")
    ).reset_index()
    shist['subtot_floor'] = shist['subtot'].apply(lambda x: clusters(x))
    shist = shist.groupby(by='subtot_floor').agg(
        clientes_unicos=pd.NamedAgg(column="telefono", aggfunc="nunique"),
        cant_pedidos=pd.NamedAgg(column="Número de pedido", aggfunc="nunique"),
        items_tot=pd.NamedAgg(column="items_tot", aggfunc="sum"),
        mean_items_pedido=pd.NamedAgg(column="items_tot", aggfunc="mean"),
        subtot=pd.NamedAgg(column='subtot', aggfunc="sum")
    ).reset_index()

    shist['pd_por_cl'] = shist.cant_pedidos / shist.clientes_unicos
    shist['mean_subtot'] = shist.subtot / shist.cant_pedidos
    # shist['mean_prod_price'] = shist.items_tot / shist.
    shist['prod_por_pedido'] = shist.items_tot / shist.cant_pedidos
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        ax=ax1,
        data=shist[['prod_por_pedido']],
        marker='o')
    ax2 = ax1.twinx()

    sns.barplot(
        ax=ax2,
        data=shist,
        x='subtot_floor',
        y='subtot',
        alpha=0.5)

    plt.show()

    return shist

    # df2['Item Name'] = df2['Item Name'].apply(lambda x: changename(x))

def orders_by_day(df):
    df = df[df.Dia_sem > 4].reset_index()
    df['Hora_float'] = df['Hora_float'].apply(lambda x: round(10*x)/10)
    orders = df.groupby(by=['Hora_float','Dia_sem']).agg(
        # Dia_sem=pd.NamedAgg(column="Dia_sem", aggfunc="mean"),
        # Hora = pd.NamedAgg(column="Hora", aggfunc="mean"),
        tot= pd.NamedAgg(column="Importe total del pedido", aggfunc="median"),
        cant = pd.NamedAgg(column="Número de pedido", aggfunc="count"),
    ).reset_index()
    cant_dias = df.day_id.max() - df.day_id.min()
    orders['cant'] = orders.cant/(cant_dias/7)
    orders['Día'] = orders.Dia_sem.apply(lambda x: to_weekday_name(x))

    orders = pd.pivot(orders,index='Hora_float' ,columns='Día', values = 'cant').fillna(0)
    for dia in orders.columns:
        orders[dia] = gaussian_filter1d(orders[dia], sigma=1.5)

    orders = orders.reset_index().melt(id_vars=['Hora_float'], var_name='Día', value_name='tot')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(
        data=orders,
        x='Hora_float',
        y='tot',
        hue='Día',
        linewidth = 2,
        palette='Set2').set_xlim(left=20)
    plt.grid(True)
    plt.show()
    return orders

def analyze_order_one_day(df, day=5):
    df = df[df.Dia_sem == day].reset_index()
    df['Hora_float'] = df['Hora_float'].apply(lambda x: round(10*x)/10)

    orders = df.groupby(by=['Hora_float','day_id']).agg(
        Dia_sem=pd.NamedAgg(column="Dia_sem", aggfunc="mean"),
        # Hora = pd.NamedAgg(column="Hora", aggfunc="mean"),
        tot= pd.NamedAgg(column="Importe total del pedido", aggfunc="median"),
        cant = pd.NamedAgg(column="Número de pedido", aggfunc="count"),
    ).reset_index()
    cant_dias = df.day_id.max() - df.day_id.min()
    orders['cant'] = orders.cant/(cant_dias/7)
    orders['Día'] = orders.Dia_sem.apply(lambda x: to_weekday_name(x))

    orders = pd.pivot(orders,index='Hora_float' ,columns='day_id', values = 'cant').fillna(0)
    for dia in orders.columns:
        orders[dia] = gaussian_filter1d(orders[dia], sigma=1)

    orders = orders.reset_index().melt(id_vars=['Hora_float'], var_name='Día', value_name='tot')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(
        data=orders,
        x='Hora_float',
        y='tot',
        # hue='Día',
        # linewidth = 2,
        err_style = 'band',
        palette='Set2').set_xlim(left=20)
    plt.grid(True)
    plt.show()
    return orders
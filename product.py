import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dateutil.relativedelta import relativedelta 
from math import ceil
import matplotlib.patches as patches


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models import poly

class Product():
    def __init__(self,df,item_name):
        self.item_name = item_name
        self.acum = self.tidy_1v(df, item_name)
        self.stock = df[df['Item Name']==self.item_name]['Stock Quantity'].sample(1).tolist()[0]
        self.poly_pr = {}

    def calculate(self):
        self.sales_pr = self.predictions.pr.tail(1).tolist()[0]-self.acum.cum.tail(1).tolist()[0]
        self.should_buy = round(self.sales_pr - self.stock)
        self.stock_dict = {
            'Item Name': self.item_name,
            'Stock': self.stock,
            'Prediccion': self.sales_pr,
            'Comprar': self.should_buy
            }
        
    def tidy_1v(self, df, item_name):
        cum = df[df['Item Name']== item_name]
        cum = cum.reset_index().sort_values(by='index', ascending=False).reset_index()     
        cum['cum'] = cum['Cantidad'].cumsum()
        cum = cum[['day_id', 'Año', 'Mes', 'Dia', 'Dia_sem','Hora', 'Hora_float', 'cum']]
        return cum

    def poly(self,lookback, deg,lookahead):
        self.poly_models = poly(self. acum, lookback, deg,lookahead)
        self.poly_models_pr = self.poly_models.predict()

    def plot_poly(self, days,lookahead,pp):
        minday = self.acum.day_id.max() - days
        fig, axes = plt.subplots(figsize =(12,9))
        plt.subplots_adjust(
                    wspace=0.25, 
                    hspace=0.2)
        plt.rc('axes', facecolor='White')
        
        ax1 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan = 1)
        ax1.grid(axis  = 'y')
        ax1.set_title('Histórico',weight = 'bold')
        
        ax1.set(ylabel = 'Cantidad', xlabel='Tiempo')
        # ax1.set_xticklabels(['{:,.2f}'.format(x) + 'K' for x in ax1.get_xticks()-1000])
        # ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        # ax1.xaxis.set_major_formatter(plt.NullFormatter())
        # ax1.get_xaxis().set_visible(False)
        
        ax1.add_patch(patches.Rectangle((max(self.acum.day_id), 0), -60, max(self.acum.cum),color='#D1D0CE'))
        ax1.add_patch(patches.Rectangle((max(self.acum.day_id), 0), -30, max(self.acum.cum),color='#CCE6FF'))
        ax1.add_patch(patches.Rectangle((max(self.acum.day_id), 0), -14, max(self.acum.cum),color='#CCFFDD'))     
        ax1.add_patch(patches.Rectangle((max(self.acum.day_id), 0), -7, max(self.acum.cum),color='#FFCCCC'))
        
        
        
        
        ax2 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan = 2)
        ax2.grid(axis = 'y')
        ax2.set(ylabel = 'Cantidad', xlabel='Tiempo')
        ax2.set_title('Últimos ' + str(days) + ' días', weight = 'bold')
        ax2.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax2.xaxis.set_major_locator(ticker.NullLocator())
        # ax2.xaxis.set_major_formatter(plt.NullFormatter())
        # ax2.get_xaxis().set_visible(False)
        ymin = min(self.acum[self.acum.day_id >= minday].groupby(by='day_id').sum().cum)
        ax2.add_patch(patches.Rectangle((max(self.acum.day_id), ymin), -30, max(self.acum.cum)-ymin,color='#CCE6FF'))
        ax2.add_patch(patches.Rectangle((max(self.acum.day_id), ymin), -14, max(self.acum.cum)-ymin,color='#CCFFDD'))     
        ax2.add_patch(patches.Rectangle((max(self.acum.day_id), ymin), -7, max(self.acum.cum)-ymin,color='#FFCCCC'))

        


        ax3 = plt.subplot2grid((2, 3), (1,2))
        # ax3.grid(True)
        ax3.set(ylabel = ' ', xlabel='Cantidad')
        ax3.set_title('Stock, predicción y compra', weight = 'bold')
        ax3.xaxis.set_major_locator(plt.MultipleLocator(2))
        
        # ax4 = plt.subplot2grid((3, 4), (1, 1))

        
        clrs = ['Red', 'Green','Blue','Grey']
        
        sns.lineplot(
            ax = ax1, 
            data=self.poly_models_pr,
            x='day_id', 
            y='pr',
            hue='pr_type',
            linewidth = 2,
            palette = clrs).set_ylim(bottom=0)
        
        sns.lineplot(
            ax = ax1,
            data=self.acum.groupby(by='day_id').last().reset_index(),
            x='day_id',
            y='cum',
            color = 'Black',
            linewidth = 5,
            ).set_ylim(bottom=0)
        
        sns.lineplot(
            ax = ax2, 
            data=self.poly_models_pr[self.poly_models_pr.day_id >= minday] ,
            x='day_id', 
            y='pr',
            hue='pr_type', 
            linewidth = 2,
            palette=clrs)
        
        sns.lineplot(
            ax = ax2,
            data=self.acum[self.acum.day_id >= minday].groupby(by='day_id').last().reset_index(),
            x='day_id',
            y='cum',
            color = 'Black',
            linewidth = 5).set_ylim(bottom=ymin)
        
        
        bars = self.poly_models.last_pr(self.stock)
        bars.pr =bars.pr.apply(lambda x: round(x))
        clrs = ['Black','Red','#FFCCCC', 'Green','#CCFFDD','Blue', '#CCE6FF','Grey','Grey']
        
        br = sns.barplot(
            ax = ax3,
            data = bars[bars['days'].str.contains('60',case=True) == False],  
            y='days', 
            x='pr', 
            palette= clrs,
            orient= 'h').set_xlim(left=0)
        
        ax3.set(ylabel = '', xlabel='Cantidad')
       
        
        def show_values_on_bars(axs):
            def _show_on_single_plot(ax):        
                for p in ax.patches:
                    _x = p.get_x() + p.get_width()
                    _y = p.get_y() + p.get_height()
                    if _x < 0:
                        _x = 0 
                    
                    if p.get_width()< 0: 
                        value = '0'
                    else:
                        value = str(round(p.get_width()))
                    
                    if p.get_width() > 0:
                        ax.text(_x / 2, _y - p.get_height()/2, value, ha="center", va = 'center', backgroundcolor = 'white', fontsize= 'small') 
                    else:
                        pass
                        # ax.text(_x + 1 , _y - p.get_height()/2, value, ha="center", va = 'center', fontsize= 'small') 
        
            if isinstance(axs, np.ndarray):
                for idx, ax in np.ndenumerate(axs):
                    _show_on_single_plot(ax)
            else:
                _show_on_single_plot(axs)
                
        show_values_on_bars(ax3)
        plt.suptitle(self.item_name, size = 'xx-large', weight = 'bold')
        # plt.tight_layout()
        if pp == 'print':
            plt.show()
        else:
            pp.savefig(fig)
        plt.close()
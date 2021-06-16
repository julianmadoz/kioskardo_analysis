# horarios de los pedidos
c.tidy()
horarios = c.df.groupby(by='Número de pedido').agg(
    dia = pd.NamedAgg(column = 'Dia_sem',aggfunc = 'mean'),
    count = pd.NamedAgg(column = 'Dia_sem',aggfunc = 'count'),
    hora = pd.NamedAgg(column = 'Hora_float', aggfunc='mean'),
    cant = pd.NamedAgg(column = 'cant_tot_items', aggfunc='mean'),
    tot = pd.NamedAgg(column = 'art_subtot', aggfunc='sum'),
    ).reset_index().drop(columns='Número de pedido')
d = 4
horarios['hora'] = round(d* horarios.hora)/d
hmap = pd.pivot_table(horarios, values ='tot', columns='hora', index='dia',aggfunc='mean')
hmap = hmap.fillna(0).transpose()
for col in hmap.columns:
    hmap[col] = gaussian_filter1d(hmap[col],0.008)

hmap = hmap.transpose()
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(hmap,square=True, center = 400,cbar_kws={"orientation": "horizontal"}).set_xlim(left=12)
plt.suptitle('Ticket promedio', size = 'xx-large', weight = 'bold')

# sns.kdeplot(data=horarios, x='dia',y='hora',fill=True).set_ylim(bottom=18)

#%%

asdsa = c.sales_hist()



#%%
from models import kmeanclass_tot

antesabril = c.df[c.df.date < '2021-01-04 10:00:00']
antesabril = antesabril[antesabril.cant_tot_items == 2]
antesabril['date'] = 'antes'

temp2 = antesabril.groupby(by='Número de pedido').aggregate({'Categoría': 'nunique','Importe de subtotal del pedido':np.mean,'cant_tot_items':np.mean})
temp2 = temp2[temp2['Importe de subtotal del pedido']<1500]
temp2 = temp2[temp2['cant_tot_items']==2]

temp2['date'] = 'Antes'
# model = kmeanclass_tot(temp,5)
# model.plot('antes de abril: Categorías')


despuesdeabril = c.df[c.df.date > '2021-01-04 10:00:00']
despuesdeabril = despuesdeabril[despuesdeabril.cant_tot_items == 2]
despuesdeabril['date'] = 'Despues'
# model = kmeanclass_tot(temp2,5)
# model.plot('despues de abril, Cantidad')

temp4 = despuesdeabril.groupby(by='Número de pedido').aggregate({'Categoría': 'nunique','Importe de subtotal del pedido':np.mean,'cant_tot_items':np.mean})
temp4 = temp4[temp4['Importe de subtotal del pedido']<500]
temp4 = temp4[temp4['cant_tot_items']==2]
temp4['date'] = 'Despues'

antesabril = antesabril.append(despuesdeabril)
temp2 = temp2.append(temp4)
# model = kmeanclass_tot(temp,5)
# model.plot('despues de abril: Categorías')


fig, ax = plt.subplots(figsize=(10,10))
sns.color_palette("tab10")
sns.kdeplot(data=antesabril,x='Importe de subtotal del pedido',y='Cantidad',hue='date',alpha=.5,common_norm=True,levels=5,thesh=0)
plt.show()
plt.close()
fig, ax = plt.subplots(figsize=(10,10))
sns.color_palette("tab10")
sns.kdeplot(data=temp2,x='Importe de subtotal del pedido',y='Categoría',hue='date',alpha=.5,common_norm=True,levels=5,thesh=0)
plt.show()
plt.close()

# kmeanclass_tot(temp,[3,16])

# rest = {}
# for i in range(3,10):
#     model = kmeanclass_tot(temp,5)
#     print(model.model.inertia_)
# model.plot()
#%%
despuesdeabril = c.df[c.df.date > '2021-01-04 10:00:00']
despuesdeabril = despuesdeabril[despuesdeabril['Importe de subtotal del pedido']>400]
despuesdeabril = despuesdeabril[despuesdeabril['Cantidad']<4]

fig, ax = plt.subplots(figsize=(10,10))
sns.color_palette("tab10")
sns.displot(data=despuesdeabril,y='Categoría',hue='Cantidad',stat='probability',common_norm=False, multiple = 'dodge')
plt.show()
plt.close()

#%%
from models import kmeanclass

despuesdeabril = c.df[c.df.date > '2021-01-04 10:00:00']
despuesdeabril = despuesdeabril[despuesdeabril['Importe de subtotal del pedido']>400]
despuesdeabril = despuesdeabril[despuesdeabril['Cantidad']<2]

model2 = kmeanclass(despuesdeabril, hue='')
model2.train(10)
aasd = model2.group(10)
model2.plot(10)








    #%%
eq = model.group(13)
model.plot(13)
# eq = model.grouped[5]
print(eq)


#%%
}


b = a
c = b.pivot_table(index= 'Número de pedido', columns = 'Categoría', values='Cantidad', aggfunc='sum').fillna(0)
# c['Dulces'] = c.Alfajores + c['Caramelos y Chupaletas'] + c['Chocolate'] + c['Galletitas'] +c['Gomitas'] + c['POSTRES']

# dulces = ['Alfajores', 'Chocolate','Caramelos y Chupaletas','Galletitas','Gomitas','POSTRES']
# bebidas = ['Bebidas sin alcohol', 'Energizantes' ]
# alcohol = ['Cerveza','Otras bebidas alcohólicas']
# c['Dulces'] = c[dulces].sum(axis=1)
# c = c.drop(columns = dulces)
# c['Bebidas'] = c[bebidas].sum(axis=1)
# c = c.drop(columns = bebidas)
# c['Alcohol'] = c[alcohol].sum(axis=1)
# c = c.drop(columns = alcohol)

# c[c>1] = 100
kdf = c.reset_index().drop(columns='Número de pedido')
kmeans = KMeans(n_clusters=11)
kmeans.fit(kdf)# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)# save new clusters for chart
c['cat'] = kmeans.fit_predict(kdf)
c['id'] = 1

#%%
cant = c.groupby(by='cat').sum().id
aaaa =  c.drop(columns= ['id']).groupby(by='cat').mean().transpose()
gr = c.drop(columns= ['id']).groupby(by='cat').agg(['mean'])
# sns.histplot( c[c.cat == 3].Alcohol)

gr['cant'] = c.groupby(by='cat').sum().id*100/3180
gr = gr.transpose().reset_index()
fig, ax = plt.subplots()
fig.set_size_inches(9,9)

#%%
distortions = {}
for k in range(2,50):
    KMeans_model = KMeans(n_clusters=k, random_state=42)
    KMeans_model.fit(kdf)
    distortions[k] = KMeans_model.inertia_

print(distortions)
distortions = pd.DataFrame.from_dict(distortions, orient='index')

sns.lineplot(data=distortions,  marker='o')
#%%
# print(distortions.reset_index())
distortions.reset_index(inplace=True)
distortions['diff'] = np.gradient(distortions[0], distortions['index'] )
print(distortions)
sns.lineplot(derv)
#%%
fig, ax = plt.subplots(figsize=(17,17))
sns.lineplot(data= distortions[distortions.index < 21], x='index',y='diff')
plt.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.set_xticks(np.arange(0, 20, 1))








# t = c.sum(axis=1)
# sns.kdeplot(t[t<10])
# sns.histplot(c.Chocolate)
# asd = c[c.Dulces < 20]
# sns.kdeplot(data= asd[((asd.Dulces + asd.SNACKS) != 0)  ], x='Dulces', y='SNACKS')


# c.analyze_products_poly()
# c.show_predictions()

# c.products['Bocadito Marroc'].poly([7,14,30],1,7)
# c.products['Bocadito Marroc'].plot_poly(30,7)
# c.products['Papas Fritas Krachitos Jamón Serrano 55g'].plot_poly(30,7)

#%%
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",

                         "bar", "bar", "bar", "bar"],

                   "B": ["one", "one", "one", "two", "two",

                         "one", "one", "two", "two"],

                   "C": ["small", "large", "large", "small",

                         "small", "large", "small", "small",

                         "large"],

                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],

                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})


asd = df.pivot_table(index='D', columns='A',values = 'E', aggfunc='sum')


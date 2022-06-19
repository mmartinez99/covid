import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
# from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error


# Lecture Fichier CSV
world = pd.read_csv('worldometer_data.csv')
full = pd.read_csv('full_grouped.csv') # utilisé pour les prédictions
covid = pd.read_csv('country_wise_latest.csv')
day = pd.read_csv('day_wise.csv')

# les premieres lignes
# print(covid.head())
# print(full.head())
# print(world.head())
# print(day.head())
#
#  info => Resume DataFramae
# print(covid.info())
#  Describe=> statistique données.
# print(covid.describe())
# print(day.info())
# print(day.describe())
#
# print(world.info())
# print(world.describe())
#
# print(full.info())
# print(full.describe())

# CAS CONFIRM2
data = dict(type='choropleth',
            locations = covid['Country/Region'],
            locationmode = 'country names',
            z = covid['Confirmed'],
            text = covid['Country/Region'],
            colorbar = {'title':'Cas Confirmés MONDE'}
            )

layout = dict(title='Cas Confirmés MONDE',
             geo=dict(showframe=False,
                     projection={'type':'natural earth'}))

choromap1=go.Figure(data=[data],layout=layout)
iplot(choromap1)

# Map CAS TOTAL

data = dict(type='choropleth',
            locations = world['Country/Region'],
            locationmode = 'country names',
            z = world['TotalCases'],
            text = world['Country/Region'],
            colorbar = {'title':'Total des Cas'},
            colorscale = 'viridis'
            )

layout = dict(title='Cas total',
             geo=dict(showframe=False,
                     projection={'type':'natural earth'}))

choromap2=go.Figure(data=[data],layout=layout)
iplot(choromap2)

# Map personnes Guérisons
data = dict(type='choropleth',
            locations = world['Country/Region'],
            locationmode = 'country names',
            z = world['TotalRecovered'],
            text = world['Country/Region'],
            colorbar = {'title':'Guérisons'},
            colorscale = 'blues'
            )

layout = dict(title='Guérisons',
             geo=dict(showframe=False,
                     projection={'type':'natural earth'}))

choromap3=go.Figure(data=[data],layout=layout)
iplot(choromap3)

# Map personne MORTS
data = dict(type='choropleth',
            locations = world['Country/Region'],
            locationmode = 'country names',
            z = world['TotalDeaths'],
            text = world['Country/Region'],
            colorbar = {'title':'Morts'},
            colorscale = 'reds'
            )

layout = dict(title='MORTS',
             geo=dict(showframe=False,
                     projection={'type':'natural earth'}))

choromap4=go.Figure(data=[data],layout=layout)
iplot(choromap4)


# fin map
# Graphique montre les cas pour chaque continant
plt.figure(figsize=(3,8))
sns.kdeplot(x = 'Recovered',data=covid, hue="WHO Region")

# CAS Confirmés
plt.figure(figsize=(10,4))
ax1 = sns.kdeplot(data=covid, x="Confirmed",color='r')
# Graphiques Morts,confirmés,cas confirmé
# Morts
plt.figure(figsize=(10,4))
ax2 = sns.kdeplot(data=covid, x="Deaths", color='g')

# Recovered
plt.figure(figsize=(10,4))
ax3 = sns.kdeplot(data=covid, x="Recovered", color='b')

# Active
plt.figure(figsize=(10,4))
ax4 = sns.kdeplot(data=covid, x="Active", color='y')

# Nouveau Cas
plt.figure(figsize=(10,4))
ax5 = sns.kdeplot(data=covid, x="New cases", color='purple')

# Nouvelles morrts
plt.figure(figsize=(10,4))
ax6 = sns.kdeplot(data=covid, x="New deaths", color='gray')

# New Recovered
plt.figure(figsize=(10,4))
ax7 = sns.kdeplot(data=covid, x="New recovered", color='pink')

#Graphique montrant les 25 pays d'europe avec le + de cas

europe = world[world['Continent'] == 'Europe']
px.pie(europe[:25], values='TotalCases', names='Country/Region',
       title='Top 25 Countries/Regions in Europe')


z_data =covid
z = z_data.values
sh_0, sh_1 = z.shape
x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

fig.update_layout(title='Cas Total')
fig.show()
plt.show()

print(day.head())

print(covid.isnull().sum()) # no null values
print(day.isnull().sum()) # no null values

day['Date']=pd.to_datetime(day['Date']) # converting Data into datetime

day_temp = day.copy()
day_temp = day_temp.drop('Date',axis=1)
print(day_temp.head())
X = day_temp.drop('Confirmed',axis=1).values
y = day_temp['Confirmed'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
X_train.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

from tensorflow.keras.callbacks import EarlyStopping
#early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[])
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

from sklearn.metrics import mean_squared_error, explained_variance_score
predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,predictions)))
print(mean_absolute_error(y_test,predictions))
print(explained_variance_score(y_test,predictions))

conf_data = full[['Date', 'Confirmed']].groupby('Date', as_index = False).sum()
conf_data.columns = ['ds', 'y']
conf_data.ds = pd.to_datetime(conf_data.ds)
print(conf_data.head())
# proph = Prophet()
# print(proph.fit(conf_data))
# confirmed_pred = proph.make_future_dataframe(periods=60)
# print(confirmed_pred.tail())
# confirmed_forecast = proph.predict(confirmed_pred)
# print(confirmed_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# fig1 = proph.plot(confirmed_forecast)
#
# fig2 = proph.plot_components(confirmed_forecast)
#
#
# deaths_data = full[['Date', 'Deaths']].groupby('Date', as_index = False).sum()
# deaths_data.columns = ['ds', 'y']
# deaths_data.ds = pd.to_datetime(deaths_data.ds)
# print(deaths_data.head())
# proph2 = Prophet()
# print(proph2.fit(deaths_data))
#
# deaths_pred = proph2.make_future_dataframe(periods=60)
# print(deaths_pred.tail())
#
# deaths_forecast = proph2.predict(deaths_pred)
# print(deaths_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
#
# fig3 = proph2.plot(deaths_forecast)
# fig4 = proph2.plot_components(deaths_forecast)
plt.show()

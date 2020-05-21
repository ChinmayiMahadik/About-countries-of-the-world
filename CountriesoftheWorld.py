# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:50:03 2020

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup

#importing the dataset through pandas package
countryData = pd.read_csv("countriesoftheworld.csv") 

#checking for null values
print(countryData.isnull().sum())
##############################################################################
# calculating the statistical mean
countryData.describe()
#Frequency per categoty
countryData['Region'].value_counts()
#printing all the columns/header data
for colName in countryData:
    print(colName)

#correlation between every columns
f, ax = plt.subplots(figsize = (15,15))## 
sns.heatmap(countryData.corr(), annot = True, linewidth = 0.5,fmt = ".1f", ax= ax)

# saving to correlation matrix in another pdf file on the local machine
plt.savefig('correlation_plot.pdf') 
##############################################################################
#UNIVARIATE ANALYSIS AND DATA SUMMARY 
##############################################################################

#Region Variable_____________________________
#Frequency per categoty
countryData['Region'].value_counts()

#Bar chart
countryData['Region'].value_counts().plot(kind='bar')
plt.xlabel('Regions')
plt.ylabel('Frequency')

#Population Variable_________________________

#Computes the satistic summary of the data.
countryData['Population'].describe()


#Finding the country with largest population

countryData['Country'].iloc[countryData.index[countryData['Population']==countryData['Population'].max()].tolist()]

#Finding the country with the smallest population

countryData['Country'].iloc[countryData.index[countryData['Population']==countryData['Population'].min()].tolist()]


#Histogram plot

sns.distplot(countryData['Population'])

#Area Variable______________

countryData['Area'].describe()


#Finding the country with largest area

countryData['Country'].iloc[countryData.index[countryData['Area']==countryData['Area'].max()].tolist()]

#Finding the country with the smallest area

countryData['Country'].iloc[countryData.index[countryData['Area']==countryData['Area'].min()].tolist()]

sns.distplot(countryData['Area'])

#Population density variable____________

countryData['Pop. Density'].describe()

#Finfiding the country with  the largest population density
countryData['Country'].iloc[countryData.index[countryData['Pop. Density']==countryData['Pop. Density'].max()].tolist()]

#Finding the country with the smallest population density

countryData['Country'].iloc[countryData.index[countryData['Pop. Density']==countryData['Pop. Density'].min()].tolist()]

#Histogram
sns.distplot(countryData['Pop. Density'])

#CoastLine variable__________________________

countryData['Coastline'].describe()


#Finfiding the country with largest coastline

countryData['Country'].iloc[countryData.index[countryData['Coastline']==countryData['Coastline'].max()].tolist()]

#Finding countries with zero coastline

Countrieswithzerocoastline=0
for i in range(len(countryData)):
    if (countryData['Coastline'].iloc[i]==0):
        Countrieswithzerocoastline+=1
    else:
        pass
print(Countrieswithzerocoastline)

#Histogram
    
sns.distplot(countryData['Coastline'])


#Net migration____________________________________________

countryData['Net migration'].describe()

#Finfiding the country with largest net migration

countryData['Country'].iloc[countryData.index[countryData['Net migration']==countryData['Net migration'].max()].tolist()]

#Finding countries with samllest net migration 

countryData['Country'].iloc[countryData.index[countryData['Net migration']==countryData['Net migration'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Net migration'])


#Infant Mortality_______________________________________________

countryData['Infant mortality'].describe()


#Finfiding the country with largest Infant Mortality_

countryData['Country'].iloc[countryData.index[countryData['Infant mortality ']==countryData['Infant mortality '].max()].tolist()]

#Finding countries with samllest Infant Mortality_

countryData['Country'].iloc[countryData.index[countryData['Infant mortality']==countryData['Infant mortality'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Infant mortality'])

#GDP__________________________________________________________

countryData['GDP'].describe()


#Finfiding the country with largest GDP_

countryData['Country'].iloc[countryData.index[countryData['GDP']==countryData['GDP'].max()].tolist()]

#Finding countries with samllest GDP_
countryData['Country'].iloc[countryData.index[countryData['GDP']==countryData['GDP'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['GDP'])

#Literacy____________________________

countryData['Literacy'].describe()


#Finfiding the country with largest Literacy

countryData['Country'].iloc[countryData.index[countryData['Literacy']==countryData['Literacy'].max()].tolist()]

#Finding countries with samllest Literacy
countryData['Country'].iloc[countryData.index[countryData['Literacy']==countryData['Literacy'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Literacy'])

#Phones_____________________________________________

countryData['Phones'].describe()


#Finfiding the country with largest number of phone per 1000 people

countryData['Country'].iloc[countryData.index[countryData['Phones']==countryData['Phones'].max()].tolist()]

#Finding countries with samllest number of phone per 1000 people

countryData['Country'].iloc[countryData.index[countryData['Phones']==countryData['Phones'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Phones'])

#Arable_____________________________________________

countryData['Arable'].describe()


#Finfiding the country with largest proportion of arable land

countryData['Country'].iloc[countryData.index[countryData['Arable']==countryData['Arable'].max()].tolist()]

#Finding countries with samllest proportion of arable land
countryData['Country'].iloc[countryData.index[countryData['Arable']==countryData['Arable'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Arable'])

#Crops________________________________________________________

countryData['Crops'].describe()


#Finfiding the country with largest proportion of crops land

countryData['Country'].iloc[countryData.index[countryData['Crops']==countryData['Crops'].max()].tolist()]

#Finding countries with samllest proportion of crops land

countryData['Country'].iloc[countryData.index[countryData['Crops']==countryData['Crops'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Crops'])

#OTHER_______________________________________________________

countryData['Other'].describe()

#Finfiding the country with largest proportion of land used for permanent meadows and pastures.

countryData['Country'].iloc[countryData.index[countryData['Other']==countryData['Other'].max()].tolist()]

#Finding countries with samllest proportion of land used for permanent meadows and pastures.

countryData['Country'].iloc[countryData.index[countryData['Other']==countryData['Other'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Other'])

#Climate________________________________________

#Frequency per categoty

countryData['Climate'].value_counts()

#Bar chart
countryData['Climate'].value_counts().plot(kind='bar')
plt.xlabel('Climate')
plt.ylabel('Frequency')

#Birth rate_______________________________________________

countryData['Birthrate'].describe()

#Finfiding the country with largest birth rate

countryData['Country'].iloc[countryData.index[countryData['Birthrate']==countryData['Birthrate'].max()].tolist()]

#Finding countries with samllest birth rate

countryData['Country'].iloc[countryData.index[countryData['Birthrate']==countryData['Birthrate'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Birthrate'])

#Death rate_______________________________________

countryData['Deathrate'].describe()

#Finfiding the country with largest death rate

countryData['Country'].iloc[countryData.index[countryData['Deathrate']==countryData['Deathrate'].max()].tolist()]

#Finding countries with samllest death rate

countryData['Country'].iloc[countryData.index[countryData['Deathrate']==countryData['Deathrate'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Deathrate'])

#Agriculture________________________________________________________

countryData['Agriculture'].describe()


#Finfiding the country with largest proportion of agriculture in its GDP

countryData['Country'].iloc[countryData.index[countryData['Agriculture']==countryData['Agriculture'].max()].tolist()]

#Finding country with samllest proportion of agriculture in its GDP

countryData['Country'].iloc[countryData.index[countryData['Agriculture']==countryData['Agriculture'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Agriculture'])


#Industry______________________________________________________

countryData['Industry'].describe()


#Finfiding the country with largest proportion of industry in its GDP

countryData['Country'].iloc[countryData.index[countryData['Industry']==countryData['Industry'].max()].tolist()]

#Finding country with samllest proportion of industry in iys GDP

countryData['Country'].iloc[countryData.index[countryData['Industry']==countryData['Industry'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Industry'])

#Service______________________________________________________

countryData['Service'].describe()

#Finfiding the country with largest proportion of service in its GDP

countryData['Country'].iloc[countryData.index[countryData['Service']==countryData['Service'].max()].tolist()]

#Finding country with samllest proportion of service in iys GDP

countryData['Country'].iloc[countryData.index[countryData['Service']==countryData['Service'].min()].tolist()]

#Histogram
    
sns.distplot(countryData['Service'])


##############################################################################
#BIVARIATE ANALYSIS#
##############################################################################
#taking log value of population density to normalize data
countryData["log population density"] = np.log(countryData["Pop. Density"])
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="log population density", y="GDP", data=countryData, kind="reg")

##############################################################################
#coverting the categorical coastline data into numeric data 
countryData["catagorical coastline"] = (countryData["Coastline"] == 0).astype(int)
# using seaborn package to plot a boxplot since the data is categorical 
sns.boxplot(x="catagorical coastline", y="GDP", data=countryData, color=".3", linewidth=1)

##############################################################################
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Net migration", y="GDP", data=countryData, kind="reg")

##############################################################################
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Infant mortality", y="GDP", data=countryData, kind="reg")
sns.lmplot(x="Infant mortality", y="GDP", data=countryData,lowess=True)

##############################################################################
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Literacy", y="GDP", data=countryData, kind="reg")

##############################################################################
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Phones", y="GDP", data=countryData, kind="reg")

###################################### not useful
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Birthrate", y="GDP", data=countryData, kind="reg") ## discrete values

###########################################3
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Deathrate", y="GDP", data=countryData, kind="reg")

############################################s############
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Industry", y="GDP", data=countryData, kind="reg")

#############################################################
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Service", y="GDP", data=countryData, kind="reg")

######################################### useless
# using seaborn package to plot a jointplot (combination of scatterplot and bargraph)
sns.jointplot(x="Agriculture", y="GDP", data=countryData, kind="reg")

##############################################################################
# using seaborn package to plot a boxplot since the data is categorical 
sns.boxplot(x="Climate", y="GDP", data=countryData, color=".3", linewidth=1)

##############################################################################
# Graph used in presentation for bivaruate analysis with scatter plot and correlation coefficiant

plt.suptitle('Scatter plot between GDP per capita and factors')
plt.figure(figsize=(18, 10)).suptitle('Scatter plot between GDP per capita and factors', fontsize="x-large")
plt.subplot(331)
plt.scatter(countryData['Phones'],countryData['GDP'])
plt.xlabel('Phones')
plt.ylabel('GDP')
plt.text(500, 50000, 'r = '+ str(countryData['Phones'].corr(countryData['GDP']))) # mentioning the dimensions of the graph

plt.subplot(332)
plt.scatter(countryData['Birthrate'],countryData['GDP'])
plt.xlabel('Birthrate')
plt.ylabel('GDP')
plt.text(30, 50000, 'r = '+ str(countryData['Birthrate'].corr(countryData['GDP'])))

plt.subplot(333)
plt.scatter(countryData['Infant mortality'],countryData['GDP'])
plt.xlabel('Infant mortality')
plt.ylabel('GDP')
plt.text(100, 50000, 'r = '+ str((countryData['Infant mortality']).corr(countryData['GDP'])))


plt.subplot(334)
plt.scatter(countryData['Agriculture'],countryData['GDP'])
plt.xlabel('Agriculture')
plt.ylabel('GDP')
plt.text(0.4, 50000, 'r = '+ str(countryData['Agriculture'].corr(countryData['GDP'])))


plt.subplot(335)
plt.scatter(countryData['Industry'],countryData['GDP'])
plt.xlabel('Industry')
plt.ylabel('GDP')
plt.text(0.4, 50000, 'r = '+ str(countryData['Industry'].corr(countryData['GDP'])))


plt.subplot(336)
plt.scatter(countryData['Service'],countryData['GDP'])
plt.xlabel('Service')
plt.ylabel('GDP')
plt.text(0.6, 50000, 'r = '+ str(countryData['Service'].corr(countryData['GDP'])))


plt.subplot(337)
plt.scatter(countryData['Net migration'],countryData['GDP'])
plt.xlabel('Net migration')
plt.ylabel('GDP')
plt.text(0.6, 50000, 'r = '+ str(countryData['Net migration'].corr(countryData['GDP'])))


plt.subplot(338)
plt.scatter(countryData['Literacy'],countryData['GDP'])
plt.xlabel('Literacy')
plt.ylabel('GDP')
plt.text(60, 50000, 'r = '+ str(countryData['Literacy'].corr(countryData['GDP'])))


plt.subplot(339)
plt.scatter(countryData['Deathrate'],countryData['GDP'])
plt.xlabel('Deathrate')
plt.ylabel('GDP')
plt.text(17, 50000, 'r = '+ str(countryData['Deathrate'].corr(countryData['GDP'])))

plt.savefig('bivariant.pdf')

###############################################################################

import plotly
from plotly.offline import plot
#init_notebook_mode(connected=True)
import plotly.graph_objs as go
import chart_studio.plotly as py


#Population per country

data = dict(type='choropleth',locations = countryData.Country,locationmode = 'country names', z = countryData.Population,text = countryData.Country, colorbar = {'title':'Population'},colorscale = 'Blackbody', reversescale = True)
layout = dict(title='Population per country',geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
plot(choromap,validate=False)

#Climate type per country

data = dict(type='choropleth',locations = countryData.Country,locationmode = 'country names', z = countryData.Climate,text = countryData.Country, colorbar = {'title':'Climate'},colorscale = 'Blackbody', reversescale = True)
layout = dict(title='Climate type per country',geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
plot(choromap,validate=False)


##############################################################################
# MULTIVARIANT ANALYSIS #
##############################################################################
# selecting the required data using seaborn package pairplot
selectedData = countryData[["GDP","Agriculture","Industry","Service","Region"]]
sns.pairplot(selectedData, hue="Region")
plt.savefig('multivariant.pdf')

##############################################################################
# ENERGY EFFICIENCY #
##############################################################################
#URL containing the data set.
url = "https://www.cia.gov/library/publications/resources/the-world-factbook/fields/253rank.html"
#The URL is converted to urlopen to get the html of the page.
html = urlopen(url)

#Creates a Beautiful Soup object to parse the html. The argument 'lxml' is the html parser.
soup = BeautifulSoup(html, 'lxml')


#Gives us the title of the web site 
title = soup.title
print(title)

#Extracts rows 
rows = soup.find_all('tr')
print(rows[:10])

#Finds the tables cells 
for row in rows:
    row_td = row.find_all('td')
print(row_td)
type(row_td)

#Removes the html tags to extract the text.
str_cells = str(row_td)
cleantext = BeautifulSoup(str_cells, "lxml").get_text()
print(cleantext)


#Extracts text in between html tags for each row, and append it to a list.
import re

list_rows = []
for row in rows:
    cells = row.find_all('td')
    str_cells = str(cells)
    clean = re.compile('<.*?>')
    clean2 = (re.sub(clean, '',str_cells))
    list_rows.append(clean2)
print(clean2)
type(clean2)

#Converts the list into a data frame
df = pd.DataFrame(list_rows)
df.head(10)


#Because data is going to be split using the criteria ' ,', some countries that include 
# a ' ,' in their name will introduce an error. This is corrected in the following lines 
#of code.

countrieswitherrors=[9,83,106,145,157,186,195,214]

previousnamelist=['\nKorea, South\n', '\nKorea, North\n', '\nCongo, Democratic Republic of the\n', '\nBahamas, The\n', '\nCongo, Republic of the\n', '\nGambia, The\n', '\nMicronesia, Federated States of\n', '\nSaint Helena, Ascension, and Tristan da Cunha\n']
newnamelist=['Korea South', 'Korea North', 'Congo Democratic Republic of the', 'Bahamas The', 'Congo Republic of the', 'Gambia The','Micronesia Federated States of','Saint Helena Ascension and Tristan da Cunha']


#The following code creates a list with names corrected.
listcountries=[]
for i in range(len(previousnamelist)):       
    x=df.iloc[countrieswitherrors[i]][0].replace(previousnamelist[i], newnamelist[i])
    listcountries.append(x)
   
#This code replace in the data frame the names corrected
for i in range(len(listcountries)):
    df.iloc[countrieswitherrors[i]][0]=listcountries[i] 

  
#Splits in several columns the previous data frame
df1 = df[0].str.split(', ',expand=True) 

df1.head(10)

#Elimiates unnecesary columns. 
del df1[0]
del df1[3]

#Assigns columns names to the data frame.
df1.columns=['Country', 'Energy Consumption']

#Eliminates rows with missing values.
df2=df1.dropna()

#Because the values for the variable energy consumption are expresed as a string
#the following code transforms them into float.
energy=[]
for i in range (len(df2)):
    x=(float(df2.iloc[i][1].replace(',', '')))
    energy.append(x)

df2['Energy Consumption'] = energy


#Importing the data original data set.

countries=pd.read_csv('countriesoftheworld.csv')



#Blanks were detected at the begining and end of the countries names.
#Those blanks needed to be elimitated because will intriduce an error when merging
#both data frames. 

#This code eliminates blanks for the data set countries in the country column.
g=[]
for i in range (len(countries)):
    g.append(countries.iloc[i][0].strip())

countries['Country']=g

#This code eliminates blanks for the data frame df2 in the country column.
t=[]
for i in range (len(df2)):
    t.append(df2.iloc[i][0].strip())

df2['Country']=t



#merging 2 data frames
newcountries = pd.merge(df2,countries, on="Country")
for i in newcountries:
    print(i)

#Because GDP column is actually GDP per capita, its name is being changed 
newcountries=newcountries.rename(columns={'GDP':'GDP/Capita'})
#A new column GDP is created.
newcountries['GDP']=newcountries['GDP/Capita']*newcountries['Population']

#New data frame is created with the columns needed for the regression model.
newcountries3=pd.concat([newcountries['Country'],newcountries['Energy Consumption'],newcountries['Population'],newcountries['GDP']],axis=1)
print(newcountries3)


#Bulting the regression model 

import seaborn as sns

#Creates a scatterplot betwwen the indepent and dependent variable.
sns.jointplot(x='GDP', y='Energy Consumption', data=newcountries3, kind='reg')

#computes the correlation between the indepent and dependent variable.
newcountries3['Energy Consumption'].corr(newcountries3['GDP'])

sns.jointplot(x='Population', y='Energy Consumption', data=newcountries3, kind='reg')
newcountries3['Energy Consumption'].corr(newcountries3['Population'])


from sklearn import linear_model

#To create the gression model it has been definied that 80% of the data will be used
#to train the model 

#Determines number of rows in the data set
rownumber=len(newcountries3)

#Rows need to be randomly shuffled before selecting the data used to train and test the model.
randomlyshuffledrows=np.random.permutation(rownumber)

#Defining rows for training the model
trainingrows=randomlyshuffledrows[0:153]
#Defining rows for testing the model
testrows=randomlyshuffledrows[153:]

#Selecting the data to traing the model 
xtraining=newcountries3.iloc[trainingrows,2:4]
ytraining=newcountries3.iloc[trainingrows,1]

#Selecting the data to test the model 
xtest=newcountries3.iloc[testrows,2:4]
ytest=newcountries3.iloc[testrows,1]

#Creating the regression model
reg=linear_model.LinearRegression()
reg.fit(xtraining,ytraining)

reg.coef_
reg.intercept_

from sklearn.metrics import r2_score

ypredictions=reg.predict(xtest)

#Computig Coefficient of determination
r2=r2_score(ytest,ypredictions)

#Computig Coefficient of determination adjusted

r2adjusted=1-(1-r2)*(len(newcountries3)-1)/(len(newcountries3)-2-1)

#Checking Regression Model assumptions 

import matplotlib.pyplot as plt

#Checking Linearity

plt.scatter(ytest,ypredictions)

#Cheking constance variance and independence 

residuals=ypredictions-ytest

plt.scatter(xtest['GDP'],residuals)

#Checking normality

from scipy import stats
stats.probplot(residuals, plot=plt)

#Transformation log of the data because linear regression asumptions were violated

newcountries3['Energy Consumption log']=np.log(newcountries3['Energy Consumption'])
newcountries3['Population log']=np.log(newcountries3['Population'])
newcountries3['GDP log']=np.log(newcountries3['GDP'])


newcountries3=newcountries3.drop(['Population','Energy Consumption','GDP'], axis=1)
print(newcountries3)

# new regression model with transformed  data 

xtraining=newcountries3.iloc[trainingrows,2:4]
ytraining=newcountries3.iloc[trainingrows,1]

xtest=newcountries3.iloc[testrows,2:4]
ytest=newcountries3.iloc[testrows,1]

reg=linear_model.LinearRegression()
reg.fit(xtraining,ytraining)

reg.coef_
reg.intercept_

from sklearn.metrics import r2_score

ypredictions=reg.predict(xtest)

#Computig Coefficient of determination
r2=r2_score(ytest,ypredictions)

#Computig Coefficient of determination adjusted
r2adjusted=1-(1-r2)*(len(newcountries3)-1)/(len(newcountries3)-2-1)

#Checking Regression Model assumptions 
#Checking Linearity

plt.scatter(ytest,ypredictions)

#Cheking constance variance and independence 

residuals=ypredictions-ytest

plt.scatter(xtest['GDP log'],residuals)

#Checking normality

from scipy import stats
stats.probplot(residuals, plot=plt)


#More regression models
#Correlation among all numerical variables is calculated to define posible regression models.

correlationdf=countries.corr()
print(correlationdf)

import seaborn as sns
sns.heatmap(correlationdf)

#Predicting literacy on the basis of gdp, infant mortality, phones, and birth rate.

countries.columns
countries2=pd.concat([countries['Literacy'],countries['Infant mortality'],countries['GDP'],countries['Phones'],countries['Birthrate']],axis=1)
print(countries2)

#Determines number of rows in the data set
rownumber=len(countries2)

#Rows need to be randomly shuffled before selecting the data used to train and test the model.
randomlyshuffledrows=np.random.permutation(rownumber)

#To create the gression model it has been definied that 80% of the data will be used
#to train the model 

#Defining rows for training the model
trainingrows=randomlyshuffledrows[0:175]
#Defining rows for testing the model
testrows=randomlyshuffledrows[175:]

#Selecting the data to traing the model 
xtraining=countries2.iloc[trainingrows,1:5]
ytraining=countries2.iloc[trainingrows,0]

#Selecting the data to test the model 
xtest=countries2.iloc[testrows,1:5]
ytest=countries2.iloc[testrows,0]

#Creating the regression model

from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(xtraining,ytraining)

reg.coef_
reg.intercept_

from sklearn.metrics import r2_score

ypredictions=reg.predict(xtest)

#Computig Coefficient of determination
r2=r2_score(ytest,ypredictions)

#Computig Coefficient of determination adjusted
r2adjusted=1-(1-r2)*(len(countries2)-1)/(len(countries2)-4-1)

#Checking Regression Model assumptions 

import matplotlib.pyplot as plt

#Checking Linearity

plt.scatter(ytest,ypredictions)

#Cheking constance variance and independence 

residuals=ypredictions-ytest

plt.scatter(xtest['GDP'],residuals)

#Checking normality

from scipy import stats
stats.probplot(residuals, plot=plt)


#Predicting infant mortality on the basis of gdp, literacy, agriculture, birth rate, and phones.


countries.columns
countries3=pd.concat([countries['Infant mortality'],countries['Literacy'],countries['GDP'],countries['Phones'],countries['Birthrate'],countries['Agriculture']],axis=1)
print(countries3)

#Determines number of rows in the data set
rownumber=len(countries3)

#Rows need to be randomly shuffled before selecting the data used to train and test the model.
randomlyshuffledrows=np.random.permutation(rownumber)

#To create the gression model it has been definied that 80% of the data will be used
#to train the model 

#Defining rows for training the model
trainingrows=randomlyshuffledrows[0:175]
#Defining rows for testing the model
testrows=randomlyshuffledrows[175:]

#Selecting the data to traing the model 
xtraining=countries3.iloc[trainingrows,1:6]
ytraining=countries3.iloc[trainingrows,0]

#Selecting the data to test the model 
xtest=countries3.iloc[testrows,1:6]
ytest=countries3.iloc[testrows,0]

#Creating the regression model

from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(xtraining,ytraining)

reg.coef_
reg.intercept_

from sklearn.metrics import r2_score

ypredictions=reg.predict(xtest)

#Computig Coefficient of determination
r2=r2_score(ytest,ypredictions)

#Computig Coefficient of determination adjusted
r2adjusted=1-(1-r2)*(len(countries3)-1)/(len(countries3)-5-1)

#Checking Regression Model assumptions 
import matplotlib.pyplot as plt

#Checking Linearity

plt.scatter(ytest,ypredictions)

#Cheking constance variance and independence 

residuals=ypredictions-ytest

plt.scatter(xtest['GDP'],residuals)

#Checking normality

from scipy import stats
stats.probplot(residuals, plot=plt)

##############################################################################
# GDP PREDICTION #
##############################################################################
##### correlation

f, ax = plt.subplots(figsize = (15,15))## 
sns.heatmap(countryData.corr(), annot = True, linewidth = 0.5,fmt = ".1f", ax= ax)
plt.savefig('correlation_plot2.pdf')  
# Dropping columns which has less collinearity to GDP data based on EDA
for colName in countryData:
    print(colName)
countryData = countryData.drop(columns =['Country','Region','Population','Area','Arable','Pop. Density',
                                         'Coastline','Crops','Other','Deathrate','Industry'])
## Transform Climate into indicator variables
countryData["isClimate1"] = (countryData["Climate"]==1).astype(int)
countryData["isClimate2"] = (countryData["Climate"]==2).astype(int)
countryData["isClimate3"] = (countryData["Climate"]==3).astype(int)
countryData["isClimate4"] = (countryData["Climate"]==4).astype(int)
del countryData["Climate"]
countryData['GDP1'] = countryData['GDP']
del countryData['GDP']

##############################################################################
##  MODEL: 1
from sklearn import linear_model
reg = linear_model.LinearRegression()
countryData.shape
for colName in countryData:
    print(colName)
numberRows = len(countryData)
#Rows need to be randomly shuffled before selecting the data
randomlyShuffledRows = np.random.permutation(numberRows)
#Defining rows for training the model
trainingRows = randomlyShuffledRows[0:190]
#Defining rows for testing the model
testRows = randomlyShuffledRows[190:]
#Selecting the data to traing the model 
xTrainingData = countryData.iloc[trainingRows,0:13]
yTrainingData = countryData.iloc[trainingRows,13]
#Selecting the data to test the model 
xTestData = countryData.iloc[testRows,0:13]
yTestData = countryData.iloc[testRows,13]
#Creating the regression model
reg.fit(xTrainingData,yTrainingData)
print(reg.coef_)
print(reg.intercept_)
predictions = reg.predict(xTestData)
errors = (predictions-yTestData)
#Computig Coefficient of determination
from sklearn.metrics import r2_score
r2 = r2_score(yTestData,predictions)     
print(r2)
## Checking for significant values
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Net migration'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Infant mortality'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Literacy'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Phones'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Birthrate'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Agriculture'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Service'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['log population density'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['catagorical coastline'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate1'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate2'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate3'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate4'],countryData['GDP1'])
print(p_value)
## Residual analysis (Making a dataframe of errors and prediction values and plotting)
er = pd.DataFrame(errors)
pr = pd.DataFrame(predictions)
sns.jointplot(x=pr[0], y=er["GDP1"], kind="reg")
## Probability plot
stats.probplot(errors, plot=plt)

##############################################################################
# correlation between every columns
f, ax = plt.subplots(figsize = (15,15))## 
sns.heatmap(countryData.corr(), annot = True, linewidth = 0.5,fmt = ".1f", ax= ax)
#plt.show()
# Based on p value and tests for assumptions we discarded few values to reduce multicolinearity
#countryData = countryData.drop(columns =['log population density'])
countryData = countryData.drop(columns =['Infant mortality'])
countryData = countryData.drop(columns =['Birthrate'])
countryData = countryData.drop(columns =['Agriculture'])
# Checking remaining rows
for colName in countryData:
    print(colName)

#############################################
##  MODEL: 2
from sklearn import linear_model
reg = linear_model.LinearRegression()
countryData.shape
for colName in countryData:
    print(colName)
numberRows = len(countryData)
#Rows need to be randomly shuffled before selecting the data
randomlyShuffledRows = np.random.permutation(numberRows)
#Defining rows for training the model
trainingRows = randomlyShuffledRows[0:190]
#Defining rows for testing the model
testRows = randomlyShuffledRows[190:]
#Selecting the data to traing the model 
xTrainingData = countryData.iloc[trainingRows,0:9]
yTrainingData = countryData.iloc[trainingRows,9]
#Selecting the data to test the model 
xTestData = countryData.iloc[testRows,0:9]
yTestData = countryData.iloc[testRows,9]
#Creating the regression model
reg.fit(xTrainingData,yTrainingData)
print(reg.coef_)
print(reg.intercept_)
predictions = reg.predict(xTestData)
errors = (predictions-yTestData)
#Computig Coefficient of determination
from sklearn.metrics import r2_score
r2 = r2_score(yTestData,predictions)
print(r2)
## Checking for significant values
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Net migration'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Literacy'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Phones'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['Service'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['catagorical coastline'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate1'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate2'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate3'],countryData['GDP1'])
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(countryData['isClimate4'],countryData['GDP1'])
print(p_value)
## Residual analysis (Making a dataframe of errors and prediction values and plotting)
er = pd.DataFrame(errors)
pr = pd.DataFrame(predictions)
sns.jointplot(x=pr[0], y=er["GDP1"], kind="reg")
## Probability plot
stats.probplot(errors, plot=plt)
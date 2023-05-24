#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
font = {'size':12}
plt.rc('font', **font)


# In[2]:


#leer analizar y corregir datos NaN en Dataset

os.chdir(r"C:\Users\chris\Downloads\1 Cursos  UNAD\52 ANÁLISIS DE DATOS\Tarea 3\Jupyter Python DS_Titanic") #se pone una r al inicio para que python lo tome como un directorio y no genere error
datos=pd.read_csv("Titanic.csv", delimiter=',')
datos.head()


# In[3]:


#Verificar tipo de datos
datos.dtypes


# In[4]:


#conteo de columnas y saca datos estadisticos (conteo, mediana, cuartiles etc)
datos.describe()


# In[5]:


#isna: devuelve datos na como true/false y con esto hacemos una suma
datos.isna().sum()


# In[6]:


#rellenamos los na en Edad por la media de edades
mediaEdades=round(datos['Age'].mean())
datos['Age']=datos['Age'].fillna(mediaEdades)
datos.isna().sum()


# In[7]:


datos['Cabin']=datos['Cabin'].fillna("NE")
datos.isna().sum()


# In[8]:


#hacemos lo mismo con Embarked rellenamos los na
datos['Embarked']=datos['Embarked'].fillna("NE")
datos.isna().sum()


# In[9]:


#conteo de los datos de Cabin
datos['Cabin'].value_counts()


# In[10]:


#conteo de los datos de Embarked
datos['Embarked'].value_counts()


# In[11]:


datos.head()


# In[12]:


#Para mejor visualizacion de la data hacemos algunos ajustes:
datos['Survived']=datos['Survived'].map({
    0:'No',
    1:'Yes'
    
})
datos.head()


# In[13]:


#Para mejor visualizacion de la data hacemos algunos ajustes:
datos['Embarked']=datos['Embarked'].map({
    'S':'Southampton',
    'C':'Cherbourg',
    'Q':'Queenstown'
})
datos.head()


# In[14]:


#luego de preparar la data revisamos
datos.groupby(['Pclass','Survived'])['Survived'].count()
ax=sns.countplot(x='Pclass', hue='Survived', palette='Set1', data=datos)
ax.set(title='Estado del pasajero (murio/sobrevivio) dado a la clase a la que pertenecia',
      xlabel='Clase del pasajero', ylabel='Total')
#Agregar etiquetas de datos:
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')
plt.show()


# In[15]:


datos.groupby(['Sex', 'Survived'])['Survived'].count()


# In[16]:


ax=sns.countplot(x='Sex', hue='Survived', palette='Set1', data=datos)
ax.set(title='Total de pasajeros con respecto al sexo',
      xlabel='Sexo', ylabel='Total')
#Agregar etiquetas de datos:
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')
plt.show()


# In[17]:


# crear varios tipos de gráficos categóricos en una única figura
ax=sns.catplot(x='Pclass', hue='Sex', col='Survived', palette='Set1',
              data=datos, kind='count')
plt.show()


# In[18]:


#mostrar etiquetas a las barras
def autolabel(bars):
    for bar in bars:
        height=bar.get_height()
        ax.annotate('{}'.format(height),
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0,3),
        textcoords="offset points",
        ha='center', va='bottom')


# In[19]:


#presenta una tabla dinamica con datos de la edad media por cabina
aux=datos.pivot_table(values='Age', index='Cabin', aggfunc='mean')
aux


# In[20]:


datos.groupby('Cabin').filter(lambda x: (x['Cabin']=="F4").any())
datos.groupby('Cabin').filter(lambda x: (x['Cabin'] =="F4").any())['Age']


# In[21]:


#muestra las primeras 5 cabinas
aux.index[:5].to_list()


# In[22]:


#edad primeras 5 cabinas
aux[:5]['Age'].to_list()


# In[23]:


#Grafica Edad promedio de las primeras 5 cabinas
fig, ax=plt.subplots()
ax.set_ylabel('Edad')
ax.set_title('Edades promedio en las diferentes cabinas')
bar1=ax.bar(aux.index[:5].to_list(), aux[:5]['Age'].to_list())

autolabel(bar1)
plt.show()


# In[24]:


#Lugares de embarque vs si sobrevivio o no
pd.crosstab(datos['Embarked'], datos['Survived'])


# In[25]:


#Se grafican los datos
ax=sns.countplot(x='Embarked', hue='Survived', data=datos)
ax.set(title='Distribucion de supervivencia segun lugar de embarque',
      xlabel='Lugar', ylabel='Total')
plt.show()


# In[26]:


#conteo de personas por cabina
datos['Cabin'].groupby(datos['Cabin']).count()


# In[27]:


#conteo Personas menores de 18 años
datos[datos['Age']<18]['Age'].count()


# In[28]:


#cantidad de personas menores de 18 años por clase
intervaloEdad1=datos[datos['Age']<18].pivot_table(values='Age', index='Pclass', aggfunc='count')
intervaloEdad1


# In[29]:


#cantidad de personas mayores igual de 18 años y menoresd e 50 años por clase
intervaloEdad2=datos[(datos['Age']>=18) & (datos['Age']<=50)].pivot_table(values='Age', index='Pclass', aggfunc='count')
intervaloEdad2


# In[30]:


#cantidad de personas mayores 50 años  por clase
intervaloEdad3=datos[datos['Age']>50].pivot_table(values='Age', index='Pclass', aggfunc='count')
intervaloEdad3


# In[31]:


#etiquetas grafica pastel/pie
def funcPie(values):
    val=iter(values)
    return lambda pct: f"{pct:.1f}% ({next(val)})"


# In[32]:


#se grafican los pies con los datos anteriores
fig, ax=plt.subplots(1, 3, figsize = (16, 7))
ax[0].pie(intervaloEdad1['Age'].to_list(), labels=intervaloEdad1.index.to_list(), 
        autopct=funcPie(intervaloEdad1['Age'].to_list()), shadow=True, startangle=90)
ax[0].axis('equal')
ax[0].set_title('Edades menores a 18')  
ax[1].pie(intervaloEdad2['Age'].to_list(), labels=intervaloEdad2.index.to_list(),
        autopct=funcPie(intervaloEdad2['Age'].to_list()), shadow=True, startangle=90)
ax[1].axis('equal') 
ax[1].set_title('Edades mayores o iguales a 18 y menores o iguales a 50')  
ax[2].pie(intervaloEdad3['Age'].to_list(), labels=intervaloEdad3.index.to_list(),
        autopct=funcPie(intervaloEdad3['Age'].to_list()), shadow=True, startangle=90)
ax[2].axis('equal')  
ax[2].set_title('Edades mayores a 50') 
plt.legend()
plt.show()


# In[33]:


#devuelve nombres que contengan Carter
datos[datos['Name'].str.match("Carter"+ r'\b', case=False)]['Name']


# In[46]:


#lista de 5 familias apellidos
familias=[]
for i in datos['Name']:
    apellido=str(i).split(',')[0]
    f=datos[datos['Name'].str.match(apellido+ r'\b', case=False)]['Name'].to_list()
    if(familias.count(f)==0):
        familias.append(f)
familias[:5]


# In[35]:


#cantidad de familias
len(familias)


# In[36]:


#lista 5 familias que comparten apellidos
for i in range(0, len(familias)-1):
    for j in range(i+1, len(familias)):
        if len(familias[j])>len(familias[i]):
            aux=familias[i]
            familias[i]=familias[j]
            familias[j]=aux
familias[:5]


# In[37]:


# conteto de integrantes por cada familia mas numerosa
cont=[]
nombreFamilias=[]
for i in familias[:3]:
    aux=""
    cont.append(len(i))
    for j in i:
        aux+=j+'\n'
    nombreFamilias.append(aux)

fig, ax=plt.subplots(figsize = (18, 7))
ax.set_ylabel('Total')
ax.set_title('Familias mas numerosas')
bar1=ax.bar(nombreFamilias, cont)
autolabel(bar1)
plt.show()


# In[38]:


#detalle de informacion por edad igual a 50 años
aux=datos[['Age', 'Sex', 'Pclass', 'Survived']].groupby('Age').filter(lambda x: (x['Age']==50).any())
aux


# In[39]:


#tabla dinamica con detalle por edad, si sobrevivio o no, y clase
aux.pivot_table(index='Sex', columns=['Survived', 'Pclass'], aggfunc='count').fillna(0)


# In[40]:


#agrupa las personas por clase y sexo
aux=datos.groupby(['Pclass', 'Sex'])['Pclass'].count()
aux


# In[41]:


#indice del resultado anterior
aux.index


# In[42]:


#indice de la clase 1 por sexo
aux[1]


# In[43]:


#se grafican en pies ls datos de la cantidad de personas en clases 1, 2 y 3 
fig, ax=plt.subplots(1, 3, figsize = (16, 7))
ax[0].pie(aux[1].to_list(), labels=aux[1].index.to_list(), 
              autopct=funcPie(aux[1].to_list()), shadow=True, startangle=90)
ax[0].axis('equal')
ax[0].set_title('Total hombres y mujeres clase 1')
ax[1].pie(aux[2].to_list(), labels=aux[2].index.to_list(), 
              autopct=funcPie(aux[2].to_list()), shadow=True, startangle=90)
ax[1].axis('equal')
ax[1].set_title('Total hombres y mujeres clase 2')
ax[2].pie(aux[3].to_list(), labels=aux[3].index.to_list(), 
              autopct=funcPie(aux[3].to_list()), shadow=True, startangle=90)
ax[2].axis('equal')
ax[2].set_title('Total hombres y mujeres clase 3')
plt.legend()
plt.show()


# In[44]:


#Diagrama de dispersion se grafica edad y tarifa cantidad de personas
fig, ax=plt.subplots(figsize = (20, 5))
ax.scatter(datos['Age'], datos['Fare'])
ax.set_xlabel('Edad')
ax.set_ylabel('Tarifa')
ax.set_title('Relacion entre edad y la tarifa')
plt.show()


# In[48]:


#Crea 3 graficos de dispersion relacionando la tarifa entre clase 1, 2 y 3 y edad 
clase1=datos[['Age', 'Fare', 'Pclass']].groupby('Pclass').filter(lambda x: (x['Pclass']==1).any())
clase2=datos[['Age', 'Fare', 'Pclass']].groupby('Pclass').filter(lambda x: (x['Pclass']==2).any())
clase3=datos[['Age', 'Fare', 'Pclass']].groupby('Pclass').filter(lambda x: (x['Pclass']==3).any())
fig, ax = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)
ax[0].scatter(clase1['Age'], clase1['Fare'])
ax[0].set_xlabel('Edad')
ax[0].set_ylabel('Tarifa')
ax[0].set_title('Relacion entre edad y la tarifa clase 1')
ax[0].grid(True)
ax[1].scatter(clase2['Age'], clase2['Fare'])
ax[1].set_xlabel('Edad')
ax[1].set_ylabel('Tarifa')
ax[1].set_title('Relacion entre edad y la tarifa clase 2')
ax[1].grid(True)
ax[2].scatter(clase3['Age'], clase3['Fare'])
ax[2].set_xlabel('Edad')
ax[2].set_ylabel('Tarifa')
ax[2].set_title('Relacion entre edad y la tarifa clase 3')
ax[2].grid(True)
plt.show()


# In[49]:


datos.head()


# In[50]:


#eliminamos datos no numericos para hacer mapa de calor
datos.drop(['Name','Ticket','PassengerId','Cabin'], 1, inplace=True)
datos.head()


# In[51]:


#limpiamos datos cambiando No, Yes, Female, Male, etc por datos binarios numericos como 0.1.2
datos['Survived'].replace(('No', 'Yes'), (0, 1), inplace=True)
datos['Sex'].replace(('male', 'female'), (0, 1), inplace=True)
datos['Embarked'].replace(('Cherbourg','Queenstown','Southampton'), (0, 1, 2), inplace=True)
datos.head()


# In[52]:


#finalmente graficamos un  mapa de calor que muestre la relacion entre variables 
plt.figure(figsize=(14,12))
sns.heatmap(datos.corr(), linewidths=0.1, square=True,  cmap='Blues', annot=True)
plt.show()


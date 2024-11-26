import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador desde archivos
with open('modelo_regres_logistica.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler_regres_logistica.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Título de la aplicación
st.title('Predicción de si un cliente contratará (1) o no un depósito bancariao (Regresión Logística)')

# Entrada de datos del usuario
housing = int(st.selectbox('¿Tiene hipoteca? (1: Sí, 0: No)',options=[1, 0]))
balance=float(st.number_input('¿Cuál es el saldo promedio anual en la cuenta bancaria?'))
cat_poutcome_success= int(st.selectbox('¿Contrató productos en campañas anteriores? (1: Sí, 0: No)',options=[1, 0]))

# Crear un DataFrame con las entradas
user_data = pd.DataFrame({
     'housing': [housing],
     'balance': [balance],
     'cat_poutcome_success': [cat_poutcome_success]
 })

#Para un único valor numérico que se le pase a scaler.transform debo recuperar primero los valores de user_data['balance] con user_data['balance'].values
# y luego hacer reshape(-1,1) para que tenga dos dimensiones tal como necesita scaler.transform
balance_np=user_data['balance'].values.reshape(1,-1)
balance_standarized = scaler.transform(balance_np)

#Canvio el valor de balance del DataFrame por el estandarizado
user_data['balance']=balance_standarized 

# Cuando cree el modelo (modelo.fit) le pase un numpy array de las características a entrenar para no tenr conflicos de los nombres de columna. 
#Por este motivo, ahora le dobo pasar un numpy array utilizando el atributo values

data = user_data.values
# Realizar la predicción
y_pred = model.predict(data)

# Mostrar la predicción
st.write(f'Predicción de la contratación del depósito: {y_pred[0]}')



import streamlit as st

import pandas as pd

from sklearn import datasets

import matplotlib.pyplot as plt 

from pycaret.classification import load_model, predict_model

modelo = load_model('Melhor Modelo para Custos')

st.title('Plano de Saúde Deploy Center')

st.sidebar.title('Menu Lateral')

idade = st.sidebar.number_input('Entre com sua idade', 18, 65, 20, 1)
imc = st.sidebar.slider('Entre com o seu IMC:', 18, 45, 25, 1)
sexo = st.sidebar.selectbox('Entre com o sexo:', ['male', 'female'])
criancas = st.sidebar.slider('Número de crianças:', 0, 5, 0, 1)
fumante = st.sidebar.selectbox('Fumante?', ['yes', 'no'])
regiao =  st.sidebar.selectbox('Região:', ['southeast', 'southwest', 'northeast', 'northwest'])


dicionario = {'age': [idade],
			  'sex': [idade],
			  'bmi': [imc],		
			  'children': [criancas],
			  'region': [regiao],
			  'smoker': [fumante]}

dados = pd.DataFrame(dicionario)

saida = predict_model(modelo, dados)

if st.button('Aplicar o Modelo'):
	pred = float(saida['Label'].round(2))

	s1 = 'Custo Estimado do Seguro por Ano: $ {:.2f}'.format(pred)
	st.markdown('### **' + s1 + '**')

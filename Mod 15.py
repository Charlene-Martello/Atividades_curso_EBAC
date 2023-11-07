import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title = 'MÓDULO XV', #titulo da página
                   page_icon= 'https://yt3.googleusercontent.com/ytc/APkrFKabi8A0AVZoNe4H7jDNUIsRknnsjPjVYvVypI0hoQ=s900-c-k-c0x00ffffff-no-rj', #usa um ícone para a página copiando o endereço da imagem
                   layout='wide') #ajusta melhor as imagens à página


#criando 3 colunas pois quero centralizar a imagem e para isso irei inseri-la na coluna 2
col1, col2, col3 = st.columns(3)
imagem = Image.open('streamlit.png')
#inserindo na coluna 2
col2.image(imagem, use_column_width=True)

st.markdown("<h1 style='text-align: center; color: purple;'>EXPERIENCIANDO STREAMLIT</h1>", unsafe_allow_html=True) #alterando a cor do título

st.write('''Para inciar a nossa experiência vamos carregar alguns dados para análise, 
        para tanto utilizaremos a função load_data que baixa os dados, coloca-os em um data frame do Pandas e converte a coluna de data do 
         texto para data e hora. A fim de evitar que os dados sejam recarregados toda vez que o app atualizar, deixaremos salvos em cache,
         utilizando a função '@st.cache_data'.''')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data #para não recarregar os dados do df toda vez, deixamos em cache, poupando tempo
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

#Crie um elemento de texto e informe ao leitor que os dados estão sendo carregados.
data_load_state = st.text('Carregando dataframe...')
#Carregue 10.000 linhas de dados no dataframe.
data = load_data(10000)
#Notifique o leitor de que os dados foram carregados com sucesso.
data_load_state.text("Pronto! (using st.cache_data)")

if st.checkbox('Mostrar dados brutos'):
    st.subheader('Impressão dos dados brutos:')
    st.write(data)

st.subheader('Histograma para ver quais são os horários de maior movimento do Uber na cidade de Nova York:')
st.write('Inicialmente agrupamos os tempos de coleta por hora (formato 24h) e posteriormente utilizamos ''st.bar_chart'' para desenhar o histograma.')
#divide os tempos de coleta agrupados por hora, formato 24h:
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values) 
st.write('Como conclusão do histograma vemos que o horário de pico é 17h.')

st.subheader('Mapa de todas as coletas')
st.map(data) #utilizando a latitude a longitude ele sobrepoe os dados no mapa. 

st.write('Agora vamos redesenhar o mapa para mostrar a concentração de captadores do Uber às 17h.')
hour_to_filter = st.slider('Selecione o horário desejado:', 0, 23, 17) #permite selecionar o horário desejado arrastando a barra
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Mapa das coletas às {hour_to_filter}:00')
st.map(filtered_data)



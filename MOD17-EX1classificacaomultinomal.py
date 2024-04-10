#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # Árvores II - Tarefa I

# 
# ![image.png](attachment:f65bd82d-aa56-4cf2-8030-bb758771f332.png)  
# [fonte](https://momentum.org/programs-services/manage-your-money/savings-app/pxfuel-creative-commons-zero-cc0-iphone-smartphone-cell-phone-mobile-technology-texting/)
# 
# Neste exercício vamos trabalhar com a base de dados de identificação de atividade humana com smartphones. Smartphones possuem acelerômetro e giroscópio, que registram dados de aceleração e giro nos eixos X, Y e Z, com frequencia de 50 Hz (ou seja, 50 registros por segundo). Os dados foram agrupados a medidas de 0.3 Hz, calculando-se variáveis derivadas como aceleração mínima, máxima, média etc por eixo no período agrupado de tempo, resultando em 561 variáveis que podem ser lidas nas bases disponíveis.
# 
# A base é oriunda de um experimento, em que os indivíduos realizavam uma de seis atividades corriqueiras:
# - andando
# - subindo escada
# - descendo escada
# - parado
# - sentado
# - deitado
# 
# O objetivo é classificar a atividade humana com base nos dados do acelerômetro e giroscópio do celular.

# ### 1. Carregar a base
# 
# Sua primeira atividade é carregar a base.
# 
# Ela está disponível neste link:
# https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# 
# dados https://archive.ics.uci.edu/ml/machine-learning-databases/00240/  
# dataset.names https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names  
# UCI HAR Dataset.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
# 
# Você vai encontrar os seguintes arquivos:
# 
# - **features.txt:** contém a lista com os nomes das variáveis
# - **features_info.txt:** contém a descrição das variáveis
# - **README.txt:** contém uma descrição do estudo e das bases
# - **activity_labels:** contém o código da atividade (inteiro entre 1 e 6) e a descrição
# - **subject_train.txt:** uma lista indicando que registro pertence a que indivíduo na base de treino
# - **X_train.txt:** as *features* (ou variáveis explicativas) da base de testes. Cada linha representa um registro das informações de um indivíduo em um intervalo de tempo de aproximadamente 1/0.3 segundo. As medidas estão em ordem temporal dentro do estudo, e o indivíduo que originou a medida está identificado na base subject_train.txt.
# - **y_train.txt:** Possui o mesmo número de linhas que X_test. Contém um número de 1 a 6 indicando a atividade que estava sendo realizada por aquele registro na base de treino.
# - **subject_test.txt:** uma lista indicando que registro pertence a que indivíduo na base de teste
# - **X_test.txt:** as *features* (ou variáveis explicativas) da base de testes. Cada linha representa um registro das informações de um indivíduo em um intervalo de tempo de aproximadamente 1/0.3 segundo. As medidas estão em ordem temporal dentro do estudo, e o indivíduo que originou a medida está identificado na base subject_test.txt.
# - **y_test.txt:** Possui o mesmo número de linhas que X_train. Contém um número de 1 a 6 indicando a atividade que estava sendo realizada por aquele registro na base de teste.
# 
# Carregue as bases:
# 
# 1. Faça o download dos arquivos.
# 2. Carregue a base ```features.txt``` em uma *Series* (por exemplo usando o comando ```pd.read_csv()```.
# 3. Carregue a base subject_train.txt em uma *Series*
# 4. Carregue a base X_train.txt
#     1. Faça com que as colunas deste *dataframe* tenham os nomes indicados em ```features.txt```
#     2. Sem alterar a ordem dos *dataframes*, coloque o indicador do indivíduo lido em ```subject_train.txt``` como uma variável a mais neste *dataframe***
#     3. Faça com que este *dataframe* tenha um índice duplo, composto pela ordem dos dados e pelo identificador do indivíduo
# 5. Com a mesma lógica, carregue a base X_test
#     1. Certifique-se de que tenha os nomes lidos em ```features.txt```
#     2. Coloque o identificador do sujeito lido em ```subject_test.txt```
#     3. Defina um índice duplo composto pela ordem do registro e o identificador do registro
# 6. Salve as bases em arquivos CSV para facilitar a leitura deles na terefa 2 deste módulo
# 7. Considere que esta base é maior que a da aula, tanto em linhas quanto em colunas. Selecione apenas as três primeiras colunas da base ('tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y' e 'tBodyAcc-mean()-Z'), para efeitos desse exercício.
# 
# **OBS:** As bases já estão divididas em treino e teste, de modo que não vamos precisar da função ```train_test_split```.

# In[36]:


#importando pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[3]:


#observar o feature
features= pd.read_csv('features.txt', delim_whitespace=True)
features.head()


# Aqui podemos perceber que nosso feature contém um índice numérico na primeira coluna, ou seja, na coluna de índice 0, por isso, precisamos que a leitura desse dataframe se dê apenas na coluna de índice 1. Da mesma forma não deve se assumir que a primeira linha é uma coluna, pois agora precisamos ajustar todas como variável para somente depois colocá-las como índice das colunas.

# In[4]:


#observar o X_train
X_train = pd.read_csv('X_train.txt', delim_whitespace=True)
X_train.head()


# No X_train, também temos a primeira linha como índice, mas não é, por isso, faremos a releitura do arquivo.

# In[5]:


#2-Carregando a base feature.txt
features= pd.read_csv('features.txt', header=None, delim_whitespace=True, usecols=[1]) 
#3-Carregando a base subject_train.txt
subject_train= pd.read_csv('subject_train.txt', header=None)
#4-carregando a base X_train.txt
X_train = pd.read_csv('X_train.txt', header=None, delim_whitespace=True)


# header=None -> Não assumir que a primeira linha do arquivo contém nomes de colunas.
# 
# delim_whitespace=True -> separador no arquivo é um espaço em branco;
# 
# usecols=[1] -> significa que apenas a segunda coluna será lida e carregada no DataFrame resultante.

# Abaixo vamos visualizar como se comporta cada um deles agora que fizemos os ajustes. 

# In[6]:


features.head()


# In[7]:


subject_train.head()


# In[8]:


X_train.head()


# Agora que todos os itens acima foram lidros da maneira que deveriam ser, podemos prosseguir juntando-os conforme pede o exercício de número 4:

# In[9]:


#4)a)Fazendo com que as colunas de X_train tenham os nomes indicados no feature:
X_train.columns = features[1] #[1] porque a primeira coluna é de índice, então usamos a 2ª. 
#b)adicionando subject_train como uma coluna a mais no x_train:
X_train['subject'] = subject_train 


# In[10]:


X_train.head(2)


# Recapitulando os passos até aqui.... 1) Lemos corretamente os DF features, subject_train e X_train. Em seguida, adicionamos o feature como índice no X_train e por fim, adicionamos o subject como uma coluna a mais no X_train. Agora, essa coluna a mais passará a ser índice também, sendo ela o identificador do indivíduo:

# In[11]:


#c)criar um índice duplo no DataFrame: o índice normal + identificador do indivíduo
X_train.index = pd.MultiIndex.from_tuples(list(zip(range(len(X_train)), X_train['subject'])))
X_train.head(2)


# Agora que os dados de subject estão como um índice, podemos excluí-la do nosso DF:

# In[12]:


#Excluindo a coluna subject
X_train = X_train.drop(columns='subject')
X_train.head()


# Agora, iremos repetir todo o processo anterior com a base X_test. Lembrando que não precisamos carregar novamente a base feature, pois como ela dá nome às colunas, é a mesma para os dois.

# In[13]:


#5#carregando subject_test
subject_test= pd.read_csv('subject_test.txt', header=None)
#carregando X_test
X_test = pd.read_csv('X_test.txt', header=None, delim_whitespace=True)
#Fazendo com que as colunas tenham os nomes indicados no feature
X_test.columns = features[1]
#adicionando subject_test como uma coluna a mais no X_test
X_test['subject'] = subject_test


# In[14]:


X_test.head(2)


# In[15]:


#criar um índice duplo no DataFram: índice normal + identificador do indivíduo
X_test.index = pd.MultiIndex.from_tuples(list(zip(range(len(X_test)), X_test['subject'])))
X_test.head(2)


# In[16]:


#Agora que subject está como um índice, podemos excluí-la do df. 
X_test = X_test.drop(columns='subject')
X_test.head()


# Refeito todo o processo anterior, agora vamos em ambas, X_train e X_test, selecionar apenas as três primeiras colunas, conforme pede o exercício e, posteriormente, salvaremos as novas bases em arquivos csv.

# In[17]:


#7('tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y' e 'tBodyAcc-mean()-Z')
X_treino= X_train[['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z']]
X_teste= X_test[['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z']]


# In[18]:


#6)Salve as bases em arquivos CSV.
X_treino.to_csv("X_treino.csv")
X_teste.to_csv("X_teste.csv")
X_treino.head(2)


# In[19]:


X_teste.head(2)


# Agora, preisamos realizar a leitura também dos y_treino e teste para uso posterior. Lembrando que y pode possuir seis saídas, que são números inteiros de 1 a 6, onde cada um deles representa um tipo específico de atividade física. 

# In[23]:


y_treino = pd.read_csv('y_train.txt', header=None, delim_whitespace=True)
y_teste = pd.read_csv('y_test.txt', header=None, delim_whitespace=True)


# In[24]:


y_treino.head()


# ### 2. Ajustar uma árvore de decisão
# 
# - 2.1 ajuste uma árvore de decisão com número mínimo de observações por folha = 20. Certifique-se de que você esteja utilizando apenas as 3 variáveis indicadas no exercício anterior.
# - 2.2 Calcule os ccp_alphas. Como feito em aula, certifique-se de que todos os valores são positivos, e selecione apenas valores únicos.
# - 2.3 Considere que vamos fazer uma árvore para cada valor de ```ccp_alpha```. Para ter um pouco mais de velocidade, crie uma coleção de dados com os ```ccp_alphas```, considerando apenas 1 a cada 5 valores. Dica: utilize o slicing do tipo ```array[::5]``` para isto. Caso se sinta seguro, fique à vontade para utilizar mais valores de ```ccp_alpha```.

# No exercício anterior criamos a variável X_treino e X_teste, colocando nelas apenas as três primeiras colunas. Agora, utilizando X_treino, iremos ajustar uma árvore com número MINIMO de observações por folha igual a 20. A nossa ideia aqui é treinar um modelo capaz de classificar qual atividade física foi executada com base nas variáveis X. 

# In[25]:


#2.1)num min de observações por folha=20
regr_1 = DecisionTreeRegressor(min_samples_leaf=20)
regr_1.fit(X_treino, y_treino)


# Agora iremos calcular os ccp_alphas da nossa árvore, eles representam os valores de complexidade que controlam a quantidade de poda aplicada à árvore. Quanto maior o valor de ccp_alpha, mais agressiva será a poda. Ao podar a árvore, nós internos que têm um custo-complexidade menor que ccp_alpha são removidos, resultando em uma árvore mais enxuta e menos propensa ao overfitting.

# In[26]:


#2.2)Calculando os ccp_alphas
ccp_alphas = regr_1.cost_complexity_pruning_path(X_treino, y_treino)['ccp_alphas']

#Salvando valores maiores que 0, ou seja, positivos
ccp_alphas = ccp_alphas[ccp_alphas > 0]

#Selecionando apenas valores únicos de ccp_alphas
ccp_alphas = np.unique(ccp_alphas)


# In[27]:


#2.3) usando slice array para selecionar apenas 1 a cada 5 valores.
ccp_alphas_5 = ccp_alphas[::5]


# ### 3. Desempenho da árvore por ccp_alpha
# 
# - 3.1: Rode uma árvore para cada ```ccp_alpha```, salvando cada árvore em uma lista
# - 3.2: Calcule a acurácia de cada árvore na base de treinamento e de teste
# - 3.3: Monte um gráfico da acurácia em função do ```ccp_alpha``` na base de validação e na base de teste
# - 3.4: Selecione a melhor árvore como sendo aquela que possui melhor acurácia na base de teste
# - 3.5: Qual a melhor acurácia que se pode obter com estas três variáveis?
# 
# **Dica:** utilize a estrutura do notebook apresentado em aula.  
# **Dica 2:** meça o tempo com a função mágica ```%%time``` na primeira linha da célula.  
# **Sugestão:** caso fique confortável com o tempo de execução, faça a busca pelo melhor ```ccp_alpha``` com mais iterações.  
# **Sugestão 2:** caso fique confortável com o tempo de execução, tente inserir uma ou mais variáveis adicionais e veja se consegue aumentar a acurácia.

# Classifiers (clf):  utilizados em problemas de classificação, onde o objetivo é atribuir uma classe ou categoria a uma observação com base em um conjunto de características.

# In[29]:


get_ipython().run_cell_magic('time', '', '#3.1) Lista para armazenar as árvores\nclfs = []\n\n# Para cada ccp_alpha, ajuste uma árvore e salve-a na lista\nfor ccp_alpha in ccp_alphas:\n    regr_1 = DecisionTreeRegressor(random_state=2360873, ccp_alpha=ccp_alpha)\n    regr_1.fit(X_treino, y_treino)\n    clfs.append(regr_1)\n')


# Agora que salvamos as árvores de acordo com os classificadores, vamos calcular a acurácia na base de treino e de teste, ou seja a porcentagem de amostras no conjunto de treinamento para as quais o modelo fez previsões corretas em relação às classes verdadeiras.:

# In[31]:


get_ipython().run_cell_magic('time', '', '#3.2)Calculo da acurácia na base de treinamento e de teste:\ntrain_scores = [clf.score(X_treino, y_treino) for clf in clfs] #p/ cada arvore clf dentro da coleção pega as acurácias\ntest_scores = [clf.score(X_teste, y_teste) for clf in clfs]\n')


# Agora, a fim de visualizar isso, iremos plotar o gráfico:

# In[32]:


#3.3) monte o gráfico:
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Acurácia")
ax.set_title("Acurácia x alpha do conjunto de dados de treino e teste")
ax.plot(ccp_alphas, train_scores, marker='o', label="treino",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="teste",
        drawstyle="steps-post")
ax.legend()
plt.show()


# Agora, vamos encontrar a melhor árvore da nossa lista. 

# In[45]:


get_ipython().run_cell_magic('time', '', '#3.4) melhor árvore:\nind_melhor_arvore = len(test_scores) - test_scores[::-1].index(max(test_scores)) - 1\nmelhor_arvore = clfs[ind_melhor_arvore]\nmelhor_arvore\n')


# Vamos plotar essa árvore apenas para ter uma ideia de como irá ficar:

# In[51]:


from sklearn.tree import export_graphviz
# DOT data
dot_data = export_graphviz(melhor_arvore, out_file=None, 
                                feature_names=X_treino.columns,  
                                class_names=['1', '2', '3', '4', '5', '6'],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph


# In[37]:


melhor_acurácia = max(test_scores)
print("Melhor acurácia obtida:", melhor_acurácia)


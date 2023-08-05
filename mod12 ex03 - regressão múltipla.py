#!/usr/bin/env python
# coding: utf-8

# # Regressão 01 - tarefa 03 - transformações em X e Y

# In[4]:


#carregando pacotes
import pandas as pd
import seaborn as sns
from seaborn import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#carregando base de gorjetas 
tips = sns.load_dataset("tips")

#adicionando novas colunas: porcentagem de gorjeta e valor da conta sem gorjeta.
tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
tips['net_bill'] = tips['total_bill'] - tips['tip']
tips.head()


# Carregue os pacotes necessários e a base de gorjetas.
# 
# ### I. Modelo no valor da gorjeta
# 
# 1. Crie a matriz de design (e a matriz y) utilizando o Patsy, para um modelo em ```tip```, explicada por ```sex, smoker, diner e net_bill```.  
# 2. Remova as variáveis não significantes.  
# 3. observe o gráfico de resíduos em função de ```net_bill```  
# 4. teste transformar ```net_bill``` no log e um polinômio. Escolha o melhor modelo.

# In[6]:


#1) criando a matriz de desing, para melhor visualização da matriz, iremos utilizar apenas as 5 primeiras linhas do DataFrame.

y, X = patsy.dmatrices('tip ~ sex + smoker + time + net_bill + 1', tips[:5]) #+1 se dá para calcular o intercepto. 
X


# Com o modelo abaixo, analisaremos 'P>|t|' que nos mostra a significancia das variáveis, valores até 5% mostram variáveis a considerar, além disso, são var menos significantes para explicar o nosso y. Agora para fins de criação de modelo, usarei todas as linhas do DataFrame. Teremos um modelo aonde a variável tip é explicada por sex, smoker, time e net_bill:

# In[8]:


modelo= smf.ols('tip ~ sex + smoker + time + net_bill + 1', data = tips).fit()
modelo.summary()


# Analisando o sumário do sumário acima percebemos que as nossas variáveis explicam 33,5% do modelo. E com base no P>|t| percemos que a única variável que é signigicante, muito significante, é net_bill.

# In[10]:


#2) Excluindo do modelo sex, smoker e time 
modelo= smf.ols('tip ~ net_bill + 1', data = tips).fit()
modelo.summary()


# Obviamente o nosso R-quadrado diminuiu um pouco, porque ele realmente tende a diminuir conforme diminui-se o nº de variáveis explicativas, mas tudo bem pois as outras variáveis não explicavam muita coisa mesmo, visto que a diminuição do R-quadrado foi quase irrelevante. 

# In[13]:


#3)salvando o resíduo do modelo acima em uma nova coluna no nosso dataFrame:
tips['residuo']=modelo.resid


# In[14]:


#exibindo o gráfico do resíduo por net_bill
sns.scatterplot(x = 'net_bill', y = 'residuo', data = tips)
plt.axhline(y=0, color='r', linestyle='--')


# In[ ]:





# INCIANDO TRANSFORMAÇÕES NA NOSSA BASE, COM LOG E POLINÔMIO:

# In[21]:


#4) TRANSFORMANDO NET_BILL PARA LOG
modelo2= smf.ols('tip ~ np.log(net_bill)', data = tips).fit()
modelo2.summary()


# In[22]:


#salvando o resíduo do modelo acima em uma nova coluna no nosso dataFrame:
tips['res_log_net_bill']=modelo2.resid


# In[23]:


#plotando o gráfico do resíduo do modelo 2 por net_bill
sns.scatterplot(x = 'net_bill', y = 'res_log_net_bill', data = tips)
plt.axhline(y=0, color='r', linestyle='--')


# O nosso modelo 2 obteve um R-quadrado menor que o primeiro modelo realizado, por hora, preferimos o primeiro modelo. Mas vamos testar a transformação de net_bill para polinômio.

# In[31]:


#4.1 transformando net_bill em polinômio:
modelo3 = smf.ols('tip ~  net_bill + np.power(net_bill,2)', data = tips).fit()
modelo3.summary


# In[32]:


#salvando o resíduo do modelo acima em uma nova coluna no nosso dataFrame:
tips['res_polinomio_net_bill'] = modelo3.resid

#plotando o gráfico do resíduo do modelo 3 por net_bill
sns.scatterplot(x = 'net_bill', y = 'res_polinomio_net_bill', data = tips, alpha = .75)
plt.axhline(y=0, color='r', linestyle='--')


# O modelo que nos trouxe um melhor R-quadrado foi o modelo3 onde transformamos net_bill (variável explicativa de tip para polinômio. Portanto, esse é o modelo escolhido.

# ### II. Modelo no valor do percentual da gorjeta
# 
# 1. Crie a matriz de design (e a matriz y) utilizando o Patsy, para um modelo no log de ```tip```, explicado por ```sex, smoker, diner e net_bill```.
# 2. Remova as variáveis não significantes.
# 3. Observe o gráfico de resíduos em função de ```net_bill```
# 4. Teste transformar ```net_bill``` no log e um polinômio. Escolha o melhor modelo.
# 5. Do modelo final deste item, calcule o $R^2$ na escala de ```tip``` (sem o log). Compare com o modelo do item 1. Qual tem melhor coeficiente de determinação?

# In[33]:


#1) criando a matriz de log de tip explicada por sex, smoker, time e net_bill - utilizando 5 primeiras linhas apenas para facilitar a visualização

y, X = patsy.dmatrices('np.log(tip) ~ sex + smoker + time + net_bill + 1', tips[:5]) #+1 se dá para calcular o intercepto. 
X


# In[34]:


#2) analisando quais variáveis não são significantes para posteriormente removê-las (utilizando todas as linhas do nosso df)
modelo4= smf.ols('np.log(tip) ~ sex + smoker + time + net_bill + 1', data = tips).fit()
modelo4.summary()


# In[35]:


#2.1) Removendo todas as variáveis não significantes, deixando portanto, apenas net_bill, que tem total relevância. 
modelo4= smf.ols('np.log(tip) ~ net_bill + 1', data = tips).fit()
modelo4.summary()


# In[36]:


#3)criando uma nova variável no data frame para guardar o redíduo de log de tip x net_bill
tips['res_log_tip']=modelo4.resid


# In[39]:


#3.1)observe o gráfico de resíduos em função de net_bill
sns.scatterplot(x = 'net_bill', y = 'res_log_tip', data = tips)
plt.axhline(y=0, color='r', linestyle='--')


# Observando o gráfico podemos ter uma base de quanto é 31% de explicação da nossa variável, muito se concentra em -0,5 e +0,5 no resíduo do modelo.

# In[38]:


#4)teste transformar net_bill em log
modelo5= smf.ols('np.log(tip) ~ np.log(net_bill) + 1', data = tips).fit()
modelo5.summary()


# In[60]:


tips['res_log_tip_e_net_bill']= modelo5.resid


# In[61]:


#4.1)observe o gráfico de resíduos em função de net_bill
sns.scatterplot(x = 'net_bill', y = 'res_log_tip_e_net_bill', data = tips)
plt.axhline(y=0, color='r', linestyle='--')


# In[41]:


#4.2) TRANSFORMANDO NET_BILL EM POLINÔMIO, ENQUANTO TIP É LOG
modelo6 = smf.ols('np.log(tip) ~  net_bill + np.power(net_bill,2)', data = tips).fit()
modelo6.summary()


# ### III. Previsão de renda
# 
# Vamos trabalhar a base que você vai usar no projeto do final deste ciclo.
# 
# Carregue a base ```previsao_de_renda.csv```.
# 
# |variavel|descrição|
# |-|-|
# |data_ref                | Data de referência de coleta das variáveis |
# |index                   | Código de identificação do cliente|
# |sexo                    | Sexo do cliente|
# |posse_de_veiculo        | Indica se o cliente possui veículo|
# |posse_de_imovel         | Indica se o cliente possui imóvel|
# |qtd_filhos              | Quantidade de filhos do cliente|
# |tipo_renda              | Tipo de renda do cliente|
# |educacao                | Grau de instrução do cliente|
# |estado_civil            | Estado civil do cliente|
# |tipo_residencia         | Tipo de residência do cliente (própria, alugada etc)|
# |idade                   | Idade do cliente|
# |tempo_emprego           | Tempo no emprego atual|
# |qt_pessoas_residencia   | Quantidade de pessoas que moram na residência|
# |renda                   | Renda em reais|
# 
# 1. Ajuste um modelo de regressão linear simples para explicar ```renda``` como variável resposta, por ```tempo_emprego``` como variável explicativa. Observe que há muitas observações nessa tabela. Utilize os recursos que achar necessário.
# 2. Faça uma análise de resíduos. Com os recursos vistos neste módulo, como você melhoraria esta regressão?
# 3. Ajuste um modelo de regressão linear múltipla para explicar ```renda``` (ou uma transformação de ```renda```) de acordo com as demais variáveis.
# 4. Remova as variáveis não significantes e ajuste novamente o modelo. Interprete os parâmetros
# 5. Faça uma análise de resíduos. Avalie a qualidade do ajuste.

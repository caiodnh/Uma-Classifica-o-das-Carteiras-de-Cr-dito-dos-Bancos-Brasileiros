import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans

# Esse mini-projeto tem como objetivo para classificar carteiras de crédito dos bancos brasileiros usando dados do Banco Central.

# Vamos começar analizando o mês de junho/2022
# Depois de analizar um pouco o arquivos, descobrimos como importá-lo

estban = pd.read_csv("202206_ESTBAN.CSV"
                  , sep=';'
                  , skiprows=[0,1]
                  , encoding="latin1"
                  )

# Estamos interassados apenas nos dados de crédito
# Lendo os metadados no site do Banco Central, esses são os verbetes 160-179.

credito = estban.filter(regex = r"(VERBETE_1(6|7))", axis = 1)

# Esse arquivo possui os dados por município
# Queremos os dados nacionais por banco

credito_nacional = credito.groupby(by=["NOME_INSTITUICAO"]).sum()

# Os nomes dos verbetes são muito longos, vamos usar só os dígitos
# Mas vamos salvar os nomes originais para ficar mais fácil de compreender depois

verbetes = credito_nacional.columns

def get_num(col):
  match = re.search(r"1(6|7)\d", col)
  return match.group(0)

credito_nacional = credito_nacional.rename(columns=get_num)

# Os metadados dão a entender que o verbete 160 é a soma dos demais, mas não explicita isso. Vale a pena verificar

credito_nao_160 = credito_nacional.filter(regex = r"1(6[1-9]|7)", axis = 1)

somas = credito_nao_160.sum(axis=1)

check = abs(somas - credito_nacional["160"])

# Verificamos que o valor máximo de `check` é 47 reais, claramente um pequeno erro de ponto flutuante dado a ordem de grandeza dos nossos dados
# Ordenando os dados, vemos que o eles variam de 200 mil a quase 1 trilhão de reais.

# Vamos manter os dados ordenados por volume total para ser mais fácil de interpretar

credito_nacional = credito_nacional.sort_values(by = "160")

# Mas queremos classificar as carteiras de crédito, ou seja, as proporções nos diferentes verbetes

carteiras = credito_nacional.filter(regex = r"1(6[1-9]|7)", axis = 1).div(credito_nacional["160"], axis=0)

# O plano agora é agrupar os bancos por carteiras semelhantes.
# Para isso vamos usar o kmeans.
# É difícil decidir quantos grupos queremos. Uma primeira euristica é o "método dos cotovelos".

inertias = []
for k in range(1, 11):
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(carteiras)
  inertias.append(kmeans.inertia_)

# plt.style.use("fivethirtyeight")
# plt.plot(range(1,11), inertias)
# plt.xticks(range(1,11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia")
# plt.show()

# Olhando para o gráfico, o número de clusters deveria ser 3.
# Vamos ver o que isso nos dá.
# Escolhemos uma seed para a aleatoriade para o resultado ser reproduzível.

seed = 131

kmeans = KMeans(n_clusters=3, random_state=seed)
kmeans.fit(carteiras)

# print(kmeans.cluster_centers_.round(2))
# O que significa cada coluna?

# print(verbetes)

# Esses clusteres não parecem com o esperado. 
# Créditos imobiliários não apareceram e os créditos para agro apareceram com peso muito pequeno. 
# Talvez realmente poucos bancos trabalham nessas áreas, mas será o caso?

print(carteiras.max())

# Muitas colunas são nulas mesmo, mas financiamentos imobiliários e agro de fato são relevantes para alguns bancos. Talvez sejam poucos, e por isso 3 grupos não são capazes de classificálos.

# Vamos olhar para os créditos imobiliários
print(carteiras.sort_values(by = "169").round(2).to_string())

# Certamente a Caixa é um outlier. E temos ainda alguns outros poucos bancos que também investem em crédito imobiliário.

# Agora vamos para o agro:

print(carteiras.sort_values(by = "163").round(2).to_string())

# O Banco John Deere também é um outlier. E também temos alguns bancos que investem em agro.


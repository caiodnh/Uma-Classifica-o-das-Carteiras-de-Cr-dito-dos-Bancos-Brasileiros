import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

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

check = abs(somas - credito["160"])
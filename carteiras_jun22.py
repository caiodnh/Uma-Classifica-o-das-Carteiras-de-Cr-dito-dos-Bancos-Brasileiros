import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Esse mini-projeto tem como objetivo para classificar carteiras de crédito
# dos bancos brasileiros usando dados do Banco Central.

# Vamos começar analizando o mês de junho/2022
# Depois de analizar um pouco o arquivos, descobrimos como importá-lo

estban = pd.read_csv("202206_ESTBAN.CSV"
                  , sep=';'
                  , skiprows=[0,1]
                  , encoding="latin1"
                  )


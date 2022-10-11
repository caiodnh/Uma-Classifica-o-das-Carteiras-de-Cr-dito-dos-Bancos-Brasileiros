import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans

# No notebook Carteiras_Junho_2022.ipynb exploramos os dados do mês de junho/2022.
# Aqui vamos abstrair em funções a parte importante do que foi feito lá.

def read_estban(YEARMONTH):
    YEARMONTH = str(YEARMONTH)
    file = YEARMONTH + "_ESTBAN.CSV"

    estban = pd.read_csv(file
                    , sep=';'
                    , skiprows=[0,1]
                    , encoding="latin1"
                    )

    return estban

def make_carteiras(estban):
  credito = estban.filter(regex = r"NOME_INSTITUICAO|VERBETE_1(6|7)", axis = 1)

  credito_sum = credito.groupby(by=["NOME_INSTITUICAO"]).sum()

  verbetes = credito_sum.columns

  def get_num(col):
    match = re.search(r"1(6|7)\d", col)
    return match.group(0)

  credito_sum = credito_sum.rename(columns=get_num).sort_values(by = "160")

  carteiras = credito_sum.filter(regex = r"1(6[1-9]|7)", axis = 1).div(credito_sum["160"], axis=0)
  
  volume_total = pd.DataFrame(credito_sum["160"]).rename(columns={"160":"Volume"})

  return carteiras, volume_total, verbetes

# O Banco Western Union não tem dados em vários meses, então decidimos removê-lo da nossa análise.
def find_groups(carteiras, volume_total, seed = None, centers = None, n_clusters = 5):
  # Remove bancos outliers
  carteiras_bulk = carteiras.drop(["CAIXA ECONOMICA FEDERAL","BANCO JOHN DEERE S.A.","BCO WESTERN UNION"])

  # Roda o KMeans
  # Se centros iniciais são dados, a gente usa eles
  if centers is None:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
  else:
    n_clusters=len(centers)
    kmeans = KMeans(n_clusters=n_clusters, init = centers, n_init = 1)

  kmeans.fit(carteiras_bulk)

  # Escreve os cetros dos grupos num DataFrame
  centers = pd.DataFrame(kmeans.cluster_centers_, columns= carteiras.columns)

  # Coloca o nome dos bancos no resultado
  clusters = pd.Series(kmeans.labels_, index=carteiras_bulk.index)

  # Coloca a Caixa e John Deere de volta
  back_in = pd.Series([n_clusters,n_clusters+1], index=["CAIXA ECONOMICA FEDERAL", "BANCO JOHN DEERE S.A."])

  more_clusters = pd.concat(objs = [clusters,back_in])

  # Série com os tamanhos dos clusters
  sizes = more_clusters.groupby(by = more_clusters).count()

  # DataFrame com os bancos com seus clusters e o volume de crédito, ordenada
  clusters_and_vol = pd.DataFrame(more_clusters, columns = ["Grupo"]).join(volume_total["Volume"])
  clusters_and_vol = clusters_and_vol.sort_values(by = ["Grupo", "Volume"], ascending = [True,False])

  return centers, sizes, clusters_and_vol

estban = read_estban(202206)
cart, vol, verbetes = make_carteiras(estban)
center, sizes, c_vol = find_groups(cart, vol, seed = 131)
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
def run_kmeans(carteiras, seed = None, centers = None, n_clusters = 5):
  # Remove bancos problemáticos
  carteiras_bulk = carteiras.drop(["CAIXA ECONOMICA FEDERAL","BANCO JOHN DEERE S.A.","BCO WESTERN UNION"])

  # Se centros são dados, o algoritmo inicia com eles
  if centers is None:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
  else:
    n_clusters=len(centers)
    kmeans = KMeans(n_clusters=n_clusters, init = centers, n_init = 1)

  kmeans.fit(carteiras_bulk)

  return kmeans

def find_centers(carteiras, kmeans = None):
  if kmeans is None:
    kmeans = run_kmeans(carteiras)
  centers = pd.DataFrame(kmeans.cluster_centers_, columns= carteiras.columns)
  return centers

def sizes(carteiras, kmeans = None):
  if kmeans is None:
    kmeans = run_kmeans(carteiras)
  clusters = pd.Series(kmeans.labels_)
  return clusters.groupby(clusters).count()

def clusters_and_vol(carteiras, volume_total, kmeans = None):
  if kmeans is None:
    kmeans = run_kmeans(carteiras)

  outliers = ["CAIXA ECONOMICA FEDERAL", "BANCO JOHN DEERE S.A."]
  index_bulk = carteiras.index.drop(outliers)
  clusters = pd.Series(kmeans.labels_, index=index_bulk)

  added_vol = pd.DataFrame(clusters, columns = ["Grupo"]).join(volume_total["Volume"])
  sorted_vol = added_vol.sort_values(by = ["Grupo", "Volume"], ascending = [True,False])

  return sorted_vol
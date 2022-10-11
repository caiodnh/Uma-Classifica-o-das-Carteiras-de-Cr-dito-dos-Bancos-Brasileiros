import pandas as pd
import re
from sklearn.cluster import KMeans

# No notebook Carteiras_Junho_2022.ipynb exploramos os dados do mês de junho/2022.
# Aqui vamos abstrair em funções a parte importante do que foi feito lá.

# A função abaixo apenas lê um arquivo ESTBAN
def read_estban(YEARMONTH):
    YEARMONTH = str(YEARMONTH)
    file = YEARMONTH + "_ESTBAN.CSV"

    estban = pd.read_csv(file
                    , sep=';'
                    , skiprows=[0,1]
                    , encoding="latin1"
                    )

    return estban

# `make_sum` retorna apenas os créditos, mas somados em todos os municípios.
# Também deixa os nomes das colunas apenas números e retorna os verbetes para leitura futura
def make_sum(estban):
  credito = estban.filter(regex = r"NOME_INSTITUICAO|VERBETE_1(6|7)", axis = 1)

  credito_sum = credito.groupby(by=["NOME_INSTITUICAO"]).sum()

  verbetes = credito_sum.columns

  def get_num(col):
    match = re.search(r"1(6|7)\d", col)
    return match.group(0)

  credito_sum = credito_sum.rename(columns=get_num).sort_values(by = "160")

  return credito_sum, verbetes

# A próxima função faz as divisões para termos apenas as carteiras. Recebe direto o estban por simplificação
def make_carteiras(estban):
  credito_sum, verbetes = make_sum(estban)

  carteiras = credito_sum.filter(regex = r"1(6[1-9]|7)", axis = 1).div(credito_sum["160"], axis=0)
  
  volume_total = pd.DataFrame(credito_sum["160"]).rename(columns={"160":"Volume"})

  return carteiras, volume_total, verbetes

# Preparar os dados e rodar o kmeans
def run_kmeans(carteiras, seed = None, centers = None, n_clusters = 5):
  # Remove bancos problemáticos
  # O Banco Western Union não tem dados em vários meses, então decidimos removê-lo da nossa análise
  carteiras_bulk = carteiras.drop(["CAIXA ECONOMICA FEDERAL","BANCO JOHN DEERE S.A.","BCO WESTERN UNION"])

  # Se centros são dados, o algoritmo inicia com eles
  if centers is None:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
  else:
    n_clusters=len(centers)
    kmeans = KMeans(n_clusters=n_clusters, init = centers, n_init = 1)

  kmeans.fit(carteiras_bulk)

  return kmeans

# Retorna os centros dos grupos como um DataFrame, para ser mais fácil de ler
def find_centers(carteiras, kmeans = None, seed = None, centers = None, n_clusters = 5):
  if kmeans is None:
    kmeans = run_kmeans(carteiras, seed = seed, centers = centers, n_clusters = n_clusters)
  centers = pd.DataFrame(kmeans.cluster_centers_, columns= carteiras.columns)
  return centers

# Retorna os tamanhos dos grupos
def sizes(carteiras, kmeans = None, seed = None, centers = None, n_clusters = 5):
  if kmeans is None:
    kmeans = run_kmeans(carteiras, seed = seed, centers = centers, n_clusters = n_clusters)
  clusters = pd.Series(kmeans.labels_)
  return clusters.groupby(clusters).count()

# Junta os grupos com os volumes totais de crédito
def clusters_and_vol(carteiras, volume_total, kmeans = None, seed = None, centers = None, n_clusters = 5):
  if kmeans is None:
    kmeans = run_kmeans(carteiras, seed = seed, centers = centers, n_clusters = n_clusters)

  outliers = ["CAIXA ECONOMICA FEDERAL","BANCO JOHN DEERE S.A.","BCO WESTERN UNION"]
  index_bulk = carteiras.index.drop(outliers)
  clusters = pd.Series(kmeans.labels_, index=index_bulk)

  added_vol = pd.DataFrame(clusters, columns = ["Grupo"]).join(volume_total["Volume"])
  sorted_vol = added_vol.sort_values(by = ["Grupo", "Volume"], ascending = [True,False])

  return sorted_vol
# 1 - Introdução

# Porque Utilizar Estatística?

# Bibliotecas

import numpy as np
import pandas as pd

from scipy import stats
import pylab
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import yfinance as yf
import vectorbt as vbt

import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sweetviz as sv

# Criando dados aleatórios para poder simular uma correlação
np.random.seed(13)
a = np.random.randint(50, size=30)
b = np.random.randint(50, size=30)
ab = zip(a,b)
fig = px.scatter(ab, x=a, y=b, template='simple_white', width=400, height=400)
fig.update_layout(paper_bgcolor='#f7f8fa', margin=dict(l=20,r=20,t=20,b=20))

# Agora utilizamos a correlação de Pearson para ver o quão correlacionado ou disperso estãos nossos dados.
corr, p = stats.pearsonr(a,b)
print('Correlação de Pearson, r=%.3f' %corr, 'p=%.3f'%p)
# Se o valor de p for menor que 0.05, podemos dizer que a correlação, o que nesse caso o resultado foi de 0.296

# Distribuição Normal

normal_dist = np.random.normal(0,1,1000)
normal_fig01 = sns.distplot(normal_dist, hist=True, kde=True)

# Medidas de posição: média, mediana e moda

'''
Média:
A média é a média aritmética de um conjunto de números

Mediana:
A Mediana é um valor numérico que separa a metade superior de um conjunto de metade inferior.

Moda:
Valor que acontece com mais frequência

'''
print(np.mean(normal_dist))
print(np.median(normal_dist))
print(stats.mode(normal_dist))


# Utilizar o plotly para gerar um gráfico dinâmico 
fig01 = px.histogram(normal_dist, color_discrete_sequence=['lightseagreen'])
fig01.add_vline(x=np.mean(normal_dist), line_width=3,  line_color="red")
fig01.add_vline(x=np.median(normal_dist), line_width=3, line_dash="dash", line_color="red")
fig01.update_layout(width=400, height=400, template = 'simple_white',
                    paper_bgcolor="#f7f8fa", margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False)

# Como saber se uma distribuição é normal?

test_normalidade = stats.normaltest(normal_dist)
print(test_normalidade)

# Medida de posição: quantis

stats.probplot(normal_dist, dist='norm', plot=pylab)
pylab.show()

# Boxplots

fig02 = go.Figure()
fig02.add_trace(go.Box(y=normal_dist, boxpoints='all', boxmean='sd', name='<b>Normal distribution', marker_color = 'blue'))
fig02.update_layout(width=400, height=400, template = 'simple_white',
                    paper_bgcolor="#f7f8fa", margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False)

print(np.quantile(normal_dist, 0.25))
print(np.percentile(normal_dist, 25))

df_normal_distrib = pd.DataFrame(normal_dist)
df_normal_distrib.describe()

# Assimetria (Skewness)

stats.skew(normal_dist)

# Medidas de dispersão(amplitude, desvio padrão e médio, coeficiente de variação e intervalo de confiança)

# Amplitude para alvos/stops

amplitude = normal_dist.max() - normal_dist.min()
amplitude

# Desvio padrão é uma medida que indica a dispersão dos dados dentro de uma amostra com relação a média

desvio_padrao = np.std(normal_dist, ddof = True)

# Coeficiente de variação
# interessante para comparações e avaliar a consistência(ex., pagamento de dividendos)

normal_dist.std(ddof = True)/normal_dist.mean()*100

# Erro padrão da média

stats.stats.sem(normal_dist)

# Intervalo de confiança de 95%
# Probabilidade de 95% da média real estar nesse intervalor


IC_95 = stats.t.interval(alpha=0.95, df=len(normal_dist)-1, loc=np.mean(normal_dist), scale=stats.sem(normal_dist))

# Curtose

stats.kurtosis(normal_dist, fisher=True)
















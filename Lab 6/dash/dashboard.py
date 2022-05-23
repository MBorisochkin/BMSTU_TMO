from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, silhouette_score
from dash import Dash, html, dcc, Input, Output

import numpy as np
import pandas as pd
import plotly.express as px

app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Загрузка данных
data = pd.read_csv('data/wine_pca.csv')
data_true = pd.read_csv('data/wine_pca_with_true_labels.csv')
labels_true = data_true['Cultivars']
data_true['Cultivars'] = data_true['Cultivars'].astype('str')

# Методы кластеризации
methods_list = ['K-Means', 'Mini-Batch K-Means', 'Bisecting K-Means', 'Agglomerative Clustering',
                'Affinity Propagation', 'Mean shift', 'Spectral Clustering', 'Gaussian Mixture']

methods_dict = {'K-Means': KMeans(n_clusters=3, random_state=4),
                'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=3, random_state=4),
                'Bisecting K-Means': BisectingKMeans(n_clusters=3, random_state=4),
                'Agglomerative Clustering': AgglomerativeClustering(n_clusters=3),
                'Affinity Propagation': AffinityPropagation(random_state=4),
                'Mean shift': MeanShift(),
                'Spectral Clustering': SpectralClustering(n_clusters=3, random_state=4),
                'Gaussian Mixture': GaussianMixture(n_components=3, random_state=4)}

# Обучение моделей и оценка метрик
clusters_dict = {}
ami_dict, vm_dict, sil_dict = {}, {}, {}

for method_name, method in methods_dict.items():
    labels_pred = method.fit_predict(data)

    clusters_dict[method_name] = labels_pred
    ami_dict[method_name] = adjusted_mutual_info_score(labels_true, labels_pred)
    vm_dict[method_name] = v_measure_score(labels_true, labels_pred)
    sil_dict[method_name] = silhouette_score(data, labels_pred)

df_metrics = pd.DataFrame({'AMI': ami_dict, 'V-measure': vm_dict, 'Silhouette Coefficient': sil_dict})
df_metrics_melt = df_metrics.melt(ignore_index=False)
df_metrics_melt = df_metrics_melt.rename(columns={'variable': 'Metric'})

# Диаграмма рассеивания изначального набора данных
fig_data = px.scatter(data, x='PC1', y='PC2', template='plotly_white', title='Данные')
fig_data.update_xaxes(showline=True, linewidth=1, mirror=True)
fig_data.update_yaxes(showline=True, linewidth=1, mirror=True)
fig_data.update_layout(xaxis_title='', yaxis_title='', title={'x': 0.5})
fig_data.update_traces(hovertemplate="%{x}; %{y}")

# Диаграмма рассеивания набора данных с метками классов
fig_true_labels = px.scatter(data_true, x='PC1', y='PC2', color='Cultivars',
                             template='plotly_white', title='Данные с метками классов')
fig_true_labels.update_xaxes(showline=True, linewidth=1, mirror=True)
fig_true_labels.update_yaxes(showline=True, linewidth=1, mirror=True)
fig_true_labels.update_layout(xaxis_title='', yaxis_title='', title={'x': 0.5}, showlegend=False)
fig_true_labels.update_traces(hovertemplate="%{x}; %{y}")

# Гистограмма с метриками
fig_metrics = px.bar(df_metrics_melt, x=df_metrics_melt.index, y='value', color='Metric',
                     barmode='group', template='plotly_white')
fig_metrics.update_layout(xaxis_title='Метод кластеризации', yaxis_title='Значение метрики')
fig_metrics.update_traces(hovertemplate="<b>%{x}</b> <br>Значение: %{y}")

app.layout = html.Div([
    html.H2('Методы кластеризации'),
    method_input := dcc.Dropdown(options=[{'label': x, 'value': x} for x in methods_list],
                                 value='K-Means', clearable=False),
    html.Div([
        dcc.Graph(figure=fig_data, className="one-third column"),
        clusters_output := dcc.Graph(className="one-third column"),
        dcc.Graph(figure=fig_true_labels, className="one-third column")
    ], className="row"),
    dcc.Graph(figure=fig_metrics)
])


@app.callback(
    Output(clusters_output, 'figure'),
    Input(method_input, 'value'))
def plot_clusters(name):
    """
    Функция построения кластеров

    :param name: Название метода
    :return: Предсказанные кластеры
    """

    m = methods_dict[name]
    pred = m.fit_predict(data)

    fig_pred = px.scatter(data, x='PC1', y='PC2', color=np.array(pred, dtype=str),
                          template='plotly_white', title='Предсказанные кластеры')
    fig_pred.update_xaxes(showline=True, linewidth=1, mirror=True)
    fig_pred.update_yaxes(showline=True, linewidth=1, mirror=True)
    fig_pred.update_layout(xaxis_title='', yaxis_title='', title={'x': 0.5}, showlegend=False)
    fig_pred.update_traces(hovertemplate="%{x}; %{y}")

    return fig_pred


if __name__ == '__main__':
    app.run_server(debug=True)

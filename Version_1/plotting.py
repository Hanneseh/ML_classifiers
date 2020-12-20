# all imports
import pandas as pd
import plotly.graph_objects as go
from joblib import dump, load

# load metrics
metricsData = load('Metrics/metrics.joblib')
metricsDataFormatted = load('Metrics/metricsFormatted.joblib')

# Show metrics in Table
metricsData
metricsDataFormatted

# Show metrics in Graph:
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=metricsData['Classifier'],
        y=metricsData['Accuracy'],
        name='Accuracy',
        marker_color='rgb(255, 0, 0)'
        ))
fig.add_trace(
    go.Bar(
        x=metricsData['Classifier'],
        y=metricsData['Precision'],
        name='Precision',
        marker_color='rgb(255, 255, 0)'
        ))
fig.add_trace(
    go.Bar(
        x=metricsData['Classifier'],
        y=metricsData['Recall'],
        name='Recall',
        marker_color='rgb(0, 255, 0)'
        ))
fig.add_trace(
    go.Bar(
        x=metricsData['Classifier'],
        y=metricsData['F1'],
        name='F1',
        marker_color='rgb(255, 0, 255)'
        ))
fig.add_trace(
    go.Bar(
        x=metricsData['Classifier'],
        y=metricsData['AUC'],
        name='AUC',
        marker_color='rgb(0, 128, 255)'
        ))

fig.update_layout(
    title='Comparing all metrics of all classifiers',
    xaxis_tickfont_size=14,

    yaxis=dict(
        title='Scale',
    ),
    barmode='group',
)
fig.show()
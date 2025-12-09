import plotly.graph_objects as go
from src.utils import to_daily_resolution
import pandas as pd

class GasVisualizer:
    @staticmethod
    def create_dashboard(df_historical, pred_series, conf_int):
        
        hist_daily = to_daily_resolution(df_historical)
        
        # CORREZIONE: Gestione corretta della concatenazione per il grafico
        col_name = df_historical.columns[0] # 'Prices'
        pred_series.name = col_name
        
        last_hist = df_historical.iloc[[-1]]
        
        # Convertiamo pred_series in DataFrame con lo stesso nome colonna
        pred_df = pred_series.to_frame(name=col_name)
        
        pred_combined = pd.concat([last_hist, pred_df])
        pred_daily = to_daily_resolution(pred_combined)

        fig = go.Figure()

        # 1. Linea Storica
        fig.add_trace(go.Scatter(
            x=hist_daily.index, y=hist_daily.iloc[:,0],
            mode='lines', name='Storico',
            line=dict(color='royalblue', width=2)
        ))

        # 2. Previsione
        fig.add_trace(go.Scatter(
            x=pred_daily.index, y=pred_daily.iloc[:,0],
            mode='lines', name='Previsione AI',
            line=dict(color='darkorange', dash='dot', width=2)
        ))
        
        # 3. Punti Reali
        fig.add_trace(go.Scatter(
            x=df_historical.index, y=df_historical.iloc[:,0],
            mode='markers', name='Dati Mensili',
            marker=dict(color='navy', size=5)
        ))

        # 4. Confidenza (gestione errori indici)
        try:
            fig.add_trace(go.Scatter(
                x=conf_int.index, y=conf_int.iloc[:, 0],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=conf_int.index, y=conf_int.iloc[:, 1],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255, 165, 0, 0.1)',
                name='Intervallo Confidenza'
            ))
        except:
            pass # Se l'intervallo confidenza fallisce, mostra comunque il grafico

        fig.update_layout(
            title='Analisi Prezzo Gas Naturale (SARIMA Refactored)',
            xaxis_title='Data',
            yaxis_title='Prezzo ($)',
            template='plotly_white',
            hovermode="x unified"
        )
        return fig
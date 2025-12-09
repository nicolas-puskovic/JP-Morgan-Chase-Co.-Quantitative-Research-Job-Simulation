import pandas as pd

def to_daily_resolution(df_monthly, method='linear'):
    """
    Centralizza la logica di conversione da frequenza Mensile a Giornaliera.
    Usa interpolazione LINEARE per evitare overshooting sui prezzi (sicurezza finanziaria).
    """
    # Resample giornaliero crea righe vuote (NaN) tra i mesi
    df_daily = df_monthly.resample('D').mean()
    
    # Interpolazione
    # 'linear' è preferibile in finanza base perché non inventa massimi/minimi
    # che non esistono (a differenza della 'cubic').
    df_interpolated = df_daily.interpolate(method=method)
    
    return df_interpolated
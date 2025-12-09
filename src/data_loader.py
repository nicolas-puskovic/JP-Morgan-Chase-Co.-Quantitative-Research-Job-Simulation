import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_clean_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"ERRORE CRITICO: Il file dati non esiste in: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
            
            # Parsing flessibile della data
            df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
            df.set_index('Dates', inplace=True)
            
            # Ordina e normalizza alla fine del mese ('ME') per coerenza SARIMA
            df = df.sort_index().asfreq('ME')
            
            # Controllo NaN post-importazione
            if df.isnull().values.any():
                print(">> Warning: Trovati valori nulli. Riempimento automatico (ffill).")
                df = df.fillna(method='ffill')
                
            return df
            
        except Exception as e:
            raise Exception(f"Errore caricamento dati: {str(e)}")
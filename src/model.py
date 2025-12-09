import os
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import mean_squared_error
from src.utils import to_daily_resolution
import warnings

# Silenzia warning dei modelli non convergenti
warnings.filterwarnings("ignore")

class GasModel:
    def __init__(self, save_dir='models', model_name='sarima_v1.pkl'):
        self.save_path = os.path.join(save_dir, model_name)
        self.results = None
        # Parametri di default un po' più robusti
        self.best_order = (1, 1, 1)
        self.best_seasonal = (1, 1, 1, 12)

    def load_or_train(self, df_prices):
        if os.path.exists(self.save_path):
            print(f">> Caricamento modello da: {self.save_path}")
            try:
                self.results = SARIMAXResults.load(self.save_path)
                return
            except:
                print(">> File modello corrotto. Riaddestramento forzato.")
        
        self._optimize_params(df_prices)
        self._validate_and_train(df_prices)

    def _optimize_params(self, df):
        print(">> Inizio Grid Search (Aumentata)...")
        
        # 1: Range più ampio per catturare pattern complessi
        # p, q fino a 2. d fino a 1 (differenziazione).
        p = q = range(0, 3) 
        d = range(0, 2)
        
        pdq = list(itertools.product(p, d, q))
        # Stagionalità: P, Q ridotti per velocità, ma D=1 è importante per la stagionalità annuale
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product([0, 1], [0, 1], [0, 1]))]
        
        best_aic = float("inf")
        
        # Contatore per feedback visivo
        total_iter = len(pdq) * len(seasonal_pdq)
        counter = 0

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(df,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    res = mod.fit(disp=False)
                    
                    if res.aic < best_aic:
                        best_aic = res.aic
                        self.best_order = param
                        self.best_seasonal = param_seasonal
                except:
                    continue
                
        print(f">> Parametri ottimali trovati: {self.best_order} x {self.best_seasonal} (AIC: {best_aic:.2f})")

    def _validate_and_train(self, df):
        # Training Finale su TUTTI i dati
        print(">> Addestramento modello finale...")
        final_model = sm.tsa.statespace.SARIMAX(df, 
                                                order=self.best_order, 
                                                seasonal_order=self.best_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
        self.results = final_model.fit(disp=False)
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.results.save(self.save_path)
        print(">> Modello salvato con successo.")

    def predict_value(self, date_str, df_history):
        try:
            target_date = pd.to_datetime(date_str, format='%m/%d/%y')
        except ValueError:
            return None, None, "Formato data errato (usa MM/GG/AA)"

        last_hist_date = df_history.index.max()
        first_hist_date = df_history.index.min()

        # CASO 1: Storico
        if first_hist_date <= target_date <= last_hist_date:
            df_daily = to_daily_resolution(df_history)
            idx = df_daily.index.get_indexer([target_date], method='nearest')[0]
            price = df_daily.iloc[idx].iloc[0]
            return price, "STORICO", "Dato recuperato dallo storico"

        # CASO 2: Futuro
        elif target_date > last_hist_date:
            if not self.results:
                return None, None, "Modello non caricato"
            
            delta_months = (target_date.year - last_hist_date.year) * 12 + (target_date.month - last_hist_date.month)
            steps = delta_months + 2 # Buffer aumentato
            
            forecast = self.results.get_forecast(steps=steps)
            pred_series = forecast.predicted_mean
            
            # CORREZIONE 2: Allineamento Nomi Colonne per Concat
            # Assicuriamo che la serie predetta abbia lo stesso nome della colonna prezzi
            col_name = df_history.columns[0] # Es: 'Prices'
            pred_series.name = col_name 
            
            history_tail = df_history.iloc[[-1]]
            
            # Ora la concatenazione funziona verticalmente perché i nomi colonne coincidono
            combined = pd.concat([history_tail, pred_series.to_frame()])
            
            combined_daily = to_daily_resolution(combined)
            
            if target_date <= combined_daily.index.max():
                idx = combined_daily.index.get_indexer([target_date], method='nearest')[0]
                val = combined_daily.iloc[idx]
                price = val if isinstance(val, (int, float, np.number)) else val.iloc[0]
                return price, "PREVISIONE", f"SARIMA {self.best_order}x{self.best_seasonal}"
            else:
                 return None, None, "Data fuori range previsionale"

        else:
            return None, None, "Data antecedente allo storico"

    def get_forecast_for_plot(self, steps=24):
        forecast = self.results.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.conf_int()
    def run_backtest(self, df, test_months=6):
        
        print(f"\n--- AVVIO BACKTEST (Ultimi {test_months} mesi nascosti) ---")
        
        # 1. Split Temporale
        train = df.iloc[:-test_months]
        test = df.iloc[-test_months:] # Questi sono i dati reali che oscuriamo
        
        print(f">> Training set: {len(train)} mesi | Test set: {len(test)} mesi")
        
        # 2. Addestramento su dati parziali (usando i parametri migliori già trovati)
        # Nota: Usiamo gli stessi best_order trovati, ma i coefficienti vengono ricalcolati
        model = sm.tsa.statespace.SARIMAX(train, 
                                        order=self.best_order, 
                                        seasonal_order=self.best_seasonal,
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False)
        res = model.fit(disp=False)
        
        # 3. Previsione sul periodo oscurato
        forecast = res.get_forecast(steps=test_months)
        pred_mean = forecast.predicted_mean
        pred_mean.index = test.index # Allineamento indici per sicurezza
        
        # 4. Calcolo Metriche di Errore
        # MAE: Errore medio assoluto (quanto sbaglio in media in $)
        mae = np.mean(np.abs(pred_mean - test.iloc[:,0]))
        
        # RMSE: Radice dell'errore quadratico medio 
        rmse = np.sqrt(mean_squared_error(test, pred_mean))
        
        # MAPE: Errore percentuale medio 
        mape = np.mean(np.abs((test.iloc[:,0] - pred_mean) / test.iloc[:,0])) * 100
        
        print(f">> MAE  (Errore Medio): ${mae:.2f}")
        print(f">> RMSE (Errore Quadratico): ${rmse:.2f}")
        print(f">> MAPE (Errore Percentuale): {mape:.2f}%")
        

        return test, pred_mean, mape

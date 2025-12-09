import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class GasValidator:
    def __init__(self, model_instance):
        """
        Riceve l'istanza del modello GIÀ addestrato.
        Non ricarica i dati, usa quelli che gli passi.
        """
        self.model_ref = model_instance

    def run_backtest(self, df, test_months=6):
        """
        Esegue il backtest 'Out-of-Sample'.
        Addestra una copia temporanea del modello sui dati passati.
        """
        print(f"\n--- 📉 AVVIO BACKTEST (Ultimi {test_months} mesi nascosti) ---")
        
        train = df.iloc[:-test_months]
        test = df.iloc[-test_months:]
        
        # Usiamo gli STESSI iperparametri ottimali trovati dal modello principale
        print(f">> Training set parziale: {len(train)} mesi")
        
        temp_model = sm.tsa.statespace.SARIMAX(train, 
                                               order=self.model_ref.best_order, 
                                               seasonal_order=self.model_ref.best_seasonal,
                                               enforce_stationarity=False, 
                                               enforce_invertibility=False)
        temp_res = temp_model.fit(disp=False)
        
        # Previsione
        forecast = temp_res.get_forecast(steps=test_months)
        pred_mean = forecast.predicted_mean
        pred_mean.index = test.index # Allineamento indici
        
        # Calcolo Metriche
        rmse = np.sqrt(mean_squared_error(test, pred_mean))
        mape = np.mean(np.abs((test.iloc[:,0] - pred_mean) / test.iloc[:,0])) * 100
        
        print(f">> RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
        
        if mape < 5:
            print("✅ ESITO BACKTEST: Eccellente (<5%)")
        elif mape < 10:
            print("⚠️ ESITO BACKTEST: Accettabile (<10%)")
        else:
            print("❌ ESITO BACKTEST: Alto errore, rivedere modello.")
            
        return test, pred_mean

    def check_ljung_box(self):
        """
        Esegue il test statistico sui residui del modello principale.
        """
        if self.model_ref.results is None:
            print("⚠️ Errore: Il modello principale non è addestrato.")
            return

        print("\n" + "="*50)
        print("  ANALISI STATISTICA (Ljung-Box Test)")
        print("="*50)
        
        resid = self.model_ref.results.resid
        
        # Test su 6 e 12 mesi (ciclo annuale)
        # return_df=True restituisce un DataFrame pandas facile da leggere
        lb_test = acorr_ljungbox(resid, lags=[6, 12], return_df=True)
        
        # Prendiamo il p-value del lag 12 (un anno)
        p_val = lb_test.loc[12, 'lb_pvalue']
        
        print(f">> P-Value (12 lags): {p_val:.4f}")
        
        if p_val > 0.05:
            print("✅ PROMOSSO: I residui sono Rumore Bianco (Random).")
            print("   Il modello ha estratto tutto il segnale disponibile.")
        else:
            print("❌ BOCCIATO: C'è ancora autocorrelazione residua.")
            print("   (Possibile stagionalità non catturata)")
            
        print("="*50)

    def plot_diagnostics(self):
        """Genera i grafici dei residui con lag ridotti per dataset piccoli"""
        if self.model_ref.results is None: 
            return
        
        print("\n>> Apertura grafici diagnostici...")
        
        try:
            # FIX: Aggiunto lags=8 per evitare l'errore "Length of endogenous variable..."
            # Questo limita l'analisi dell'autocorrelazione a 8 mesi indietro, 
            # compatibile con uno storico breve (48 mesi).
            fig = self.model_ref.results.plot_diagnostics(figsize=(12, 8), lags=8)
            fig.tight_layout()
            plt.show()
            
        except ValueError as e:
            # Fallback se i dati sono davvero troppo pochi
            print(f"⚠️ Impossibile generare grafico diagnostico completo: {e}")
            print(">> Genero grafico semplificato dei soli residui...")
            
            plt.figure(figsize=(10, 4))
            plt.plot(self.model_ref.results.resid)
            plt.title("Residui del Modello (Semplificato)")
            plt.xlabel("Tempo")
            plt.ylabel("Errore")
            plt.show()
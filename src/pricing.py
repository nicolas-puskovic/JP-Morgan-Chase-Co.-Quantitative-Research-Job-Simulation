import pandas as pd
import numpy as np
from datetime import timedelta

class StorageContract:
    def __init__(self, 
                 max_volume=1000000,   # Volume massimo stoccabile (MMBtu)
                 inj_rate=50000,       # Tasso di iniezione giornaliero
                 with_rate=50000,      # Tasso di prelievo giornaliero
                 inj_cost=0.01,        # Costo per iniettare ($/MMBtu)
                 with_cost=0.01,       # Costo per prelevare ($/MMBtu)
                 storage_cost=0.05):   # Costo di stoccaggio mensile ($/MMBtu/Mese)
        
        self.max_volume = max_volume
        self.inj_rate = inj_rate
        self.with_rate = with_rate
        self.inj_cost = inj_cost
        self.with_cost = with_cost
        self.storage_cost_monthly = storage_cost
        self.inventory = 0 # Partiamo con il magazzino vuoto

    def calculate_valuation(self, injection_dates, withdrawal_dates, price_model, df_history):
        """
        Calcola il valore del contratto basandosi su date pianificate.
        Usa il modello SARIMA per stimare i prezzi futuri.
        """
        print("\n" + "="*50)
        print("  VALUTAZIONE CONTRATTO STORAGE")
        print("="*50)

        ledger = [] # Qui salveremo ogni movimento (giornale di cassa)
        total_value = 0
        current_vol = 0
        
        # Uniamo e ordiniamo tutte le date (cronologia eventi)
        # Tupla: (Data, Tipo_Azione)
        all_actions = [(d, 'INJECTION') for d in injection_dates] + \
                      [(d, 'WITHDRAWAL') for d in withdrawal_dates]
        
        # Ordiniamo per data
        all_actions.sort(key=lambda x: pd.to_datetime(x[0]))
        
        last_date = None

        for date_str, action in all_actions:
            current_date = pd.to_datetime(date_str)
            
            # 1. Calcolo Costi di Stoccaggio (Tempo trascorso dall'ultima azione)
            if last_date is not None:
                days_passed = (current_date - last_date).days
                # Costo = Volume * (CostoMensile / 30) * Giorni
                carrying_cost = current_vol * (self.storage_cost_monthly / 30) * days_passed
                total_value -= carrying_cost
                if days_passed > 0:
                    ledger.append({
                        "Date": current_date.strftime('%Y-%m-%d'),
                        "Action": "CARRYING COST",
                        "Volume": 0,
                        "Price": 0,
                        "CashFlow": -carrying_cost,
                        "Inventory": current_vol
                    })

            # 2. Ottenere Prezzo dal Modello (AI)
            predicted_price, _, _ = price_model.predict_value(date_str, df_history)
            
            if predicted_price is None:
                print(f"⚠️ Saltato evento del {date_str}: Data non valida per il modello.")
                continue

            cash_flow = 0
            vol_moved = 0

            # 3. Logica INIEZIONE (Compriamo gas)
            if action == 'INJECTION':
                # Quanto possiamo iniettare? Minimo tra Rateo e Spazio Rimasto
                space_left = self.max_volume - current_vol
                vol_moved = min(self.inj_rate, space_left)
                
                # Costo Totale = (Prezzo Gas + Costo Iniezione) * Volume
                cost_basis = predicted_price + self.inj_cost
                cash_flow = -(vol_moved * cost_basis)
                
                current_vol += vol_moved

            # 4. Logica PRELIEVO (Vendiamo gas)
            elif action == 'WITHDRAWAL':
                # Quanto possiamo prelevare? Minimo tra Rateo e Gas Disponibile
                vol_moved = min(self.with_rate, current_vol)
                
                # Ricavo Netto = (Prezzo Gas - Costo Prelievo) * Volume
                revenue_basis = predicted_price - self.with_cost
                cash_flow = (vol_moved * revenue_basis)
                
                current_vol -= vol_moved

            # Aggiornamento Totali
            total_value += cash_flow
            last_date = current_date
            
            ledger.append({
                "Date": current_date.strftime('%Y-%m-%d'),
                "Action": action,
                "Volume": vol_moved,
                "Price": predicted_price,
                "CashFlow": cash_flow,
                "Inventory": current_vol
            })

        # Creiamo un DataFrame leggibile per l'utente
        df_ledger = pd.DataFrame(ledger)
        return total_value, df_ledger
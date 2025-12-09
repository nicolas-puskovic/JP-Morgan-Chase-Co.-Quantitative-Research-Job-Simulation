import os
import sys
from src.pricing import StorageContract
# Aggiungi l'import del nuovo validator
from src.data_loader import DataLoader
from src.model import GasModel
from src.visualizer import GasVisualizer
from src.validator import GasValidator 

def main():
    # ... (Setup Percorsi rimangono uguali) ...
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'Nat_Gas.csv')
    
    print("\n" + "="*50)
    print("   QUANT GAS PRICING SYSTEM - FINAL")
    print("="*50)

    # 1. Caricamento Dati
    loader = DataLoader(DATA_PATH)
    try:
        df = loader.get_clean_data()
        print(f"✅ Dataset caricato: {len(df)} mesi.")
    except Exception as e:
        print(f"❌ Errore: {e}")
        return

    # 2. Modello (Core Logic)
    model_system = GasModel(save_dir=os.path.join(BASE_DIR, 'models'))
    model_system.load_or_train(df['Prices'])

    # 3. Validazione & Quality Assurance (Opzionale ma raccomandata)
    # Creiamo il validatore passandogli il modello GIA' pronto
    validator = GasValidator(model_system)
    
    # Eseguiamo i test
    validator.check_ljung_box() # Test Statistico (veloce)
    
    # Domandiamo all'utente se vuole il backtest (che è più lento)
    check_bt = input("\n>> Vuoi eseguire il Backtest (richiede ricalcolo)? [y/n]: ").lower()
    if check_bt == 'y':
        validator.run_backtest(df[['Prices']], test_months=6)
        validator.plot_diagnostics()

    # 4. Generazione Grafico Dashboard
    print("\n>> Generazione Dashboard...")
    pred, conf = model_system.get_forecast_for_plot(steps=24)
    fig = GasVisualizer.create_dashboard(df[['Prices']], pred, conf)
    fig.show()

    # 5. Sezione Pricing (Nuova)
    print("\n" + "-"*50)
    print(" MODULO PRICING CONTRATTI")
    print("-" * 50)
    
    do_pricing = input(">> Vuoi valutare una strategia di stoccaggio? [y/n]: ").lower()
    
    if do_pricing == 'y':
        # Esempio di parametri (puoi renderli input utente se vuoi)
        print(">> Inizializzazione contratto standard:")
        print("   - Max Vol: 1.000.000 MMBtu")
        print("   - Rateo Iniezione/Prelievo: 100.000 MMBtu/day")
        print("   - Costo Stoccaggio: $0.10/mese")
        
        contract = StorageContract(max_volume=1000000, 
                                   inj_rate=100000, 
                                   with_rate=100000,
                                   storage_cost=0.10)
        
        # Simuliamo una strategia semplice:
        # Comprare in Estate (Prezzi bassi), Vendere in Inverno (Prezzi alti)
        # Nota: L'utente dovrebbe inserire date future rispetto al dataset!
        # Assumiamo che l'utente inserisca date valide (es. Estate 2025, Inverno 2026)
        
        print("\nInserisci le date (formato MM/GG/AA). Lascia vuoto per terminare la lista.")
        
        inj_dates = []
        print(">> Inserisci date di INIEZIONE (Acquisto):")
        while True:
            d = input("   Data Inj: ")
            if d == "": break
            inj_dates.append(d)
            
        wit_dates = []
        print(">> Inserisci date di PRELIEVO (Vendita):")
        while True:
            d = input("   Data With: ")
            if d == "": break
            wit_dates.append(d)
            
        if inj_dates and wit_dates:
            # Calcolo
            try:
                value, df_ledger = contract.calculate_valuation(inj_dates, wit_dates, model_system, df[['Prices']])
                
                print("\n" + "="*30)
                print(" RISULTATO VALUTAZIONE")
                print("="*30)
                print(df_ledger[['Date', 'Action', 'Price', 'CashFlow', 'Inventory']].to_string(index=False))
                print("-" * 30)
                print(f"💰 VALORE NETTO TOTALE (NPV): ${value:,.2f}")
                print("="*30)
            except Exception as e:
                print(f"❌ Errore nel calcolo pricing: {e}")
        else:
            print("⚠️ Date insufficienti per la valutazione.")

    # 6. Interfaccia Utente (Loop di previsione)
    print("\n" + "-"*50)
    print(" SISTEMA PRONTO. Inserisci una data target.")
    
    while True:
        user_input = input("\n> Data (MM/GG/AA): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        
        price, type_lbl, msg = model_system.predict_value(user_input, df[['Prices']])
        
        if price:
            print(f"   📊 {type_lbl}: ${price:.2f} | {msg}")
        else:
            print(f"   ❌ {msg}")

if __name__ == "__main__":
    main()
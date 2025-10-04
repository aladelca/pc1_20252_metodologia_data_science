# prediction.py CORREGIDO
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

def predict(product_sku, date, historical_data_path=None):
    """
    Predice unidades para un producto y fecha específicos
    """
    # 1. Cargar modelo
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "artifacts" / "trained_models" / "xgboost_model.pkl"
    model, features = joblib.load(model_path)
    
    pred_date = pd.to_datetime(date)
    
    # 2. VERIFICAR SI EL PRODUCTO EXISTE EN HISTÓRICO
    product_exists = False
    product_stats = {}
    
    if historical_data_path and Path(historical_data_path).exists():
        try:
            historical_data = pd.read_parquet(historical_data_path)
            historical_data['parsed_date'] = pd.to_datetime(historical_data['parsed_date'])
            
            product_data = historical_data[historical_data['product_sku'] == product_sku]
            if not product_data.empty:
                product_exists = True
                product_data = product_data.sort_values('parsed_date')
                last_row = product_data.iloc[-1]
                
                # Obtener estadísticas del producto
                product_stats = {
                    'avg_units': product_data['units_sold'].mean(),
                    'max_units': product_data['units_sold'].max(),
                    'min_units': product_data['units_sold'].min(),
                    'last_units': last_row['units_sold'] if 'units_sold' in last_row else 0
                }
                
        except Exception as e:
            print(f"⚠️ Error cargando histórico: {e}")
    
    # 3. SI EL PRODUCTO NO EXISTE, DEVOLVER PREDICCIÓN POR DEFECTO
    if not product_exists:
        print(f"❌ PRODUCTO DESCONOCIDO: {product_sku}")
        print(f"💡 No hay datos históricos para este producto")
        return 0  # o un valor por defecto basado en productos similares
    
    # 4. CREAR FEATURES SOLO SI EL PRODUCTO EXISTE
    input_data = {feature: 0 for feature in features}
    
    # Features básicas
    input_data.update({
        'dayofweek': pred_date.dayofweek,
        'is_weekend': 1 if pred_date.dayofweek >= 5 else 0,
        'month': pred_date.month,
        'year': pred_date.year,
        'day': pred_date.day,
        'month_sin': np.sin(2 * np.pi * pred_date.month / 12),
        'month_cos': np.cos(2 * np.pi * pred_date.month / 12),
    })
    
    # Copiar features del histórico
    for feature in features:
        if feature in last_row and pd.notna(last_row[feature]):
            input_data[feature] = last_row[feature]
    
    # 5. PREDECIR
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df[features])[0]
    predicted_units = max(0, round(prediction, 1))
    
    print(f"✅ PRODUCTO: {product_sku} (EXISTE)")
    print(f"📅 FECHA: {pred_date.date()}")
    print(f"📊 Últimas ventas: {product_stats.get('last_units', 'N/A')}")
    print(f"📦 UNIDADES PREDICHAS: {predicted_units}")
    
    return predicted_units

# Función para explorar productos disponibles
def list_available_products(historical_data_path):
    """Mostrar productos que SÍ pueden predecirse"""
    try:
        historical_data = pd.read_parquet(historical_data_path)
        products = historical_data['product_sku'].unique()
        print(f"📋 PRODUCTOS DISPONIBLES para predicción ({len(products)}):")
        for i, product in enumerate(products[:10]):  # Mostrar primeros 10
            print(f"   {i+1}. {product}")
        if len(products) > 10:
            print(f"   ... y {len(products) - 10} más")
        return products
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    historical_path = project_root / "data" / "processed" / "data_training.parquet"
    
    # Mostrar productos disponibles
    available_products = list_available_products(historical_path)
    
    print("\n" + "="*50)
    print("🔮 PRUEBAS DE PREDICCIÓN:")
    print("="*50)
    
    # Probar con producto que SÍ existe
    #if len(available_products) > 0:
    #    real_product = available_products[0]
    #    predict(real_product, "2024-01-20", historical_path)
    
    print("\n" + "-"*30)
    
    # Probar con producto que NO existe
    #fake_product = "PRODUCTO_INEXISTENTE_123"
    #predict(fake_product, "2024-01-20", historical_path)
    predict("GGOEAAAJ031914", "2025-10-04", historical_path)
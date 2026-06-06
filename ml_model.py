import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

EXPIRY_DATE = pd.to_datetime('2026-01-27 15:30:00', format='%Y-%m-%d %H:%M:%S')

def load_data(path="dataset.csv"):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
    return df

def extract_strikes(cols):
    strikes = {}
    for col in cols:
        if 'CE' in col or 'PE' in col:
            strike = int(col.replace('NIFTY27JAN26', '').replace('CE', '').replace('PE', ''))
            strikes[col] = strike
    return strikes

def create_validation_set(df, mask_prob=0.1, random_seed=42):
    np.random.seed(random_seed)
    df_val = df.copy()
    option_cols = [c for c in df.columns if 'CE' in c or 'PE' in c]
    
    mask = np.random.rand(*df_val[option_cols].shape) < mask_prob
    orig_isna = df_val[option_cols].isna()
    mask = mask & ~orig_isna
    
    true_values = df_val[option_cols].values[mask]
    
    masked_data = df_val[option_cols].values.copy()
    masked_data[mask] = np.nan
    df_val[option_cols] = masked_data
    
    return df_val, mask, true_values

def cross_sectional_interpolate(df, method='index'):
    df_filled = df.copy()
    ce_cols = [c for c in df.columns if 'CE' in c]
    pe_cols = [c for c in df.columns if 'PE' in c]
    
    ce_strikes = extract_strikes(ce_cols)
    pe_strikes = extract_strikes(pe_cols)
    
    ce_cols_sorted = sorted(ce_cols, key=lambda c: ce_strikes[c])
    pe_cols_sorted = sorted(pe_cols, key=lambda c: pe_strikes[c])
    
    ce_strike_vals = [ce_strikes[c] for c in ce_cols_sorted]
    pe_strike_vals = [pe_strikes[c] for c in pe_cols_sorted]
    
    for idx, row in df.iterrows():
        # CE
        y_ce = row[ce_cols_sorted].values.astype(float)
        valid_ce = ~np.isnan(y_ce)
        if valid_ce.sum() > 1:
            s = pd.Series(y_ce, index=ce_strike_vals)
            s = s.interpolate(method=method, limit_direction='both')
            df_filled.loc[idx, ce_cols_sorted] = s.values
                
        # PE
        y_pe = row[pe_cols_sorted].values.astype(float)
        valid_pe = ~np.isnan(y_pe)
        if valid_pe.sum() > 1:
            s = pd.Series(y_pe, index=pe_strike_vals)
            s = s.interpolate(method=method, limit_direction='both')
            df_filled.loc[idx, pe_cols_sorted] = s.values
                
    return df_filled

def prepare_features(df_input):
    # Cross-sectional feature
    df_cs = cross_sectional_interpolate(df_input, method='index')
    # Use ffill for any remaining nans in CS interpolation (edges/blank rows)
    option_cols = [c for c in df_cs.columns if 'CE' in c or 'PE' in c]
    df_cs[option_cols] = df_cs[option_cols].ffill()
    
    # Melt input
    df_long = df_input.melt(id_vars=['datetime', 'underlying_price'], var_name='option', value_name='iv')
    
    # Melt cross-sectional estimates
    df_cs_long = df_cs.melt(id_vars=['datetime', 'underlying_price'], var_name='option', value_name='cs_iv')
    df_long['cs_iv'] = df_cs_long['cs_iv']
    
    # Extract properties
    df_long['strike'] = df_long['option'].str.extract(r'(\d+)').astype(int)
    df_long['is_ce'] = df_long['option'].str.contains('CE').astype(int)
    
    # Features
    df_long['moneyness'] = df_long['strike'] / df_long['underlying_price']
    df_long['log_moneyness'] = np.log(df_long['moneyness'])
    
    df_long['time_to_expiry'] = (EXPIRY_DATE - df_long['datetime']).dt.total_seconds() / 60.0 # in minutes
    
    # Sort to create lagged features safely (no lookahead)
    df_long = df_long.sort_values(by=['option', 'datetime']).reset_index(drop=True)
    
    # Lagged features (using groupby)
    # We want lagged true IV, so we forward fill the 'iv' column per option
    df_long['lagged_iv'] = df_long.groupby('option')['iv'].shift(1).ffill()
    
    # Lagged underlying
    df_long['lagged_underlying'] = df_long.groupby('option')['underlying_price'].shift(1)
    df_long['underlying_return'] = np.log(df_long['underlying_price'] / df_long['lagged_underlying'].replace(0, np.nan))
    
    # Drop where cs_iv is missing (extremely rare, start of time series)
    # But wait, we need to predict all missing, so let's fillna with some naive value or ffill/bfill
    df_long['cs_iv'] = df_long['cs_iv'].bfill()
    df_long['lagged_iv'] = df_long['lagged_iv'].fillna(df_long['cs_iv'])
    df_long['underlying_return'] = df_long['underlying_return'].fillna(0)
    
    return df_long

def main():
    df = load_data("dataset.csv")
    df_val, mask, true_vals = create_validation_set(df)
    
    print("Preparing features...")
    df_long = prepare_features(df_val)
    
    # Split into train/test
    # Train: where 'iv' is not missing
    # Test: where 'iv' was masked
    # We will get the mask indices by matching datetime and option
    
    # Wait, instead of joining masks, we can just use df_long['iv'].notna() for training.
    # The masked values are NaNs in df_val, so they are NaNs in df_long['iv'].
    
    train_data = df_long[df_long['iv'].notna()].copy()
    test_data = df_long[df_long['iv'].isna()].copy() # this includes original NaNs and masked ones
    
    features = ['moneyness', 'log_moneyness', 'time_to_expiry', 'is_ce', 'cs_iv', 'lagged_iv', 'underlying_return', 'underlying_price', 'strike']
    # Train on residuals
    train_data['iv_residual'] = train_data['iv'] - train_data['cs_iv']
    test_data['iv_residual'] = test_data['iv'] - test_data['cs_iv'] # This is technically NaN, but we will predict it
    
    target = 'iv_residual'
    
    print("Training LightGBM on residuals...")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42)
    model.fit(train_data[features], train_data[target])
    
    # Predict on all missing data
    test_data['pred_residual'] = model.predict(test_data[features])
    test_data['pred_iv'] = test_data['cs_iv'] + test_data['pred_residual']
    
    # Reconstruct the filled dataframe
    df_filled_long = df_long.copy()
    df_filled_long.loc[df_filled_long['iv'].isna(), 'iv'] = test_data['pred_iv']
    
    df_filled_wide = df_filled_long.pivot(index='datetime', columns='option', values='iv').reset_index()
    # Merge underlying_price back
    df_filled_wide = pd.merge(df[['datetime', 'underlying_price']], df_filled_wide, on='datetime')
    # Reorder columns
    df_filled_wide = df_filled_wide[df.columns]
    
    # Calculate MSE on the masked values
    pred_values = df_filled_wide[df.columns[2:]].values[mask] # Exclude datetime and underlying
    
    mse = mean_squared_error(true_vals, pred_values)
    print(f"LightGBM MSE: {mse:.8f}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_excel('SuDS CSO Clen Data base.xlsx')
df.columns = df.columns.str.strip()

# Convert numeric columns safely
numeric_cols = [
    '% Impermeable total contributing',
    '% Ground Infiltration contributing',
    'Total Permeable (ha)',
    'Total Impermeable (Roads,Roofs) (ha)',
    'Total Ground Infiltration (ha)',
    'Total model catchment (ha)',
    'Total catchment area removed (ha)',
    'Spill reduction (%)'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

@st.cache_data
def load_and_train():
    df_model = df.copy()

    # Feature engineering
    if 'Total catchment area removed (ha)' in df_model.columns and 'Total model catchment (ha)' in df_model.columns:
        df_model['removed_frac_total'] = df_model['Total catchment area removed (ha)'] / df_model['Total model catchment (ha)']
    else:
        df_model['removed_frac_total'] = 0.0

    def compute_removed_frac_imperv(row):
        imp = row.get('Total Impermeable (Roads,Roofs) (ha)', None)
        rem = row.get('Total catchment area removed (ha)', None)
        if pd.notnull(imp) and imp > 0 and pd.notnull(rem):
            return min(1.0, max(0.0, rem / imp))
        return 0.0

    df_model['removed_frac_imperv'] = df_model.apply(compute_removed_frac_imperv, axis=1)

    if '% Impermeable total contributing' in df_model.columns:
        df_model['remaining_perc_imperv'] = df_model['% Impermeable total contributing'] * (1 - df_model['removed_frac_imperv'])
    else:
        df_model['remaining_perc_imperv'] = 0.0

    intended_features = [
        '% Impermeable total contributing',
        '% Ground Infiltration contributing',
        'Total Permeable (ha)',
        'Total Impermeable (Roads,Roofs) (ha)',
        'Total Ground Infiltration (ha)',
        'Total model catchment (ha)',
        'Total catchment area removed (ha)',
        'removed_frac_total',
        'removed_frac_imperv',
        'remaining_perc_imperv'
    ]

    # Use only existing columns
    actual_features = [f for f in intended_features if f in df_model.columns]
    existing_cols = ['Spill reduction (%)'] + actual_features
    if len(actual_features) < len(intended_features):
        missing = set(intended_features) - set(actual_features)
        st.warning(f"Missing columns in dataset: {missing}. Model will use available features only.")

    df_model = df_model.dropna(subset=existing_cols)
    X = df_model[actual_features].values
    y = df_model['Spill reduction (%)'].values

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    defaults = {f: float(df_model[f].median()) for f in actual_features}
    return model, actual_features, defaults

model, FEATURES, DEFAULTS = load_and_train()

# Streamlit UI
st.title('Spill Reduction Predictor')
st.write('Estimate spill reduction based on catchment characteristics.')

# Inputs: areas only
st.header('Input Areas (ha)')
imperv_area = st.number_input('Total Impermeable Area (Roads, Roofs)', value=DEFAULTS.get('Total Impermeable (Roads,Roofs) (ha)', 0.0), min_value=0.0)
permeable_area = st.number_input('Total Permeable Area', value=DEFAULTS.get('Total Permeable (ha)', 0.0), min_value=0.0)
infiltration_area = st.number_input('Total Ground Infiltration Area', value=DEFAULTS.get('Total Ground Infiltration (ha)', 0.0), min_value=0.0)
removed_area = st.number_input('Total Catchment Area Removed', value=DEFAULTS.get('Total catchment area removed (ha)', 0.0), min_value=0.0)

if st.button('Predict'):
    # Calculate total catchment
    total_catchment = imperv_area + permeable_area + infiltration




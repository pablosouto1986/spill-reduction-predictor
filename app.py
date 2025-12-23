import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel('SuDS CSO Clen Data base.xlsx')
    df.columns = df.columns.str.strip()
    # Convert all possible numeric columns to numeric, leave others unchanged
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            # If conversion fails (e.g. for non-numeric columns) leave unchanged
            pass
    return df

df = load_data()

@st.cache_data
def load_and_train():
    df_model = df.copy()

    # Engineered features
    if 'Total catchment area removed (ha)' in df_model and 'Total model catchment (ha)' in df_model:
        df_model['removed_frac_total'] = df_model['Total catchment area removed (ha)'] / df_model['Total model catchment (ha)']
    else:
        df_model['removed_frac_total'] = 0.0

    def compute_removed_frac_imperv(row):
        imp = row.get('Total Impermeable (Roads,Roofs) (ha)')
        rem = row.get('Total catchment area removed (ha)')
        if pd.notnull(imp) and imp > 0 and pd.notnull(rem):
            return min(1.0, max(0.0, rem / imp))
        return 0.0

    df_model['removed_frac_imperv'] = df_model.apply(compute_removed_frac_imperv, axis=1)
    df_model['remaining_perc_imperv'] = df_model.get('% Impermeable total contributing', 0) * (1 - df_model['removed_frac_imperv'])

    # Define intended features based on what exists in the file
    intended = [
        '% Impermeable total contributing',
        '% Permeable/GI total contributing',
        'Total Impermeable (Roads,Roofs) (ha)',
        'Total Permeable/GI (ha)',
        'Total model catchment (ha)',
        'Total catchment area removed (ha)',
        'removed_frac_total',
        'removed_frac_imperv',
        'remaining_perc_imperv'
    ]
    actual = [f for f in intended if f in df_model.columns]
    # Warn about missing columns
    if len(actual) < len(intended):
        missing = set(intended) - set(actual)
        # Because st.write/warning cannot be called in cache context, return or store missing elsewhere
        # We'll handle messaging in the UI context
    df_model = df_model.dropna(subset=['Spill reduction (%)'] + actual)
    X = df_model[actual].values
    y = df_model['Spill reduction (%)'].values
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    defaults = {f: float(df_model[f].median()) for f in actual}
    return model, actual, defaults

model, FEATURES, DEFAULTS = load_and_train()

# Streamlit UI
st.title('Spill Reduction Predictor')
st.write('Estimate spill reduction based on catchment characteristics.')
st.header('Input Areas (ha)')
imperv_area = st.number_input('Total Impermeable Area (Roads, Roofs)', min_value=0.0)
permeable_area = st.number_input('Total Permeable Area', min_value=0.0)
infiltration_area = st.number_input('Total Ground Infiltration Area', min_value=0.0)
removed_area = st.number_input('Total Catchment Area Removed', min_value=0.0)

if st.button('Predict'):
    total_catchment = imperv_area + permeable_area + infiltration_area
    # Compute the % contributions
    perc_imp = (imperv_area / total_catchment) * 100 if total_catchment else 0
    perc_perm_gi = ((permeable_area + infiltration_area) / total_catchment) * 100 if total_catchment else 0
    # Build scenario dict
    scenario = {
        '% Impermeable total contributing': perc_imp,
        '% Permeable/GI total contributing': perc_perm_gi,
        'Total Impermeable (Roads,Roofs) (ha)': imperv_area,
        'Total Permeable/GI (ha)': permeable_area + infiltration_area,
        'Total model catchment (ha)': total_catchment,
        'Total catchment area removed (ha)': removed_area,
        'removed_frac_total': removed_area / total_catchment if total_catchment else 0,
        'removed_frac_imperv': min(1.0, max(0.0, removed_area / imperv_area)) if imperv_area else 0,
        'remaining_perc_imperv': perc_imp * (1 - (min(1.0, max(0.0, removed_area / imperv_area)) if imperv_area else 0))
    }
    # Prepare input for prediction
    X_input = np.array([[scenario.get(f, DEFAULTS.get(f, 0.0)) for f in FEATURES]])
    pred = model.predict(X_input)[0] * 100
    preds_trees = np.array([t.predict(X_input) for t in model.estimators_]).ravel() * 100
    certainty = max(0.0, 100.0 - preds_trees.std())
    st.subheader('Result')
    st.write(f"**Predicted Spill Reduction:** {pred:.2f} %")
    st.write(f"**Certainty Score:** {certainty:.2f} / 100")

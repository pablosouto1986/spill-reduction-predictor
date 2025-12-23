import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel('SuDS CSO Clen Data base.xlsx')
    df.columns = df.columns.str.strip()
    # Convert numeric columns where possible
    for col in df.columns:
        # Attempt conversion to numeric if not object
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass
    return df

df = load_data()

@st.cache_data
def load_and_train(df):
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
    # Define intended features according to file structure
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
    # Drop rows with NaN in target or features
    df_model = df_model.dropna(subset=['Spill reduction (%)'] + actual)
    X = df_model[actual].values
    y = df_model['Spill reduction (%)'].values
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    defaults = {f: float(df_model[f].median()) for f in actual}
    return model, actual, defaults

model, FEATURES, DEFAULTS = load_and_train(df)

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
    # Avoid division by zero
    if total_catchment <= 0:
        st.error('Total catchment must be greater than zero.')
    else:
        # Compute fractions (no *100 here)
        perc_imp = imperv_area / total_catchment
        perc_perm_gi = (permeable_area + infiltration_area) / total_catchment
        removed_frac_total = removed_area / total_catchment
        removed_frac_imperv = min(1.0, max(0.0, removed_area / imperv_area)) if imperv_area else 0.0
        remaining_perc_imperv = perc_imp * (1 - removed_frac_imperv)
        # Build scenario dict
        scenario = {
            '% Impermeable total contributing': perc_imp,
            '% Permeable/GI total contributing': perc_perm_gi,
            'Total Impermeable (Roads,Roofs) (ha)': imperv_area,
            'Total Permeable/GI (ha)': permeable_area + infiltration_area,
            'Total model catchment (ha)': total_catchment,
            'Total catchment area removed (ha)': removed_area,
            'removed_frac_total': removed_frac_total,
            'removed_frac_imperv': removed_frac_imperv,
            'remaining_perc_imperv': remaining_perc_imperv,
        }
        # Apply rule-based override
        if perc_imp < 0.05 and removed_frac_total < 0.02:
            pred_frac = 0.05
            # Certainty could be set high since rule applies
            certainty = 100.0
        else:
            # Prepare input aligned with FEATURES
            X_input = np.array([[scenario.get(f, DEFAULTS.get(f, 0.0)) for f in FEATURES]])
            pred_frac = model.predict(X_input)[0]
            # Compute certainty: lower std leads to higher certainty
            preds_trees = np.array([t.predict(X_input) for t in model.estimators_]).ravel()
            certainty = max(0.0, 100.0 - preds_trees.std() * 100)
        # Convert to percentage for display
        pred_percent = pred_frac * 100
        st.subheader('Result')
        st.write(f'**Predicted Spill Reduction:** {pred_percent:.2f} %')
        st.write(f'**Certainty Score:** {certainty:.2f} / 100')

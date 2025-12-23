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
    df_model['removed_frac_total'] = df_model['Total catchment area removed (ha)'] / df_model['Total model catchment (ha)']
    def compute_removed_frac_imperv(row):
        imp = row['Total Impermeable (Roads,Roofs) (ha)']
        rem = row['Total catchment area removed (ha)']
        if pd.notnull(imp) and imp > 0:
            return min(1.0, max(0.0, rem / imp))
        return 0.0
    df_model['removed_frac_imperv'] = df_model.apply(compute_removed_frac_imperv, axis=1)
    df_model['remaining_perc_imperv'] = df_model['% Impermeable total contributing'] * (1 - df_model['removed_frac_imperv'])

    features = [
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

    df_model = df_model.dropna(subset=['Spill reduction (%)'] + features)
    X = df_model[features].values
    y = df_model['Spill reduction (%)'].values

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    defaults = {f: float(df_model[f].median()) for f in features}
    return model, features, defaults

model, FEATURES, DEFAULTS = load_and_train()

# Streamlit UI
st.title('Spill Reduction Predictor')
st.write('Estimate spill reduction based on catchment characteristics.')

# Inputs: areas only
st.header('Input Areas (ha)')
imperv_area = st.number_input('Total Impermeable Area (Roads, Roofs)', value=DEFAULTS['Total Impermeable (Roads,Roofs) (ha)'], min_value=0.0)
permeable_area = st.number_input('Total Permeable Area', value=DEFAULTS['Total Permeable (ha)'], min_value=0.0)
infiltration_area = st.number_input('Total Ground Infiltration Area', value=DEFAULTS['Total Ground Infiltration (ha)'], min_value=0.0)
removed_area = st.number_input('Total Catchment Area Removed', value=DEFAULTS['Total catchment area removed (ha)'], min_value=0.0)

if st.button('Predict'):
    # Calculate total catchment
    total_catchment = imperv_area + permeable_area + infiltration_area

    # Compute percentages
    perc_imperv = (imperv_area / total_catchment) * 100 if total_catchment > 0 else 0.0
    perc_infiltration = (infiltration_area / total_catchment) * 100 if total_catchment > 0 else 0.0

    # Build full feature vector
    scenario = DEFAULTS.copy()
    scenario['% Impermeable total contributing'] = perc_imperv
    scenario['% Ground Infiltration contributing'] = perc_infiltration
    scenario['Total Impermeable (Roads,Roofs) (ha)'] = imperv_area
    scenario['Total Permeable (ha)'] = permeable_area
    scenario['Total Ground Infiltration (ha)'] = infiltration_area
    scenario['Total model catchment (ha)'] = total_catchment
    scenario['Total catchment area removed (ha)'] = removed_area
    scenario['removed_frac_total'] = removed_area / total_catchment if total_catchment > 0 else 0.0
    scenario['removed_frac_imperv'] = min(1.0, max(0.0, removed_area / imperv_area if imperv_area > 0 else 0.0))
    scenario['remaining_perc_imperv'] = perc_imperv * (1 - scenario['removed_frac_imperv'])

    X_input = np.array([[scenario[f] for f in FEATURES]])
    pred = model.predict(X_input)[0]

    # Certainty score = std deviation across trees
    preds_trees = np.array([tree.predict(X_input) for tree in model.estimators_]).ravel()
    certainty = max(0.0, 100.0 - preds_trees.std())  # Higher std = lower certainty

    st.subheader('Prediction Result')
    st.write(f"**Predicted Spill Reduction:** {pred:.2f} %")
    st.write(f"**Certainty Score:** {certainty:.2f} / 100")



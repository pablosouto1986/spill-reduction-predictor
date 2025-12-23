import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load data once
df = pd.read_excel('SuDS CSO Clen Data base.xlsx')

# Convert numeric columns safely
numeric_cols = [
    '% Impermeable total contributing',
    '% Permeable/GI total contributing',
    'Total Permeable/GI (ha)',
    'Total Impermeable (Roads,Roofs) (ha)',
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
    # Engineer features
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
        '% Permeable/GI total contributing',
        'Total Permeable/GI (ha)',
        'Total Impermeable (Roads,Roofs) (ha)',
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

st.title('Spill Reduction Predictor')
st.write('Estimate spill reduction based on pre-modelling catchment characteristics.')

# Scenario A
st.header('Scenario A')
scenario_a = {}
for f in FEATURES:
    vals = pd.to_numeric(df[f], errors='coerce').dropna()
    min_v, max_v = (0.0, 1.0) if vals.empty else (float(vals.min()), float(vals.max()))
    default = DEFAULTS.get(f, min_v)
    scenario_a[f] = st.number_input(f + ' (A)', value=default, min_value=min_v, max_value=max_v)

# Scenario B
st.header('Scenario B')
scenario_b = {}
for f in FEATURES:
    vals = pd.to_numeric(df[f], errors='coerce').dropna()
    min_v, max_v = (0.0, 1.0) if vals.empty else (float(vals.min()), float(vals.max()))
    default = DEFAULTS.get(f, min_v)
    scenario_b[f] = st.number_input(f + ' (B)', value=default, min_value=min_v, max_value=max_v)

if st.button('Predict'):
    X_a = np.array([[scenario_a[f] for f in FEATURES]])
    X_b = np.array([[scenario_b[f] for f in FEATURES]])
    pred_a = model.predict(X_a)[0]
    pred_b = model.predict(X_b)[0]
    preds_a_trees = np.array([tree.predict(X_a) for tree in model.estimators_]).ravel()
    preds_b_trees = np.array([tree.predict(X_b) for tree in model.estimators_]).ravel()
    std_a, std_b = float(preds_a_trees.std()), float(preds_b_trees.std())

    st.subheader('Predictions')
    st.write(f'Scenario A: {pred_a:.2f} % ± {std_a:.2f}')
    st.write(f'Scenario B: {pred_b:.2f} % ± {std_b:.2f}')
    st.write(f'Difference (B - A): {pred_b - pred_a:.2f} %')

st.subheader('Feature Importance')
feat_imp = model.feature_importances_
imp_df = pd.DataFrame({'feature': FEATURES, 'importance': feat_imp}).sort_values('importance', ascending=False).set_index('feature')
st.bar_chart(imp_df)

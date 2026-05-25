import streamlit as st
import joblib
import numpy as np

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🔍 Credit Card Fraud Detection')
st.write('Enter transaction details to detect fraud')

st.subheader('Transaction Details')

amount = st.number_input('Amount', min_value=0.0)

v_features = []
for i in range(1, 29):
    v = st.number_input(f'V{i}', value=0.0)
    v_features.append(v)

if st.button('Predict'):
    features = v_features + [amount]
    features = np.array(features).reshape(1, -1)
    features[0, -1] = scaler.transform([[features[0, -1]]])[0][0]
    
    prob = model.predict_proba(features)[0][1]
    st.write(f'Fraud Probability: {prob:.4f}')
    
    if prob >= 0.3:
        st.error('🚨 Fraud Detected!')
    else:
        st.success('✅ Transaction is Normal')
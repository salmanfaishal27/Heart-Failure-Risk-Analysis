import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load All Files

with open('model_xgb.pkl', 'rb') as file_model:
  model_xgb = pickle.load(file_model)

with open('model_scaler.pkl', 'rb') as file_scaler:
  model_scaler = pickle.load(file_scaler)

with open('data.txt', 'r') as file_data:
  data = json.load(file_data)

  st.title('Heart failure prediction')

def run():
    # Membuat form
    with st.form(key='form_parameters'):
        age =st.number_input('Age', min_value=0, max_value=100, value=25, step =1,help='Umur Pasien')
        anaemia =st.selectbox('Anaemia', (0,1), index=1 ,help='0 jika tidak mempunyai anemia, 1 jika punya anemia')
        creatinine_phosphokinase = st.number_input('Creatine Phosphokinase', min_value=0, max_value=None, value=25, step =1)
        diabetes = st.selectbox('Diabetes', (0,1), index=1 ,help='0 jika tidak mempunyai diabetes, 1 jika punya diabetes')
        ejection_fraction = st.number_input('Ejection Fraction', min_value=0, value=25, max_value=None, step=1)
        high_blood_pressure = st.selectbox('High Blood Pressure', (0,1), index=1, help='0 jika tidak mempunyai darah tinggi, 1 jika punya darah tinggi')
        platelets = st.number_input('Platelets', min_value=0, max_value=None, value=25, step =1, help='jumlah trombosit')
        serum_creatinine = st.number_input('Serum Creatinine', min_value=0, max_value=None,value=50, step=1)
        serum_sodium = st.number_input('Serum Sodium', min_value=0, max_value=None,value=50)
        sex = st.selectbox('Sex', (0,1), index=1 ,help='0 jika perempuan, 1 jika pria')
        smoking = st.selectbox('Smoking', (0,1), index=1 ,help='0 jika tidak merokok, 1 jika merokok')
        time = st.number_input('time',min_value=0, max_value=None, value=25, step =1, help='Follow up period (days)')

        

        submitted = st.form_submit_button('Predict')

    data_inf = {
    'age': age,
    'anaemia': anaemia, 
    'creatinine_phosphokinase': creatinine_phosphokinase, 
    'diabetes': diabetes, 
    'ejection_fraction': ejection_fraction, 
    'high_blood_pressure': high_blood_pressure, 
    'platelets': platelets,
    'serum_creatinine': serum_creatinine, 
    'serum_sodium': serum_sodium, 
    'sex': sex, 
    'smoking': smoking,
    'time': time
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        data_inf_num = data_inf[data]
        data_inf_scaled = model_scaler.transform(data_inf_num)        
        y_pred_inf = model_xgb.predict(data_inf_scaled)
        st.write('# Death Event :', str(int(y_pred_inf)))

      
if __name__ == '__main__':
    run()
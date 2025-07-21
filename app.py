import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="wide"
)

# Título principal
st.title("🩺 Predictor de Diabetes")
st.write("Ingrese los datos médicos para evaluar el riesgo de diabetes")

# Función para cargar o entrenar el modelo
@st.cache_resource
def load_or_train_model():
    # Datos de ejemplo para entrenar el modelo (dataset)
    # En un caso real, cargarías tu dataset desde un archivo CSV
    # data = pd.read_csv('diabetes_dataset.csv')
    
    # Para este ejemplo, creo datos sintéticos
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'pregnancies': np.random.randint(0, 15, n_samples),
        'glucose': np.random.randint(70, 200, n_samples),
        'blood_pressure': np.random.randint(50, 120, n_samples),
        'skin_thickness': np.random.randint(10, 50, n_samples),
        'insulin': np.random.randint(15, 300, n_samples),
        'bmi': np.random.uniform(18, 45, n_samples),
        'diabetes_pedigree': np.random.uniform(0.08, 2.5, n_samples),
        'age': np.random.randint(21, 70, n_samples)
    })
    
    # Crear target basado en reglas (simulando diagnóstico real)
    data['outcome'] = ((data['glucose'] > 140) | 
                      (data['bmi'] > 30) | 
                      (data['age'] > 50)).astype(int)
    
    X = data.drop('outcome', axis=1)
    y = data['outcome']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Función para guardar nuevos datos
def save_new_data(data_dict, prediction, probability):
    """Guarda los nuevos datos ingresados en un CSV"""
    
    # Nombre del archivo donde guardar
    filename = 'nuevos_datos_diabetes.csv'
    
    # Agregar timestamp y predicción
    data_dict['fecha_ingreso'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_dict['prediccion'] = prediction
    data_dict['probabilidad'] = probability
    
    # Crear DataFrame
    new_data = pd.DataFrame([data_dict])
    
    # Si el archivo existe, agregar datos; si no, crear nuevo
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data
    
    # Guardar archivo
    updated_data.to_csv(filename, index=False)
    return len(updated_data)

# Cargar modelo
model, scaler = load_or_train_model()

# Crear formulario de entrada
st.subheader("📋 Ingrese los datos del paciente:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Embarazos previos", min_value=0, max_value=15, value=0)
    glucose = st.number_input("Nivel de glucosa (mg/dL)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Presión arterial (mmHg)", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("Grosor de piel (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulina (μU/mL)", min_value=0, max_value=500, value=50)
    bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input("Función de pedigrí diabético", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Edad", min_value=1, max_value=100, value=30)

# Botón para hacer predicción
if st.button("🔍 Realizar Predicción", type="primary"):
    # Preparar datos
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Escalar datos
    input_scaled = scaler.transform(input_data)
    
    # Hacer predicción
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Mostrar resultados
    st.subheader("📊 Resultado de la Predicción:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ ALTO RIESGO de diabetes")
        else:
            st.success("✅ BAJO RIESGO de diabetes")
    
    with col2:
        st.metric("Probabilidad de diabetes", f"{probability[1]*100:.1f}%")
    
    with col3:
        st.metric("Probabilidad de no diabetes", f"{probability[0]*100:.1f}%")
    
    # Guardar datos
    data_dict = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree': diabetes_pedigree,
        'age': age
    }
    
    try:
        total_records = save_new_data(data_dict, prediction, probability[1])
        st.success(f"✅ Datos guardados exitosamente. Total de registros: {total_records}")
    except Exception as e:
        st.error(f"Error al guardar datos: {e}")
    
    # Mostrar gráfico de probabilidades
    st.subheader("📈 Análisis de Probabilidades:")
    prob_df = pd.DataFrame({
        'Resultado': ['Sin Diabetes', 'Con Diabetes'],
        'Probabilidad': [probability[0]*100, probability[1]*100]
    })
    st.bar_chart(prob_df.set_index('Resultado'))

# Sidebar con información adicional
st.sidebar.header("ℹ️ Información")
st.sidebar.info("""
Esta aplicación utiliza Machine Learning para predecir el riesgo de diabetes basado en factores médicos.

**Parámetros evaluados:**
- Embarazos previos
- Nivel de glucosa
- Presión arterial
- Grosor de piel
- Insulina
- IMC
- Función de pedigrí diabético
- Edad

**Nota:** Esta es una herramienta de apoyo. Siempre consulte con un profesional médico.
""")

# Mostrar datos guardados (opcional)
if st.sidebar.button("📁 Ver datos guardados"):
    if os.path.exists('nuevos_datos_diabetes.csv'):
        saved_data = pd.read_csv('nuevos_datos_diabetes.csv')
        st.subheader("📋 Historial de predicciones:")
        st.dataframe(saved_data)
        
        # Opción para descargar
        csv = saved_data.to_csv(index=False)
        st.download_button(
            label="⬇️ Descargar datos",
            data=csv,
            file_name='historial_diabetes.csv',
            mime='text/csv'
        )
    else:
        st.info("No hay datos guardados aún.")

# Footer
st.markdown("---")
st.markdown("🩺 **Predictor de Diabetes** - Desarrollado para presentación final")
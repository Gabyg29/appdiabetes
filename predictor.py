import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_history = []
        self.model_metrics = {}
    
    def create_sample_data(self, n_samples=1000):
        """
        Crea datos de muestra para diabetes si no hay archivo CSV
        """
        print("Creando datos de muestra...")
        np.random.seed(42)
        
        # Generar datos sintéticos realistas
        data = {
            'Pregnancies': np.random.poisson(3, n_samples),
            'Glucose': np.random.normal(120, 32, n_samples),
            'BloodPressure': np.random.normal(69, 19, n_samples),
            'SkinThickness': np.random.normal(20, 16, n_samples),
            'Insulin': np.random.exponential(80, n_samples),
            'BMI': np.random.normal(32, 8, n_samples),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
            'Age': np.random.gamma(2, 15, n_samples)
        }
        
        # Asegurar valores positivos y rangos realistas
        data['Glucose'] = np.clip(data['Glucose'], 0, 300)
        data['BloodPressure'] = np.clip(data['BloodPressure'], 0, 200)
        data['SkinThickness'] = np.clip(data['SkinThickness'], 0, 100)
        data['Insulin'] = np.clip(data['Insulin'], 0, 850)
        data['BMI'] = np.clip(data['BMI'], 10, 70)
        data['Age'] = np.clip(data['Age'], 18, 90)
        
        # Crear variable objetivo con lógica médica
        outcome = []
        for i in range(n_samples):
            risk_score = 0
            
            # Factores de riesgo principales
            if data['Glucose'][i] > 126: risk_score += 3  # Diabetes si glucosa > 126
            elif data['Glucose'][i] > 100: risk_score += 1  # Prediabetes
            
            if data['BMI'][i] > 30: risk_score += 2  # Obesidad
            elif data['BMI'][i] > 25: risk_score += 1  # Sobrepeso
            
            if data['Age'][i] > 45: risk_score += 1
            if data['Pregnancies'][i] > 0: risk_score += 0.5
            if data['BloodPressure'][i] > 140: risk_score += 1
            
            # Probabilidad basada en score de riesgo
            prob = min(risk_score / 8, 0.9)
            outcome.append(1 if np.random.random() < prob else 0)
        
        data['Outcome'] = outcome
        
        self.data = pd.DataFrame(data)
        print(f"Datos de muestra creados: {self.data.shape}")
        print(f"Distribución de diabetes: {self.data['Outcome'].value_counts().to_dict()}")
        
        return True
    
    def load_data(self, csv_path=None):
        """
        Carga los datos desde un archivo CSV o crea datos de muestra
        """
        if csv_path is None:
            return self.create_sample_data()
            
        try:
            # Intentar diferentes codificaciones
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(csv_path, encoding=encoding)
                    print(f"Datos cargados exitosamente con codificación {encoding}")
                    print(f"Shape: {self.data.shape}")
                    return True
                except UnicodeDecodeError:
                    continue
                    
        except FileNotFoundError:
            print(f"Archivo '{csv_path}' no encontrado. Usando datos de muestra...")
            return self.create_sample_data()
        except Exception as e:
            print(f"Error al cargar datos: {e}. Usando datos de muestra...")
            return self.create_sample_data()
    
    def explore_data(self):
        """
        Explora y muestra información sobre los datos
        """
        print("=== EXPLORACIÓN DE DATOS ===")
        print(f"Forma del dataset: {self.data.shape}")
        print(f"Columnas: {list(self.data.columns)}")
        
        print("\nPrimeras 5 filas:")
        print(self.data.head())
        
        print("\nInformación estadística:")
        print(self.data.describe())
        
        print("\nValores nulos:")
        print(self.data.isnull().sum())
        
        # Distribución de la variable objetivo
        if 'Outcome' in self.data.columns:
            print(f"\nDistribución de diabetes:")
            print(self.data['Outcome'].value_counts())
            print(f"Porcentaje con diabetes: {self.data['Outcome'].mean()*100:.1f}%")
    
    def preprocess_data(self):
        """
        Preprocesa los datos para el entrenamiento
        """
        print("Preprocesando datos...")
        
        # Manejar valores nulos
        if self.data.isnull().sum().sum() > 0:
            print("Rellenando valores nulos con la mediana...")
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(
                self.data[numeric_columns].median()
            )
        
        # Separar características y variable objetivo
        if 'Outcome' in self.data.columns:
            X = self.data.drop('Outcome', axis=1)
            y = self.data['Outcome']
        else:
            # Si no hay columna 'Outcome', usar la última columna
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar los datos
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Datos de entrenamiento: {X_train_scaled.shape}")
        print(f"Datos de prueba: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self):
        """
        Entrena el modelo de Random Forest
        """
        print("=== ENTRENANDO MODELO ===")
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        # Crear y entrenar el modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'  # Para manejar clases desbalanceadas
        )
        
        print("Entrenando Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calcular métricas
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Guardar métricas
        self.model_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        }
        
        # Guardar historial
        self.training_history.append({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'model_type': 'RandomForest'
        })
        
        print(f"Accuracy entrenamiento: {train_accuracy:.4f}")
        print(f"Accuracy prueba: {test_accuracy:.4f}")
        print(f"Diferencia (overfitting): {abs(train_accuracy - test_accuracy):.4f}")
        
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred_test))
        
        return test_accuracy
    
    def predict_single(self, features):
        """
        Hace predicción para una sola muestra
        """
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        # Convertir a DataFrame si es necesario
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features], columns=self.feature_names)
        elif isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Escalar los datos
        features_scaled = self.scaler.transform(features)
        
        # Hacer predicción
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'prediction_text': 'Diabetes' if prediction == 1 else 'Sin Diabetes',
            'probability_diabetes': float(probability[1]),
            'probability_no_diabetes': float(probability[0]),
            'confidence': float(max(probability))
        }
    
    def predict_batch(self, features_df):
        """
        Hace predicciones para múltiples muestras
        """
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        features_scaled = self.scaler.transform(features_df)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def save_model(self, model_path="diabetes_model.pkl"):
        """
        Guarda el modelo entrenado
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, model_path)
        print(f"Modelo guardado en: {model_path}")
    
    def load_model(self, model_path="diabetes_model.pkl"):
        """
        Carga un modelo previamente guardado
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data.get('model_metrics', {})
            self.training_history = model_data.get('training_history', [])
            print("Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
    
    def get_feature_importance(self):
        """
        Obtiene la importancia de las características
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self):
        """
        Visualiza la importancia de las características
        """
        importance_df = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title('Importancia de las Características para Predicción de Diabetes')
        plt.xlabel('Importancia')
        plt.ylabel('Características')
        plt.tight_layout()
        plt.show()
        
        return importance_df

def main():
    """
    Función principal para ejecutar el pipeline completo
    """
    print("=== PREDICTOR DE DIABETES ===")
    
    # Crear instancia del predictor
    predictor = DiabetesPredictor()
    
    # Cargar datos (usará datos de muestra si no encuentra CSV)
    predictor.load_data()
    
    # Explorar datos
    predictor.explore_data()
    
    # Entrenar modelo
    accuracy = predictor.train_model()
    
    # Guardar modelo
    predictor.save_model()
    
    # Mostrar importancia de características
    print("\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
    importance_df = predictor.get_feature_importance()
    print(importance_df)
    
    # Ejemplo de predicción
    print("\n=== EJEMPLO DE PREDICCIÓN ===")
    
    # Datos de ejemplo para predicción
    sample_patient = {
        'Pregnancies': 2,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    try:
        result = predictor.predict_single(sample_patient)
        print(f"Datos del paciente: {sample_patient}")
        print(f"Predicción: {result['prediction_text']}")
        print(f"Probabilidad de diabetes: {result['probability_diabetes']:.3f}")
        print(f"Confianza: {result['confidence']:.3f}")
        
    except Exception as e:
        print(f"Error en predicción: {e}")
    
    print("\n=== PIPELINE COMPLETADO EXITOSAMENTE ===")
    print(f"Accuracy del modelo: {accuracy:.4f}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
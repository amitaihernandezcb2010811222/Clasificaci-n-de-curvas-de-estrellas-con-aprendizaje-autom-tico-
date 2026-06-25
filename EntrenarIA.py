import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("Iniciando entrenamiento de la red neuronal...")

# Cargar datos extraídos, codigo (extractordedatos.py) 
df = pd.read_csv("datos_para_entrenar.csv")

# Quitar filas con valores faltantes
original_count = len(df)
df = df.dropna(subset=['Clase'])

if df.empty:
    raise ValueError(
        f"No hay datos de entrenamiento con etiquetas. "
    )

# Variables de entrada y salida
X = df[['logP', 'Amplitud', 'Asimetria', 'Mag_Media']].values
y = df['Clase'].values

# Convertimos de texto a números 
encoder = LabelEncoder()
y_num = encoder.fit_transform(y)

#Extandarizamos los datos 
scaler = StandardScaler()
X_esc = scaler.fit_transform(X)

# Creamos la red neuronal
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), # 4 entradas
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_num)), activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamos con 1000 epocas
modelo.fit(X_esc, y_num, epochs=2500, verbose=0)

# Guardamos el conocimiento 
modelo.save("cerebro_ia.keras")
joblib.dump(scaler, "escalador.pkl")
joblib.dump(encoder, "etiquetas.pkl")

print("La red neuronal ha sido entrenada y guardada.")
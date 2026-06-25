import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.optimize import minimize_scalar

# Calcular la fase 
def calcular_fase(t, P):
    return ((t - np.min(t)) / P) % 1.0

#Calcular la longitud para elegir el mejor periodo
def longitud_cuerda(P, t, mag):
    fase = calcular_fase(t, P)
    idx = np.argsort(fase)
    mag_norm = (mag[idx] - np.min(mag)) / (np.max(mag) - np.min(mag))
    return np.sum(np.sqrt(np.diff(fase[idx])**2 + np.diff(mag_norm)**2))

def extraer_fisica(t, mag):
    # Elegimos el periodo óptimo
    res = minimize_scalar(longitud_cuerda, bounds=(0.01, 1.1), args=(t, mag), method='bounded')
    P = res.x
    # Extracción de 4 variables importantes para que la red reconozca que tipo de estrella variable es
    return [np.log10(P), np.max(mag) - np.min(mag), skew(mag), np.mean(mag)]

# EEtiquetamos y extraemos los datos 
etiquetas_oficiales = {
    'V1': 'RRab', 'V2': 'RRab', 'V3': 'RRab', 'V4': 'RRab', 'V5': 'RRab',
    'V6': 'RRab', 'V7': 'AC', 'V8': 'RRab', 'V9': 'RRab', 'V10': 'RRc',
    'V11': 'RRc', 'V12': 'RRc', 'V13': 'RRc', 'V25': 'RRab', 'V29': 'RRab',
    'V30': 'BL Her', 'V31': 'RRc', 'V32': 'RRc', 'V33': 'SX Phe', 'V34': 'SX Phe',
    'V35': 'Sx Phe', 'V36': 'SX Phe', 'V37': 'SX Phe', 'V38': 'SX Phe', 'V39': 'RRc',
    'V40': 'RRd', 'V41': 'SX Phe', 'V42': 'SR'
}

if __name__ == "__main__":
    carpeta_datos = r"C:\Users\Usuario\Downloads\M92_Todas\Todas"  #Cambiar ruta de los archivos de datos 
    archivos = glob.glob(os.path.join(carpeta_datos, '*_V.dat'))
    datos = []

    print("Extraemos los datos y generamos el archivo 'datos_para_entrenar.csv'...")
    for archivo in archivos:
        nombre = os.path.basename(archivo).split('_')[0]
        
        # Leemos SOLO las primeras dos columnas
        df = pd.read_csv(archivo, delim_whitespace=True, usecols=[0, 1], names=['HJD', 'Mag'])
        features = extraer_fisica(df['HJD'].values, df['Mag'].values)
        
        # En caso de que no aparezca, la marcamos como estrella (desconocida)
        clase_real = etiquetas_oficiales.get(nombre, 'Desconocida')
        
        datos.append([nombre] + features + [clase_real])

    columnas = ['Estrella', 'logP', 'Amplitud', 'Asimetria', 'Mag_Media', 'Clase']
    df_final = pd.DataFrame(datos, columns=columnas)
    
    # Filtramos las desconocidas para que no tengamos ningun problema a la hora de entrenar nuestra red neuronal
    df_final = df_final[df_final['Clase'] != 'Desconocida']
    
    df_final.to_csv("datos_para_entrenar_red_neuronal_V.csv", index=False)
    print("Listo. Archivo 'datos_para_entrenar_red_neuronal_V.csv' generado con todas las clases incluidas.")
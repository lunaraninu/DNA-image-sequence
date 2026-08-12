import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------- CONFIGURACIÓN --------------------
INPUT_CSV = "caracteristicas_glcm.csv"  # Archivo con características GLCM
OUTPUT_DIR = "resultados_clasificacion"

# Características a usar
FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Configuración KNN
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = [3, 5, 7, 9, 11, 13, 15]  # Valores de K a probar
WEIGHTS = ['uniform', 'distance']        # Tipos de peso
METRICS = ['euclidean', 'manhattan', 'minkowski']

# Crear directorio de salida
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------

def load_and_prepare_data(csv_path):
    """
    Carga los datos y prepara las características para clasificación.
    """
    # Cargar datos
    df = pd.read_csv(csv_path)
    
    print(f"📊 Datos cargados: {len(df)} imágenes")
    print(f"   Características disponibles: {df.columns.tolist()}")
    
    # Verificar que existan las características
    available_features = [f for f in FEATURES if f in df.columns]
    if not available_features:
        print("❌ No se encontraron características GLCM en el archivo")
        return None, None, None
    
    print(f"   Características usadas: {available_features}")
    
    # Preparar X (características) e y (etiquetas)
    X = df[available_features].values
    y = df['species'].values  # Usar especie como etiqueta
    
    # Verificar que haya suficientes clases
    unique_species = np.unique(y)
    print(f"   Especies únicas: {len(unique_species)}")
    print(f"   Distribución: {dict(pd.Series(y).value_counts())}")
    
    # Verificar balanceo de clases
    species_counts = pd.Series(y).value_counts()
    min_samples = species_counts.min()
    if min_samples < 2:
        print(f"⚠️ Advertencia: Hay especies con solo {min_samples} muestra. Considerar agrupar especies raras.")
    
    return X, y, df

def find_best_knn(X_train, y_train, X_test, y_test):
    """
    Encuentra los mejores parámetros para KNN usando validación cruzada.
    """
    print("\n🔍 Buscando mejores parámetros para KNN...")
    
    results = []
    best_score = 0
    best_params = None
    
    # Crear un DataFrame para resultados
    results_df = pd.DataFrame()
    
    # Probar diferentes combinaciones
    for n_neighbors in tqdm(N_NEIGHBORS, desc="Probando K"):
        for weights in WEIGHTS:
            for metric in METRICS:
                # Crear y entrenar modelo
                knn = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )
                
                # Validación cruzada estratificada
                cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                # Entrenar con todos los datos de entrenamiento y probar en test
                knn.fit(X_train, y_train)
                test_score = knn.score(X_test, y_test)
                
                # Guardar resultado
                result = {
                    'n_neighbors': n_neighbors,
                    'weights': weights,
                    'metric': metric,
                    'cv_mean': mean_score,
                    'cv_std': std_score,
                    'test_score': test_score
                }
                results.append(result)
                
                # Actualizar mejor modelo
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'weights': weights,
                        'metric': metric
                    }
                    best_model = knn
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Mostrar mejores resultados
    print("\n✅ Mejores parámetros encontrados:")
    print(f"   K = {best_params['n_neighbors']}")
    print(f"   Peso = {best_params['weights']}")
    print(f"   Métrica = {best_params['metric']}")
    print(f"   Precisión CV = {best_score:.4f}")
    
    # Mostrar top 5 resultados
    print("\n🏆 Top 5 configuraciones:")
    print(results_df.sort_values('cv_mean', ascending=False).head(5).to_string(index=False))
    
    return best_params, best_model, results_df

def train_and_evaluate_knn(X, y, best_params=None):
    """
    Entrena y evalúa el clasificador KNN.
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {len(X_train)} muestras")
    print(f"   Prueba: {len(X_test)} muestras")
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encontrar mejores parámetros si no se proporcionan
    if best_params is None:
        best_params, best_model, search_results = find_best_knn(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
    else:
        # Usar parámetros proporcionados
        best_model = KNeighborsClassifier(**best_params)
        best_model.fit(X_train_scaled, y_train)
        search_results = None
    
    # Predicciones
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📈 Métricas de rendimiento:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision (weighted): {precision:.4f}")
    print(f"   Recall (weighted): {recall:.4f}")
    print(f"   F1-Score (weighted): {f1:.4f}")
    
    # Reporte de clasificación detallado
    print("\n📋 Reporte de clasificación detallado:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': best_model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'search_results': search_results,
        'best_params': best_params
    }

def visualize_results(results, df, X, y):
    """
    Visualiza los resultados de la clasificación.
    """
    # 1. Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
    plt.show()
    
    # 2. Visualización con PCA
    plt.figure(figsize=(12, 5))
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(results['X_test'])
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=[hash(s) % 100 for s in results['y_test']], 
                         cmap='tab20', alpha=0.7)
    plt.title('Visualización PCA de las predicciones')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # 3. t-SNE
    plt.subplot(1, 2, 2)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(X)//2))
    X_tsne = tsne.fit_transform(results['X_test'])
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=[hash(s) % 100 for s in results['y_test']],
                         cmap='tab20', alpha=0.7)
    plt.title('Visualización t-SNE de las predicciones')
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/visualization_pca_tsne.png", dpi=300)
    plt.show()
    
    # 4. Análisis de errores
    errors = results['y_test'] != results['y_pred']
    if errors.any():
        plt.figure(figsize=(10, 6))
        error_df = pd.DataFrame({
            'Real': results['y_test'][errors],
            'Predicción': results['y_pred'][errors]
        })
        error_counts = error_df.groupby(['Real', 'Predicción']).size().reset_index(name='count')
        error_counts = error_counts.sort_values('count', ascending=False).head(10)
        
        plt.barh(range(len(error_counts)), error_counts['count'])
        plt.yticks(range(len(error_counts)), 
                  [f"{row['Real']}→{row['Predicción']}" for _, row in error_counts.iterrows()])
        plt.xlabel('Número de errores')
        plt.title('Principales confusiones entre especies')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/error_analysis.png", dpi=300)
        plt.show()

def save_results(results, df, X, y):
    """
    Guarda todos los resultados en archivos.
    """
    # 1. Guardar métricas
    metrics_df = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Valor': [results['accuracy'], results['precision'], 
                 results['recall'], results['f1']]
    })
    metrics_df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
    
    # 2. Guardar predicciones
    predictions_df = pd.DataFrame({
        'Real': results['y_test'],
        'Predicción': results['y_pred'],
        'Correcto': results['y_test'] == results['y_pred']
    })
    predictions_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    
    # 3. Guardar mejores parámetros
    params_df = pd.DataFrame([results['best_params']])
    params_df.to_csv(f"{OUTPUT_DIR}/best_params.csv", index=False)
    
    # 4. Guardar reporte de clasificación
    report = classification_report(results['y_test'], results['y_pred'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")
    
    # 5. Guardar resumen de resultados
    with open(f"{OUTPUT_DIR}/resumen.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("RESULTADOS DE CLASIFICACIÓN KNN\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total de imágenes: {len(df)}\n")
        f.write(f"Número de especies: {len(np.unique(y))}\n")
        f.write(f"Especies: {', '.join(np.unique(y))}\n\n")
        f.write(f"Mejores parámetros:\n")
        f.write(f"  K: {results['best_params']['n_neighbors']}\n")
        f.write(f"  Peso: {results['best_params']['weights']}\n")
        f.write(f"  Métrica: {results['best_params']['metric']}\n\n")
        f.write(f"Métricas de rendimiento:\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall: {results['recall']:.4f}\n")
        f.write(f"  F1-Score: {results['f1']:.4f}\n")
    
    print(f"\n✅ Resultados guardados en '{OUTPUT_DIR}/'")

# -------------------- FUNCIONES ADICIONALES --------------------

def cross_validation_analysis(X, y):
    """
    Análisis detallado de validación cruzada.
    """
    print("\n🔍 Análisis de Validación Cruzada...")
    
    # Escalar datos completos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probar diferentes K
    k_values = range(1, 21)
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'b-', marker='o')
    plt.xlabel('Número de Vecinos (K)')
    plt.ylabel('Precisión de Validación Cruzada')
    plt.title('Optimización de K en KNN')
    plt.grid(True, alpha=0.3)
    
    # Marcar mejor K
    best_k = k_values[np.argmax(cv_scores)]
    best_score = max(cv_scores)
    plt.axvline(x=best_k, color='r', linestyle='--', 
                label=f'Mejor K={best_k} (Acc={best_score:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/k_optimization.png", dpi=300)
    plt.show()
    
    return best_k, best_score

def feature_importance_analysis(X, y, feature_names):
    """
    Análisis de importancia de características usando diferentes métodos.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    
    print("\n📊 Análisis de Importancia de Características...")
    
    # Método 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Método 2: SelectKBest
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    f_scores = selector.scores_
    
    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Forest
    axes[0].barh(feature_names, rf_importance)
    axes[0].set_title('Importancia (Random Forest)')
    axes[0].set_xlabel('Importancia')
    
    # F-Scores
    axes[1].barh(feature_names, f_scores)
    axes[1].set_title('Importancia (F-Score)')
    axes[1].set_xlabel('F-Score')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
    plt.show()
    
    # Crear DataFrame de importancia
    importance_df = pd.DataFrame({
        'Característica': feature_names,
        'RF_Importance': rf_importance,
        'F_Score': f_scores
    })
    importance_df = importance_df.sort_values('RF_Importance', ascending=False)
    print("\n📊 Importancia de características:")
    print(importance_df.to_string(index=False))
    
    return importance_df

# -------------------- EJECUCIÓN PRINCIPAL --------------------

if __name__ == "__main__":
    print("="*60)
    print("🧬 CLASIFICACIÓN DE ESPECIES CON KNN")
    print("   Basado en características GLCM")
    print("="*60)
    
    # 1. Cargar datos
    X, y, df = load_and_prepare_data(INPUT_CSV)
    
    if X is not None and len(np.unique(y)) > 1:
        # 2. Análisis de validación cruzada
        best_k, best_cv_score = cross_validation_analysis(X, y)
        
        # 3. Entrenar y evaluar modelo
        best_params = {
            'n_neighbors': best_k,
            'weights': 'distance',
            'metric': 'euclidean'
        }
        
        results = train_and_evaluate_knn(X, y, best_params)
        
        # 4. Visualizar resultados
        visualize_results(results, df, X, y)
        
        # 5. Análisis de importancia de características
        importance_df = feature_importance_analysis(X, y, FEATURES)
        
        # 6. Guardar todos los resultados
        save_results(results, df, X, y)
        
        print("\n✅ ¡Análisis completado exitosamente!")
        print(f"📁 Todos los resultados guardados en '{OUTPUT_DIR}/'")
        
    else:
        print("❌ No se pudieron procesar los datos. Verifica el archivo CSV.")

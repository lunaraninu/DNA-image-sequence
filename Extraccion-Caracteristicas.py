import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------- CONFIGURACIÓN --------------------
INPUT_DIR = "todas_imagenes_hilbert_auto"  # Directorio con las imágenes
OUTPUT_CSV = "caracteristicas_glcm.csv"    # Archivo de salida
OUTPUT_PLOT = "analisis_glcm.png"          # Gráfico de resultados

# Parámetros GLCM
DISTANCES = [1, 2, 3, 5, 10]               # Distancias a analizar
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ángulos: 0°, 45°, 90°, 135°
LEVELS = 256                               # Niveles de gris (0-255)
SYMMETRIC = True                           # Matriz simétrica

# Características a extraer
FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# ------------------------------------------------------

def extract_glcm_features(image_path, distances=DISTANCES, angles=ANGLES, levels=LEVELS):
    """
    Extrae características GLCM de una imagen.
    Retorna un diccionario con las características promedio.
    """
    try:
        # Cargar imagen y convertir a escala de grises
        img = Image.open(image_path)
        img_gray = rgb2gray(np.array(img))
        
        # Normalizar a 0-255 y convertir a uint8
        img_gray = (img_gray * 255).astype(np.uint8)
        
        # Calcular GLCM
        glcm = graycomatrix(img_gray, distances=distances, angles=angles, 
                           levels=levels, symmetric=SYMMETRIC)
        
        # Extraer características
        features = {}
        for feature in FEATURES:
            # Calcular la característica para todas las distancias y ángulos
            values = graycoprops(glcm, feature)
            # Promedio sobre todos los ángulos y distancias
            features[feature] = np.mean(values)
        
        return features
    
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def extract_all_features(input_dir, output_csv=None, show_progress=True):
    """
    Procesa todas las imágenes en el directorio y extrae características GLCM.
    """
    # Buscar todas las imágenes PNG
    image_paths = list(Path(input_dir).rglob("*.png"))
    
    if not image_paths:
        print(f"No se encontraron imágenes PNG en {input_dir}")
        return None
    
    print(f"📊 Procesando {len(image_paths)} imágenes...")
    
    # Lista para almacenar resultados
    results = []
    
    # Procesar cada imagen
    iterator = tqdm(image_paths, desc="Extrayendo características") if show_progress else image_paths
    
    for img_path in iterator:
        # Extraer metadatos del nombre del archivo
        filename = img_path.stem
        parts = filename.split('_')
        
        # Intentar extraer información del nombre
        try:
            # Formato esperado: especie_seq#_###bp_###x###_hilbert
            species = parts[0]
            seq_id = parts[1] if len(parts) > 1 else "seq1"
            bp = parts[2].replace('bp', '') if len(parts) > 2 else "0"
            size = parts[3] if len(parts) > 3 else "0x0"
            width, height = size.split('x') if 'x' in size else ("0", "0")
        except:
            species = filename
            seq_id = "seq1"
            bp = "0"
            width, height = "0", "0"
        
        # Extraer características GLCM
        features = extract_glcm_features(img_path)
        
        if features:
            # Crear registro
            record = {
                'filename': filename,
                'path': str(img_path),
                'species': species,
                'sequence_id': seq_id,
                'length_bp': int(bp) if bp.isdigit() else 0,
                'width': int(width) if width.isdigit() else 0,
                'height': int(height) if height.isdigit() else 0,
                **features
            }
            results.append(record)
    
    # Convertir a DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("⚠️ No se pudo extraer características de ninguna imagen")
        return None
    
    # Guardar CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ Datos guardados en: {output_csv}")
    
    return df

def analyze_and_visualize(df, output_plot=None):
    """
    Genera visualizaciones de las características GLCM.
    """
    if df is None or df.empty:
        print("No hay datos para visualizar")
        return
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Análisis de Características GLCM', fontsize=16, fontweight='bold')
    
    # 1. Distribución de cada característica
    for idx, feature in enumerate(FEATURES):
        row = idx // 3
        col = idx % 3
        if idx < 5:  # Solo 5 características
            ax = axes[row, col]
            df[feature].hist(bins=30, ax=ax, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'Distribución de {feature.capitalize()}')
            ax.set_xlabel(feature.capitalize())
            ax.set_ylabel('Frecuencia')
    
    # 6. Boxplot comparativo
    ax = axes[1, 2]
    df_melted = pd.melt(df[FEATURES], var_name='Característica', value_name='Valor')
    sns.boxplot(data=df_melted, x='Característica', y='Valor', ax=ax)
    ax.set_title('Comparación de Características')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Guardar figura
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {output_plot}")
    
    plt.show()
    
    # Estadísticas descriptivas
    print("\n📊 Estadísticas descriptivas:")
    print(df[FEATURES].describe().round(4))
    
    # Matriz de correlación
    print("\n🔗 Matriz de correlación entre características:")
    corr_matrix = df[FEATURES].corr().round(3)
    print(corr_matrix)
    
    return corr_matrix

def batch_process_images(input_dir, output_csv, output_plot=None):
    """
    Función principal para procesar todas las imágenes y generar análisis.
    """
    print("="*60)
    print("🔬 ANÁLISIS DE MATRIZ DE COOCURRENCIA (GLCM)")
    print("="*60)
    
    # Extraer características
    df = extract_all_features(input_dir, output_csv)
    
    if df is None:
        return
    
    # Mostrar resumen
    print(f"\n📊 Resumen de datos procesados:")
    print(f"   - Total de imágenes: {len(df)}")
    print(f"   - Especies únicas: {df['species'].nunique()}")
    print(f"   - Rango de tamaños: {df['width'].min()}x{df['height'].min()} a {df['width'].max()}x{df['height'].max()}")
    
    # Analizar y visualizar
    corr_matrix = analyze_and_visualize(df, output_plot)
    
    # Guardar resumen estadístico
    stats_file = output_csv.replace('.csv', '_estadisticas.csv')
    df[FEATURES].describe().to_csv(stats_file)
    print(f"✅ Estadísticas guardadas en: {stats_file}")
    
    return df

# -------------------- FUNCIONES ADICIONALES --------------------

def extract_features_per_species(df):
    """
    Agrupa características por especie.
    """
    species_stats = df.groupby('species')[FEATURES].agg(['mean', 'std', 'count']).round(4)
    return species_stats

def plot_features_by_species(df, top_n=10):
    """
    Visualiza características por especie (top N especies).
    """
    # Seleccionar top N especies por número de imágenes
    top_species = df['species'].value_counts().head(top_n).index
    df_top = df[df['species'].isin(top_species)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Características GLCM por Especie (Top {top_n})', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(FEATURES):
        row = idx // 3
        col = idx % 3
        if idx < 5:
            ax = axes[row, col]
            df_top.boxplot(column=feature, by='species', ax=ax, rot=45)
            ax.set_title(f'{feature.capitalize()} por Especie')
            ax.set_xlabel('Especie')
            ax.set_ylabel(feature.capitalize())
    
    plt.tight_layout()
    plt.show()

# -------------------- EJECUCIÓN PRINCIPAL --------------------

if __name__ == "__main__":
    # Procesar todas las imágenes
    df = batch_process_images(
        input_dir=INPUT_DIR,
        output_csv=OUTPUT_CSV,
        output_plot=OUTPUT_PLOT
    )
    
    if df is not None:
        # Análisis adicional por especie
        print("\n📊 Estadísticas por especie:")
        species_stats = extract_features_per_species(df)
        print(species_stats)
        
        # Guardar estadísticas por especie
        species_stats.to_csv(OUTPUT_CSV.replace('.csv', '_por_especie.csv'))
        
        # Visualizar por especie (opcional)
        # plot_features_by_species(df)

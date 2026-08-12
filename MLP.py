import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------- CONFIGURACIÓN --------------------
INPUT_CSV = "caracteristicas_glcm.csv"
OUTPUT_DIR = "resultados_red_neuronal_kfold"

# Características a usar
FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Parámetros de K-Fold
N_FOLDS = 5                     # Número de folds
RANDOM_STATE = 42

# Parámetros de la Red Neuronal
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20

# Arquitectura de la red
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.3
ACTIVATION = 'relu'

# Configuración de GPU
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("💻 Usando CPU")

# Crear directorio de salida
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------

# -------------------- PREPARACIÓN DE DATOS --------------------

def load_and_prepare_data(csv_path):
    """
    Carga y prepara los datos para K-Fold.
    """
    df = pd.read_csv(csv_path)
    
    print(f"📊 Datos cargados: {len(df)} imágenes")
    
    # Verificar características disponibles
    available_features = [f for f in FEATURES if f in df.columns]
    if not available_features:
        print("❌ No se encontraron características GLCM")
        return None, None, None, None
    
    print(f"   Características usadas: {available_features}")
    
    # Preparar X e y
    X = df[available_features].values
    y = df['species'].values
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"   Especies únicas: {len(label_encoder.classes_)}")
    print(f"   Distribución: {dict(pd.Series(y).value_counts())}")
    
    # Escalar características (importante hacerlo dentro de cada fold)
    # Guardamos el scaler para usarlo en cada fold
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, label_encoder, df

# -------------------- DATASET PERSONALIZADO --------------------

class GLCMDataset(Dataset):
    """
    Dataset personalizado para características GLCM.
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------- ARQUITECTURA DE LA RED --------------------

class GLCMClassifier(nn.Module):
    """
    Red Neuronal para clasificación de especies basada en GLCM.
    """
    def __init__(self, input_size, num_classes, hidden_layers=HIDDEN_LAYERS, 
                 dropout_rate=DROPOUT_RATE, activation=ACTIVATION):
        super(GLCMClassifier, self).__init__()
        
        # Construir capas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        """Retorna la función de activación solicitada."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        return self.network(x)

# -------------------- ENTRENAMIENTO POR FOLD --------------------

def train_single_fold(model, train_loader, val_loader, epochs=EPOCHS, 
                      lr=LEARNING_RATE, patience=PATIENCE, device=DEVICE):
    """
    Entrena el modelo para un solo fold.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        # Métricas
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss_avg)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Cargar mejor modelo
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc

def evaluate_fold(model, test_loader, device=DEVICE):
    """
    Evalúa el modelo en el fold de prueba.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# -------------------- K-FOLD CROSS VALIDATION --------------------

def kfold_cross_validation(X, y, label_encoder, n_folds=N_FOLDS):
    """
    Realiza validación cruzada con K-Folds.
    """
    print(f"\n🔬 Iniciando K-Fold Cross Validation (K={n_folds})")
    print("="*50)
    
    # Configurar K-Fold estratificado
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Almacenar resultados
    fold_results = []
    all_y_test = []
    all_y_pred = []
    all_y_probs = []
    
    # Para cada fold
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n📊 Fold {fold + 1}/{n_folds}")
        print("-" * 30)
        
        # Dividir datos
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Dividir entrenamiento en train/val (80/20)
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size
        
        indices = np.random.permutation(len(X_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train_fold = X_train[train_indices]
        y_train_fold = y_train[train_indices]
        X_val_fold = X_train[val_indices]
        y_val_fold = y_train[val_indices]
        
        print(f"   Entrenamiento: {len(X_train_fold)} muestras")
        print(f"   Validación: {len(X_val_fold)} muestras")
        print(f"   Prueba: {len(X_test)} muestras")
        
        # Crear datasets
        train_dataset = GLCMDataset(X_train_fold, y_train_fold)
        val_dataset = GLCMDataset(X_val_fold, y_val_fold)
        test_dataset = GLCMDataset(X_test, y_test)
        
        # Crear dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Crear modelo
        input_size = X.shape[1]
        num_classes = len(label_encoder.classes_)
        model = GLCMClassifier(input_size, num_classes)
        
        # Entrenar
        model, history, best_val_acc = train_single_fold(
            model, train_loader, val_loader
        )
        print(f"   Mejor accuracy en validación: {best_val_acc:.4f}")
        
        # Evaluar
        y_test_fold, y_pred_fold, y_probs_fold = evaluate_fold(model, test_loader)
        
        # Métricas del fold
        accuracy = accuracy_score(y_test_fold, y_pred_fold)
        precision = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
        recall = recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
        f1 = f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
        
        # Guardar resultados del fold
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_val_acc': best_val_acc,
            'history': history
        })
        
        # Acumular predicciones
        all_y_test.extend(y_test_fold)
        all_y_pred.extend(y_pred_fold)
        all_y_probs.extend(y_probs_fold)
        
        print(f"   Accuracy en prueba: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
    
    # Calcular métricas promedio
    results_df = pd.DataFrame(fold_results)
    
    print("\n" + "="*50)
    print("📊 RESULTADOS DE K-FOLD CROSS VALIDATION")
    print("="*50)
    
    print("\n📈 Métricas por fold:")
    print(results_df[['fold', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
    
    print("\n📊 Estadísticas promedio:")
    print(f"   Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"   Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"   Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"   F1-Score: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    
    # Métricas globales (todas las predicciones juntas)
    global_accuracy = accuracy_score(all_y_test, all_y_pred)
    global_precision = precision_score(all_y_test, all_y_pred, average='weighted', zero_division=0)
    global_recall = recall_score(all_y_test, all_y_pred, average='weighted', zero_division=0)
    global_f1 = f1_score(all_y_test, all_y_pred, average='weighted', zero_division=0)
    
    print("\n🌍 Métricas globales (todas las predicciones):")
    print(f"   Accuracy: {global_accuracy:.4f}")
    print(f"   Precision: {global_precision:.4f}")
    print(f"   Recall: {global_recall:.4f}")
    print(f"   F1-Score: {global_f1:.4f}")
    
    return {
        'fold_results': fold_results,
        'results_df': results_df,
        'all_y_test': np.array(all_y_test),
        'all_y_pred': np.array(all_y_pred),
        'all_y_probs': np.array(all_y_probs),
        'global_metrics': {
            'accuracy': global_accuracy,
            'precision': global_precision,
            'recall': global_recall,
            'f1': global_f1
        },
        'mean_metrics': {
            'accuracy': results_df['accuracy'].mean(),
            'accuracy_std': results_df['accuracy'].std(),
            'precision': results_df['precision'].mean(),
            'precision_std': results_df['precision'].std(),
            'recall': results_df['recall'].mean(),
            'recall_std': results_df['recall'].std(),
            'f1': results_df['f1'].mean(),
            'f1_std': results_df['f1'].std()
        }
    }

# -------------------- VISUALIZACIÓN --------------------

def visualize_kfold_results(kfold_results, label_encoder):
    """
    Visualiza los resultados de K-Fold.
    """
    results_df = kfold_results['results_df']
    
    # 1. Boxplot de métricas por fold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Métricas por fold
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results_df_melted = pd.melt(results_df[metrics], 
                                var_name='Métrica', 
                                value_name='Valor')
    
    sns.boxplot(data=results_df_melted, x='Métrica', y='Valor', ax=axes[0])
    axes[0].set_title('Distribución de Métricas por Fold')
    axes[0].set_ylabel('Valor')
    axes[0].grid(True, alpha=0.3)
    
    # Evolución de accuracy por fold
    for fold in results_df['fold']:
        history = results_df[results_df['fold'] == fold]['history'].values[0]
        axes[1].plot(history['val_acc'], label=f'Fold {fold}', alpha=0.7)
    
    axes[1].set_title('Evolución del Accuracy en Validación')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kfold_metrics.png", dpi=300)
    plt.show()
    
    # 2. Matriz de confusión global
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(kfold_results['all_y_test'], kfold_results['all_y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusión (K-Fold Global)')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kfold_confusion_matrix.png", dpi=300)
    plt.show()
    
    # 3. Comparación de folds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df))
    width = 0.2
    
    ax.bar(x - width*1.5, results_df['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x - width*0.5, results_df['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width*0.5, results_df['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width*1.5, results_df['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas por Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(results_df))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kfold_comparison.png", dpi=300)
    plt.show()

def save_kfold_results(kfold_results, label_encoder):
    """
    Guarda todos los resultados de K-Fold.
    """
    # 1. Resumen por fold
    kfold_results['results_df'].to_csv(f"{OUTPUT_DIR}/kfold_results.csv", index=False)
    
    # 2. Métricas globales
    global_metrics_df = pd.DataFrame([kfold_results['global_metrics']])
    global_metrics_df.to_csv(f"{OUTPUT_DIR}/global_metrics.csv", index=False)
    
    # 3. Métricas promedio
    mean_metrics_df = pd.DataFrame([kfold_results['mean_metrics']])
    mean_metrics_df.to_csv(f"{OUTPUT_DIR}/mean_metrics.csv", index=False)
    
    # 4. Predicciones globales
    predictions_df = pd.DataFrame({
        'Real': label_encoder.inverse_transform(kfold_results['all_y_test']),
        'Predicción': label_encoder.inverse_transform(kfold_results['all_y_pred']),
        'Correcto': kfold_results['all_y_test'] == kfold_results['all_y_pred']
    })
    predictions_df.to_csv(f"{OUTPUT_DIR}/global_predictions.csv", index=False)
    
    # 5. Reporte de clasificación
    report = classification_report(kfold_results['all_y_test'], 
                                  kfold_results['all_y_pred'],
                                  target_names=label_encoder.classes_,
                                  output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{OUTPUT_DIR}/global_classification_report.csv")
    
    # 6. Resumen
    with open(f"{OUTPUT_DIR}/resumen_kfold.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("RESULTADOS DE K-FOLD CROSS VALIDATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Número de folds: {N_FOLDS}\n")
        f.write(f"Total de imágenes: {len(kfold_results['all_y_test'])}\n")
        f.write(f"Número de especies: {len(label_encoder.classes_)}\n\n")
        f.write("MÉTRICAS GLOBALES:\n")
        for metric, value in kfold_results['global_metrics'].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nMÉTRICAS PROMEDIO (± std):\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean = kfold_results['mean_metrics'][metric]
            std = kfold_results['mean_metrics'][f'{metric}_std']
            f.write(f"  {metric}: {mean:.4f} ± {std:.4f}\n")
        f.write("\n" + "="*60)
    
    print(f"\n✅ Resultados guardados en '{OUTPUT_DIR}/'")

# -------------------- FUNCIÓN PRINCIPAL --------------------

def main():
    print("="*60)
    print("🧠 K-FOLD CROSS VALIDATION CON RED NEURONAL")
    print("   Basado en características GLCM")
    print("="*60)
    
    # 1. Cargar datos
    X, y, label_encoder, df = load_and_prepare_data(INPUT_CSV)
    
    if X is None:
        return
    
    # 2. K-Fold Cross Validation
    kfold_results = kfold_cross_validation(X, y, label_encoder, n_folds=N_FOLDS)
    
    # 3. Visualizar resultados
    visualize_kfold_results(kfold_results, label_encoder)
    
    # 4. Guardar resultados
    save_kfold_results(kfold_results, label_encoder)
    
    print("\n✅ ¡Análisis K-Fold completado exitosamente!")
    print(f"📁 Todos los resultados guardados en '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()

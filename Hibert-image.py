import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from PIL import Image

# -------------------- CONFIGURACIÓN --------------------
INPUT_DIR = "seq_genom"
OUTPUT_DIR = "todas_imagenes_hilbert_auto"

BIN = 1                 # Compactación (1 = sin compactar)
PIXEL_SIZE = 1          # Escalado de píxel
OVERWRITE = False       # Sobrescribir imágenes existentes

FIXED_WIDTH = 2048      # Ancho fijo de la imagen
MIN_HEIGHT = 16         # Altura mínima
MAX_HEIGHT = 4096       # Altura máxima (por seguridad)

EXTENSIONS = ".fna,.fa,.fasta"

COLOR_MAP = {
    'A': (255, 0, 0),      # Rojo
    'C': (0, 255, 0),      # Verde
    'G': (0, 0, 255),      # Azul
    'T': (255, 255, 0),    # Amarillo
    'N': (187, 187, 187)   # Gris
}
# ------------------------------------------------------

def calculate_optimal_height(seq_length, width):
    """
    Calcula la altura óptima para una secuencia dado un ancho fijo.
    """
    # Calcular altura mínima necesaria
    height = math.ceil(seq_length / width)
    
    # Aplicar límites
    height = max(height, MIN_HEIGHT)
    height = min(height, MAX_HEIGHT)
    
    return height

def hilbert_index(x, y, pow2, rotate=0):
    """
    Calcula el índice de Hilbert para una coordenada (x,y) en una cuadrícula de 2^pow2 x 2^pow2.
    """
    if pow2 == 0:
        return 0
    
    hpow2 = pow2 - 1
    seg = 0
    
    if x < (1 << hpow2):
        if y < (1 << hpow2):
            seg = 0
            new_x, new_y = y, x
            rot = 0
        else:
            seg = 1
            new_x, new_y = x, y - (1 << hpow2)
            rot = 1
    else:
        if y < (1 << hpow2):
            seg = 3
            new_x, new_y = (1 << hpow2) - 1 - (y - (1 << hpow2)), (1 << hpow2) - 1 - x
            rot = 3
        else:
            seg = 2
            new_x, new_y = x - (1 << hpow2), y - (1 << hpow2)
            rot = 2
    
    sub_index = hilbert_index(new_x, new_y, hpow2, (rotate + rot) % 4)
    return (seg << (2 * hpow2)) + sub_index

def generate_hilbert_order(side):
    """
    Genera el orden de los píxeles según la curva de Hilbert.
    Retorna un array de (side*side, 2) con las coordenadas en orden Hilbert.
    """
    power = int(math.ceil(math.log2(side)))
    size = 1 << power
    
    coords = np.zeros((side * side, 2), dtype=np.int32)
    
    for i in range(side * side):
        x = i % side
        y = i // side
        coords[i] = [x, y]
    
    hilbert_indices = np.array([hilbert_index(x, y, power) for x, y in coords])
    order = np.argsort(hilbert_indices)
    
    return coords[order]

def to_rgb_array(seq, color_map):
    """Convierte secuencia ADN a RGB."""
    get = color_map.get
    return np.array(
        [get(base if base in color_map else 'N', color_map['N']) for base in seq],
        dtype=np.uint8
    )

def bin_sequence_colors(rgb_arr, bin_size):
    """Reduce resolución promediando bloques."""
    if bin_size <= 1:
        return rgb_arr

    n = len(rgb_arr)
    m = n // bin_size

    if m > 0:
        trimmed = rgb_arr[:m * bin_size]
        blocks = trimmed.reshape(m, bin_size, 3)
        binned = blocks.mean(axis=1).astype(np.uint8)
    else:
        binned = np.empty((0, 3), dtype=np.uint8)

    rem = n - m * bin_size
    if rem > 0:
        tail = rgb_arr[m * bin_size:]
        tail_mean = tail.mean(axis=0, keepdims=True).astype(np.uint8)
        binned = np.vstack([binned, tail_mean]) if binned.size else tail_mean

    return binned

def make_hilbert_image_auto(rgb_linear, color_map, fixed_width, height=None, pixel_size=1):
    """
    Crea imagen RGB usando curva de Hilbert con ancho fijo y alto automático.
    """
    seq_length = len(rgb_linear)
    
    # Calcular la altura si no se proporciona
    if height is None:
        height = calculate_optimal_height(seq_length, fixed_width)
    
    total_pixels = fixed_width * height
    
    # Si la secuencia es más larga que la imagen, truncar
    if seq_length > total_pixels:
        print(f"  ⚠️ Secuencia muy larga ({seq_length} bases), truncando a {total_pixels}")
        rgb_linear = rgb_linear[:total_pixels]
        seq_length = total_pixels
    
    # Si la secuencia es más corta, no rellenar con N
    # Simplemente usamos menos píxeles
    
    # Generar el orden de Hilbert (necesita lado cuadrado, pero usaremos el más cercano)
    max_side = max(fixed_width, height)
    power = int(math.ceil(math.log2(max_side)))
    hilbert_size = 1 << power
    
    # Para imágenes rectangulares, mapeamos la curva de Hilbert de tamaño cuadrado
    # y luego la recortamos al tamaño deseado
    hilbert_coords_full = generate_hilbert_order(hilbert_size)
    
    # Filtrar coordenadas que están dentro del rectángulo
    mask = (hilbert_coords_full[:, 0] < fixed_width) & (hilbert_coords_full[:, 1] < height)
    hilbert_coords = hilbert_coords_full[mask]
    
    # Si no hay suficientes coordenadas, usar un método alternativo
    if len(hilbert_coords) < seq_length:
        # Generar orden por filas (fallback)
        coords = []
        for y in range(height):
            for x in range(fixed_width):
                coords.append([x, y])
        hilbert_coords = np.array(coords, dtype=np.int32)
    
    # Crear imagen vacía (negra)
    img_array = np.zeros((height, fixed_width, 3), dtype=np.uint8)
    
    # Rellenar la imagen según la curva de Hilbert solo con las bases disponibles
    for idx, (x, y) in enumerate(hilbert_coords):
        if idx < len(rgb_linear):
            img_array[y, x] = rgb_linear[idx]
        # Si no hay más bases, dejamos el píxel en negro (0,0,0)
    
    # Crear imagen PIL
    img = Image.fromarray(img_array, mode="RGB")
    
    # Escalar si es necesario
    if pixel_size != 1:
        img = img.resize(
            (img.width * pixel_size, img.height * pixel_size),
            resample=Image.NEAREST
        )
    
    return img, fixed_width, height

def find_fasta_files(root_dir, exts):
    """Busca archivos FASTA recursivamente."""
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path

def safe_name(s):
    """Nombre seguro para archivos."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def process_all():
    root = Path(INPUT_DIR)
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    exts = {e if e.startswith('.') else '.' + e for e in EXTENSIONS.split(',')}
    fasta_paths = list(find_fasta_files(root, exts))

    if not fasta_paths:
        print("No se encontraron archivos FASTA.")
        return

    species_counters = defaultdict(int)
    total_saved = 0
    total_sequences = 0

    print(f"📐 Ancho fijo: {FIXED_WIDTH} píxeles")
    print(f"📏 Altura: automática (mín {MIN_HEIGHT}, máx {MAX_HEIGHT})")
    print(f"{'='*50}")

    for fasta_path in fasta_paths:
        species = fasta_path.parent.name or fasta_path.stem
        species_safe = safe_name(species)

        print(f"\n📁 Procesando: {fasta_path}")

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq = str(record.seq).upper()
            if not seq:
                continue

            total_sequences += 1
            species_counters[species_safe] += 1
            seq_id = species_counters[species_safe]
            seq_length = len(seq)

            # Calcular altura necesaria
            needed_height = calculate_optimal_height(seq_length, FIXED_WIDTH)
            
            print(f"  📊 Secuencia {seq_id}: {seq_length:,} bases")
            print(f"  📐 Altura necesaria: {needed_height} píxeles")
            print(f"  📐 Total píxeles: {FIXED_WIDTH * needed_height:,}")

            # Crear subcarpeta espejo
            relative_path = fasta_path.parent.relative_to(INPUT_DIR)
            out_subdir = outdir / relative_path
            out_subdir.mkdir(parents=True, exist_ok=True)

            # Convertir toda la secuencia a RGB
            rgb = to_rgb_array(seq, COLOR_MAP)
            rgb = bin_sequence_colors(rgb, BIN)

            # Generar imagen con ancho fijo y alto automático
            img, width_used, height_used = make_hilbert_image_auto(
                rgb,
                COLOR_MAP,
                fixed_width=FIXED_WIDTH,
                height=needed_height,  # Usar la altura calculada
                pixel_size=PIXEL_SIZE
            )

            # Nombre del archivo con información del tamaño
            out_name = f"{species_safe}_seq{seq_id}_{seq_length}bp_{width_used}x{height_used}_hilbert.png"
            out_path = out_subdir / out_name

            if out_path.exists() and not OVERWRITE:
                print(f"  ⏭️  Archivo existe, saltando: {out_name}")
                continue

            img.save(out_path)
            total_saved += 1
            print(f"  ✅ Guardado: {out_name}")

    print(f"\n{'='*50}")
    print(f"✅ Proceso terminado.")
    print(f"📊 Secuencias procesadas: {total_sequences}")
    print(f"🖼️  Imágenes generadas: {total_saved}")
    print(f"📁 Carpeta de salida: {outdir.resolve()}")
    print(f"{'='*50}")

if __name__ == "__main__":
    process_all()

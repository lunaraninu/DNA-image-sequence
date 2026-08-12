import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from PIL import Image

# -------------------- CONFIGURACIÓN --------------------
ANCHO_IMAGEN = 2048  # Ancho fijo (n)
INPUT_DIR = "seq_genom"
OUTPUT_DIR = f"Secuencial_imagenes-{ANCHO_IMAGEN}"

BIN = 1                 # Compactación (1 = sin compactar)
PIXEL_SIZE = 1          # Escalado de píxel
OVERWRITE = False       # Sobrescribir imágenes existentes

EXTENSIONS = ".fna,.fa,.fasta"

COLOR_MAP = {
    'A': (255, 0, 0),    # Rojo
    'C': (0, 255, 0),    # Verde
    'G': (0, 0, 255),    # Azul
    'T': (255, 255, 0),  # Amarillo
    'N': (187, 187, 187) # Gris
}
# ------------------------------------------------------


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


def chunk_sequence(seq, chunk_size):
    """Divide secuencia en fragmentos de tamaño fijo."""
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def make_variable_image(rgb_linear, color_map, width=128, pixel_size=1):
    """
    Crea imagen RGB de ancho fijo y alto variable.
    El alto se calcula automáticamente para acomodar la secuencia.
    """
    L = rgb_linear.shape[0]
    
    # Calcular el alto necesario (redondear hacia arriba)
    height = math.ceil(L / width)
    
    # Crear array con el tamaño exacto
    total_pixels = height * width
    pad_size = total_pixels - L
    
    # Rellenar con 'N' si es necesario
    if pad_size > 0:
        pad_rgb = np.tile(np.array(color_map['N'], dtype=np.uint8), (pad_size, 1))
        rgb_linear = np.vstack([rgb_linear, pad_rgb])
    
    # Reshape a la forma final
    arr = rgb_linear.reshape(height, width, 3)
    
    # Crear imagen
    img = Image.fromarray(arr, mode="RGB")
    
    # Escalar si es necesario
    if pixel_size != 1:
        img = img.resize(
            (img.width * pixel_size, img.height * pixel_size),
            resample=Image.NEAREST
        )
    
    return img, height


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

    for fasta_path in fasta_paths:
        species = fasta_path.parent.name or fasta_path.stem
        species_safe = safe_name(species)

        print(f"\nProcesando: {fasta_path}")

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq = str(record.seq).upper()
            if not seq:
                continue

            species_counters[species_safe] += 1
            seq_id = species_counters[species_safe]

            # Crear subcarpeta espejo
            relative_path = fasta_path.parent.relative_to(INPUT_DIR)
            out_subdir = outdir / relative_path
            out_subdir.mkdir(parents=True, exist_ok=True)

            # Procesar toda la secuencia de una vez (sin fragmentar)
            rgb = to_rgb_array(seq, COLOR_MAP)
            rgb = bin_sequence_colors(rgb, BIN)

            img, height = make_variable_image(
                rgb,
                COLOR_MAP,
                width=ANCHO_IMAGEN,
                pixel_size=PIXEL_SIZE
            )

            out_name = f"{species_safe}_seq{seq_id}.png"
            out_path = out_subdir / out_name

            if out_path.exists() and not OVERWRITE:
                continue

            img.save(out_path)
            total_saved += 1
            print(f"  ✔ {out_path} (ancho: {ANCHO_IMAGEN}, alto: {height})")

    print(f"\n✔ Proceso terminado. Imágenes generadas: {total_saved}")
    print(f"✔ Carpeta de salida: {outdir.resolve()}")


if __name__ == "__main__":
    process_all()

"""
Training entrypoints for forecasting models.

Untuk saat ini, script training utama masih di `models/train_forecast.py`.
Modul ini disiapkan untuk nantinya memindahkan/membungkus logic training
ke dalam fungsi-fungsi yang bisa dipanggil ulang (mis. dari CLI atau API).
"""


def train():
    """
    Placeholder fungsi training.

    Rencana ke depan:
    - Baca fitur dari `data/processed/weekly_features.parquet`
    - Split train/validation
    - Latih model (LightGBM / RandomForest)
    - Simpan model & statistik ke `models/artifacts/`
    """

    raise NotImplementedError("Training pipeline belum dipindahkan ke src/forecasting/train.py")



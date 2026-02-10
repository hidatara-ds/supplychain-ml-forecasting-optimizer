## Data Dictionary

Dokumen ini menjelaskan **format dataset** yang dipakai untuk training forecasting dan sebagai input ke optimizer. Jika data aktual belum final, skema di bawah ini berfungsi sebagai **sample schema** yang nanti bisa di-map ke sumber data sebenarnya.

### 1. Format Dataset Inti (Core)

Setiap baris mewakili observasi permintaan untuk satu kombinasi **tanggal × SKU × lokasi**.

#### Tabel Kolom Core

| Kolom         | Tipe (disarankan) | Contoh          | Deskripsi                                                                 |
|---------------|-------------------|-----------------|---------------------------------------------------------------------------|
| `date`        | `date` / `string` (YYYY-MM-DD) | `2025-01-06`   | Tanggal observasi demand (granularity harian atau awal minggu).          |
| `sku_id`      | `string`          | `P001`          | ID unik produk / SKU.                                                    |
| `location_id` | `string`          | `S001`          | ID unik lokasi (toko, DC, region).                                       |
| `demand_qty`  | `float` / `int`   | `12.0`          | Jumlah permintaan / penjualan aktual pada tanggal tersebut (target forecast). |

> Catatan: Granularity yang disarankan konsisten dengan **Problem Contract** (weekly). Jika data historis harian, agregasi ke mingguan dapat dilakukan di tahap ETL.

### 2. Kolom Opsional (Feature Tambahan)

Kolom-kolom di bawah ini bersifat opsional. Jika tersedia, bisa digunakan sebagai fitur tambahan untuk model forecasting atau sebagai parameter untuk optimizer.

#### 2.1. Harga & Promo

| Kolom          | Tipe (disarankan) | Contoh  | Deskripsi                                                              |
|----------------|-------------------|---------|------------------------------------------------------------------------|
| `price`        | `float`           | `49.90` | Harga jual per unit SKU pada tanggal/lokasi tersebut.                  |
| `promo_flag`   | `int` / `bool`    | `1`     | Indikator periode promo (1 = promo, 0 = tidak promo).                 |
| `holiday_flag` | `int` / `bool`    | `0`     | Indikator hari libur / high season (1 = libur/high season, 0 = bukan).|

#### 2.2. Inventory & Supply

| Kolom            | Tipe (disarankan) | Contoh | Deskripsi                                                                 |
|------------------|-------------------|--------|---------------------------------------------------------------------------|
| `on_hand`        | `float` / `int`   | `35`   | Stok on-hand di awal/periode observasi.                                  |
| `on_order`       | `float` / `int`   | `20`   | Kuantitas yang sudah di-order tetapi belum diterima (open PO).          |
| `lead_time_days` | `int`             | `7`    | Lead time rata-rata (hari) dari order sampai stok tiba di lokasi.       |

#### 2.3. Master Data Produk & Supplier

| Kolom        | Tipe (disarankan) | Contoh     | Deskripsi                                             |
|--------------|-------------------|------------|-------------------------------------------------------|
| `supplier_id`| `string`          | `SUP01`    | ID pemasok utama untuk SKU tersebut.                  |
| `category`   | `string`          | `Beverage` | Kategori produk (mis. category, subcategory, segment).|

### 3. Contoh Skema (Sample Schema)

Contoh skema tabel dalam format pseudo-SQL:

```sql
CREATE TABLE demand_history (
    date           DATE           NOT NULL,
    sku_id         VARCHAR(50)    NOT NULL,
    location_id    VARCHAR(50)    NOT NULL,
    demand_qty     FLOAT          NOT NULL,

    -- Optional features
    price          FLOAT          NULL,
    promo_flag     TINYINT        NULL,
    holiday_flag   TINYINT        NULL,

    on_hand        FLOAT          NULL,
    on_order       FLOAT          NULL,
    lead_time_days INT            NULL,

    supplier_id    VARCHAR(50)    NULL,
    category       VARCHAR(100)   NULL
);
```

Skema di atas bisa diadaptasi ke format lain (CSV, Parquet, Pandas DataFrame). Yang penting, setiap kolom mengikuti **nama** dan **arti** seperti yang didefinisikan di tabel data dictionary ini, sehingga pipeline ETL, model, dan optimizer dapat saling terhubung secara konsisten.


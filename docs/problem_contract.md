## Problem Contract

Dokumen ini mendefinisikan “kontrak masalah” yang menjadi acuan untuk seluruh pipeline forecasting dan replenishment optimizer di proyek ini.

### A. Apa yang mau diprediksi?

- **Target prediksi**: `demand_qty` (jumlah permintaan / penjualan) per kombinasi:
  - **SKU** (product_id)
  - **Location** (store_id)
  - **Periode waktu**: minggu
- **Horizon**: \(H = 4\) minggu ke depan untuk setiap SKU–location.
- **Granularity waktu**: **mingguan** (weekly), konsisten dengan horizon dan struktur data dummy/training.

Secara operasional, model akan menerima daftar pasangan `(store_id, product_id)` dan mengembalikan deret waktu forecast demand mingguan selama 4 minggu ke depan untuk setiap pasangan tersebut.

### B. Output optimizer apa?

- **Output utama**: `order_qty` per kombinasi:
  - SKU
  - Location
  - Periode (misalnya, order mingguan)

- **Objective function (utama)**:
  - **Minimisasi total biaya persediaan**, yang meliputi:
    - biaya simpan (**holding cost**),
    - biaya kehabisan stok (**stockout / shortage cost**),
    - biaya pemesanan (**ordering cost**),
  - atau ekuivalennya mencapai **target service level** minimum dengan biaya total serendah mungkin.

Secara matematis, optimizer menyelesaikan masalah alokasi budget/capacity ke kombinasi SKU–location berbasis forecast demand untuk menghasilkan keputusan pemesanan (`order_qty`) yang ekonomis namun tetap memenuhi target service.

### C. Constraint minimal

Optimizer setidaknya mempertimbangkan (atau siap diperluas untuk mempertimbangkan) constraint di bawah ini:

- **Lead time**:
  - Jeda antara order ditempatkan sampai stok tersedia di lokasi.
  - Mempengaruhi kapan `order_qty` perlu ditempatkan agar stok cukup ketika permintaan terjadi.

- **MOQ (Minimum Order Quantity)**:
  - Batas minimal pemesanan per SKU (misal per karton / pack).
  - `order_qty` dibulatkan atau dibatasi agar memenuhi MOQ tersebut.

- **Capacity constraint**:
  - Batas maksimum total volume atau nilai order (mis. kapasitas gudang, slot pallet, atau kapasitas distribusi).
  - Dapat dimodelkan sebagai:
    - batas total kuantitas, atau
    - batas total nilai biaya (budget/cap).

- **Shelf-life / expiry** (jika relevan untuk kategori produk tertentu):
  - Membatasi seberapa jauh ke depan kita berani menumpuk stok.
  - Mengurangi atau melarang overstock yang berpotensi kadaluarsa sebelum terjual.

- **Budget cap**:
  - Batas atas total belanja (mis. per minggu / per siklus replenishment).
  - Optimizer mengalokasikan budget ini ke SKU–location dengan prioritas tertinggi secara ekonomi (mis. margin, criticality, atau impact service level).

### D. KPI yang jadi “lulus / gagal”

Kontrak keberhasilan solusi diukur dari gabungan KPI forecasting dan KPI operasional / inventory berikut:

- **KPI Forecasting**:
  - **WAPE (Weighted Absolute Percentage Error)**:
    - Lebih stabil untuk data dengan banyak zero/low demand dibanding MAPE.
    - Dievaluasi per horizon (mis. 1–4 minggu ke depan) dan diaggregate per SKU–location dan total portfolio.
  - **MASE (Mean Absolute Scaled Error)**:
    - Membandingkan error model terhadap naive baseline (mis. naive last week).
    - Nilai \< 1 menandakan model lebih baik dari baseline naïf.

- **KPI Inventory / Operational**:
  - **Fill-rate**:
    - Proporsi permintaan yang bisa dipenuhi dari stok on-hand.
    - Target tipikal: mis. \(\ge 95\%\) (dapat diubah sesuai kebutuhan bisnis).
  - **Stockout days**:
    - Jumlah hari/minggu dimana stok = 0 saat ada permintaan.
    - Semakin sedikit semakin baik; dapat dijadikan constraint maksimum.
  - **Total cost**:
    - Penjumlahan biaya holding + stockout + ordering dalam horizon tertentu.
    - Digunakan untuk membandingkan skenario (baseline vs optimizer) dan sebagai objective yang diminimasi.

Kombinasi KPI di atas mendefinisikan apakah solusi “lulus/gagal”: model forecast dianggap baik jika WAPE/MASE memenuhi target; optimizer dianggap berhasil jika mampu mencapai atau melampaui target fill-rate/stockout dengan total cost yang masuk akal atau lebih rendah dari kebijakan baseline.


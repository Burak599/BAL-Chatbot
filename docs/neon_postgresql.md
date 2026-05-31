# Neon PostgreSQL Kurulumu

Bu proje kullanici hesaplarini ve kota sayaclarini `DATABASE_URL` ile baglanilan veritabaninda saklar.

## 1. Neon'da Veritabani Ac

1. Neon dashboard'da yeni bir project olustur.
2. `Connection string` alanindan PostgreSQL URL'sini kopyala.
3. URL'nin sonunda `sslmode=require` oldugundan emin ol.

Ornek format:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST.neon.tech/DB?sslmode=require
```

Kod bu URL'yi otomatik olarak SQLAlchemy'nin `postgresql+psycopg://` surucusune cevirir.

## 2. .env Ayari

`.env` dosyasinda su alanlari doldur:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST.neon.tech/DB?sslmode=require
FLASK_SECRET_KEY=uzun-rastgele-bir-deger
GOOGLE_CLIENT_ID=google-web-client-id
ADMIN_EMAILS=admin@example.com
```

`FLASK_SECRET_KEY` icin ornek uretim:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

## 3. Tablolar

Uygulama ilk acilista tablolari otomatik olusturur.

Neon Table Editor'da sunlari goreceksin:

- `users`: hesap acan veya Google ile giris yapan kullanicilar
- `usage_counters`: gunluk ve dakikalik soru limitleri

Misafirler de `users` tablosunda tutulur:

- `provider = 'guest'`
- `role = 'guest'`
- `email = 'guest-...@guest.local'`

Bu sayede misafir kotasi sadece tarayici ekraninda degil, veritabaninda da kalici izlenir.

## 4. Kontrol SQL'leri

Son kayit olan kullanicilar:

```sql
SELECT id, email, provider, role, created_at
FROM users
ORDER BY created_at DESC;
```

Misafir kullanicilar:

```sql
SELECT id, email, created_at
FROM users
WHERE role = 'guest'
ORDER BY created_at DESC;
```

Bugunku kullanımlar:

```sql
SELECT *
FROM usage_counters
WHERE period_type = 'day'
ORDER BY updated_at DESC;
```

Hesapli kullanicilarin gunluk soru sayilari:

```sql
SELECT u.email, c.period_key, c.count
FROM usage_counters c
JOIN users u ON c.subject_id = u.id::text
WHERE c.subject_type = 'user'
  AND c.period_type = 'day'
ORDER BY c.period_key DESC, c.count DESC;
```

Misafirlerin gunluk soru sayilari:

```sql
SELECT u.id, u.email, c.period_key, c.count
FROM usage_counters c
JOIN users u ON c.subject_id = u.id::text
WHERE u.role = 'guest'
  AND c.period_type = 'day'
ORDER BY c.period_key DESC, c.count DESC;
```

## 5. Notlar

- Sifreler duz metin olarak saklanmaz; `password_hash` olarak tutulur.
- `DATABASE_URL` bos kalirsa uygulama lokal gelistirme icin `data/app.db` SQLite dosyasini kullanir.
- Production'da `DATABASE_URL` mutlaka Neon PostgreSQL URL'si olmalidir.

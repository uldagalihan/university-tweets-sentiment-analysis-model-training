import pandas as pd

# --- Ayarlar ---

# Lütfen 'veriseti.xlsx' adını kendi Excel dosyanızın adıyla değiştirin.
input_filename = 'temizlenecek_veri.xlsx'

# Tekrarlanan verilerin aranacağı sütun adı
column_to_deduplicate = 'text'

# Temizlenmiş verilerin kaydedileceği dosya adı
output_filename = 'raw.xlsx'

# --- Kod ---

try:
    # 1. Excel dosyasını oku
    df = pd.read_excel(input_filename)

    # 'text' sütununun varlığını kontrol et
    if column_to_deduplicate not in df.columns:
        print(f"Hata: '{column_to_deduplicate}' sütunu '{input_filename}' dosyasında bulunamadı.")
        print(f"Lütfen sütun adını kontrol edin. Dosyadaki mevcut sütunlar: {df.columns.tolist()}")
    else:
        # 2. İşlem öncesi orijinal satır sayısını al
        original_rows = len(df)
        print(f"İşlem öncesi orijinal satır sayısı: {original_rows}")

        # 3. 'text' sütununa göre tekrarlananları kaldır (ilk kaydı tut)
        # 'keep='first'' -> aynı metne sahip satırlardan ilkini tutar, diğerlerini siler.
        df_cleaned = df.drop_duplicates(subset=[column_to_deduplicate], keep='first')

        # 4. İşlem sonrası satır sayısını al
        cleaned_rows = len(df_cleaned)
        duplicates_removed = original_rows - cleaned_rows

        print(f"Kaldırılan tekrar eden satır sayısı: {duplicates_removed}")
        print(f"İşlem sonrası temizlenmiş satır sayısı: {cleaned_rows}")

        # 5. Temizlenmiş veriyi yeni bir Excel dosyasına kaydet
        # index=False -> Excel'e yazarken pandas'ın eklediği satır numaralarını kaldırır.
        df_cleaned.to_excel(output_filename, index=False)

        print(f"\nBaşarılı! Temizlenmiş veri '{output_filename}' dosyasına kaydedildi.")

except FileNotFoundError:
    print(f"Hata: '{input_filename}' adında bir dosya bulunamadı.")
    print("Lütfen dosya adını kontrol edin veya dosyayı doğru dizine yüklediğinizden emin olun.")
except Exception as e:
    print(f"Beklenmeyen bir hata oluştu: {e}")
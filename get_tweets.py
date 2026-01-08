import requests
import pandas as pd
import time
import os
from datetime import datetime, date
import locale  # <-- YENİ MODÜL

# ============== 1. AYARLAR ==============

API_KEY = "*YOUR_API_KEY"
API_ENDPOINT = "https://api.twitterapi.io/twitter/tweet/advanced_search"

CSV_DOSYA_ADI = "temizlenecek1.csv"

MAX_TWEETS_PER_COMBO = 100

# --- YENİ TARİH FORMATI (SADECE GÜN) ---
SINCE_DATE_STR = "2021-01-01"  # örn: YYYY-MM-DD
UNTIL_DATE_STR = "2022-01-01"  # örn: YYYY-MM-DD

# ============== YENİ: TARİH SINIRLARINI PARSE ETME ==============

# Tarih formatımız: YYYY-MM-DD
TARIH_FORMAT_STR = "%Y-%m-%d"
since_date = None
until_date = None

try:
    if SINCE_DATE_STR:
        # String'i bir 'date' objesine çeviriyoruz
        since_date = datetime.strptime(SINCE_DATE_STR, TARIH_FORMAT_STR).date()

    if UNTIL_DATE_STR:
        # String'i bir 'date' objesine çeviriyoruz
        until_date = datetime.strptime(UNTIL_DATE_STR, TARIH_FORMAT_STR).date()

except ValueError as e:
    print(f"KRİTİK HATA: SINCE/UNTIL tarih formatı yanlış. Hata: {e}")
    print("Format 'YYYY-MM-DD' olmalı. Script durduruluyor.")
    exit()

print("--- Lokal tarih filtresi devrede (Sadece Gün) ---")
if since_date: print(f"    Başlangıç (Dahil): {since_date}")
if until_date: print(f"    Bitiş (Hariç):    {until_date}")

# ============== 2. BASE FILTER ==============

BASE_FILTERS = (
    "lang:tr "
    "-filter:retweets -filter:quote -filter:media -filter:replies -filter:links "
    "-\"kiralık\" -\"satılık\" -\"iş ilanı\" -\"personel alımı\" "
    "-\"kampanya\" -\"çekiliş\" -\"sponsorlu\" -\"reklam\" "
    "-\"özel ders\" -\"dershane\" -\"duyuru\" -\"haber\" "
    "-\"YKS\" -\"TYT\" -\"AYT\""
)

# ============== 3. ÜNİVERSİTELER ==============

UNIVERSITIES = {
    "YTU": '"Yıldız Teknik" OR YTÜ OR ytü',
    "ODTU": '"Orta Doğu Teknik" OR ODTÜ OR METU OR metu',
    "BOUN": '"Boğaziçi Üniversitesi" OR "Boğaziçi" OR BOUN',
    "ITU": '"İstanbul Teknik Üniversitesi" OR "İTÜ" OR ITU',
    "HACETTEPE": '"Hacettepe Üniversitesi" OR Hacettepe',
    "MARMARA": '"Marmara Üniversitesi" OR "Marmara Üni"',
    "GAZI": '"Gazi Üniversitesi" OR "Gazi Üni"',
    "DEU": '"Dokuz Eylül Üniversitesi" OR "Dokuz Eylül" OR DEÜ',
    "ISTANBUL_UNI": '"İstanbul Üniversitesi" OR "İstanbul Üni"',
    "BILKENT": '"Bilkent Üniversitesi" OR Bilkent',
    "EGE": '"Ege Üniversitesi" OR "Ege Üni" OR "Ege Uni"',
    "KOC": '"Koç Üniversitesi" OR "Koç Uni"',
    "YEDITEPE": '"Yeditepe Üniversitesi" OR "Yeditepe Uni"',
    "OZYEGIN": '"Özyeğin Üniversitesi" OR "Ozyegin University" OR "Özyeğin" OR Ozyegin',
}

# ============== 4. GRUPLAR ==============

QUERY_GROUPS = {
    "guclu_negatif": (
        '"rezalet" OR "berbat" OR "skandal" OR "iğrenç" OR "şikayet" '
        'OR "bıktım" OR "usandım" OR "yetersiz" OR "pişmanım" OR "nefret" OR "kötü"'
    ),
    "guclu_pozitif": (
        '"mükemmel" OR "harika" OR "muhteşem" OR "efsane" OR "gurur" '
        'OR "seviyorum" OR "iyi ki" OR "teşekkürler" OR "memnunum"'
    ),
    "akademik": (
        '"hoca" OR "hocası" OR "ders" OR "müfredat" OR "sınav" OR "vize" OR "final" '
        'OR "proje" OR "ödev" OR "quiz" OR "staj" OR "tez"'
    ),
    "idari": (
        '"öğrenci işleri" OR "rektörlük" OR "dekanlık" '
        'OR "burs" OR "transkript" OR "OBS"'
    ),
    "kampus": (
        '"yemekhane" OR "yemekler" OR "kantin" OR "kütüphane" OR "yurt" '
        'OR "kampüs" OR "internet" OR "wifi" OR "ulaşım" OR "ring" OR "temizlik" OR "güvenlik"'
    ),
}

# ============== 5. VARSA ESKİ CSV (APPEND) ==============

if os.path.exists(CSV_DOSYA_ADI):
    try:
        eski_df = pd.read_csv(CSV_DOSYA_ADI, dtype=str)
    except Exception:
        eski_df = pd.DataFrame()
else:
    eski_df = pd.DataFrame()

all_rows = []

print("\n--- Veri toplama başladı ---")
print(
    f"API Zaman filtresi (Best-Effort): "
    f"{'since:' + SINCE_DATE_STR if SINCE_DATE_STR else 'yok'} "
    f"{'until:' + UNTIL_DATE_STR if UNTIL_DATE_STR else 'yok'}"
)

# ============== 6. ÜNİ x GRUP: SADE ÇEKME ==============

for uni_code, uni_pat in UNIVERSITIES.items():
    for group_name, kw_pat in QUERY_GROUPS.items():

        # --- API SORGUSU İÇİN TARİH KISMI GÜNCELLENDİ ---
        time_clause = ""
        if SINCE_DATE_STR:
            time_clause += f" since:{SINCE_DATE_STR}"
        if UNTIL_DATE_STR:
            time_clause += f" until:{UNTIL_DATE_STR}"

        query = f"({uni_pat}) ({kw_pat}) {BASE_FILTERS}{time_clause}"

        if len(query) > 3800:
            print(f"[SKIP] {uni_code}-{group_name} query çok uzun ({len(query)} char)")
            continue

        combo_name = f"{uni_code}_{group_name}"
        print(f"\n[{combo_name}]")

        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }

        cursor = ""
        added_for_combo = 0

        while True:
            if added_for_combo >= MAX_TWEETS_PER_COMBO:
                break

            params = {
                "query": query,
                "queryType": "Latest",
                "cursor": cursor,
            }

            try:
                resp = requests.get(API_ENDPOINT, headers=headers, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"KRİTİK HATA: {e}")
                break

            if resp.status_code != 200:
                print(f"HATA! HTTP {resp.status_code} - {resp.text[:200]}")
                break

            data = resp.json()
            tweets = data.get("tweets", [])
            if not tweets:
                break

            for tw in tweets:
                if added_for_combo >= MAX_TWEETS_PER_COMBO:
                    break

                text = (tw.get("text") or "").strip()
                if not text:
                    continue

                # ============== YENİ LOKAL FİLTRELEME (strptime KULLANILARAK) ==============
                tweet_created_at_str = (tw.get("createdAt") or "").strip()
                if not tweet_created_at_str:
                    continue  # Tarihi olmayan tweet'i atla

                tweet_date = None
                try:
                    # Gelen format: 'Sat Dec 21 10:39:33 +0000 2024'
                    # Bu formatı parse etmek için locale'i 'C' (İngilizce) olarak ayarlıyoruz
                    locale.setlocale(locale.LC_TIME, 'C')
                    tweet_dt = datetime.strptime(tweet_created_at_str, "%a %b %d %H:%M:%S %z %Y")
                    tweet_date = tweet_dt.date()
                    # Locale'i varsayılana geri döndür
                    locale.setlocale(locale.LC_TIME, '')

                except ValueError:
                    # Eğer 'Sat Dec...' formatı başarısız olursa, 'ISO' formatını (Z ile biten) dene
                    try:
                        tweet_date = datetime.fromisoformat(tweet_created_at_str.replace('Z', '+00:00')).date()
                    except ValueError:
                        # İkisi de başarısız olursa, hatayı bas ve atla
                        print(f"Uyarı: Anlaşılamayan tarih formatı: {tweet_created_at_str}. Tweet atlanıyor.")
                        continue

                # Lokal tarih kontrolü (date objeleri ile)
                # 'since' (dahil)
                if since_date and tweet_date < since_date:
                    # Bu tweet çok eski, atla
                    continue

                    # 'until' (hariç)
                if until_date and tweet_date >= until_date:
                    # Bu tweet çok yeni (veya tam sınırda), atla
                    continue

                    # ============== FİLTRELEME SONU ==============

                row = {
                    "type": tw.get("type"),
                    "id": str(tw.get("id") or "").strip(),
                    "url": (tw.get("url") or "").strip(),
                    "tags": "",  # buraya manuel etiket yazacaksınız
                    "text": text,
                    "createdAt": tweet_created_at_str,  # Orijinal string'i kaydet
                    "location": (
                        (tw.get("author") or {}).get("location") or ""
                        if isinstance(tw.get("author"), dict) else ""
                    ).strip(),
                    "authorUserName": (
                        (tw.get("author") or {}).get("userName") or ""
                        if isinstance(tw.get("author"), dict) else ""
                    ).strip(),
                    "university": uni_code,
                    "group": group_name,
                }

                all_rows.append(row)
                added_for_combo += 1

            if not data.get("has_next_page") or not data.get("next_cursor"):
                break

            cursor = data.get("next_cursor")
            time.sleep(1)

        print(f"{combo_name}: {added_for_combo} tweet")

# ============== 7. CSV APPEND & SAVE ==============

print("\n--- Toplama bitti ---")
print(f"Bu run'da (lokal filtreli) eklenen yeni satır: {len(all_rows)}")

yeni_df = pd.DataFrame(all_rows)

if not eski_df.empty and not yeni_df.empty:
    combined = pd.concat([eski_df, yeni_df], ignore_index=True)
elif not eski_df.empty:
    combined = eski_df
else:
    combined = yeni_df

cols = [
    "type",
    "id",
    "url",
    "tags",  # text'in solu
    "text",
    "createdAt",
    "location",
    "authorUserName",
    "university",
    "group",
]
combined = combined.reindex(columns=cols)

combined.to_csv(CSV_DOSYA_ADI, index=False, encoding="utf-8-sig")
print(f"CSV güncellendi: {CSV_DOSYA_ADI} (toplam {len(combined)} tweet)")
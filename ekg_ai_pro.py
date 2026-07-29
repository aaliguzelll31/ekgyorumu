import os
import sys
import json
import argparse
import base64
import logging
import io
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
import google.generativeai as genai
from PIL import Image

# ==============================================================================
# API AYARLARI
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ Birincil AI hazır")

if GROQ_API_KEY:
    print("✓ Yedek AI hazır")

# ==============================================================================
# DETAYLI MI VE RİTİM ALGORİTMALARI
# ==============================================================================
TEDAVI_ALGORITMALARI = {
    "ANTERIOR_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Anterior STEMI",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: ANTERIOR STEMI
🎯 LOKALIZASYON: V1-V4 derivasyonları
🩸 TIKANIK DAMAR: LAD (Sol Ön İnen Arter)
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ EN CIDDI MI TİPİ!
• Sol ventrikülün büyük kısmı etkilenir
• Kardiyojenik şok riski yüksek
• Aritmi riski yüksek (VT/VF)

🎯 ANA KARAR:
→ ACİL PCI (Perkütan Koroner Girişim)
→ Hedef: Kapı-Balon <90 dakika
→ PCI yoksa: Fibrinoliz <30 dakika

💊 İLK MÜDAHALE (MONA):
• MORFİN 2-4 mg IV
• OKSİJEN (SpO2 <90 ise)
• NİTROGLİSERİN 0.4 mg SL (5 dk arayla max 3)
• ASPİRİN 300 mg çiğnet

💊 ANTİAGREGAN YÜKLEME:
• Aspirin 300 mg + Clopidogrel 600 mg
• VEYA Ticagrelor 180 mg
• VEYA Prasugrel 60 mg

💊 ANTİKOAGÜLAN:
• Enoksaparin 30 mg IV bolus + 1 mg/kg SC
• VEYA Fondaparinuks 2.5 mg SC

💊 EK TEDAVİ:
• Beta-bloker (Metoprolol 5 mg IV)
• Statin (Atorvastatin 80 mg PO)
• ACE inhibitör (ilk 24 saatte)

🚑 TRANSPORT:
• PCI merkezine ACIL
• Kırmızı kod, ışık+siren
• 2 IV yol açık

⚠️ KOMPLİKASYONLAR:
• Kardiyojenik şok → İnotrop
• VT/VF → Defibrilasyon HAZIR
• Kalp yetmezliği → Furosemid IV
• Papiller kas rüptürü → Cerrahi"""
    },
    "INFERIOR_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - İnferior STEMI",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: İNFERİOR STEMI
🎯 LOKALIZASYON: II, III, aVF derivasyonları
🩸 TIKANIK DAMAR: RCA (Sağ Koroner Arter)
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ ÖZEL DİKKAT!
• Sağ ventrikül tutulumu olabilir!
• AV blok riski yüksek
• Bradikardi sık görülür

🔍 SAĞ VENTRİKÜL MI EKARTASYONU:
→ V4R çek (sağ göğüs derivasyonu)
→ ST elevasyonu >1mm → Sağ V MI

⛔ SAĞ VENTRİKÜL MI'DA:
• NİTROGLİSERİN KULLANMA!
• Morfin dikkatli kullan
• Preload'a bağımlı → Sıvı VER!

🎯 ANA KARAR:
→ ACİL PCI
→ Hedef: Kapı-Balon <90 dakika

💊 İLK MÜDAHALE:
• ASPİRİN 300 mg çiğnet
• OKSİJEN (SpO2 <90 ise)
• MORFİN 2-4 mg IV (dikkatli!)

⚠️ NİTROGLİSERİN:
• Sağ V MI YOKSA: 0.4 mg SL
• Sağ V MI VARSA: VERME!

💊 ANTİAGREGAN:
• Aspirin 300 mg + Clopidogrel 600 mg
• VEYA Ticagrelor 180 mg

💊 ANTİKOAGÜLAN:
• Enoksaparin 1 mg/kg SC

💊 BRADİKARDİ VARSA:
• ATROPİN 0.5-1 mg IV
• Etkisizse: Transkütan pacing

💊 HİPOTANSİYON + Sağ V MI:
• %0.9 NaCl 250-500 ml BOLUS
• Yanıt izle
• Gerekirse Dopamin

🚑 TRANSPORT:
• PCI merkezine ACIL
• Yolda: AV blok, bradikardi izle

⚠️ KOMPLİKASYONLAR:
• AV blok (2°-3°) → Pacing
• Bradikardi → Atropin
• Hipotansiyon → Sıvı"""
    },
    "LATERAL_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Lateral STEMI",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: LATERAL STEMI
🎯 LOKALIZASYON: I, aVL, V5-V6
🩸 TIKANIK DAMAR: Cx (Sirkumfleks) veya LAD Diagonal
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ DİKKAT:
• Genelde diğer MI'lara eşlik eder
• İzole lateral MI nadir
• Cx tıkanıklığı EKG'de sessiz olabilir!

🎯 ANA KARAR:
→ ACİL PCI
→ Hedef: Kapı-Balon <90 dakika

💊 İLK MÜDAHALE (MONA):
• MORFİN 2-4 mg IV
• OKSİJEN (SpO2 <90 ise)
• NİTROGLİSERİN 0.4 mg SL
• ASPİRİN 300 mg çiğnet

💊 ANTİAGREGAN:
• Aspirin 300 mg + Clopidogrel 600 mg
• VEYA Ticagrelor 180 mg

💊 ANTİKOAGÜLAN:
• Enoksaparin 1 mg/kg SC

💊 EK:
• Beta-bloker
• Statin 80 mg
• ACE inhibitör

🚑 TRANSPORT:
• PCI merkezine ACIL

⚠️ DİKKAT:
• Yaygın MI'ın parçası olabilir
• Anterolateral, İnferolateral kontrolü yap"""
    },
    "POSTERIOR_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Posterior STEMI",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: POSTERIOR STEMI
🎯 LOKALIZASYON: V7-V9 (posterior derivasyonlar)
🩸 TIKANIK DAMAR: RCA veya Cx
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ KOLAY KAÇIRILIR!

🔍 EKG BULGULARI:
• V1-V3'te ST DEPRESYONU (ayna görüntüsü!)
• V1-V3'te R dalgası PROMİNENT
• V1-V3'te T dalgası POZİTİF
→ Posterior derivasyon (V7-V9) ÇEK!
→ V7-V9'da >0.5mm ST elevasyonu = Posterior MI

⚠️ SIKLIKLA İNFERİOR MI'YA EŞLİK EDER!

🎯 ANA KARAR:
→ ACİL PCI
→ Hedef: Kapı-Balon <90 dakika

💊 İLK MÜDAHALE:
• ASPİRİN 300 mg çiğnet
• MORFİN 2-4 mg IV
• NİTROGLİSERİN 0.4 mg SL (Sağ V MI yoksa)
• OKSİJEN (SpO2 <90)

💊 ANTİAGREGAN:
• Aspirin 300 mg + Clopidogrel 600 mg
• VEYA Ticagrelor 180 mg

💊 ANTİKOAGÜLAN:
• Enoksaparin 1 mg/kg SC

🚑 TRANSPORT:
• PCI merkezine ACIL

⚠️ MUTLAKA KONTROL:
• Sağ V MI (V4R)
• İnferior MI (II, III, aVF)"""
    },
    "SAG_V_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Sağ Ventrikül MI",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: SAĞ VENTRİKÜL MI
🎯 LOKALIZASYON: V4R (sağ göğüs derivasyonu)
🩸 TIKANIK DAMAR: RCA proximal
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ ⚠️ ⚠️ ÖZEL DURUM! ⚠️ ⚠️ ⚠️

⛔ MUTLAK KONTRENDİKASYONLAR:
❌ NİTROGLİSERİN VERME!
❌ MORFİN dikkatli (hipotansiyon)
❌ Diüretik VERME
❌ ACE inhibitör dikkatli

🔍 EKG BULGULARI:
• II, III, aVF'te ST elevasyonu
• V4R'de ST elevasyonu >1mm
• Hipotansiyon
• Boyun venöz distansiyon
• Akciğer temiz (ödem YOK)

🎯 ANA KARAR:
→ ACİL PCI (RCA'ya)
→ PRELOAD'U KORU!

💊 ÖNCELIK: SIVI TEDAVİSİ!
• %0.9 NaCl 250-500 ml IV BOLUS
• Yanıt yoksa tekrar 500 ml
• Toplam 1-2 L verebilir
• Amaç: TA artması

💊 EĞER HİPOTANSİYON DEVAM EDERSE:
• DOPAMIN 5-20 mcg/kg/dk
• VEYA Dobutamin 2-20 mcg/kg/dk
• Norepinefrin (son çare)

💊 İZİN VERİLEN İLAÇLAR:
• ASPİRİN 300 mg
• Clopidogrel 600 mg
• Enoksaparin (dikkatli)
• Oksijen

⛔ VERME:
• Nitrogliserin
• Furosemid
• Yüksek doz morfin

🚑 TRANSPORT:
• PCI merkezine ACIL
• Yatay pozisyon (bacaklar yukarı)
• 2 IV yol (sıvı için)

⚠️ İZLE:
• TA sık (5 dk)
• Bradikardi → Atropin
• AV blok → Pacing hazır"""
    },
    "YAYGIN_ANTERIOR_MI": {
        "aciliyeti": "🚨🚨 SÜPER KIRMIZI - Yaygın Anterior MI",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: YAYGIN ANTERIOR STEMI
🎯 LOKALIZASYON: V1-V6, I, aVL
🩸 TIKANIK DAMAR: LAD PROXIMAL (Ana LAD)
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ ⚠️ ⚠️ EN CIDDI MI! ⚠️ ⚠️ ⚠️

• Sol ventrikülün BÜYÜK KISMI etkilenir
• Kardiyojenik şok riski ÇOK YÜKSEK
• Ölüm riski YÜKSEK
• Mekanik komplikasyon riski

🎯 ANA KARAR:
→ SUPER ACIL PCI
→ Hedef: Kapı-Balon <60 dakika (İdeal)
→ Fibrinoliz: Yalnızca PCI mümkün değilse

💊 İLK MÜDAHALE (MONA):
• MORFİN 2-4 mg IV
• OKSİJEN (SpO2 <90 ise)
• NİTROGLİSERİN 0.4 mg SL
• ASPİRİN 300 mg çiğnet

💊 ÇİFT ANTİAGREGAN:
• Aspirin 300 mg
• + Ticagrelor 180 mg (tercih)
• VEYA Prasugrel 60 mg
• VEYA Clopidogrel 600 mg

💊 ANTİKOAGÜLAN:
• Enoksaparin 30 mg IV + 1 mg/kg SC
• VEYA UFH 60 U/kg IV

💊 STATIN:
• Atorvastatin 80 mg PO

💊 KARDİYOJENİK ŞOK VARSA:
• Dopamin 5-20 mcg/kg/dk
• Norepinefrin
• İntra-aortik balon pompası (IABP)
• Mekanik destek düşün

🚑 TRANSPORT:
• EN YAKIN PCI merkezine
• Kırmızı kod
• Yolda: Defibrilatör HAZIR
• Havayolu ekipmanı HAZIR

⚠️ KOMPLİKASYON RİSKİ YÜKSEK:
• Kardiyojenik şok
• VT/VF (defibrilasyon hazır!)
• Akut mitral yetmezlik
• Ventriküler septal rüptür
• Serbest duvar rüptürü

📊 MORTALİTE:
• Zamanında PCI: %5-10
• Geç PCI: %20-30
• Tedavisiz: >%50"""
    },
    "NSTEMI": {
        "aciliyeti": "⚠️ SARI KOD - NSTEMI/USAP",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: NSTEMI / USAP
🔍 ST DEPRESYONU / T İNVERSİYONU
━━━━━━━━━━━━━━━━━━━━━━━

⚠️ NSTEMI DE KALP KRİZİDİR!

🎯 ANA KARAR:
→ Risk stratifikasyonu (GRACE skoru)
→ Erken invaziv strateji

📊 GRACE SKORU:
• >140 → ACİL (<2 saat)
• 109-140 → Erken (<24 saat)
• <109 → Selektif (72 saat)

💊 İLK MÜDAHALE:
• ASPİRİN 300 mg çiğnet
• NİTROGLİSERİN 0.4 mg SL
• OKSİJEN (SpO2 <90)
• Morfin (ağrı için)

💊 ANTİAGREGAN:
• Aspirin 300 mg
• + Clopidogrel 300-600 mg
• VEYA Ticagrelor 180 mg

💊 ANTİKOAGÜLAN:
• Enoksaparin 1 mg/kg SC 2x1
• VEYA Fondaparinuks 2.5 mg SC

💊 EK:
• Atorvastatin 80 mg PO
• Beta-bloker (kontrendikasyon yoksa)
• ACE inhibitör (LV disfonksiyon)

🚑 TRANSPORT:
• Kardiyoloji merkezi
• Sarı kod

⚠️ TAKİP:
• Troponin (0-3-6 saat)
• Seri EKG
• STEMI'ye dönüşebilir!"""
    },
    "AF": {
        "aciliyeti": "⚠️ Atriyal Fibrilasyon",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: Atriyal Fibrilasyon (AF)
🔍 Düzensiz RR, P dalgası YOK
━━━━━━━━━━━━━━━━━━━━━━━

🎯 ANA KARAR:

⚠️ ANSTABİL (TA<90, bilinç değ., göğüs ağrısı):
→ SENKRONİZE KARDİYOVERSİYON
   • Başlangıç: 100-200 J
   • Sedasyon: Midazolam 2-5 mg IV

✅ STABİL:
→ HIZ KONTROLÜ tercih edilir

💊 HIZ KONTROLÜ:
• Metoprolol 5 mg IV (2 dk'da, 3 doz)
• VEYA Diltiazem 0.25 mg/kg IV
• VEYA Digoksin 0.5 mg IV yavaş
• Hedef: <110/dk

💊 RİTİM KONTROLÜ:
• Amiodaron 150 mg IV (10 dk)
• Sonra: 1 mg/dk infüzyon

💊 ANTİKOAGÜLASYON:
• CHA₂DS₂-VASc ≥2 (E) veya ≥3 (K)
• DOAK (Apixaban, Rivaroxaban)
• VEYA Warfarin (INR 2-3)

⚠️ KARDİYOVERSİYON ÖNCESİ:
• <48 saat: Direkt yapılabilir
• >48 saat: 3 hafta OAK veya TEE"""
    },
    "VT": {
        "aciliyeti": "🚨 KIRMIZI KOD - VT",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: Ventriküler Taşikardi (VT)
🔍 Geniş QRS (>120ms), Hız >100/dk
━━━━━━━━━━━━━━━━━━━━━━━

🎯 ANA KARAR:

❌ NABIZ YOK → VF/Nabızsız VT!
   → DEFİBRİLASYON 150-200 J
   → CPR başla

✅ NABIZ VAR:

⚠️ ANSTABİL:
→ SENKRONİZE KARDİYOVERSİYON
   • Monomorfik: 100 J başla
   • Polimorfik: 200 J
   • Sedasyon hazırla

✅ STABİL:
→ İLAÇ TEDAVİSİ

💊 STABİL VT İLAÇLARI:
• AMİODARON 150 mg IV (10 dk)
  → Tekrar: 150 mg
  → İnfüzyon: 1 mg/dk × 6 saat

• LİDOKAİN 1-1.5 mg/kg IV (alternatif)

💊 TORSADES DE POINTES:
• Magnezyum sülfat 2 g IV
• QT uzatan ilaçları kes"""
    },
    "VF": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - VF",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: Ventriküler Fibrilasyon
━━━━━━━━━━━━━━━━━━━━━━━

🎯 ANA KARAR:
⚡ HEMEN DEFİBRİLASYON!

⚡ DEFİBRİLASYON:
• Bifazik: 150-200 J
• Monofazik: 360 J
• Şok → 2 dk CPR → Ritm kontrol → Şok

🫀 CPR:
• Hız: 100-120/dk
• Derinlik: 5-6 cm
• 30:2 (kompresyon:ventilasyon)

💊 İLAÇLAR:

ADRENALİN:
• 1 mg IV/IO
• Her 3-5 dakikada TEKRAR
• 2. şoktan sonra başla

AMİODARON:
• 300 mg IV bolus
• 3. şoktan sonra
• Tekrar: 150 mg

VEYA LİDOKAİN:
• 1-1.5 mg/kg IV
• Tekrar: 0.5-0.75 mg/kg

🔍 5H-5T ARA:
H: Hipovolemi, Hipoksi, H+, Hipo/HiperK, Hipotermi
T: Toksin, Tamponad, Tension pnö, Tromboz (kor/pulm)"""
    },
    "BRADIKARDI": {
        "aciliyeti": "⚠️ Semptomatik Bradikardi",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: Bradikardi (<50/dk)
━━━━━━━━━━━━━━━━━━━━━━━

🎯 ANA KARAR:

❓ SEMPTOM VAR MI?
• TA<90, bilinç değ., göğüs ağrısı, nefes darlığı

✅ ASEMPTOMATİK → İzle
⚠️ SEMPTOMATİK → Aktif tedavi

💊 1. BASAMAK:
ATROPİN 1 mg IV
• Her 3-5 dk tekrar
• Maksimum: 3 mg

⛔ ATROPİN ETKİSİZ OLABİLİR:
• 2. derece Mobitz II blok
• 3. derece AV blok
• Kalp transplant hastası

⚡ 2. BASAMAK (Atropin yetersizse):
TRANSKÜTAN PACING
• Hız: 60-80/dk
• Akım: 40-80 mA
• Sedasyon: Midazolam

💊 3. BASAMAK (İnfüzyon):
• Dopamin 5-20 mcg/kg/dk
• VEYA Epinefrin 2-10 mcg/dk

🚑 KESİN TEDAVİ:
• Transvenöz pacing
• Kalıcı pacemaker"""
    },
    "SVT": {
        "aciliyeti": "⚠️ Supraventriküler Taşikardi",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: SVT (Dar QRS, 150-250/dk)
━━━━━━━━━━━━━━━━━━━━━━━

🎯 ANA KARAR:

⚠️ ANSTABİL:
→ SENKRONİZE KARDİYOVERSİYON 50-100 J

✅ STABİL:
→ Sırayla ilerle

1️⃣ VAGAL MANEVRALAR:
• Valsalva manevrası (modifiye)
• Yüze soğuk uygulama
• Öksürme
⛔ Karotis masajı - dikkat (yaşlıda)

2️⃣ ADENOZİN (Vagal başarısızsa):
• 1. doz: 6 mg IV HIZLI PUSH
• 2. doz: 12 mg IV (1-2 dk sonra)
• 3. doz: 12 mg IV

⚠️ HASTAYI UYAR:
• Kısa süreli göğüs basıncı normal

3️⃣ ADENOZİN YETERSİZ:
• Diltiazem 0.25 mg/kg IV
• VEYA Metoprolol 5 mg IV
• VEYA Amiodaron 150 mg IV"""
    },
    "AV_BLOK": {
        "aciliyeti": "⚠️ AV Blok",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: AV Blok
━━━━━━━━━━━━━━━━━━━━━━━

🔍 BLOK TİPİ:

1° BLOK (PR>200ms):
→ Tedavi gereksiz, izle

2° MOBİTZ I (Wenckebach):
→ Genelde stabil, izle

2° MOBİTZ II:
⚠️ Yüksek risk!
→ Pacemaker hazırla

3° TAM BLOK:
🚨 ACİL!
→ ATROPİN çoğunlukla etkisiz
→ Transkütan pacing HEMEN

💊 TEDAVİ (Semptomlu):

1️⃣ ATROPİN 1 mg IV
   (Mobitz II ve 3° blokta ETKİSİZ)

2️⃣ TRANSKÜTAN PACING
   • Hız: 60-80/dk
   • Sedasyon: Midazolam

3️⃣ İNFÜZYON:
   • Dopamin 5-20 mcg/kg/dk
   • Epinefrin 2-10 mcg/dk

🎯 KESİN TEDAVİ:
• Kalıcı pacemaker

🔍 NEDENLERİ:
• İnferior MI (RCA)
• İlaçlar (BB, CCB, digoksin)
• Hiperkalemi"""
    },
    "ASISTOLI": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: Asistoli / PEA
━━━━━━━━━━━━━━━━━━━━━━━

🎯 ANA KARAR:
⚡ HEMEN CPR!
⛔ DEFİBRİLASYON YAPILMAZ!

🫀 CPR:
• Hız: 100-120/dk
• Derinlik: 5-6 cm
• 30:2

💊 İLAÇ:
ADRENALİN 1 mg IV/IO
• Her 3-5 dakikada TEKRAR
• HEMEN başla (erken adrenalin faydalı)

⛔ ATROPİN ARTIK YOK!
(Kılavuzdan çıkarıldı)

🔍 5H-5T ARA (KRİTİK!):

H'ler:
• Hipovolemi → Sıvı
• Hipoksi → O2, ventilasyon
• H+ (Asidoz) → Ventilasyon
• Hipo/HiperK → K+/Ca++
• Hipotermi → Isıtma

T'ler:
• Toksin → Antidot
• Tamponad → Perikardiyosentez
• Tension pnö → İğne dekompresyon
• Tromboz (kor) → PCI
• Tromboz (pulm) → Fibrinolitik"""
    },
    "NORMAL": {
        "aciliyeti": "✅ Normal Sinüs Ritmi",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 TANI: Normal EKG
━━━━━━━━━━━━━━━━━━━━━━━

✅ NORMAL BULGULAR:
• Hız: 60-100/dk
• Ritim: Düzenli sinüs
• PR: 120-200 ms
• QRS: <120 ms
• QTc: <440 ms (E), <460 ms (K)
• ST: İzoelektrik

🎯 ANA KARAR:
→ Tedavi gerekmez
→ Klinik korelasyon önemli

⚠️ DİKKAT:
"Normal EKG" ≠ "Sorun yok"

📋 SEMPTOMLU HASTA (EKG normal):
• Aralıklı ritim bozukluğu olabilir
• NSTEMI ilk saatte normal olabilir
• Anstabil angina düşün
• Pulmoner emboli düşün

📋 EK TETKİKLER:
• Troponin (kardiyak enzim)
• Ekokardiyografi
• Holter (aralıklı aritmiler)
• D-dimer (PE şüphesi)"""
    },
    "GENEL": {
        "aciliyeti": "ℹ️ Genel Değerlendirme",
        "algoritma": """━━━━━━━━━━━━━━━━━━━━━━━
📌 GENEL YAKLAŞIM
━━━━━━━━━━━━━━━━━━━━━━━

🎯 SİSTEMATİK EKG:
1. Hız (60-100)
2. Ritim (düzenli/düzensiz)
3. Aks (-30° ile +90°)
4. P dalgası
5. PR aralığı
6. QRS süresi
7. ST segmenti
8. T dalgası
9. QT/QTc

💊 GENEL TEDAVİ:
• ABC değerlendirme
• Vital takibi
• IV yol açılması
• O2 (SpO2 <94)
• Monitörizasyon

🔍 TETKİKLER:
• Troponin
• Elektrolit
• Tam kan
• Koagülasyon

🚑 KONSÜLTASYON:
• Kardiyoloji"""
    }
}

def tedavi_algoritmasi_bul(tahmin_metni):
    tahmin_upper = tahmin_metni.upper()
    
    # MI türlerini spesifik olarak yakala
    if "YAYGIN ANTERIOR" in tahmin_upper or "YAYGIN ANTERİOR" in tahmin_upper or ("ANTERIOR" in tahmin_upper and ("LATERAL" in tahmin_upper or "V1-V6" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["YAYGIN_ANTERIOR_MI"]
    elif "SAĞ VENTRİKÜL" in tahmin_upper or "SAG VENTRIKUL" in tahmin_upper or "SAĞ V" in tahmin_upper or "RIGHT VENTRICULAR" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["SAG_V_MI"]
    elif "POSTERIOR" in tahmin_upper or "POSTERİOR" in tahmin_upper or "ARKA DUVAR" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["POSTERIOR_MI"]
    elif "ANTERIOR" in tahmin_upper or "ANTERİOR" in tahmin_upper or "ÖN DUVAR" in tahmin_upper or ("STEMI" in tahmin_upper and ("V1" in tahmin_upper or "V2" in tahmin_upper or "V3" in tahmin_upper or "V4" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["ANTERIOR_MI"]
    elif "INFERIOR" in tahmin_upper or "İNFERİOR" in tahmin_upper or "ALT DUVAR" in tahmin_upper or ("STEMI" in tahmin_upper and ("II" in tahmin_upper or "III" in tahmin_upper or "AVF" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["INFERIOR_MI"]
    elif "LATERAL" in tahmin_upper or "YAN DUVAR" in tahmin_upper or ("STEMI" in tahmin_upper and ("V5" in tahmin_upper or "V6" in tahmin_upper or "AVL" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["LATERAL_MI"]
    elif "STEMI" in tahmin_upper or ("ST" in tahmin_upper and ("ELEVASYON" in tahmin_upper or "YÜKSELME" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["ANTERIOR_MI"]  # Varsayılan
    elif "NSTEMI" in tahmin_upper or "USAP" in tahmin_upper or "UNSTABLE ANGINA" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["NSTEMI"]
    elif "FİBRİL" in tahmin_upper or "FIBRIL" in tahmin_upper or "ATRİYAL" in tahmin_upper:
        if "VENTRİKÜLER" in tahmin_upper or "VF" in tahmin_upper:
            return TEDAVI_ALGORITMALARI["VF"]
        return TEDAVI_ALGORITMALARI["AF"]
    elif "VENTRİKÜLER TAŞİKARDİ" in tahmin_upper or "VT" in tahmin_upper.split():
        return TEDAVI_ALGORITMALARI["VT"]
    elif "BRADİKARDİ" in tahmin_upper or "BRADYKARDI" in tahmin_upper or "YAVAŞ" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["BRADIKARDI"]
    elif "SVT" in tahmin_upper or "SUPRAVENTRİKÜLER" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["SVT"]
    elif "AV BLOK" in tahmin_upper or "BLOK" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["AV_BLOK"]
    elif "ASİSTOLİ" in tahmin_upper or "ASYSTOLE" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["ASISTOLI"]
    elif "NORMAL" in tahmin_upper or "SİNÜS" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["NORMAL"]
    else:
        return TEDAVI_ALGORITMALARI["GENEL"]

# ==============================================================================
# HTML ARAYÜZÜ
# ==============================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EKG Analiz Asistanı</title>
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#dc3545">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #c31432 0%, #240b36 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { background: white; padding: 30px; border-radius: 12px 12px 0 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-bottom: 4px solid #dc3545; }
        .header h1 { color: #2c3e50; margin-bottom: 8px; font-size: 28px; }
        .header p { color: #7f8c8d; font-size: 14px; }
        .badge { background: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; vertical-align: middle; }
        .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 4px; font-size: 13px; color: #856404; }
        .main-content { background: white; padding: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .upload-zone { border: 2px dashed #dc3545; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; background: #fff5f5; }
        .upload-zone:hover { border-color: #a71d2a; background: #ffe5e5; }
        input[type="file"] { display: none; }
        .button-group { display: flex; gap: 10px; margin-top: 20px; justify-content: center; }
        button { padding: 12px 24px; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-analyze { background: #dc3545; color: white; }
        .btn-analyze:hover { background: #a71d2a; }
        .btn-analyze:disabled { background: #95a5a6; cursor: not-allowed; }
        .btn-reset { background: #95a5a6; color: white; }
        .preview-img { max-width: 100%; max-height: 300px; border-radius: 6px; margin: 15px 0; display: none; }
        .result-section { margin-top: 30px; display: none; }
        .result-box { background: #f8f9fa; padding: 25px; border-radius: 8px; border-left: 4px solid #dc3545; margin-bottom: 15px; }
        .result-box.urgent { border-left-color: #e74c3c; background: #fdf2f2; }
        .result-box.safe { border-left-color: #27ae60; background: #f0fdf4; }
        .algorithm-box { background: #fff5f5; padding: 25px; border-radius: 8px; border-left: 4px solid #dc3545; margin-top: 15px; }
        .algorithm-title { font-size: 16px; font-weight: 700; color: #dc3545; margin-bottom: 10px; }
        .algorithm-urgency { display: inline-block; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-bottom: 12px; background: white; }
        .algorithm-content { font-size: 14px; color: #2c3e50; line-height: 1.8; white-space: pre-wrap; font-family: 'Segoe UI', sans-serif; }
        .result-label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: bold; }
        .result-value { font-size: 22px; font-weight: 700; color: #2c3e50; margin-bottom: 15px; }
        .analysis-text { font-size: 14px; color: #34495e; line-height: 1.7; white-space: pre-wrap; margin-top: 10px; }
        .error-message { background: #fdf2f2; border-left: 4px solid #e74c3c; color: #c0392b; padding: 15px; border-radius: 4px; margin-top: 15px; }
        .footer { background: white; padding: 20px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-size: 12px; color: #7f8c8d; text-align: center; }
        .loading { text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #dc3545; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚑 EKG Analiz Asistanı <span class="badge">DETAYLI TANI</span></h1>
            <p>Ritim + MI Türü + Tedavi Algoritması</p>
        </div>
        <div class="warning-box">
            ⚠️ <strong>Klinik Uyarı:</strong> Bu sistem karar destek amaçlıdır. Tıbbi karar mutlaka hekim/uzman tarafından verilmelidir.
        </div>
        <div class="main-content">
            <div class="upload-zone" id="uploadZone">
                <div style="font-size: 48px; margin-bottom: 10px;">📸</div>
                <div style="font-size: 16px; font-weight: 500; color: #2c3e50;">EKG Fotoğrafını Yükleyin</div>
                <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">Tıklayarak dosya seçin veya fotoğraf çekin</div>
                <input type="file" id="ekgFile" accept="image/*" capture="environment">
            </div>
            <img id="previewImg" class="preview-img" src="" alt="Önizleme">
            <div class="button-group" id="buttonGroup" style="display: none;">
                <button class="btn-analyze" id="analyzeBtn">🧠 EKG'yi Analiz Et</button>
                <button class="btn-reset" id="resetBtn">↻ Temizle</button>
            </div>
            <div class="result-section" id="resultSection">
                <div id="resultContent"></div>
            </div>
        </div>
        <div class="footer"><p>EKG Analiz Av8.0 | EKG + MI Türü + Tedavi Asistanı | Prm. Ali GÜZEL tarafından hazırlanmıştırsistanı</p></div>
    </div>
    <script>
        const uploadZone = document.getElementById('uploadZone');
        const ekgFile = document.getElementById('ekgFile');
        const previewImg = document.getElementById('previewImg');
        const buttonGroup = document.getElementById('buttonGroup');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resetBtn = document.getElementById('resetBtn');
        const resultSection = document.getElementById('resultSection');
        const resultContent = document.getElementById('resultContent');

        uploadZone.addEventListener('click', () => ekgFile.click());
        ekgFile.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    previewImg.src = event.target.result;
                    previewImg.style.display = 'block';
                    buttonGroup.style.display = 'flex';
                    resultSection.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });

        analyzeBtn.addEventListener('click', async () => {
            const file = ekgFile.files[0];
            if (!file) return alert('Lütfen bir dosya seçin!');
            
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '🧠 Analiz ediliyor...';
            resultSection.style.display = 'block';
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>EKG detaylı analiz ediliyor (15-30 saniye)...</p></div>';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/analyze', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (result.status === 'success') {
                    const pred = result.prediction;
                    const urgent = pred.acil_mudahale;
                    const algo = pred.algoritma || {};
                    
                    resultContent.innerHTML = `
                        <div class="result-box ${urgent ? 'urgent' : 'safe'}">
                            <div class="result-label">🎯 EKG TANISI</div>
                            <div class="result-value">${pred.tahmin}</div>
                            
                            <div class="result-label">📊 ACİLİYET</div>
                            <div style="font-size: 16px; font-weight: 600; margin-bottom: 15px; color: ${urgent ? '#e74c3c' : '#27ae60'};">
                                ${pred.risk_seviyesi}
                            </div>
                            
                            <div class="result-label">📝 EKG DEĞERLENDİRMESİ</div>
                            <div class="analysis-text">${pred.detay}</div>
                        </div>
                        
                        <div class="algorithm-box">
                            <div class="algorithm-title">💊 TEDAVİ ALGORİTMASI</div>
                            <div class="algorithm-urgency">${algo.aciliyeti || 'Genel Yaklaşım'}</div>
                            <div class="algorithm-content">${algo.algoritma || 'Algoritma yüklenemedi.'}</div>
                            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #f5c6cb; font-size: 12px; color: #555;">
                                📚 <strong>Kaynak:</strong> ESC 2023, ACLS 2020, Sağlık Bakanlığı<br>
                                ⚠️ <strong>Uyarı:</strong> ${pred.uyari}
                            </div>
                        </div>
                    `;
                } else {
                    resultContent.innerHTML = `<div class="error-message"><strong>Hata:</strong> ${result.message}</div>`;
                }
            } catch (error) {
                resultContent.innerHTML = `<div class="error-message"><strong>Bağlantı Hatası:</strong> ${error.message}</div>`;
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '🧠 EKG\'yi Analiz Et';
            }
        });

        resetBtn.addEventListener('click', () => {
            ekgFile.value = ''; previewImg.style.display = 'none'; buttonGroup.style.display = 'none'; resultSection.style.display = 'none';
        });
    </script>
</body>
</html>
"""

# ==============================================================================
# MI TÜRLERİNİ TANIYAN DETAYLI PROMPT
# ==============================================================================
DETAYLI_PROMPT = """Sen deneyimli bir kardiyolog ve acil tıp uzmanısın.

Bu EKG fotoğrafını DETAYLI analiz et. Özellikle MI türlerini SPESIFIK olarak belirt.

🔍 MI TÜRLERİ (ÇOK ÖNEMLİ - SPESIFIK BELİRT!):

STEMI TÜRLERİ:
1. ANTERIOR STEMI → V1-V4'te ST elevasyonu → LAD tıkanıklığı
2. YAYGIN ANTERIOR (ANTEROLATERAL) → V1-V6, I, aVL → LAD proximal
3. İNFERİOR STEMI → II, III, aVF'te ST elevasyonu → RCA tıkanıklığı
4. LATERAL STEMI → I, aVL, V5-V6 → Cx tıkanıklığı
5. POSTERIOR STEMI → V1-V3'te ST DEPRESYONU (ayna) + V7-V9'da ST elevasyonu → RCA/Cx
6. SAĞ VENTRİKÜL MI → İnferior MI + V4R'de ST elevasyonu → RCA proximal (NİTRO VERİLMEZ!)

⚠️ MI TANISI KOYARKEN:
- Hangi derivasyonlarda ST elevasyonu?
- Hangi anatomik alan?
- Hangi koroner damar tıkalı?
- Reciprocal (ayna) değişiklikler var mı?
- Sağ ventrikül tutulumu şüphesi var mı? (İnferior MI'da MUTLAKA belirt)

DİĞER TANILAR:
- Ritim türü (sinüs, AF, VT, VF, SVT, asistoli)
- Bradikardi/AV blok
- ST depresyonu (NSTEMI/USAP)
- T inversiyonu

Sadece aşağıdaki JSON formatında cevap ver:

{
  "tahmin": "SPESIFIK MI türü veya ritim (örn: 'Anterior STEMI (V1-V4) - LAD Tıkanıklığı', 'İnferior STEMI (II, III, aVF) - RCA', 'Sağ Ventrikül MI Eşlik', 'Yaygın Anterior MI', 'Posterior MI', 'Lateral MI', 'Ventriküler Taşikardi', 'Hızlı AF', vb.)",
  "risk_seviyesi": "Düşük veya Orta veya Yüksek",
  "acil_mudahale": true veya false,
  "detay": "KISA değerlendirme (4-6 cümle). Şunları belirt: 1) Kalp hızı, 2) Ritim, 3) ST bulguları (hangi derivasyonlar, kaç mm), 4) Anatomik lokalizasyon ve etkilenen koroner arter, 5) Sağ V MI şüphesi (İnferior MI'da), 6) Kritik uyarılar (Nitro kontrendikasyonu vb.)",
  "uyari": "Klinik onay zorunludur."
}

Fotoğraf net değilse: 'Analiz Yapılamadı' de.
SADECE JSON döndür."""

# ==============================================================================
# ÇOKLU AI MOTORU
# ==============================================================================
class MultiAIAnalyzer:
    def __init__(self):
        self.gemini_model = None
        if GEMINI_API_KEY:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
                print("✓ Birincil AI hazır")
            except Exception as e:
                print(f"⚠ Birincil AI yüklenemedi: {e}")

    def analyze_with_gemini(self, image_bytes):
        if not self.gemini_model:
            raise Exception("Birincil AI yüklenmemiş")
        
        image = Image.open(io.BytesIO(image_bytes))
        response = self.gemini_model.generate_content([DETAYLI_PROMPT, image])
        response_text = response.text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        return json.loads(response_text)

    def analyze_with_groq(self, image_bytes):
        if not GROQ_API_KEY:
            raise Exception("Yedek AI anahtarı yok")
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DETAYLI_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        
        if response.status_code != 200:
            raise Exception(f"API hatası: {response.status_code}")
        
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        return json.loads(result_text)

    def analyze_ecg_image(self, image_bytes):
        errors = []
        result = None
        
        if self.gemini_model:
            try:
                print("🔵 Birincil AI ile analiz...")
                result = self.analyze_with_gemini(image_bytes)
                print("✓ Birincil AI başarılı")
            except Exception as e:
                errors.append(str(e))
                print(f"⚠ Birincil AI başarısız: {e}")
        
        if not result and GROQ_API_KEY:
            try:
                print("🟢 Yedek AI ile deneniyor...")
                result = self.analyze_with_groq(image_bytes)
                print("✓ Yedek AI başarılı")
            except Exception as e:
                errors.append(str(e))
                print(f"⚠ Yedek AI başarısız: {e}")
        
        if not result:
            return {
                "tahmin": "Analiz Yapılamadı",
                "risk_seviyesi": "Bilinmiyor",
                "acil_mudahale": False,
                "detay": "Sistem şu anda yoğun. Lütfen tekrar deneyin.",
                "uyari": "Sistem geçici olarak kullanılamıyor.",
                "algoritma": TEDAVI_ALGORITMALARI["GENEL"]
            }
        
        tahmin = result.get("tahmin", "")
        algoritma = tedavi_algoritmasi_bul(tahmin)
        result["algoritma"] = algoritma
        
        print(f"📋 Algoritma: {algoritma['aciliyeti']}")
        
        return result

# ==============================================================================
# FASTAPI
# ==============================================================================
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EKG Analiz", version="8.0.0")
analyzer = MultiAIAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "EKG Analiz Asistanı",
        "short_name": "EKG",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#dc3545",
        "theme_color": "#dc3545",
        "icons": [
            {"src": "https://img.icons8.com/color/192/heart-with-pulse.png", "sizes": "192x192", "type": "image/png"},
            {"src": "https://img.icons8.com/color/512/heart-with-pulse.png", "sizes": "512x512", "type": "image/png"}
        ]
    })

@app.post("/api/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Dosya boş.")
        
        print(f"📸 EKG alındı: {file.filename} ({len(contents)} bytes)")
        print("🧠 Analiz başlıyor...")
        
        prediction = analyzer.analyze_ecg_image(contents)
        
        print(f"✓ Analiz: {prediction['tahmin']}")
        
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚑 EKG ANALİZ ASİSTANI v8.0")
    print("="*60)
    print(f"📍 http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
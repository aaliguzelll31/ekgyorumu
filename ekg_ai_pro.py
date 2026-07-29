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
# PARAMEDİK SAHA ALGORİTMALARI (T.C. Sağlık Bakanlığı 112 & UMKE Protokolleri)
# ==============================================================================
TEDAVI_ALGORITMALARI = {
    "STEMI": {
        "aciliyeti": "🚨 KIRMIZI KOD - STEMI (Kalp Krizi) - HASTANE ÖNCESİ",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - STEMI

⏱️ İLK 5 DAKİKA (Olay Yerinde):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Güvenlik değerlendirmesi (kendini koru!)
✓ Hasta pozisyonu: Yarı oturur (semi-Fowler)
✓ ABC değerlendirmesi
✓ Vital bulgular: TA, nabız, SpO2, solunum, KŞ
✓ 12 derivasyon EKG çek + KAYIT AL
✓ Hastane ile TELEFON BAĞLANTISI KUR (STEMI aktivasyonu)

💊 PARAMEDİK YETKİSİ - İLAÇLAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ASPİRİN (ASA) 300 mg
   → ÇİĞNETEREK ver (dil altı değil!)
   → Kontrendikasyon: Alerji, aktif kanama

2. OKSİJEN
   → SADECE SpO2 <90 ise
   → 2-4 L/dk nazal kanül
   → Rutin O2 ÖNERİLMİYOR (2024 kılavuzu)

3. NİTROGLİSERİN 0.4 mg SL
   → 5 dk arayla max 3 doz
   → KONTRENDİKASYON:
     ⛔ Sistolik TA <90 mmHg
     ⛔ Bradikardi <50/dk
     ⛔ Sildenafil (Viagra) son 24 saatte
     ⛔ Sağ ventrikül infarktüsü şüphesi

4. AĞRI YÖNETİMİ
   → Morfin 2-4 mg IV (varsa)
   → VEYA Fentanil 25-50 mcg IV

🚨 IV YOL AÇ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 2 adet geniş çaplı damar yolu (18G)
✓ %0.9 NaCl KVO hızında
✓ KAN ÖRNEĞİ AL (hastaneye götürmek için)

🚑 HIZLI TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 HEDEF: En yakın PCI YAPAN hastane
⏱️ Süre: Semptom-Balon <120 dk
📞 Hastaneye ÖN BİLGİ VER:
   "STEMI aktivasyonu, [X] dakika içinde varış,
   Anterior/İnferior/Lateral MI şüphesi"

⚠️ YOL BOYUNCA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Sürekli monitör (aritmi riski!)
✓ Defibrilatör HAZIRDA
✓ Kardiyak arrest hazırlığı
✓ VF/VT olursa: ANINDA ŞOK 150-200 J

📋 KAYIT ALTINA AL:
   • EKG'nin çekildiği zaman
   • Semptom başlangıç zamanı
   • Verilen tüm ilaçlar + saat
   • Vital değişimler"""
    },
    "NSTEMI": {
        "aciliyeti": "⚠️ SARI KOD - NSTEMI / USAP",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - NSTEMI/USAP

⏱️ OLAY YERİ DEĞERLENDİRMESİ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ ABC + Vital bulgular
✓ 12 derivasyon EKG (ST DEPRESYONU ara)
✓ Semptom başlangıç zamanı
✓ Risk faktörleri sorgusu

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ASPİRİN 300 mg çiğneterek
   ⛔ Alerji, aktif kanama varsa VERME

2. OKSİJEN (SpO2 <90 ise)

3. NİTROGLİSERİN 0.4 mg SL
   → Ağrı geçmezse 5 dk sonra tekrar (max 3)

4. AĞRI: Morfin/Fentanil (varsa)

🚨 IV YOL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 18G damar yolu
✓ %0.9 NaCl KVO
✓ Kan örneği (Troponin için)

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Kardiyoloji olan en yakın hastane
📞 Bildirim: "USAP/NSTEMI şüphesi, [X] dk sonra varış"

⚠️ İZLEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Her 5 dk vital kontrol
✓ Ağrı skoru takibi
✓ EKG değişikliği tekrar bak
✓ STEMI'ye dönüşebilir!

📝 UNUTMA:
   NSTEMI de KALP KRİZİDİR!
   Ciddiye al, hızlı transport yap."""
    },
    "AF": {
        "aciliyeti": "⚠️ Atriyal Fibrilasyon - Saha Yönetimi",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - AF

🔍 İLK DEĞERLENDİRME:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Hemodinamik durum KRİTİK!
✓ Vital bulgular: TA, nabız, SpO2
✓ 12 derivasyon EKG (RR düzensiz, P dalgası yok)
✓ Semptom sorgusu:
   - Çarpıntı? Ne zaman başladı?
   - Göğüs ağrısı?
   - Nefes darlığı?
   - Baş dönmesi/senkop?

🚨 DURUM DEĞERLENDİRMESİ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

❗ ANSTABİL HASTA (Aşağıdakiler varsa):
   ⚠️ TA <90 mmHg
   ⚠️ Bilinç değişikliği
   ⚠️ Göğüs ağrısı devam eden
   ⚠️ Akut kalp yetmezliği bulguları
   
   → ACİL TRANSPORT!
   → Kardiyoversiyon HASTANEDE yapılır
   → Yolda defibrilatör HAZIR olsun
   → Sedasyon HAZIRLA (midazolam)

✅ STABİL HASTA:
   → Sakin, güven ver
   → Vital izlem
   → Hastaneye rutin transport

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ IV yol: 18G, %0.9 NaCl KVO
✓ O2 (SpO2 <94 ise)
✓ Sürekli monitör

⛔ PARAMEDİK YETKİSİ DIŞINDA:
   • Amiodaron
   • Digoksin
   • Antikoagülan
   (Doktor orderıyla verilir)

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Kardiyoloji olan hastane
📞 Bildirim: "AF, hemodinamik [stabil/anstabil]"

⚠️ YOL BOYUNCA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Ventriküler hız takibi
✓ TA sık ölçüm (5 dk)
✓ Ani düşüş → ACİL MÜDAHALE
✓ Bilinç kaybı → CPR hazırlığı"""
    },
    "VT": {
        "aciliyeti": "🚨 KIRMIZI KOD - Ventriküler Taşikardi",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - VT

⚡ HEMEN NABIZ KONTROL ET!
━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ NABIZ YOK:
   → NABIZSIZ VT = KARDİYAK ARREST!
   → Hemen CPR başla
   → DEFİBRİLASYON: 150-200 J
   → VF/Nabızsız VT algoritması

✅ NABIZ VAR:
   → Aşağıdaki adımlara devam

🔍 HEMODİNAMİK DURUM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

❗ ANSTABİL (KRİTİK!):
   ⚠️ TA <90 mmHg
   ⚠️ Bilinç değişikliği
   ⚠️ Şok bulguları
   ⚠️ Göğüs ağrısı + akut nefes darlığı
   
   → HEMEN HASTANEYE!
   → SENKRONİZE KARDİYOVERSİYON (hastanede)
   → Yolda: Defibrilatör pedleri YAPIŞTIR
   → Sedasyon HAZIRLA

✅ STABİL:
   → Hızlı ama sakin transport
   → Vital sürekli izlem

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 2 IV yol (18G)
✓ %0.9 NaCl KVO
✓ O2 desteği
✓ 12 derivasyon EKG (KAYIT AL!)
✓ Defibrilatör HAZIRDA

⛔ AMİODARON: Doktor orderı gerekli
   (Bazı ilde 112 protokolü izin verir)

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 En yakın hastane (kardiyoloji tercih)
📞 URGENT: "VT hastası, [stabil/anstabil]"
🚨 Kırmızı ışıklarla, sirenle!

⚠️ YOL BOYUNCA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Sürekli EKG monitör
✓ Nabız her dakika kontrol
✓ Nabız kaybolursa: ANINDA CPR + ŞOK
✓ VF'ye dönüşebilir!

📝 SEBEP DÜŞÜN:
   • Akut MI
   • Elektrolit bozukluğu (K+, Mg+)
   • İlaç zehirlenmesi
   • Kalp yetmezliği"""
    },
    "VF": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - VF/Nabızsız VT",
        "algoritma": """🚑 112 KARDİYAK ARREST PROTOKOLÜ

⚡ ZAMAN = HAYAT! HEMEN BAŞLA!
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ CPR BAŞLAT (İLK 10 SANİYE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Hastayı sert zemine yatır
✓ Göğüs kompresyonu:
   • Hız: 100-120/dk
   • Derinlik: 5-6 cm
   • Sternum orta hattı
   • Tam geri çekilme
✓ 30:2 (kompresyon:ventilasyon)
✓ Kesintileri MINIMIZE ET (max 10 sn)

2️⃣ DEFİBRİLATÖR TAK (İLK 2 DAKİKA):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Ped 1: Sağ klavikula altı
✓ Ped 2: Sol midaksiller 5. IC aralık
✓ Ritm analizi yap
✓ VF/Nabızsız VT? → HEMEN ŞOK

⚡ DEFİBRİLASYON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Bifazik: 150-200 J
• Monofazik: 360 J
• "ÇEKİLİN!" bağır
• Şoktan HEMEN sonra 2 dk CPR
• 2 dk sonra ritm kontrol

3️⃣ HAVAYOLU (CPR'ı KESME!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Ambu maske ile ventilasyon
✓ Nazofarengeal/orofarengeal airway
✓ Uygun durumda: Laringeal maske
✓ %100 O2

4️⃣ IV/IO YOL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ IV yol açılamıyorsa: İNTRAOSSEÖZ (IO)
✓ Tibia proximal (tercih)
✓ %0.9 NaCl açık infüzyon

💊 İLAÇLAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADRENALİN (Epinephrine):
   → 1 mg IV/IO
   → Her 3-5 dakikada TEKRAR
   → Şoklanamayan ritimde HEMEN
   → Şoklanan ritimde: 2. şok sonrası

AMİODARON (VF/Nabızsız VT için):
   → 300 mg IV/IO bolus
   → 3. şoktan sonra
   → Tekrar: 150 mg IV

⏱️ CPR DÖNGÜSÜ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 dk CPR → Ritm kontrol → Şok (gerekirse) →
2 dk CPR → İlaç → Ritm kontrol → ...

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ CPR SIRASINDA HASTANEYE!
📞 "Kardiyak arrest, resüsitasyon devam ediyor"
🎯 En yakın hastane

📋 SEBEP ARA (5H-5T):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
H'ler: Hipovolemi, Hipoksi, H+ (asidoz), 
       Hipo/HiperK, Hipotermi
T'ler: Toksin, Tamponad, Tension pnömotoraks,
       Tromboz (koroner/pulmoner)

⚠️ ROSC OLURSA:
✓ Vital tam kontrol
✓ 12 derivasyon EKG
✓ HEMEN hastane bildirimi
✓ Hedefli sıcaklık yönetimi hazırlığı"""
    },
    "BRADIKARDI": {
        "aciliyeti": "⚠️ Semptomatik Bradikardi",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - BRADİKARDİ

🔍 DEĞERLENDİRME:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Kalp hızı <50/dk MI?
✓ SEMPTOM VAR MI?
   - TA <90 mmHg?
   - Bilinç değişikliği?
   - Göğüs ağrısı?
   - Nefes darlığı?
   - Baş dönmesi/senkop?

⚠️ SEMPTOMLU DURUM = AKTİF MÜDAHALE
✅ ASEMPTOMATİK = İZLE VE TAŞI

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GENEL:
   ✓ IV yol açılması (18G)
   ✓ %0.9 NaCl KVO
   ✓ O2 (SpO2 <94 ise)
   ✓ Sürekli EKG monitör

2. ATROPİN (SEMPTOMLU ise):
   💊 1 mg IV bolus
   → 3-5 dakikada tekrar
   → Maksimum: 3 mg
   
   ⚠️ ETKİSİZ OLABİLİR:
   • 2. derece Mobitz II blok
   • 3. derece AV blok
   • Kalp transplant hastası

3. ATROPİN YETERSİZ İSE:
   🚨 TRANSKÜTAN PACING (varsa):
   • Ped yerleşimi
   • Hız: 60-80/dk
   • Akım: 40-80 mA
   • Hastayı sedasyon HAZIRLA
   
   VEYA HASTANEYE ACIL!

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Kardiyoloji olan hastane
📞 Bildirim: "Semptomatik bradikardi, [Kalp hızı X], [TA]"

⚠️ YOL BOYUNCA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Vital her 5 dk
✓ Bilinç takibi
✓ Ani kötüleşme → Pacing/CPR hazır
✓ TA kritik ise: Sıvı bolus (250 ml)

📝 SEBEP DÜŞÜN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• İnferior MI (RCA)
• İlaçlar (Beta-bloker, digoksin, CCB)
• Elektrolit (K+ yüksek)
• Hipotermi
• Yüksek intrakraniyal basınç"""
    },
    "SVT": {
        "aciliyeti": "⚠️ Supraventriküler Taşikardi",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - SVT

🔍 DEĞERLENDİRME:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Kalp hızı 150-250/dk
✓ QRS dar (<120 ms)
✓ P dalgası görülmüyor
✓ RR düzenli

🚨 HEMODİNAMİK DURUM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

❗ ANSTABİL:
   → HEMEN HASTANEYE!
   → Kardiyoversiyon (hastanede)
   → Yolda: Defibrilatör HAZIR

✅ STABİL: Aşağıdaki adımlar

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. VAGAL MANEVRALAR (GÜVENLİ!):
   
   a) VALSALVA MANEVRASI:
   • Hastayı sırt üstü yatır
   • 15 saniye ıkındır (40 mmHg)
   • MODİFİYE: Bacakları kaldır (%40 daha etkili)
   
   b) YÜZE SOĞUK UYGULAMA:
   • Buz torbası yüze
   • 15-30 saniye
   • Özellikle çocuklarda etkili
   
   c) ÖKSÜRTME:
   • Güçlü öksürme
   
   ⛔ KAROTİS MASAJI:
   • Sadece doktor yetkisi
   • Yaşlıda RİSKLİ (inme!)

2. IV YOL:
   ✓ Büyük çaplı damar (18G)
   ✓ Antekübital bölge tercih
   ✓ %0.9 NaCl

3. ADENOZİN (Doktor orderı):
   💊 6 mg IV HIZLI PUSH
   → 20 ml NaCl ile flush
   → 1-2 dk sonra 12 mg
   → Gerekirse tekrar 12 mg
   
   ⚠️ HASTAYI UYAR:
   • "Kısa süreli göğüste basınç"
   • "Yüzde kızarma"
   • Bu NORMAL, geçici!

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Kardiyoloji olan hastane
📞 "SVT, vagal manevra [başarılı/başarısız]"

⚠️ İZLEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Sürekli monitör
✓ Vital her 5 dk
✓ SVT geçerse: EKG tekrar çek"""
    },
    "AV_BLOK": {
        "aciliyeti": "⚠️ AV Blok",
        "algoritma": """🚑 112 PARAMEDİK PROTOKOLÜ - AV BLOK

🔍 BLOK TİPİ AYIRIMI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DERECE (PR >200 ms):
   → GENELDE TEDAVİ GEREKSİZ
   → İzlem yeterli

2. DERECE MOBİTZ I (Wenckebach):
   → PR uzayarak QRS düşer
   → GENELDE STABİL
   → İzle, taşı

2. DERECE MOBİTZ II:
   ⚠️ YÜKSEK RİSK!
   → 3. dereceye ilerleyebilir

3. DERECE (TAM BLOK):
   🚨 ACİL DURUM!
   → P ve QRS BAĞIMSIZ
   → Ventrikül hızı 20-40/dk

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENEL:
✓ IV yol (18G)
✓ %0.9 NaCl KVO
✓ O2 desteği
✓ Sürekli monitör
✓ 12 derivasyon EKG

SEMPTOMLU HASTA (TA<90, bilinç değ., göğüs ağrısı):

1. ATROPİN 1 mg IV
   ⚠️ 2. Mobitz II ve 3. derece blokta ETKİSİZ!
   
2. TRANSKÜTAN PACING (varsa):
   • Hemen hazırla
   • Ped yerleşimi
   • Hız: 60-80/dk
   • Sedasyon: Midazolam 2-5 mg IV
   
3. DOPAMİN infüzyon (order gerekli):
   • 5-20 mcg/kg/dk

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Kardiyoloji + Pacemaker olan hastane
📞 URGENT: "Tam AV blok, hız [X]"
🚨 Kırmızı ışık, siren

⚠️ YOL BOYUNCA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Kardiyak arrest hazırlığı
✓ Pacing HAZIRDA
✓ Vital her 3-5 dk
✓ Ani asistoli riski!"""
    },
    "ASISTOLI": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - Asistoli",
        "algoritma": """🚑 112 ASİSTOLİ / PEA PROTOKOLÜ

⚡ HEMEN CPR!
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ CPR BAŞLAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Göğüs kompresyonu 100-120/dk
✓ Derinlik 5-6 cm
✓ 30:2 (kompresyon:ventilasyon)
✓ KESİNTİSİZ

2️⃣ RİTM KONTROL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 2 dakikada bir
✓ Asistoli? → CPR DEVAM
✓ PEA (Nabızsız Elektriksel Aktivite)? → CPR

⛔ DEFİBRİLASYON YAPILMAZ!
   (Şoklanamayan ritim)

3️⃣ HAVAYOLU:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Ambu maske + %100 O2
✓ Nazofarengeal/orofarengeal
✓ İleri havayolu (LMA/ETT)

4️⃣ IV/IO YOL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ IV yol açılamıyorsa: IO (tibia)
✓ %0.9 NaCl açık infüzyon

💊 İLAÇLAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADRENALİN:
   → 1 mg IV/IO
   → Her 3-5 DAKİKADA TEKRAR
   → HEMEN ver (kanıt: erken adrenalin faydalı)

⛔ ATROPİN ARTIK YOK!
   (Kılavuzdan çıkarıldı)

🔍 5H-5T ARA (MUTLAKA!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━

H'ler:
• Hipovolemi → Sıvı bolus
• Hipoksi → O2, ventilasyon
• H+ (Asidoz) → İyi ventilasyon
• Hipo/HiperK → Bilgi hastaneye
• Hipotermi → Aktif ısıtma

T'ler:
• Toksin → Antidot (varsa)
• Tamponad → Perikardiyosentez (hastanede)
• Tension pnömotoraks → İğne dekompresyon
• Tromboz (koroner) → PCI için hastane
• Tromboz (pulmoner) → Fibrinolitik

🚑 TRANSPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ CPR SIRASINDA TAŞI!
   → LUCAS (varsa mekanik CPR)
   → Manuel CPR: Kesintisiz
📞 "Kardiyak arrest, asistoli, CPR devam ediyor"
🎯 En yakın hastane

📋 CPR SÜRESİ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• En az 20-30 dakika
• Doktor sonlandırma kararı verir
• Özel durumlar: Hipotermi (uzun CPR!)

⚠️ ROSC OLURSA:
✓ Vital tam kontrol
✓ 12 derivasyon EKG (sebep ara!)
✓ HEMEN hastane bildirimi
✓ %100 O2 → SpO2 94-98 hedef
✓ Hedefli sıcaklık yönetimi hazırlığı"""
    },
    "NORMAL": {
        "aciliyeti": "✅ Normal Sinüs Ritmi",
        "algoritma": """🚑 112 PARAMEDİK YAKLAŞIMI - NORMAL EKG

✅ EKG NORMAL SINIRLARDA

📊 NORMAL EKG KRİTERLERİ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Kalp hızı: 60-100/dk
✓ Ritim: Düzenli sinüs
✓ P dalgası: Uniform, her QRS öncesi
✓ PR: 120-200 ms
✓ QRS: <120 ms
✓ ST: İzoelektrik
✓ T: Normal morfoloji
✓ QTc: <440 ms (E), <460 ms (K)

⚠️ ANCAK DİKKAT!
━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Normal EKG" ≠ "Sorun yok"

📋 KLİNİĞE BAK:
• Göğüs ağrısı VAR MI?
• Nefes darlığı?
• Baş dönmesi?
• Çarpıntı hissi?

⚠️ SEMPTOM VARSA:
• Aralıklı ritim bozukluğu olabilir
• NSTEMI ilk saatte normal EKG olabilir!
• Anstabil angina
• Pulmoner emboli

💊 GENEL YAKLAŞIM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEMPTOMLU HASTA (EKG normal olsa da):
✓ IV yol açılması
✓ O2 (gerekliyse)
✓ Vital takibi
✓ Aspirin (göğüs ağrısı varsa, kontrendikasyon yoksa)
✓ Hastaneye transport

ASEMPTOMATIK HASTA:
✓ Vital kontrol
✓ Sorgulama detaylı
✓ Ambulans gerekliyse: rutin transport
✓ Hastane değerlendirmesi öner

🚑 TRANSPORT KRİTERLERİ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 HEMEN HASTANEYE:
• Devam eden göğüs ağrısı
• Bilinç değişikliği
• Anormal vitaller
• Yeni semptom

📞 HASTA REDDİYE:
• Bilinç açık, oryante
• Vital normal
• Semptom yok
• Aile onayı
• KAYIT ALTINA AL

📝 UNUTMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tek EKG YETERLİ DEĞİL
• Seri EKG önemli
• Klinik korelasyon şart
• Şüpheye düş → HASTANEYE"""
    },
    "GENEL": {
        "aciliyeti": "ℹ️ Genel Paramedik Yaklaşımı",
        "algoritma": """🚑 112 PARAMEDİK GENEL EKG PROTOKOLÜ

🔍 SİSTEMATİK YAKLAŞIM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GÜVENLİK:
   ✓ Kendini koru!
   ✓ Olay yeri güvenliği
   ✓ Eldivenler, maske

2. HIZLI DEĞERLENDİRME:
   ✓ Bilinç seviyesi (AVPU)
   ✓ ABC (Havayolu, Solunum, Dolaşım)
   ✓ Vital bulgular:
     - TA
     - Nabız (hız, ritim)
     - Solunum sayısı
     - SpO2
     - Kan şekeri
     - Ateş

3. HİKAYE (SAMPLE):
   S: Semptomlar
   A: Alerjiler
   M: Medikasyon (ilaçlar)
   P: Geçmiş hastalıklar
   L: Son yenilen yemek
   E: Olayı anlatın

4. EKG:
   ✓ 12 derivasyon
   ✓ ELEKTROD YERLEŞİMİ DOĞRU!
   ✓ Kağıt hızı 25 mm/s
   ✓ Kalibrasyon 10 mm/mV
   ✓ ARTİFAKT AZALT

💊 PARAMEDİK MÜDAHALE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

STANDART BAKIM:
✓ IV yol (18G, geniş çap)
✓ %0.9 NaCl KVO
✓ O2 (SpO2 <94 ise)
✓ Monitör (sürekli)
✓ Vital her 5 dk

SEMPTOMA GÖRE:
• Göğüs ağrısı → ASA + Nitro
• Nefes darlığı → O2, salbutamol
• Şok → Sıvı bolus
• Bilinç kaybı → Glukoz kontrol

🚑 TRANSPORT KARARI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 KIRMIZI KOD (Işık+Siren):
• STEMI
• Anstabil aritmi
• Şok
• Bilinç kaybı
• Solunum yetmezliği

⚠️ SARI KOD (Işıksız hızlı):
• NSTEMI şüphesi
• Stabil aritmi
• Kontrollü ağrı

✅ MAVİ KOD (Rutin):
• Stabil hasta
• Asemptomatik EKG değişikliği

📞 HASTANE BİLDİRİMİ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIST FORMATI:
M: Mekanizma (ne oldu?)
I: İnjury/Illness (yaralanma/hastalık)
S: Signs/Symptoms (bulgular)
T: Treatment (yapılan tedavi)

📋 KAYIT (ÖNEMLİ!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Zaman damgaları
✓ Vital değişimler
✓ Verilen ilaçlar
✓ Hasta yanıtı
✓ EKG kayıtları (SAKLA!)
✓ Kimlik bilgileri"""
    }
}

def tedavi_algoritmasi_bul(tahmin_metni):
    """AI tahminine göre uygun paramedik algoritmayı bulur"""
    tahmin_upper = tahmin_metni.upper()
    
    if "STEMI" in tahmin_upper or ("ST" in tahmin_upper and ("ELEVASYON" in tahmin_upper or "YÜKSELME" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["STEMI"]
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
# HTML ARAYÜZÜ (PARAMEDİK ODAKLI)
# ==============================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paramedik EKG Asistanı</title>
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
        .algorithm-content { font-size: 13px; color: #2c3e50; line-height: 1.7; white-space: pre-wrap; font-family: 'Segoe UI', monospace; }
        .result-label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: bold; }
        .result-value { font-size: 22px; font-weight: 700; color: #2c3e50; margin-bottom: 15px; }
        .analysis-text { font-size: 14px; color: #34495e; line-height: 1.8; white-space: pre-wrap; margin-top: 10px; }
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
            <h1>🚑 Paramedik EKG Asistanı <span class="badge">SAHA</span></h1>
            <p>112 Paramedik Saha Protokolleri | UMKE & Sağlık Bakanlığı</p>
        </div>
        <div class="warning-box">
            ⚠️ <strong>Saha Uyarısı:</strong> Bu sistem 112 Acil Sağlık Hizmetleri ve UMKE protokollerine göre paramedikler için hazırlanmıştır. Tıbbi karar mutlaka merkezi arayarak veya doktor onayı ile alınmalıdır.
        </div>
        <div class="main-content">
            <div class="upload-zone" id="uploadZone">
                <div style="font-size: 48px; margin-bottom: 10px;">📸</div>
                <div style="font-size: 16px; font-weight: 500; color: #2c3e50;">EKG Fotoğrafını Yükleyin</div>
                <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">Ambulansta çekilen EKG fotoğrafını yükleyin</div>
                <input type="file" id="ekgFile" accept="image/*" capture="environment">
            </div>
            <img id="previewImg" class="preview-img" src="" alt="Önizleme">
            <div class="button-group" id="buttonGroup" style="display: none;">
                <button class="btn-analyze" id="analyzeBtn">🚨 SAHA ANALİZİ YAP</button>
                <button class="btn-reset" id="resetBtn">↻ Temizle</button>
            </div>
            <div class="result-section" id="resultSection">
                <div id="resultContent"></div>
            </div>
        </div>
        <div class="footer"><p>v6.0 | Paramedik Saha Asistanı | 112 & UMKE Protokolleri</p></div>
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
            analyzeBtn.textContent = '🚨 Saha Analizi...';
            resultSection.style.display = 'block';
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>EKG hızlıca analiz ediliyor, lütfen bekleyin (15-30 saniye)...</p></div>';
            
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
                            <div class="algorithm-title">🚑 SAHA MÜDAHALE PROTOKOLÜ</div>
                            <div class="algorithm-urgency">${algo.aciliyeti || 'Genel Yaklaşım'}</div>
                            <div class="algorithm-content">${algo.algoritma || 'Algoritma yüklenemedi.'}</div>
                            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #f5c6cb; font-size: 12px; color: #555;">
                                📚 <strong>Kaynak:</strong> T.C. Sağlık Bakanlığı 112 Acil Sağlık Hizmetleri, UMKE Protokolleri, ACLS 2020<br>
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
                analyzeBtn.textContent = '🚨 SAHA ANALİZİ YAP';
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
# PARAMEDİK ODAKLI EKG ANALİZ PROMPTU
# ==============================================================================
DETAYLI_PROMPT = """Sen deneyimli bir 112 ACİL SAĞLIK EKİPLERİ hekimisin ve UMKE protokollerini biliyorsun. Aynı zamanda kardiyoloji uzmanısın.

Sana gösterilen EKG fotoğrafını PARAMEDİK BAKIŞ AÇISIYLA, SAHA ODAKLI şekilde analiz et.

🔍 SİSTEMATİK PARAMEDİK YAKLAŞIMI:

1️⃣ HIZLI DEĞERLENDİRME (5 saniyede):
   - Ritm düzenli mi? (RR aralıkları)
   - Hız normal mi? (60-100 arası mı?)
   - QRS geniş mi dar mı?
   - ST elevasyonu/depresyonu var mı?

2️⃣ DETAYLI EKG ANALİZİ:
   - Kalp hızı (bpm)
   - Ritm türü
   - P dalgası varlığı
   - PR aralığı
   - QRS süresi
   - QRS aksı
   - ST segmenti (hangi derivasyonlarda?)
   - T dalgası

3️⃣ TANI ODAKLI (Paramedik için ÖNEMLİ):
   - STEMI mi? (Anatomik lokalizasyon: Anterior/İnferior/Lateral)
   - Anstabil aritmi mi?
   - Şoklanabilir ritim mi? (VF/VT)
   - Bradikardi/Blok var mı?
   - Vagal manevra düşünülür mü?

4️⃣ SAHA ÖNCELİKLERİ:
   - HEMEN müdahale gerekli mi?
   - Hangi hastaneye götürülmeli? (PCI merkezi?)
   - Hangi kod ile transport? (Kırmızı/Sarı/Mavi)
   - Yolda ne kontrol edilmeli?

⚠️ PARAMEDİK ODAKLI YAZ:
- Sadece EKG bulgusu değil, KLİNİK KARAR odaklı yaz
- Ambulansta yapılabilecek adımları vurgu
- Riskli durumları özellikle belirt
- Hangi hastane tipi gerektiğini söyle

Sadece aşağıdaki JSON formatında cevap ver, başka hiçbir şey yazma:

{
  "tahmin": "Ana tanı (spesifik ve net, örn: 'Anterior STEMI (V1-V4) - LAD tıkanıklığı', 'Hızlı Ventrikül Yanıtlı AF', 'Normal Sinüs Ritmi', 'Semptomatik Bradikardi - AV Blok Şüphesi', vb.)",
  "risk_seviyesi": "KIRMIZI KOD - ACİL veya SARI KOD - Hızlı veya MAVİ KOD - Rutin",
  "acil_mudahale": true veya false,
  "detay": "PARAMEDİK ODAKLI Türkçe analiz (8-12 cümle). Şunları MUTLAKA belirt: 1) Kalp hızı (bpm) ve ritim, 2) Ana EKG bulguları (spesifik derivasyonlarla), 3) Anatomik lokalizasyon (hangi koroner damar alanı - LAD/RCA/Cx), 4) Ambulansta hangi vitallere dikkat edilmeli, 5) Hangi ilaç önceliği, 6) Hangi hastane tipine götürülmeli (PCI merkezi mi, sıradan mı), 7) Yolda hangi komplikasyonlar beklenmeli, 8) STEMI ise kapı-balon süresi hedefi. Paramedik dilinde, saha odaklı yaz.",
  "uyari": "Merkezi arayarak/doktor onayı ile hareket edin. Bu bir AI karar destek aracıdır."
}

Fotoğraf net değilse veya EKG değilse, tahmini 'Analiz Yapılamadı - Görüntü kalitesi yetersiz' olarak belirt.
SADECE JSON döndür, başka açıklama YOK."""

# ==============================================================================
# ÇOKLU AI ANALİZ MOTORU
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
            "max_tokens": 2048
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
                print("🔵 Birincil AI ile saha analizi...")
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
                "detay": "Sistem şu anda yoğun. Lütfen birkaç dakika sonra tekrar deneyin.",
                "uyari": "Sistem geçici olarak kullanılamıyor.",
                "algoritma": TEDAVI_ALGORITMALARI["GENEL"]
            }
        
        tahmin = result.get("tahmin", "")
        algoritma = tedavi_algoritmasi_bul(tahmin)
        result["algoritma"] = algoritma
        
        print(f"📋 Saha algoritması: {algoritma['aciliyeti']}")
        
        return result

# ==============================================================================
# FASTAPI SUNUCUSU
# ==============================================================================
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Paramedik EKG", version="6.0.0")
analyzer = MultiAIAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "Paramedik EKG Asistanı",
        "short_name": "Paramedik EKG",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#dc3545",
        "theme_color": "#dc3545",
        "icons": [
            {"src": "https://img.icons8.com/color/192/ambulance.png", "sizes": "192x192", "type": "image/png"},
            {"src": "https://img.icons8.com/color/512/ambulance.png", "sizes": "512x512", "type": "image/png"}
        ]
    })

@app.post("/api/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Dosya boş.")
        
        print(f"📸 EKG alındı: {file.filename} ({len(contents)} bytes)")
        print("🚑 Saha analizi başlıyor...")
        
        prediction = analyzer.analyze_ecg_image(contents)
        
        print(f"✓ Analiz: {prediction['tahmin']}")
        
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ==============================================================================
# GİRİŞ NOKTASI
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚑 PARAMEDİK EKG ASİSTANI v6.0")
    print("="*60)
    print(f"📍 Yerel adres: http://localhost:8000")
    print(f"🌐 Ağdaki adres: http://0.0.0.0:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
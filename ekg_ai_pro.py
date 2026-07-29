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
# SAĞLIK BAKANLIĞI / ESC / ACLS ALGORİTMA VERİTABANI
# ==============================================================================
TEDAVI_ALGORITMALARI = {
    "STEMI": {
        "aciliyeti": "🚨 ACİL - ST Elevasyonlu Miyokard Enfarktüsü",
        "algoritma": """📋 TÜRKİYE SAĞLIK BAKANLIĞI / ESC 2023 STEMI PROTOKOLÜ

1️⃣ İLK 10 DAKİKA (Değerlendirme):
   • Vital bulgular (TA, nabız, SpO2, solunum)
   • IV yol açılması (2 tercihen büyük çaplı)
   • 12 derivasyonlu EKG (10 dk içinde)
   • Kan gazı + Troponin + Tam kan sayımı

2️⃣ İLK MÜDAHALE (MONA-B):
   • MORFİN: 2-4 mg IV (ağrı için)
   • OKSİJEN: SpO2 <90 ise 2-4 L/dk
   • NİTROGLİSERİN: 0.4 mg sublingual (5 dk arayla max 3 doz)
   • ASPİRİN: 300 mg çiğneterek
   • BETA-BLOKER: Kontrendikasyon yoksa

3️⃣ ANTİAGREGAN YÜKLEME:
   • Aspirin 300 mg + Clopidogrel 600 mg
   • VEYA Ticagrelor 180 mg
   • VEYA Prasugrel 60 mg

4️⃣ REPERFÜZYON (KRİTİK):
   • Primer PCI hedef: <90 dakika (door-to-balloon)
   • PCI mümkün değilse: Fibrinoliz <30 dakika (door-to-needle)
   • Semptom başlangıcından <12 saat içinde

5️⃣ ANTİKOAGÜLASYON:
   • Enoksaparin 1 mg/kg SC (2x1)
   • VEYA Fondaparinuks 2.5 mg SC

6️⃣ HEDEFLER:
   ✓ Kapı-EKG: <10 dk
   ✓ Kapı-Balon: <90 dk
   ✓ Kapı-İğne: <30 dk"""
    },
    "NSTEMI": {
        "aciliyeti": "⚠️ ACİL - Non-ST Elevasyonlu MI",
        "algoritma": """📋 ESC 2023 NSTEMI/USAP PROTOKOLÜ

1️⃣ RİSK STRATİFİKASYONU (GRACE Skoru):
   • Yüksek risk (>140): Acil invaziv <2 saat
   • Orta risk (109-140): Erken invaziv <24 saat
   • Düşük risk (<109): Selektif invaziv

2️⃣ İLK MÜDAHALE:
   • Aspirin 300 mg yükleme
   • Clopidogrel 300-600 mg yükleme
   • Atorvastatin 80 mg
   • Beta-bloker (kontrendikasyon yoksa)
   • Nitrat (semptomatik)

3️⃣ ANTİKOAGÜLASYON:
   • Enoksaparin 1 mg/kg 2x1 SC
   • VEYA Fondaparinuks 2.5 mg SC

4️⃣ KORONER ANJİYOGRAFİ:
   • GRACE >140: Acil (<2 saat)
   • Yüksek risk: Erken (<24 saat)
   • Orta risk: <72 saat"""
    },
    "AF": {
        "aciliyeti": "⚠️ Atriyal Fibrilasyon Yönetimi",
        "algoritma": """📋 ESC 2024 ATRİYAL FİBRİLASYON PROTOKOLÜ

1️⃣ HEMODİNAMİK DEĞERLENDİRME:
   ⚠️ Anstabilse → ACİL KARDİYOVERSİYON
   • Senkronize DC şok 100-200 J
   
2️⃣ HIZ KONTROLÜ (Stabil hasta):
   • Metoprolol 5 mg IV (2 dk'da, tekrar edilebilir)
   • VEYA Diltiazem 0.25 mg/kg IV
   • VEYA Digoksin 0.5 mg IV (yavaş)
   • Hedef: <110/dk (istirahat)

3️⃣ RİTİM KONTROLÜ:
   • Amiodaron 150 mg IV (10 dk'da) → 1 mg/dk infüzyon
   • VEYA Propafenon 2 mg/kg IV
   • Elektif kardiyoversiyon planı

4️⃣ ANTİKOAGÜLASYON (CHA2DS2-VASc):
   • Erkek ≥2, Kadın ≥3: OAK ENDİKASYON VAR
   • DOAK (Apixaban, Rivaroxaban, Dabigatran)
   • VEYA Warfarin (INR 2-3)

5️⃣ KARDİYOVERSİYON ÖNCESİ:
   • <48 saat: Direkt kardiyoversiyon
   • >48 saat: 3 hafta OAK VEYA TEE"""
    },
    "VT": {
        "aciliyeti": "🚨 ACİL - Ventriküler Taşikardi",
        "algoritma": """📋 ACLS 2020 VT PROTOKOLÜ

1️⃣ NABIZ VAR MI? KONTROL ET!
   ❌ Nabız YOK → VF/Nabızsız VT protokolü
   ✓ Nabız VAR → Devam

2️⃣ HEMODİNAMİK DURUM:
   ⚠️ ANSTABİL (hipotansiyon, göğüs ağrısı, bilinç):
   → SENKRONİZE KARDİYOVERSİYON
   • Monomorfik: 100 J başla, artır
   • Polimorfik: 200 J

3️⃣ STABİL VT:
   • Amiodaron 150 mg IV (10 dk'da)
   • Tekrar: 150 mg IV
   • İnfüzyon: 1 mg/dk × 6 saat, sonra 0.5 mg/dk
   
   ALTERNATİF:
   • Lidokain 1-1.5 mg/kg IV
   • Prokainamid 20-50 mg/dk

4️⃣ TORSADES DE POINTES:
   • Magnezyum sülfat 2 g IV
   • Elektrolit düzeltmesi
   • QT uzatan ilaçları kes

5️⃣ TETİK: Sürekli monitörizasyon + Kardiyoloji"""
    },
    "VF": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - Ventriküler Fibrilasyon",
        "algoritma": """📋 ACLS 2020 VF/NABIZSIZ VT PROTOKOLÜ

⚡ HEMEN DEFİBRİLASYON!

1️⃣ CPR + DEFİBRİLASYON:
   • Bifazik: 150-200 J
   • Monofazik: 360 J
   • 2 dakika CPR (30:2 veya sürekli 100-120/dk)

2️⃣ ADRENALİN:
   • 1 mg IV/IO her 3-5 dakikada
   • Vazopressin ARTIK ÖNERİLMİYOR

3️⃣ ANTİARİTMİK (3. şoktan sonra):
   • AMİODARON 300 mg IV bolus
   • Tekrar: 150 mg IV
   
   VEYA
   • Lidokain 1-1.5 mg/kg IV
   • Tekrar: 0.5-0.75 mg/kg

4️⃣ 5H-5T (Geri Döndürülebilir Nedenler):
   H'ler: Hipovolemi, Hipoksi, H+ (asidoz), Hipo/hiperK, Hipotermi
   T'ler: Toksin, Tamponad, Tension pnömotoraks, Tromboz (pulm/koroner)

5️⃣ ROSC SONRASI:
   • Hedefli sıcaklık yönetimi (32-36°C)
   • MAP >65 mmHg
   • Etkin oksijenasyon"""
    },
    "BRADIKARDI": {
        "aciliyeti": "⚠️ Semptomatik Bradikardi",
        "algoritma": """📋 ACLS BRADİKARDİ PROTOKOLÜ

1️⃣ DEĞERLENDİRME:
   Kalp hızı <50/dk + Semptom var mı?
   • Hipotansiyon
   • Akut mental durum değişikliği
   • Şok bulguları
   • İskemik göğüs ağrısı
   • Akut kalp yetmezliği

2️⃣ İLK MÜDAHALE:
   ✓ Havayolu, solunum, IV yol
   ✓ Monitör, 12 derivasyonlu EKG
   ✓ Oksijen (hipoksemi varsa)

3️⃣ İLAÇ TEDAVİSİ:
   💊 ATROPİN 1 mg IV bolus
   • Her 3-5 dakikada tekrar
   • Maksimum: 3 mg
   
   ⚠️ 2. veya 3. derece AV blokta atropin ETKİSİZ OLABİLİR!

4️⃣ ATROPİN YETERSİZSE:
   • TRANSKÜTAN PACING (hazır ol!)
   • Dopamin 5-20 mcg/kg/dk infüzyon
   • Epinefrin 2-10 mcg/dk infüzyon

5️⃣ KESİN TEDAVİ:
   • Transvenöz pacing
   • Kalıcı pacemaker değerlendirmesi
   • Kardiyoloji konsültasyonu"""
    },
    "SVT": {
        "aciliyeti": "⚠️ Supraventriküler Taşikardi",
        "algoritma": """📋 ACLS SVT PROTOKOLÜ

1️⃣ HEMODİNAMİK DURUM:
   ⚠️ ANSTABİL → Senkronize Kardiyoversiyon 50-100 J
   ✓ Stabil → Devam

2️⃣ VAGAL MANEVRALAR:
   • Karotid sinüs masajı (yaşlı hastada dikkat!)
   • Valsalva manevrası (modifiye)
   • Yüze soğuk uygulama
   • Öksürme, ıkınma

3️⃣ ADENOZİN:
   • 1. doz: 6 mg IV hızlı push (antekübital)
   • 2. doz: 12 mg IV (1-2 dk sonra)
   • 3. doz: 12 mg IV (tekrar)
   ⚠️ Astımda dikkat, WPW'de kontrendike olabilir

4️⃣ ADENOZİN YETERSİZ:
   • Diltiazem 0.25 mg/kg IV
   • VEYA Metoprolol 5 mg IV
   • VEYA Amiodaron 150 mg IV

5️⃣ REKÜRRENSE:
   • Elektrofizyoloji çalışması
   • Kateter ablasyon değerlendirmesi"""
    },
    "AV_BLOK": {
        "aciliyeti": "⚠️ AV Blok",
        "algoritma": """📋 AV BLOK YÖNETİMİ

1️⃣ BLOK DERECESİNİ BELİRLE:
   • 1. Derece: PR >200 ms (genelde tedavi gereksiz)
   • 2. Derece Mobitz Tip 1 (Wenckebach)
   • 2. Derece Mobitz Tip 2 (yüksek risk!)
   • 3. Derece (Tam blok) - ACİL!

2️⃣ SEMPTOMATİK MI?
   ✓ Asemptomatik: İzlem
   ⚠️ Semptomatik: Aktif tedavi

3️⃣ ACİL TEDAVİ (3. derece / Mobitz II):
   💊 Atropin 1 mg IV (etkisi kısıtlı olabilir)
   
   HEMEN:
   • Transkütan pacing
   • Dopamin/Epinefrin infüzyon
   • Transvenöz pacing hazırlığı

4️⃣ KALICI ÇÖZÜM:
   • Kalıcı pacemaker endikasyonları:
     - Semptomatik 2. derece Mobitz II
     - 3. derece AV blok
     - Semptomatik yavaş ritim

5️⃣ NEDENLERİ ARAŞTIR:
   • İskemi (özellikle inferior MI)
   • İlaçlar (Beta-bloker, CCB, digoksin)
   • Elektrolit bozukluğu (K+)
   • Miyokardit, Lyme hastalığı"""
    },
    "ASISTOLI": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - Asistoli",
        "algoritma": """📋 ACLS ASİSTOLİ/PEA PROTOKOLÜ

⚡ HEMEN CPR BAŞLA!

1️⃣ YÜKSEK KALİTELİ CPR:
   • Hız: 100-120/dk
   • Derinlik: 5-6 cm
   • Tam geri çekilme
   • Kesintileri minimize et
   • 30:2 (2 kurtarıcı = sürekli göğüs, 10/dk ventilasyon)

2️⃣ İLAÇ:
   💊 ADRENALİN 1 mg IV/IO her 3-5 dk
   ⚠️ Atropin ARTIK ÖNERİLMİYOR!
   ⚠️ Defibrilasyon YAPILMAZ (şoklanamaz ritim)

3️⃣ 5H-5T MUTLAKA ARAŞTIR:
   H'ler:
   • Hipovolemi → IV sıvı
   • Hipoksi → Oksijen, entübasyon
   • H+ (Asidoz) → Ventilasyon, NaHCO3
   • Hipo/Hiperkalemi → K+/Ca++, insülin
   • Hipotermi → Aktif ısıtma
   
   T'ler:
   • Toksin → Antidot
   • Tamponad → Perikardiyosentez
   • Tension pnömotoraks → İğne dekompresyon
   • Tromboz (Koroner) → Reperfüzyon
   • Tromboz (Pulmoner) → Fibrinolitik

4️⃣ İLERİ HAVAYOLU:
   • Endotrakeal tüp veya supraglottik
   • ETCO2 monitörizasyonu (>10 mmHg iyi CPR)

5️⃣ ROSC OLURSA:
   • Post-kardiyak arrest bakımı
   • Hedefli sıcaklık yönetimi
   • Neden araştırması (özellikle EKG)"""
    },
    "NORMAL": {
        "aciliyeti": "✅ Normal Sinüs Ritmi",
        "algoritma": """📋 NORMAL EKG BULGULARI

✓ EKG normal sınırlarda görünmektedir.

DEĞERLENDİRME KRİTERLERİ:
• Kalp hızı: 60-100/dk
• P dalgası: Her QRS'ten önce, uniform
• PR aralığı: 120-200 ms (0.12-0.20 sn)
• QRS süresi: <120 ms (0.12 sn)
• QT aralığı: Düzeltilmiş (QTc) <440 ms (erkek), <460 ms (kadın)
• ST segmenti: İzoelektrik hatta

📌 ÖNERİLER:
• Klinik korelasyon önemli
• Semptom varsa ileri tetkik (Troponin, EKO)
• Rutin sağlık kontrolü
• Kardiyovasküler risk faktörleri değerlendirmesi

⚠️ TEK BİR EKG YETERLİ OLMAYABİLİR!
• Aralıklı ritm bozuklukları için Holter
• Efor testi (koroner risk için)
• Ekokardiyografi (yapısal değerlendirme)"""
    },
    "GENEL": {
        "aciliyeti": "ℹ️ Genel Yaklaşım",
        "algoritma": """📋 GENEL EKG DEĞERLENDİRME PROTOKOLÜ

1️⃣ SİSTEMATİK YAKLAŞIM:
   • Hız (60-100 normal)
   • Ritim (düzenli/düzensiz)
   • Aks (normal: -30 ile +90)
   • P dalgası morfolojisi
   • PR aralığı
   • QRS süresi ve morfolojisi
   • ST segmenti
   • T dalgası
   • QT aralığı

2️⃣ İLK MÜDAHALE:
   • ABC değerlendirmesi
   • Vital bulgular
   • IV yol açılması
   • O2 (SpO2 <94 ise)
   • Monitörizasyon

3️⃣ TETKİKLER:
   • Kardiyak enzimler (Troponin)
   • Elektrolit paneli
   • Tam kan sayımı
   • Koagülasyon
   • Böbrek fonksiyon testleri

4️⃣ KONSÜLTASYON:
   • Kardiyoloji (ritim bozuklukları)
   • Kardiyovasküler cerrahi (gerekirse)

⚠️ Her zaman klinik korelasyon önemlidir!"""
    }
}

def tedavi_algoritmasi_bul(tahmin_metni):
    """AI tahminine göre uygun algoritmayı bulur"""
    tahmin_upper = tahmin_metni.upper()
    
    if "STEMI" in tahmin_upper or ("ST" in tahmin_upper and ("ELEVASYON" in tahmin_upper or "YÜKSELME" in tahmin_upper)):
        return TEDAVI_ALGORITMALARI["STEMI"]
    elif "NSTEMI" in tahmin_upper or "USAP" in tahmin_upper or "UNSTABLE ANGINA" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["NSTEMI"]
    elif "FİBRİL" in tahmin_upper or "FIBRIL" in tahmin_upper or "AF" in tahmin_upper.split() or "ATRİYAL" in tahmin_upper:
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
    <meta name="theme-color" content="#1e3c72">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { background: white; padding: 30px; border-radius: 12px 12px 0 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-bottom: 4px solid #e74c3c; }
        .header h1 { color: #2c3e50; margin-bottom: 8px; font-size: 28px; }
        .header p { color: #7f8c8d; font-size: 14px; }
        .badge { background: #4285f4; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; vertical-align: middle; }
        .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 4px; font-size: 13px; color: #856404; }
        .main-content { background: white; padding: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .upload-zone { border: 2px dashed #3498db; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; background: #f8f9fa; }
        .upload-zone:hover { border-color: #2980b9; background: #e9ecef; }
        input[type="file"] { display: none; }
        .button-group { display: flex; gap: 10px; margin-top: 20px; justify-content: center; }
        button { padding: 12px 24px; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-analyze { background: #27ae60; color: white; }
        .btn-analyze:hover { background: #229954; }
        .btn-analyze:disabled { background: #95a5a6; cursor: not-allowed; }
        .btn-reset { background: #95a5a6; color: white; }
        .preview-img { max-width: 100%; max-height: 300px; border-radius: 6px; margin: 15px 0; display: none; }
        .result-section { margin-top: 30px; display: none; }
        .result-box { background: #f8f9fa; padding: 25px; border-radius: 8px; border-left: 4px solid #4285f4; margin-bottom: 15px; }
        .result-box.urgent { border-left-color: #e74c3c; background: #fdf2f2; }
        .result-box.safe { border-left-color: #27ae60; background: #f0fdf4; }
        .algorithm-box { background: #eaf4ff; padding: 25px; border-radius: 8px; border-left: 4px solid #1a73e8; margin-top: 15px; }
        .algorithm-title { font-size: 16px; font-weight: 700; color: #1a73e8; margin-bottom: 10px; }
        .algorithm-urgency { display: inline-block; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-bottom: 12px; background: white; }
        .algorithm-content { font-size: 13px; color: #2c3e50; line-height: 1.7; white-space: pre-wrap; font-family: 'Segoe UI', monospace; }
        .result-label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: bold; }
        .result-value { font-size: 22px; font-weight: 700; color: #2c3e50; margin-bottom: 15px; }
        .analysis-text { font-size: 14px; color: #34495e; line-height: 1.8; white-space: pre-wrap; margin-top: 10px; }
        .error-message { background: #fdf2f2; border-left: 4px solid #e74c3c; color: #c0392b; padding: 15px; border-radius: 4px; margin-top: 15px; }
        .footer { background: white; padding: 20px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-size: 12px; color: #7f8c8d; text-align: center; }
        .loading { text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #4285f4; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 EKG Analiz Asistanı <span class="badge">AI DESTEKLİ</span></h1>
            <p>Kardiyoloji Uzmanı Seviyesinde EKG Analizi + Sağlık Bakanlığı Tedavi Algoritmaları</p>
        </div>
        <div class="warning-box">
            ⚠️ <strong>Klinik Uyarı:</strong> Bu sistem eğitim ve karar destek amaçlıdır. Gösterilen algoritmalar T.C. Sağlık Bakanlığı, ESC 2023 ve ACLS 2020 protokollerine dayanır. Kesin tanı ve tedavi mutlaka hekim tarafından yapılmalıdır.
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
        <div class="footer"><p>v5.0 | EKG Analiz Sistemi | ESC 2023 & ACLS 2020 Protokolleri</p></div>
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
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>Yapay zeka EKG\'nizi detaylı inceliyor, lütfen bekleyin (15-30 saniye)...</p></div>';
            
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
                            <div class="result-label">🎯 EKG Tanısı</div>
                            <div class="result-value">${pred.tahmin}</div>
                            
                            <div class="result-label">📊 Risk Seviyesi</div>
                            <div style="font-size: 16px; font-weight: 600; margin-bottom: 15px; color: ${urgent ? '#e74c3c' : '#27ae60'};">
                                ${pred.risk_seviyesi}
                            </div>
                            
                            <div class="result-label">📝 Detaylı EKG Analizi</div>
                            <div class="analysis-text">${pred.detay}</div>
                        </div>
                        
                        <div class="algorithm-box">
                            <div class="algorithm-title">🏥 KLİNİK YAKLAŞIM & TEDAVİ ALGORİTMASI</div>
                            <div class="algorithm-urgency">${algo.aciliyeti || 'Genel Yaklaşım'}</div>
                            <div class="algorithm-content">${algo.algoritma || 'Algoritma yüklenemedi.'}</div>
                            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #b8d4f0; font-size: 12px; color: #555;">
                                📚 <strong>Kaynak:</strong> T.C. Sağlık Bakanlığı Protokolleri, ESC 2023 Kılavuzu, ACLS 2020 Guidelines<br>
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
# GELİŞMİŞ EKG ANALİZ PROMPTU (Kardiyoloji Uzmanı Seviyesi)
# ==============================================================================
DETAYLI_PROMPT = """Sen 25 yıl deneyimli, sertifikalı bir KARDİYOLOJİ UZMANI ve ACİL TIP UZMANISIN. Türkiye'nin en büyük üniversite hastanesinde çalışıyorsun.

Sana gösterdiğim EKG fotoğrafını KARDİYOLOJİ UZMANI SEVİYESİNDE, ÇOK DETAYLI ve DİKKATLİ şekilde analiz et.

🔍 SİSTEMATİK EKG DEĞERLENDİRME (Mutlaka sırayla yap):

1️⃣ TEKNİK KALİTE:
   - EKG kalibrasyonu (10 mm/mV, 25 mm/s)
   - Kağıt hızı
   - Artifakt varlığı
   - Derivasyon yerleşimi

2️⃣ KALP HIZI:
   - Ventriküler hız (bpm)
   - Atriyal hız (bpm)
   - RR ve PP aralıklarının düzeni

3️⃣ RİTİM ANALİZİ:
   - Sinüs mü, sinüs dışı mı?
   - Düzenli mi, düzensiz mi?
   - Ektopi (VES/APS) var mı?

4️⃣ P DALGASI:
   - Varlığı ve morfolojisi
   - Süresi (<120 ms normal)
   - Amplitüdü (D2'de <2.5 mm)
   - P mitrale, P pulmonale bulgusu

5️⃣ PR ARALIĞI:
   - Süresi (120-200 ms normal)
   - AV blok değerlendirmesi

6️⃣ QRS KOMPLEKSİ:
   - Süresi (<120 ms normal)
   - Morfolojisi (RBBB, LBBB, LAFB, LPFB)
   - Aksı (-30° ile +90° normal)
   - Q dalgaları (patolojik mi?)
   - R dalgası progresyonu

7️⃣ ST SEGMENTİ (KRİTİK!):
   - Elevasyon (>1 mm ekstremite, >2 mm göğüs)
   - Depresyon (yatay/downsloping)
   - Hangi derivasyonlarda?
   - Anatomik lokalizasyon:
     * Anterior: V1-V4 (LAD)
     * Inferior: II, III, aVF (RCA)
     * Lateral: I, aVL, V5-V6 (Cx)
     * Posterior: V7-V9 (RCA/Cx)

8️⃣ T DALGASI:
   - Yön (pozitif/negatif)
   - Amplitüd (Hiperakut? Ters?)
   - Simetri

9️⃣ QT ARALIĞI:
   - QTc (Bazett): erkek <440 ms, kadın <460 ms
   - Uzun QT? Kısa QT?

🔟 ÖZEL DURUMLAR:
   - WPW sendromu (delta dalgası)
   - Brugada patern
   - Erken repolarizasyon
   - Perikardit
   - Pulmoner emboli (S1Q3T3)
   - Hiperkalemi (yüksek T)
   - Hipokalemi (U dalgası)

⚠️ ÇOK ÖNEMLİ TALIMATLAR:
- HER BULGUYU spesifik derivasyonla belirt (örn: "V2-V4'te ST elevasyonu")
- Ölçümleri MUTLAKA milisaniye/mm cinsinden ver
- Anatomik lokalizasyon yap
- Ayırıcı tanı düşün
- Şüpheli durumları belirt

Sadece aşağıdaki JSON formatında cevap ver, başka hiçbir şey yazma:

{
  "tahmin": "Ana tanı (spesifik ve net, örn: 'Anterior STEMI (V1-V4)', 'Atriyal Fibrilasyon hızlı ventriküler yanıtla', 'Normal Sinüs Ritmi', 'İnferior MI şüphesi', '3. Derece AV Blok', vb.)",
  "risk_seviyesi": "Düşük veya Orta veya Yüksek",
  "acil_mudahale": true veya false,
  "detay": "ÇOK DETAYLI Türkçe analiz yaz (en az 8-10 cümle). Şunları MUTLAKA belirt: 1) Kalp hızı (bpm), 2) Ritim türü ve düzeni, 3) P dalgası, 4) PR aralığı (ms), 5) QRS süresi ve morfolojisi (ms), 6) QRS aksı, 7) ST segmenti (hangi derivasyonlarda, kaç mm), 8) T dalgası değişiklikleri, 9) QTc, 10) Anatomik lokalizasyon (hangi koroner arter alanı), 11) Ayırıcı tanılar, 12) Ek tetkik önerileri. Kardiyoloji raporu gibi profesyonel yaz.",
  "uyari": "KLİNİK ONAY ZORUNLUDUR. Bu bir AI analizidir, kesin tanı için kardiyolog değerlendirmesi ve klinik korelasyon gereklidir."
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
        """Birincil AI ile analiz"""
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
        """Yedek AI ile analiz"""
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
        """Çoklu AI ile analiz + Tedavi algoritması"""
        errors = []
        result = None
        
        # 1. Önce birincili dene
        if self.gemini_model:
            try:
                print("🔵 Birincil AI ile detaylı analiz yapılıyor...")
                result = self.analyze_with_gemini(image_bytes)
                print("✓ Birincil AI başarılı")
            except Exception as e:
                errors.append(str(e))
                print(f"⚠ Birincil AI başarısız: {e}")
        
        # 2. Yedek AI'yi dene
        if not result and GROQ_API_KEY:
            try:
                print("🟢 Yedek AI ile deneniyor...")
                result = self.analyze_with_groq(image_bytes)
                print("✓ Yedek AI başarılı")
            except Exception as e:
                errors.append(str(e))
                print(f"⚠ Yedek AI başarısız: {e}")
        
        # 3. Sonuç kontrolü
        if not result:
            return {
                "tahmin": "Analiz Yapılamadı",
                "risk_seviyesi": "Bilinmiyor",
                "acil_mudahale": False,
                "detay": "Sistem şu anda yoğun. Lütfen birkaç dakika sonra tekrar deneyin.",
                "uyari": "Sistem geçici olarak kullanılamıyor.",
                "algoritma": TEDAVI_ALGORITMALARI["GENEL"]
            }
        
        # 4. Tedavi algoritmasını ekle
        tahmin = result.get("tahmin", "")
        algoritma = tedavi_algoritmasi_bul(tahmin)
        result["algoritma"] = algoritma
        
        print(f"📋 Algoritma seçildi: {algoritma['aciliyeti']}")
        
        return result

# ==============================================================================
# FASTAPI SUNUCUSU
# ==============================================================================
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EKG AI", version="5.0.0")
analyzer = MultiAIAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "EKG AI Asistanı",
        "short_name": "EKG AI",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1e3c72",
        "theme_color": "#1e3c72",
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
        
        print(f"📸 Fotoğraf alındı: {file.filename} ({len(contents)} bytes)")
        print("🧠 Detaylı analiz başlıyor...")
        
        prediction = analyzer.analyze_ecg_image(contents)
        
        print(f"✓ Analiz tamamlandı: {prediction['tahmin']}")
        
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
    print("🚀 EKG AI Analiz Sistemi Başlatılıyor v5.0")
    print("="*60)
    print(f"📍 Yerel adres: http://localhost:8000")
    print(f"🌐 Ağdaki adres: http://0.0.0.0:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
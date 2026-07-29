import os
import sys
import json
import argparse
import base64
import logging
import io
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
# PARAMEDİK YETKİLİ TEDAVİ ALGORİTMALARI
# ==============================================================================
TEDAVI_ALGORITMALARI = {
    "ANTERIOR_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Anterior STEMI (Ön Duvar Kalp Krizi)",
        "algoritma": """📌 TANI: ANTERIOR STEMI
   (Ön Duvar Kalp Krizi)

🩸 TIKALI DAMAR: LAD
   (Sol Ön İnen Kalp Damarı)

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
   (Paramedik Yetkisi)
━━━━━━━━━━━━━━━━━━━━

1️⃣ İLK DEĞERLENDİRME (5 dk):
✓ ABC kontrolü
✓ Vital bulgular (TA, nabız, SpO2)
✓ 12 derivasyon EKG çek
✓ Hastayı yarı oturur pozisyona al

2️⃣ İZLEM:
✓ Sürekli monitörizasyon
✓ Defibrilatör pedlerini takıp bekle
✓ 2 IV yol aç (18G tercih)
✓ Serum fizyolojik başla (KVO)

3️⃣ PARAMEDİK YETKİSİYLE VEREBİLECEĞİN:
💊 ASPİRİN 300 mg (Aspirin)
   → Hastaya çiğnettir
   → Yutmasın!
   ⛔ Alerji, aktif kanama varsa VERME

💊 OKSİJEN
   → SpO2 %90'ın altında ise ver
   → 2-4 L/dk nazal kanül
   ⚠️ Rutin O2 ARTIK ÖNERİLMİYOR

💊 NİTROGLİSERİN (Doktor orderı ile)
   → 0.4 mg dil altı
   ⛔ VERME EĞER:
      - Tansiyon 90'ın altında
      - Nabız 50'nin altında
      - Viagra kullanmışsa (24 saat)

4️⃣ 112 KOMUTA MERKEZİYLE İLETİŞİM:
📞 "STEMI şüphesi var"
📞 Vital bulguları söyle
📞 EKG'yi WhatsApp/faks ile gönder
📞 Doktor onayı ile ek ilaç isteyebilirsin

5️⃣ HASTANE TRANSFERİ:
🚑 ANJİYO YAPAN hastaneye götür
🚑 Kırmızı kod, ışık + siren
🚑 Yolda hastaneyi ARA
📞 "STEMI aktivasyonu, X dakika sonra varış"

━━━━━━━━━━━━━━━━━━━━
📚 HASTANEDE YAPILACAKLAR
   (Bilgi Amaçlı)
━━━━━━━━━━━━━━━━━━━━

• Anjiyo (Damar açma işlemi)
• Clopidogrel/Ticagrelor yükleme
• Enoksaparin (kan sulandırıcı)
• Statin (kolesterol ilacı)
• Beta-bloker

⚠️ YOLDA DİKKAT:
• Aritmi (VT/VF) olabilir
• Defibrilatör HAZIR olsun!
• Bilinç kaybı → CPR başla"""
    },
    "INFERIOR_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Inferior STEMI (Alt Duvar Kalp Krizi)",
        "algoritma": """📌 TANI: INFERIOR STEMI
   (Alt Duvar Kalp Krizi)

🩸 TIKALI DAMAR: RCA
   (Sağ Kalp Damarı)

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
   (Paramedik Yetkisi)
━━━━━━━━━━━━━━━━━━━━

⚠️ ÖZEL DİKKAT!
Bu MI'da sağ kalp de etkilenmiş olabilir.
Bu yüzden bazı ilaçlar TEHLİKELİ!

1️⃣ İLK DEĞERLENDİRME:
✓ ABC kontrolü
✓ Vital bulgular
✓ 12 derivasyon EKG
✓ TANSİYONA ÇOK DİKKAT ET!

⚠️ TA DÜŞÜKSE (90 altı):
Sağ kalp krizi şüphesi yüksek!
Nitrogliserin ASLA VERME!

2️⃣ İZLEM:
✓ Sürekli monitörizasyon
✓ Defibrilatör hazır
✓ 2 IV yol aç
✓ Serum fizyolojik hazır

3️⃣ PARAMEDİK YETKİSİYLE VEREBİLECEĞİN:

💊 ASPİRİN 300 mg
   → Çiğnettir
   ⛔ Alerji/kanama varsa VERME

💊 OKSİJEN
   → SpO2 <90 ise

💊 SERUM FİZYOLOJİK (Önemli!)
   → TA düşükse: 250 ml BOLUS
   → Cevap yoksa tekrar
   → Sağ kalp krizi için sıvı ŞART

💊 NİTROGLİSERİN
   ⛔ ÇOK DİKKAT!
   ⛔ VERME EĞER:
      - TA 90'ın altında
      - Sağ kalp krizi şüphesi var
      - Nabız 50'nin altında
   ✅ VER EĞER:
      - TA 100 üstü
      - Bilinç açık
      - Doktor onayladı

⚠️ NABIZ YAVAŞLARSA (Bradikardi):
💊 ATROPİN 0.5-1 mg IV
   → Doktor orderı ile
   → Etkisiz ise pacing hazırla

4️⃣ 112 KOMUTA MERKEZİ:
📞 "İnferior STEMI şüphesi"
📞 "TA: X/Y, Nabız: Z"
📞 "Sağ V MI için V4R çekildi/çekilemedi"
📞 EKG'yi paylaş

5️⃣ HASTANE TRANSFERİ:
🚑 ANJİYO YAPAN hastaneye
🚑 Kırmızı kod
🚑 Hastayı YATAY tut
🚑 Bacaklar hafif yukarı (TA düşükse)

━━━━━━━━━━━━━━━━━━━━
📚 HASTANEDE YAPILACAKLAR
━━━━━━━━━━━━━━━━━━━━

• V4R kontrolü (sağ kalp EKG)
• Anjiyo
• İlaç tedavisi

⚠️ YOLDA DİKKAT:
• AV blok gelişebilir
• Bradikardi olabilir
• Hipotansiyon olabilir → Sıvı ver"""
    },
    "LATERAL_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Lateral STEMI (Yan Duvar Kalp Krizi)",
        "algoritma": """📌 TANI: LATERAL STEMI
   (Yan Duvar Kalp Krizi)

🩸 TIKALI DAMAR: Cx
   (Sirkumfleks Kalp Damarı)

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

1️⃣ İLK DEĞERLENDİRME:
✓ ABC kontrolü
✓ Vital bulgular
✓ 12 derivasyon EKG
✓ Yarı oturur pozisyon

2️⃣ İZLEM:
✓ Sürekli monitör
✓ Defibrilatör hazır
✓ 2 IV yol aç

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 ASPİRİN 300 mg
   → Çiğnettir
   ⛔ Alerji/kanama varsa VERME

💊 OKSİJEN
   → SpO2 <90 ise

💊 NİTROGLİSERİN (Doktor onayı)
   → 0.4 mg dil altı
   ⛔ TA<90, nabız<50 ise VERME

4️⃣ 112 KOMUTA MERKEZİ:
📞 "Lateral STEMI şüphesi"
📞 EKG paylaş
📞 Vital bildir

5️⃣ HASTANE TRANSFERİ:
🚑 ANJİYO merkezine ACİL
🚑 Kırmızı kod

━━━━━━━━━━━━━━━━━━━━
📚 HASTANEDE
━━━━━━━━━━━━━━━━━━━━

• Anjiyo
• İlaç tedavisi

⚠️ DİKKAT:
Genelde yaygın MI'ın parçası
Aritmi riski var"""
    },
    "POSTERIOR_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Posterior STEMI (Arka Duvar Kalp Krizi)",
        "algoritma": """📌 TANI: POSTERIOR STEMI
   (Arka Duvar Kalp Krizi)

🩸 TIKALI DAMAR: RCA/Cx

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⚠️ Bu MI kolayca kaçırılır!
EKG'de klasik ST elevasyonu olmayabilir

🔍 EKG'DE ARA:
• V1-V3'te ST çizgisi AŞAĞI
• V1-V3'te R dalgası büyük
• (Ayna görüntüsü)

1️⃣ İLK DEĞERLENDİRME:
✓ ABC kontrolü
✓ Vital bulgular
✓ 12 derivasyon EKG
✓ İnferior MI kontrolü de yap

2️⃣ İZLEM:
✓ Sürekli monitör
✓ Defibrilatör hazır
✓ 2 IV yol aç

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 ASPİRİN 300 mg
   → Çiğnettir

💊 OKSİJEN
   → SpO2 <90 ise

💊 NİTROGLİSERİN (Doktor onayı)
   ⛔ TA<90, nabız<50 ise VERME

4️⃣ 112 KOMUTA MERKEZİ:
📞 "Posterior STEMI şüphesi"
📞 EKG paylaş

5️⃣ HASTANE TRANSFERİ:
🚑 ANJİYO merkezine ACİL

━━━━━━━━━━━━━━━━━━━━
📚 HASTANEDE
━━━━━━━━━━━━━━━━━━━━

• V7-V9 arka derivasyon EKG
• Anjiyo
• İlaç tedavisi"""
    },
    "SAG_V_MI": {
        "aciliyeti": "🚨 KIRMIZI KOD - Sağ Ventrikül MI (ÖZEL DURUM!)",
        "algoritma": """📌 TANI: SAĞ VENTRİKÜL MI
   (Sağ Kalp Krizi)

🩸 TIKALI DAMAR: RCA Proximal
   (Sağ Kalp Damarı Ana Kolu)

━━━━━━━━━━━━━━━━━━━━
⚠️ ÖZEL DURUM! ÇOK DİKKAT!
━━━━━━━━━━━━━━━━━━━━

⛔ ASLA VERİLMEMESİ GEREKENLER:
• NİTROGLİSERİN (ÖLDÜRÜCÜ OLABİLİR!)
• Morfin (dikkatli)
• Diüretik (Furosemid)

✅ ÖNCELİK: SIVI VER!

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

1️⃣ TANI KONTROLÜ:
✓ 12 derivasyon EKG
✓ ⚠️ V4R MUTLAKA ÇEK!
   (Sağ göğüs derivasyonu)
✓ Boyun venöz distansiyon var mı?
✓ Akciğerler temiz mi?

2️⃣ İZLEM:
✓ TANSİYON ÇOK ÖNEMLİ!
✓ Her 5 dakikada ölç
✓ Sürekli monitör
✓ Defibrilatör hazır

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💧 SERUM FİZYOLOJİK (ÖNCELİK!)
   → 250-500 ml IV BOLUS
   → Hızlıca ver
   → Cevap yoksa tekrar 500 ml
   → Toplam 1-2 L verebilir
   → HEDEF: TA yükselmesi

💊 ASPİRİN 300 mg
   → Çiğnettir

💊 OKSİJEN
   → SpO2 <90 ise

⛔ NİTROGLİSERİN
   → KESİNLİKLE VERME!
   → Ölümcül olabilir!

4️⃣ 112 KOMUTA MERKEZİ:
📞 "Sağ Ventrikül MI şüphesi"
📞 "TA düşük" bildir
📞 "Sıvı verildi, cevap...."

5️⃣ HASTANE TRANSFERİ:
🚑 ANJİYO merkezine ACİL
🚑 Hastayı YATAY tut
🚑 Bacaklar YUKARI kaldır
🚑 Sıvı desteği devam

⚠️ YOLDA DİKKAT:
• TA sürekli düşük olabilir
• Nabız yavaşlayabilir
• Bilinç kaybı olabilir
• CPR hazırlığı yap"""
    },
    "YAYGIN_ANTERIOR_MI": {
        "aciliyeti": "🚨🚨 SÜPER ACİL - Yaygın Anterior MI",
        "algoritma": """📌 TANI: YAYGIN ANTERIOR MI
   (Yaygın Ön Duvar Kalp Krizi)

🩸 TIKALI DAMAR: LAD PROXIMAL
   (Ana LAD damarı)

━━━━━━━━━━━━━━━━━━━━
⚠️ EN CIDDI KALP KRİZİ!
   Ölüm riski YÜKSEK
━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

1️⃣ İLK DEĞERLENDİRME (3 dk):
✓ ABC kontrolü
✓ Vital bulgular
✓ 12 derivasyon EKG
✓ Yarı oturur pozisyon

2️⃣ İZLEM (ÇOK ÖNEMLİ):
✓ Sürekli monitör
✓ Defibrilatör pedlerini TAK
✓ 2 büyük çaplı IV yol
✓ Havayolu ekipmanı hazır

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 ASPİRİN 300 mg
   → Çiğnettir
   ⛔ Alerji/kanama varsa VERME

💊 OKSİJEN
   → SpO2 <90 ise
   → Gerekirse maske

💊 NİTROGLİSERİN (Doktor onayı)
   → 0.4 mg dil altı
   ⛔ TA<90 ise VERME

4️⃣ 112 KOMUTA MERKEZİ:
📞 "SÜPER ACİL - Yaygın Anterior MI"
📞 "Kardiyojenik şok riski var"
📞 EKG paylaş
📞 En yakın anjiyo merkezini iste

5️⃣ HASTANE TRANSFERİ:
🚑 EN YAKIN ANJİYO merkezine
🚑 Kırmızı kod, ışık+siren
🚑 SÜPER HIZLI TRANSPORT
📞 Yolda hastaneyi ARA

━━━━━━━━━━━━━━━━━━━━
⚠️ YOLDA HAZIRLIK
━━━━━━━━━━━━━━━━━━━━

• VF/VT çok olası
• DEFİBRİLATÖR HAZIR!
• Kardiyak arrest hazırlığı
• Havayolu ekipmanı hazır
• Adrenalin hazır

⚠️ EĞER KARDİYAK ARREST OLURSA:
1. Hemen CPR başla
2. Defibrilasyon (VF ise)
3. Adrenalin 1 mg IV
4. Devam et"""
    },
    "NSTEMI": {
        "aciliyeti": "⚠️ SARI KOD - NSTEMI (Hafif Kalp Krizi)",
        "algoritma": """📌 TANI: NSTEMI / USAP
   (ST Yükselmesiz Kalp Krizi)

🔍 EKG: ST çizgisi AŞAĞI
   veya T dalgası TERS

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⚠️ NSTEMI de KALP KRİZİDİR!
Ciddiye al, hızlı transport yap

1️⃣ İLK DEĞERLENDİRME:
✓ ABC kontrolü
✓ Vital bulgular
✓ 12 derivasyon EKG
✓ Seri EKG (5 dakikada bir)
✓ STEMI'ye dönüşebilir!

2️⃣ İZLEM:
✓ Sürekli monitör
✓ Defibrilatör hazır
✓ IV yol aç

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 ASPİRİN 300 mg
   → Çiğnettir

💊 OKSİJEN
   → SpO2 <90 ise

💊 NİTROGLİSERİN (Doktor onayı)
   → 0.4 mg dil altı (ağrı için)
   ⛔ TA<90, nabız<50 ise VERME

4️⃣ 112 KOMUTA MERKEZİ:
📞 "NSTEMI/USAP şüphesi"
📞 EKG paylaş
📞 Vital bildir

5️⃣ HASTANE TRANSFERİ:
🚑 Kardiyoloji merkezine
🚑 Sarı kod (STEMI değilse)

━━━━━━━━━━━━━━━━━━━━
📚 HASTANEDE
━━━━━━━━━━━━━━━━━━━━

• Troponin testi
• Seri EKG
• Kan sulandırıcı ilaçlar
• Anjiyo değerlendirmesi

⚠️ YOLDA DİKKAT:
• STEMI'ye dönüşebilir
• Sık sık EKG kontrol et
• Ağrı artışında EKG tekrarla"""
    },
    "AF": {
        "aciliyeti": "⚠️ AF - Atriyal Fibrilasyon",
        "algoritma": """📌 TANI: AF
   (Atriyal Fibrilasyon)
   (Düzensiz Kalp Ritmi)

🔍 EKG: Düzensiz RR
   P dalgası YOK

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⚠️ HEMODİNAMİK DEĞERLENDİRME KRİTİK!

1️⃣ HASTA STABİL Mİ?
✓ TA ölç
✓ Bilinç durumu
✓ Göğüs ağrısı var mı?
✓ Nefes darlığı var mı?

❌ ANSTABİL HASTA:
   (TA<90, bilinç bulanık,
    göğüs ağrısı, nefes darlığı)
   → HEMEN HASTANEYE!
   → Yolda defibrilatör HAZIR
   → Kardiyoversiyon HASTANEDE

✅ STABİL HASTA:
   → Sakin ol
   → Güven ver
   → Vital izle

2️⃣ İZLEM:
✓ Sürekli monitör
✓ Vital her 5 dk
✓ IV yol aç
✓ Defibrilatör hazır

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 OKSİJEN
   → SpO2 <94 ise

💊 SERUM FİZYOLOJİK
   → KVO hızında
   → TA düşükse bolus

⛔ Paramedik VEREMEZ:
• Metoprolol
• Diltiazem
• Amiodaron
• Digoksin
(Doktor orderı gerekli)

4️⃣ 112 KOMUTA MERKEZİ:
📞 "AF hastası"
📞 "Hemodinamik [stabil/anstabil]"
📞 EKG paylaş
📞 Doktor orderı iste

5️⃣ HASTANE TRANSFERİ:
🚑 ANSTABİL: ACİL kardiyoloji
🚑 STABİL: Rutin kardiyoloji

━━━━━━━━━━━━━━━━━━━━
⚠️ YOLDA DİKKAT
━━━━━━━━━━━━━━━━━━━━

• Ventriküler hız takibi
• Ani düşüş → ACİL müdahale
• Bilinç kaybı → CPR hazırlığı
• Nabız hızı >150 → dikkat"""
    },
    "VT": {
        "aciliyeti": "🚨 KIRMIZI KOD - VT (Ventriküler Taşikardi)",
        "algoritma": """📌 TANI: VT
   (Ventriküler Taşikardi)

🔍 EKG: Geniş QRS
   Hız >100/dk

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⚡ HEMEN NABIZ KONTROL ET!

❌ NABIZ YOK:
   → KARDİYAK ARREST!
   → CPR başla
   → DEFİBRİLASYON (150-200 J)
   → Adrenalin protokolü

✅ NABIZ VAR:

1️⃣ HEMODİNAMİK DEĞERLENDİRME:

❌ ANSTABİL:
   (TA<90, bilinç bulanık)
   → HEMEN HASTANEYE
   → Yolda defibrilatör HAZIR
   → Sedasyon HAZIR (midazolam)
   → Kardiyoversiyon HASTANEDE

✅ STABİL:
   → Hızlı ama sakin transport

2️⃣ İZLEM:
✓ Sürekli EKG monitör
✓ Nabız HER DAKIKA kontrol
✓ IV yol aç (2 tane)
✓ Defibrilatör pedleri TAK

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 OKSİJEN
   → SpO2 <94 ise

💊 SERUM FİZYOLOJİK
   → KVO hızında

⛔ AMİODARON: Doktor orderı gerekli
   (Bazı illerde protokol var)

4️⃣ 112 KOMUTA MERKEZİ:
📞 "VT hastası"
📞 "[Stabil/Anstabil]"
📞 EKG paylaş

5️⃣ HASTANE TRANSFERİ:
🚑 EN YAKIN hastane
🚑 Kardiyoloji tercih
🚑 Kırmızı kod, siren

━━━━━━━━━━━━━━━━━━━━
⚠️ YOLDA HAZIRLIK
━━━━━━━━━━━━━━━━━━━━

• Nabız kaybolursa: HEMEN ŞOK
• VF'ye dönüşebilir
• Kardiyak arrest hazırlığı"""
    },
    "VF": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - VF",
        "algoritma": """📌 TANI: VF
   (Ventriküler Fibrilasyon)
   (Kalp Durması)

⚡ HEMEN CPR + DEFİBRİLASYON!

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⏱️ ZAMAN = HAYAT!

1️⃣ İLK 10 SANİYE:
✓ Nabız yok, solunum yok mu?
✓ CPR başla
✓ Defibrilatörü aç

2️⃣ CPR:
✓ Hız: 100-120/dk
✓ Derinlik: 5-6 cm
✓ 30 kompresyon : 2 solunum
✓ Kesintileri MİNİMİZE et
✓ 2 dakikada bir yer değiştir

3️⃣ DEFİBRİLASYON:
⚡ Pedleri yerleştir
   • Ped 1: Sağ klavikula altı
   • Ped 2: Sol koltuk altı
⚡ Bifazik: 150-200 J
⚡ Monofazik: 360 J
⚡ "ÇEKİLİN!" bağır
⚡ ŞOK VER
⚡ Hemen 2 dk CPR

4️⃣ HAVAYOLU:
✓ Ambu maske + %100 O2
✓ Nazofarengeal airway
✓ Uygun ise LMA (Laringeal maske)

5️⃣ IV/IO YOL:
✓ IV açılamıyorsa: IO (tibia)
✓ Serum fizyolojik açık

6️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 ADRENALİN (Epinefrin)
   → 1 mg IV/IO
   → Her 3-5 dakikada TEKRAR
   → Şoklanan ritimde: 2. şok sonrası

⛔ AMİODARON: Doktor orderı gerekli
   (Bazı illerde protokol var)

7️⃣ CPR DÖNGÜSÜ:
   Şok → 2 dk CPR → Ritm →
   Şok → 2 dk CPR → Adrenalin →
   Şok → 2 dk CPR → Amiodaron

8️⃣ 112 KOMUTA MERKEZİ:
📞 "Kardiyak arrest - VF"
📞 "CPR devam ediyor"
📞 "X şok yapıldı"

━━━━━━━━━━━━━━━━━━━━
🔍 5H-5T ARA (BILGI)
━━━━━━━━━━━━━━━━━━━━

H'ler:
• Hipovolemi (kan azlığı)
• Hipoksi (O2 azlığı)
• H+ (asidoz)
• Hipo/HiperK (potasyum)
• Hipotermi

T'ler:
• Toksin
• Tamponad
• Tension pnö
• Tromboz (koroner)
• Tromboz (pulmoner)

━━━━━━━━━━━━━━━━━━━━
🚑 HASTANEYE TRANSPORT
━━━━━━━━━━━━━━━━━━━━

⚠️ CPR SIRASINDA TAŞI!
🚑 En yakın hastane
📞 "Kardiyak arrest, CPR devam"

⚡ ROSC OLURSA:
✓ Vital tam kontrol
✓ 12 derivasyon EKG
✓ Hastane bildirimi"""
    },
    "BRADIKARDI": {
        "aciliyeti": "⚠️ Semptomatik Bradikardi (Yavaş Nabız)",
        "algoritma": """📌 TANI: BRADİKARDİ
   (Yavaş Kalp Atışı)
   <50/dk

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

1️⃣ SEMPTOM DEĞERLENDİRMESİ:

❓ ŞU BULGULAR VAR MI?
• TA <90 mmHg
• Bilinç değişikliği
• Göğüs ağrısı
• Nefes darlığı
• Baş dönmesi/senkop
• Solukluk, terleme

✅ SEMPTOMSUZ: Sadece izle
⚠️ SEMPTOMLU: AKTİF müdahale

2️⃣ İZLEM:
✓ Sürekli monitör
✓ Vital her 5 dk
✓ IV yol aç
✓ Defibrilatör hazır

3️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 OKSİJEN
   → SpO2 <94 ise

💊 SERUM FİZYOLOJİK
   → TA düşükse: 250 ml bolus

💊 ATROPİN
   → 0.5-1 mg IV
   → Doktor orderı ile
   → 3-5 dk sonra tekrar
   → Maksimum: 3 mg
   
   ⛔ ATROPİN ETKİSİZ OLABİLİR:
   • Mobitz II blokta
   • Tam AV blokta

4️⃣ TRANSKÜTAN PACING (Varsa):
   → Atropin etkisizse
   → Hız: 60-80/dk
   → Akım: 40-80 mA
   → Hastayı UYAR/UYUT

5️⃣ 112 KOMUTA MERKEZİ:
📞 "Semptomatik bradikardi"
📞 "Kalp hızı: X"
📞 "TA: Y/Z"
📞 EKG paylaş

6️⃣ HASTANE TRANSFERİ:
🚑 Kardiyoloji merkezi
📞 Bildirim yap

━━━━━━━━━━━━━━━━━━━━
⚠️ YOLDA DİKKAT
━━━━━━━━━━━━━━━━━━━━

• Ani asistoli riski
• Pacing HAZIRDA olsun
• Vital her 3-5 dk

━━━━━━━━━━━━━━━━━━━━
🔍 SEBEBİ DÜŞÜN
━━━━━━━━━━━━━━━━━━━━

• İnferior MI olabilir
• İlaç yan etkisi
  (Beta-bloker, digoksin)
• Yüksek potasyum
• Hipoksemi"""
    },
    "SVT": {
        "aciliyeti": "⚠️ SVT (Hızlı Kalp Atışı)",
        "algoritma": """📌 TANI: SVT
   (Supraventriküler Taşikardi)

🔍 EKG: Dar QRS
   Hız 150-250/dk

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

1️⃣ HEMODİNAMİK DEĞERLENDİRME:

❌ ANSTABİL:
   (TA<90, bilinç bulanık,
    göğüs ağrısı, nefes darlığı)
   → HEMEN HASTANEYE!
   → Kardiyoversiyon hastanede

✅ STABİL:
   → Vagal manevralar dene

2️⃣ VAGAL MANEVRALAR (GÜVENLİ!):

🔹 VALSALVA MANEVRASI:
   • Hastayı yatır
   • 15 saniye ıkındır
   • MODİFİYE: Bacakları kaldır
     (%40 daha etkili)

🔹 YÜZE SOĞUK UYGULAMA:
   • Buz torbası yüze
   • 15-30 saniye
   • Özellikle çocuklarda etkili

🔹 ÖKSÜRTME:
   • Güçlü öksürsün

⛔ KAROTİS MASAJI:
   → Sadece doktor
   → Yaşlıda RİSKLİ (inme!)

3️⃣ İZLEM:
✓ Sürekli monitör
✓ Vital her 5 dk
✓ IV yol aç (18G)
✓ Antekübital tercih

4️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 OKSİJEN
   → SpO2 <94 ise

💊 ADENOZİN (Doktor orderı ile!)
   → 6 mg IV HIZLI PUSH
   → 20 ml NaCl ile flush
   → 1-2 dk sonra 12 mg
   → Gerekirse 12 mg daha
   
   ⚠️ HASTAYI UYAR:
   "Kısa süreli göğüste basınç"
   "Yüzde kızarma olabilir"
   "Bu NORMAL, geçici!"

5️⃣ 112 KOMUTA MERKEZİ:
📞 "SVT hastası"
📞 "Vagal [başarılı/başarısız]"
📞 EKG paylaş

6️⃣ HASTANE TRANSFERİ:
🚑 Kardiyoloji merkezi

━━━━━━━━━━━━━━━━━━━━
⚠️ YOLDA
━━━━━━━━━━━━━━━━━━━━

• Sürekli monitör
• SVT geçerse: EKG tekrar
• Ani bilinç kaybı → CPR"""
    },
    "AV_BLOK": {
        "aciliyeti": "⚠️ AV Blok",
        "algoritma": """📌 TANI: AV BLOK
   (Kalp Bloğu)

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

1️⃣ BLOK TİPİNİ TANI:

🔹 1° BLOK: PR uzun
   → Genelde tedavi yok
   → İzle, taşı

🔹 2° MOBİTZ I:
   (Wenckebach)
   → Genelde stabil
   → İzle, taşı

🔹 2° MOBİTZ II:
   ⚠️ Yüksek risk!
   → Pacing hazırla

🔹 3° TAM BLOK:
   🚨 ACİL!
   → P ve QRS bağımsız
   → Pacing HEMEN

2️⃣ SEMPTOM DEĞERLENDİRMESİ:
• TA <90 mmHg?
• Bilinç değişikliği?
• Göğüs ağrısı?

✅ SEMPTOMSUZ: İzle
⚠️ SEMPTOMLU: Aktif tedavi

3️⃣ İZLEM:
✓ Sürekli monitör
✓ Vital her 5 dk
✓ IV yol aç
✓ Defibrilatör hazır

4️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 OKSİJEN
   → SpO2 <94 ise

💊 ATROPİN (Doktor orderı)
   → 1 mg IV
   ⚠️ Mobitz II ve Tam blokta
      GENELDE ETKİSİZ!

⚡ TRANSKÜTAN PACING (Varsa):
   → Hemen hazırla
   → Hız: 60-80/dk
   → Sedasyon gerek

5️⃣ 112 KOMUTA MERKEZİ:
📞 "AV Blok"
📞 "Tip: [1°/2° Mobitz I-II/3°]"
📞 "Vital: TA/nabız"
📞 EKG paylaş

6️⃣ HASTANE TRANSFERİ:
🚑 Kardiyoloji + Pacemaker
🚑 Tam blok ise KIRMIZI KOD

━━━━━━━━━━━━━━━━━━━━
⚠️ YOLDA DİKKAT
━━━━━━━━━━━━━━━━━━━━

• Ani asistoli riski
• Pacing HAZIRDA
• CPR hazırlığı"""
    },
    "ASISTOLI": {
        "aciliyeti": "🚨🚨 KARDİYAK ARREST - Asistoli",
        "algoritma": """📌 TANI: ASİSTOLİ / PEA
   (Düz Çizgi Kalp Durması)

⚡ HEMEN CPR!

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⛔ DEFİBRİLASYON YAPILMAZ!
(Şoklanmayan ritim)

1️⃣ İLK 10 SANİYE:
✓ Bilinç, solunum, nabız kontrol
✓ CPR başla

2️⃣ CPR:
✓ Hız: 100-120/dk
✓ Derinlik: 5-6 cm
✓ 30:2 (kompresyon:solunum)
✓ Kesintileri minimize et

3️⃣ HAVAYOLU:
✓ Ambu maske + %100 O2
✓ Nazofarengeal airway
✓ LMA (varsa)

4️⃣ IV/IO YOL:
✓ IV açılamıyorsa: IO (tibia)
✓ Serum fizyolojik açık

5️⃣ PARAMEDİK VEREBİLECEĞİ:

💊 ADRENALİN (Epinefrin)
   → 1 mg IV/IO
   → Her 3-5 dakikada TEKRAR
   → HEMEN ver

⛔ ATROPİN ARTIK KULLANILMIYOR!
   (Kılavuzdan çıkarıldı)

6️⃣ 112 KOMUTA MERKEZİ:
📞 "Kardiyak arrest - Asistoli"
📞 "CPR devam ediyor"

━━━━━━━━━━━━━━━━━━━━
🔍 5H-5T MUTLAKA ARA
━━━━━━━━━━━━━━━━━━━━

H'ler:
• Hipovolemi → Sıvı ver
• Hipoksi → Oksijen
• H+ (asidoz) → İyi ventilasyon
• Hipo/HiperK → Bilgi paylaş
• Hipotermi → Isıt

T'ler:
• Toksin → Antidot (varsa)
• Tamponad → Hastanede
• Tension pnö → İğne dekompresyon
• Tromboz (koroner) → Anjiyo
• Tromboz (pulmoner) → Fibrinolitik

━━━━━━━━━━━━━━━━━━━━
🚑 HASTANEYE TRANSPORT
━━━━━━━━━━━━━━━━━━━━

⚠️ CPR SIRASINDA TAŞI!
🚑 En yakın hastane
📞 "Asistoli, CPR devam"

📋 CPR SÜRESİ:
En az 20-30 dakika
Doktor sonlandırma kararı verir

⚡ ROSC OLURSA:
✓ Vital tam kontrol
✓ 12 derivasyon EKG
✓ Hastane bildirimi
✓ %100 O2 → SpO2 94-98 hedef"""
    },
    "NORMAL": {
        "aciliyeti": "✅ Normal Sinüs Ritmi",
        "algoritma": """📌 TANI: NORMAL SİNÜS RİTMİ

━━━━━━━━━━━━━━━━━━━━
✅ NORMAL EKG BULGULARI
━━━━━━━━━━━━━━━━━━━━

• Hız: 60-100/dk
• Ritim: Düzenli sinüs
• P dalgası: Uniform
• PR: 120-200 ms
• QRS: <120 ms
• ST: İzoelektrik

━━━━━━━━━━━━━━━━━━━━
🚑 SEN NE YAPMALISIN?
━━━━━━━━━━━━━━━━━━━━

⚠️ ANCAK DİKKAT!
"Normal EKG" ≠ "Sorun yok"

1️⃣ KLİNİK DEĞERLENDİRME:
• Göğüs ağrısı VAR MI?
• Nefes darlığı?
• Baş dönmesi?
• Çarpıntı hissi?

2️⃣ SEMPTOMLU HASTA:
(EKG normal olsa bile)
✓ IV yol aç
✓ Vital takibi
✓ Aspirin (göğüs ağrısı varsa)
✓ Hastaneye transport

3️⃣ ASEMPTOMATIK HASTA:
✓ Vital kontrol
✓ Detaylı sorgulama
✓ Ambulans gerekli mi?
✓ Hasta reddiye seçeneği

4️⃣ 112 KOMUTA MERKEZİ:
📞 "EKG normal, semptom [var/yok]"
📞 Vital paylaş
📞 Karar iste

5️⃣ HASTANE TRANSFERİ:
🚑 Semptom varsa: Hastane
🚑 Yoksa: Doktor kararı

━━━━━━━━━━━━━━━━━━━━
⚠️ UNUTMA!
━━━━━━━━━━━━━━━━━━━━

• NSTEMI ilk saatte EKG normal olabilir
• Aralıklı ritim bozuklukları
• Anstabil angina
• Pulmoner emboli

📋 EK TETKİKLER (Hastanede):
• Troponin
• Kalp ultrasonu
• 24 saat EKG (Holter)
• D-dimer (PE şüphesi)"""
    },
    "GENEL": {
        "aciliyeti": "ℹ️ Genel Paramedik Yaklaşımı",
        "algoritma": """📌 GENEL EKG YAKLAŞIMI

━━━━━━━━━━━━━━━━━━━━
🚑 PARAMEDİK PROTOKOLÜ
━━━━━━━━━━━━━━━━━━━━

1️⃣ GÜVENLİK:
✓ Kendini koru!
✓ Olay yeri güvenliği
✓ Eldiven, maske

2️⃣ HIZLI DEĞERLENDİRME:
✓ Bilinç (AVPU)
✓ ABC
✓ Vital bulgular:
   - TA
   - Nabız
   - SpO2
   - Solunum
   - Kan şekeri

3️⃣ HİKAYE (SAMPLE):
S: Semptomlar
A: Alerjiler
M: İlaçlar
P: Geçmiş hastalıklar
L: Son yemek
E: Olayı anlat

4️⃣ EKG:
✓ 12 derivasyon
✓ Doğru yerleşim
✓ 25 mm/s hız
✓ 10 mm/mV kalibrasyon

5️⃣ PARAMEDİK VEREBİLECEĞİ:
💊 Aspirin (göğüs ağrısı)
💊 Oksijen (SpO2 <94)
💊 Nitrogliserin (doktor onayı)
💊 Serum fizyolojik
💊 Salbutamol (nefes darlığı)
💊 Adrenalin (arrest)
💊 Atropin (bradikardi, order)
💊 Glukoz (hipoglisemi)

6️⃣ 112 KOMUTA MERKEZİ:
📞 MIST formatı:
   M: Mekanizma
   I: İnjury/İllness
   S: Signs/Symptoms
   T: Treatment

7️⃣ HASTANE TRANSFERİ:
🚑 KIRMIZI KOD: STEMI, arrest, şok
⚠️ SARI KOD: NSTEMI, stabil aritmi
✅ MAVİ KOD: Stabil hasta"""
    }
}

def tedavi_algoritmasi_bul(tahmin_metni):
    tahmin_upper = tahmin_metni.upper()
    
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
        return TEDAVI_ALGORITMALARI["ANTERIOR_MI"]
    elif "NSTEMI" in tahmin_upper or "USAP" in tahmin_upper or "UNSTABLE ANGINA" in tahmin_upper:
        return TEDAVI_ALGORITMALARI["NSTEMI"]
    elif "FİBRİL" in tahmin_upper or "FIBRIL" in tahmin_upper or "ATRİYAL" in tahmin_upper:
        if "VENTRİKÜLER" in tahmin_upper or "VF" in tahmin_upper:
            return TEDAVI_ALGORITMALARI["VF"]
        return TEDAVI_ALGORITMALARI["AF"]
    elif "VENTRİKÜLER TAŞİKARDİ" in tahmin_upper or " VT " in " " + tahmin_upper + " " or tahmin_upper.strip() == "VT":
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
        
        .tabs { display: flex; gap: 5px; margin-bottom: 20px; border-bottom: 2px solid #eee; }
        .tab { padding: 12px 20px; cursor: pointer; background: #f8f9fa; border: none; border-radius: 6px 6px 0 0; font-size: 14px; font-weight: 600; color: #7f8c8d; transition: all 0.3s; }
        .tab.active { background: #dc3545; color: white; }
        .tab:hover:not(.active) { background: #e9ecef; color: #2c3e50; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .upload-zone { border: 2px dashed #dc3545; border-radius: 8px; padding: 40px; text-align: center; background: #fff5f5; }
        input[type="file"] { display: none; }
        
        .upload-options { display: flex; gap: 10px; margin-top: 15px; justify-content: center; flex-wrap: wrap; }
        .upload-btn { padding: 12px 22px; background: #dc3545; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
        .upload-btn:hover { background: #a71d2a; }
        .upload-btn.camera { background: #007bff; }
        .upload-btn.camera:hover { background: #0056b3; }
        
        .manual-select { padding: 15px; background: #fff5f5; border-radius: 8px; margin-bottom: 15px; }
        .manual-select label { display: block; margin-bottom: 8px; font-weight: 600; color: #2c3e50; font-size: 14px; }
        .manual-select select { width: 100%; padding: 12px; border: 2px solid #dc3545; border-radius: 6px; font-size: 15px; background: white; color: #2c3e50; cursor: pointer; }
        .manual-select select:focus { outline: none; border-color: #a71d2a; }
        
        .button-group { display: flex; gap: 10px; margin-top: 20px; justify-content: center; }
        button { padding: 12px 24px; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-analyze { background: #27ae60; color: white; }
        .btn-analyze:hover { background: #229954; }
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
        .algorithm-content { font-size: 15px; color: #2c3e50; line-height: 1.9; white-space: pre-wrap; font-family: 'Segoe UI', sans-serif; }
        .result-label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: bold; }
        .result-value { font-size: 22px; font-weight: 700; color: #2c3e50; margin-bottom: 15px; }
        .analysis-text { font-size: 14px; color: #34495e; line-height: 1.7; white-space: pre-wrap; margin-top: 10px; }
        .error-message { background: #fdf2f2; border-left: 4px solid #e74c3c; color: #c0392b; padding: 15px; border-radius: 4px; margin-top: 15px; }
        .footer { background: white; padding: 20px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-size: 12px; color: #7f8c8d; text-align: center; }
        .loading { text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #dc3545; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .info-note { background: #e7f3ff; border-left: 4px solid #007bff; padding: 12px 15px; border-radius: 4px; margin-bottom: 15px; font-size: 13px; color: #004085; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚑 Paramedik EKG Asistanı <span class="badge">v12</span></h1>
            <p>Paramedik Yetkisi Odaklı EKG Analiz + Öğrenme Sistemi</p>
        </div>
        <div class="warning-box">
            ⚠️ <strong>Klinik Uyarı:</strong> Bu sistem 112 paramedik yetkileri çerçevesinde hazırlanmıştır. Doktor orderı gereken ilaçlar mutlaka komuta merkezi onayı ile verilmelidir.
        </div>
        <div class="main-content">
            
            <div class="tabs">
                <button class="tab active" onclick="switchTab('photo')">📸 Fotoğraf Analizi</button>
                <button class="tab" onclick="switchTab('manual')">📚 Manuel Seçim (Öğren)</button>
            </div>
            
            <div id="photoTab" class="tab-content active">
                <div class="upload-zone">
                    <div style="font-size: 48px; margin-bottom: 10px;">📸</div>
                    <div style="font-size: 16px; font-weight: 500; color: #2c3e50;">EKG Fotoğrafını Yükleyin</div>
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">Galeriden seçin veya kamera ile çekin</div>
                    <div class="upload-options">
                        <button class="upload-btn" onclick="document.getElementById('ekgFileGallery').click();">
                            📁 Galeriden Seç
                        </button>
                        <button class="upload-btn camera" onclick="document.getElementById('ekgFileCamera').click();">
                            📷 Kamera ile Çek
                        </button>
                    </div>
                    <input type="file" id="ekgFileGallery" accept="image/*">
                    <input type="file" id="ekgFileCamera" accept="image/*" capture="environment">
                </div>
                <img id="previewImg" class="preview-img" src="" alt="Önizleme">
                <div class="button-group" id="buttonGroup" style="display: none;">
                    <button class="btn-analyze" id="analyzeBtn">🧠 EKG'yi Analiz Et</button>
                    <button class="btn-reset" id="resetBtn">↻ Temizle</button>
                </div>
            </div>
            
            <div id="manualTab" class="tab-content">
                <div class="info-note">
                    📚 <strong>Öğrenme Modu:</strong> Paramedik olarak sahada karşılaşabileceğin ritimleri seç ve müdahale protokolünü öğren.
                </div>
                <div class="manual-select">
                    <label for="ritmSelect">🎯 EKG Ritmi / Tanısı Seçin:</label>
                    <select id="ritmSelect">
                        <option value="">-- Bir ritim seçin --</option>
                        <optgroup label="🚨 Kalp Krizleri (MI/STEMI)">
                            <option value="ANTERIOR_MI">Anterior STEMI (Ön Duvar)</option>
                            <option value="INFERIOR_MI">Inferior STEMI (Alt Duvar)</option>
                            <option value="LATERAL_MI">Lateral STEMI (Yan Duvar)</option>
                            <option value="POSTERIOR_MI">Posterior STEMI (Arka Duvar)</option>
                            <option value="SAG_V_MI">Sağ Ventrikül MI</option>
                            <option value="YAYGIN_ANTERIOR_MI">Yaygın Anterior MI</option>
                            <option value="NSTEMI">NSTEMI / USAP</option>
                        </optgroup>
                        <optgroup label="⚠️ Aritmiler">
                            <option value="AF">AF (Atriyal Fibrilasyon)</option>
                            <option value="SVT">SVT (Supraventriküler Taşikardi)</option>
                            <option value="VT">VT (Ventriküler Taşikardi)</option>
                            <option value="VF">VF (Kardiyak Arrest)</option>
                            <option value="BRADIKARDI">Bradikardi</option>
                        </optgroup>
                        <optgroup label="🔌 Bloklar ve Diğer">
                            <option value="AV_BLOK">AV Blok</option>
                            <option value="ASISTOLI">Asistoli (Kardiyak Arrest)</option>
                            <option value="NORMAL">Normal Sinüs Ritmi</option>
                        </optgroup>
                    </select>
                </div>
                <div class="button-group">
                    <button class="btn-analyze" id="showManualBtn">📖 Paramedik Protokolünü Göster</button>
                </div>
            </div>
            
            <div class="result-section" id="resultSection">
                <div id="resultContent"></div>
            </div>
        </div>
        <div class="footer"><p>v12.0 | Paramedik EKG Asistanı | Prm. Ali GÜZEL tarafından hazırlanmıştır</p></div>
    </div>
    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
            document.getElementById('resultSection').style.display = 'none';
        }
        
        const ekgFileGallery = document.getElementById('ekgFileGallery');
        const ekgFileCamera = document.getElementById('ekgFileCamera');
        const previewImg = document.getElementById('previewImg');
        const buttonGroup = document.getElementById('buttonGroup');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resetBtn = document.getElementById('resetBtn');
        const resultSection = document.getElementById('resultSection');
        const resultContent = document.getElementById('resultContent');
        const ritmSelect = document.getElementById('ritmSelect');
        const showManualBtn = document.getElementById('showManualBtn');
        
        let selectedFile = null;
        
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (event) => {
                    previewImg.src = event.target.result;
                    previewImg.style.display = 'block';
                    buttonGroup.style.display = 'flex';
                    resultSection.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        }
        
        ekgFileGallery.addEventListener('change', handleFileSelect);
        ekgFileCamera.addEventListener('change', handleFileSelect);

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return alert('Lütfen bir dosya seçin!');
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '🧠 Analiz ediliyor...';
            resultSection.style.display = 'block';
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>EKG analiz ediliyor (15-30 saniye)...</p></div>';
            const formData = new FormData();
            formData.append('file', selectedFile);
            try {
                const response = await fetch('/api/analyze', { method: 'POST', body: formData });
                const result = await response.json();
                if (result.status === 'success') {
                    displayResult(result.prediction, false);
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
            ekgFileGallery.value = '';
            ekgFileCamera.value = '';
            selectedFile = null;
            previewImg.style.display = 'none';
            buttonGroup.style.display = 'none';
            resultSection.style.display = 'none';
        });
        
        showManualBtn.addEventListener('click', async () => {
            const selected = ritmSelect.value;
            if (!selected) return alert('Lütfen bir ritim seçin!');
            showManualBtn.disabled = true;
            showManualBtn.textContent = '📖 Yükleniyor...';
            resultSection.style.display = 'block';
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>Protokol yükleniyor...</p></div>';
            try {
                const response = await fetch('/api/manual/' + selected);
                const result = await response.json();
                if (result.status === 'success') {
                    displayResult(result.prediction, true);
                } else {
                    resultContent.innerHTML = `<div class="error-message"><strong>Hata:</strong> ${result.message}</div>`;
                }
            } catch (error) {
                resultContent.innerHTML = `<div class="error-message"><strong>Bağlantı Hatası:</strong> ${error.message}</div>`;
            } finally {
                showManualBtn.disabled = false;
                showManualBtn.textContent = '📖 Paramedik Protokolünü Göster';
            }
        });
        
        function displayResult(pred, isManual) {
            const urgent = pred.acil_mudahale;
            const algo = pred.algoritma || {};
            let html = `
                <div class="result-box ${urgent ? 'urgent' : 'safe'}">
                    <div class="result-label">${isManual ? '📚 SEÇİLEN TANI' : '🎯 EKG TANISI'}</div>
                    <div class="result-value">${pred.tahmin}</div>`;
            if (!isManual) {
                html += `
                    <div class="result-label">📊 ACİLİYET</div>
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 15px; color: ${urgent ? '#e74c3c' : '#27ae60'};">
                        ${pred.risk_seviyesi}
                    </div>
                    <div class="result-label">📝 EKG DEĞERLENDİRMESİ</div>
                    <div class="analysis-text">${pred.detay}</div>`;
            }
            html += `</div>
                <div class="algorithm-box">
                    <div class="algorithm-title">🚑 PARAMEDİK MÜDAHALE PROTOKOLÜ</div>
                    <div class="algorithm-urgency">${algo.aciliyeti || 'Genel Yaklaşım'}</div>
                    <div class="algorithm-content">${algo.algoritma || 'Protokol yüklenemedi.'}</div>
                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #f5c6cb; font-size: 12px; color: #555;">
                        📚 <strong>Kaynak:</strong> T.C. Sağlık Bakanlığı 112 Paramedik Protokolleri, ACLS 2020<br>
                        ⚠️ <strong>Uyarı:</strong> ${pred.uyari || 'Doktor orderı gereken ilaçlar için komuta merkezi ile iletişime geç.'}
                    </div>
                </div>
            `;
            resultContent.innerHTML = html;
        }
    </script>
</body>
</html>
"""

DETAYLI_PROMPT = """Sen deneyimli bir kardiyolog ve acil tıp uzmanısın.
Bu EKG fotoğrafını analiz et. MI türlerini SPESIFIK olarak belirt.

🔍 MI TÜRLERİ:
1. Anterior STEMI → V1-V4 → LAD
2. Yaygın Anterior MI → V1-V6, I, aVL → LAD proximal
3. Inferior STEMI → II, III, aVF → RCA
4. Lateral STEMI → I, aVL, V5-V6 → Cx
5. Posterior STEMI → V7-V9 → RCA/Cx
6. Sağ Ventrikül MI → V4R → RCA proximal

DİĞER: Normal, AF, VT, VF, SVT, Bradikardi, AV Blok, Asistoli

Sadece JSON:

{
  "tahmin": "SPESIFIK tanı",
  "risk_seviyesi": "Düşük/Orta/Yüksek",
  "acil_mudahale": true/false,
  "detay": "KISA değerlendirme (4-6 cümle)",
  "uyari": "Klinik onay zorunludur."
}

SADECE JSON döndür."""

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
        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        return json.loads(response_text.strip())

    def analyze_with_groq(self, image_bytes):
        if not GROQ_API_KEY:
            raise Exception("Yedek AI anahtarı yok")
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": DETAYLI_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}],
            "temperature": 0.2, "max_tokens": 1024
        }
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        if response.status_code != 200:
            raise Exception(f"API hatası: {response.status_code}")
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        if result_text.startswith("```json"): result_text = result_text[7:]
        if result_text.startswith("```"): result_text = result_text[3:]
        if result_text.endswith("```"): result_text = result_text[:-3]
        return json.loads(result_text.strip())

    def analyze_ecg_image(self, image_bytes):
        result = None
        if self.gemini_model:
            try:
                print("🔵 Birincil AI ile analiz...")
                result = self.analyze_with_gemini(image_bytes)
                print("✓ Birincil AI başarılı")
            except Exception as e:
                print(f"⚠ Birincil AI başarısız: {e}")
        if not result and GROQ_API_KEY:
            try:
                print("🟢 Yedek AI ile deneniyor...")
                result = self.analyze_with_groq(image_bytes)
                print("✓ Yedek AI başarılı")
            except Exception as e:
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

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Paramedik EKG", version="12.0.0")
analyzer = MultiAIAnalyzer()

MANUEL_ISIMLER = {
    "ANTERIOR_MI": "Anterior STEMI (Ön Duvar Kalp Krizi)",
    "INFERIOR_MI": "Inferior STEMI (Alt Duvar Kalp Krizi)",
    "LATERAL_MI": "Lateral STEMI (Yan Duvar Kalp Krizi)",
    "POSTERIOR_MI": "Posterior STEMI (Arka Duvar Kalp Krizi)",
    "SAG_V_MI": "Sağ Ventrikül MI",
    "YAYGIN_ANTERIOR_MI": "Yaygın Anterior MI",
    "NSTEMI": "NSTEMI / USAP",
    "AF": "AF (Atriyal Fibrilasyon)",
    "SVT": "SVT (Supraventriküler Taşikardi)",
    "VT": "VT (Ventriküler Taşikardi)",
    "VF": "VF (Ventriküler Fibrilasyon)",
    "BRADIKARDI": "Bradikardi",
    "AV_BLOK": "AV Blok",
    "ASISTOLI": "Asistoli",
    "NORMAL": "Normal Sinüs Ritmi"
}

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

@app.get("/api/manual/{ritm_kodu}")
async def get_manual_algorithm(ritm_kodu: str):
    try:
        if ritm_kodu not in TEDAVI_ALGORITMALARI:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Ritim bulunamadı"})
        algoritma = TEDAVI_ALGORITMALARI[ritm_kodu]
        tahmin_isim = MANUEL_ISIMLER.get(ritm_kodu, ritm_kodu)
        prediction = {
            "tahmin": tahmin_isim,
            "risk_seviyesi": "Manuel Seçim",
            "acil_mudahale": "ACİL" in algoritma["aciliyeti"] or "KIRMIZI" in algoritma["aciliyeti"] or "ARREST" in algoritma["aciliyeti"],
            "detay": "",
            "uyari": "Bu bir öğrenme modudur. Doktor orderı gereken ilaçlar için komuta merkezi ile iletişime geç.",
            "algoritma": algoritma
        }
        print(f"📚 Manuel seçim: {tahmin_isim}")
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Dosya boş.")
        print(f"📸 EKG alındı: {file.filename} ({len(contents)} bytes)")
        prediction = analyzer.analyze_ecg_image(contents)
        print(f"✓ Analiz: {prediction['tahmin']}")
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚑 PARAMEDİK EKG ASİSTANI v12.0")
    print("="*60)
    print(f"📍 http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
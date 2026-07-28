import os
import sys
import json
import argparse
import base64
import logging
import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime
import google.generativeai as genai
from PIL import Image

# ==============================================================================
# GEMINI API AYARLARI
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠ UYARI: GEMINI_API_KEY ortam değişkeni bulunamadı!")
    print("  Lütfen anahtarınızı Windows ortam değişkenlerine ekleyin.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ Gemini API anahtarı yüklendi.")

# ==============================================================================
# HTML ARAYÜZÜ
# ==============================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Acil Tıp EKG Analiz Asistanı (Gemini AI)</title>
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
        .result-box { background: #f8f9fa; padding: 25px; border-radius: 8px; border-left: 4px solid #4285f4; }
        .result-box.urgent { border-left-color: #e74c3c; background: #fdf2f2; }
        .result-box.safe { border-left-color: #27ae60; background: #f0fdf4; }
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
            <h1>🏥 Acil Tıp EKG Analiz Asistanı <span class="badge">GEMINI AI</span></h1>
            <p>Google Gemini Vision ile Gerçek Zamanlı EKG Yorumlama</p>
        </div>
        <div class="warning-box">
            ⚠️ <strong>Klinik Uyarı:</strong> Bu sistem bir karar destek aracıdır. Google Gemini yapay zekası tarafından analiz yapılır. Kesin tanı hekim tarafından konulmalıdır.
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
                <button class="btn-analyze" id="analyzeBtn">🧠 Gemini AI ile Analiz Et</button>
                <button class="btn-reset" id="resetBtn">↻ Temizle</button>
            </div>
            <div class="result-section" id="resultSection">
                <div id="resultContent"></div>
            </div>
        </div>
        <div class="footer"><p>v3.0 Gemini AI Edition | Powered by Google Gemini Vision</p></div>
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
            analyzeBtn.textContent = '🧠 Gemini analiz ediyor...';
            resultSection.style.display = 'block';
            resultContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>Yapay zeka EKG\'nizi inceliyor, lütfen bekleyin...</p></div>';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/analyze', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (result.status === 'success') {
                    const pred = result.prediction;
                    const urgent = pred.acil_mudahale;
                    resultContent.innerHTML = `
                        <div class="result-box ${urgent ? 'urgent' : 'safe'}">
                            <div class="result-label">🎯 AI Tahmini</div>
                            <div class="result-value">${pred.tahmin}</div>
                            
                            <div class="result-label">📊 Risk Seviyesi</div>
                            <div style="font-size: 16px; font-weight: 600; margin-bottom: 15px; color: ${urgent ? '#e74c3c' : '#27ae60'};">
                                ${pred.risk_seviyesi}
                            </div>
                            
                            <div class="result-label">📝 Detaylı Analiz</div>
                            <div class="analysis-text">${pred.detay}</div>
                            
                            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 12px; color: #7f8c8d;">
                                ⚠️ ${pred.uyari}
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
                analyzeBtn.textContent = '🧠 Gemini AI ile Analiz Et';
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
# GEMINI AI ANALİZ MOTORU
# ==============================================================================
class GeminiEKGAnalyzer:
    def __init__(self):
        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-flash-latest')
                print("✓ Gemini modeli hazır: gemini-flash-latest")
            except Exception as e:
                print(f"⚠ Gemini modeli yüklenemedi: {e}")
                try:
                    self.model = genai.GenerativeModel('gemini-2.5-flash')
                    print("✓ Yedek Gemini modeli hazır: gemini-2.5-flash")
                except Exception as e2:
                    print(f"⚠ Yedek model de yüklenemedi: {e2}")

    def analyze_ecg_image(self, image_bytes):
        if not self.model:
            return {
                "tahmin": "Hata",
                "risk_seviyesi": "Bilinmiyor",
                "acil_mudahale": False,
                "detay": "Gemini API anahtarı yapılandırılmamış. Lütfen GEMINI_API_KEY ortam değişkenini kontrol edin.",
                "uyari": "Sistem yapılandırma hatası."
            }
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            prompt = """Sen deneyimli bir acil tıp uzmanı ve kardiyoloğsun. Sana bir EKG (elektrokardiyogram) fotoğrafı gösteriyorum. 

Lütfen bu EKG'yi analiz et ve aşağıdaki JSON formatında CEVAP VER (başka hiçbir metin ekleme, sadece JSON):

{
  "tahmin": "Buraya EKG'nin ana bulgusunu yaz (örn: Normal Sinüs Ritmi, Atriyal Fibrilasyon, STEMI Şüphesi, Ventriküler Taşikardi, AV Blok, vb.)",
  "risk_seviyesi": "Düşük veya Orta veya Yüksek",
  "acil_mudahale": true veya false (acil müdahale gerekiyorsa true),
  "detay": "Buraya EKG'nin detaylı analizini yaz. Kalp hızı, ritim, P dalgaları, QRS kompleksleri, ST segmenti, T dalgaları hakkında gözlemlerini paylaş. Eğer bir patoloji tespit ettiyeysen, hangi derivasyonlarda görüldüğünü belirt. Türkçe yaz, tıbbi terimleri kullan ama açıklayıcı ol.",
  "uyari": "KLİNİK ONAY ZORUNLUDUR. Bu bir AI analizidir, kesin tanı için kardiyolog değerlendirmesi gereklidir."
}

ÖNEMLİ NOTLAR:
- Fotoğraf net değilse veya EKG değilse, tahmini "Analiz Yapılamadı" olarak belirt.
- Emin olmadığın durumlarda "şüphesi" ifadesini kullan.
- Türkçe cevap ver.
- SADECE JSON döndür, başka açıklama ekleme."""
            
            response = self.model.generate_content([prompt, image])
            
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            return {
                "tahmin": result.get("tahmin", "Bilinmiyor"),
                "risk_seviyesi": result.get("risk_seviyesi", "Bilinmiyor"),
                "acil_mudahale": result.get("acil_mudahale", False),
                "detay": result.get("detay", "Detay bilgisi alınamadı."),
                "uyari": result.get("uyari", "KLİNİK ONAY ZORUNLUDUR.")
            }
        
        except json.JSONDecodeError as e:
            return {
                "tahmin": "Analiz Sonucu",
                "risk_seviyesi": "Belirsiz",
                "acil_mudahale": False,
                "detay": response.text if 'response' in dir() else "Yanıt işlenemedi.",
                "uyari": "AI yanıtı JSON formatında değildi, ham metin gösteriliyor."
            }
        except Exception as e:
            return {
                "tahmin": "Hata",
                "risk_seviyesi": "Bilinmiyor",
                "acil_mudahale": False,
                "detay": f"Analiz sırasında hata oluştu: {str(e)}",
                "uyari": "Lütfen tekrar deneyin veya farklı bir fotoğraf yükleyin."
            }

# ==============================================================================
# FASTAPI SUNUCUSU
# ==============================================================================
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EKG Gemini AI", version="3.0.0")
analyzer = GeminiEKGAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "Acil EKG AI Asistanı",
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
        print("🧠 Gemini AI'a gönderiliyor...")
        
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
    print("🚀 EKG Gemini AI Sunucusu Başlatılıyor")
    print("="*60)
    print(f"📍 Yerel adres: http://localhost:8000")
    print(f"🌐 Ağdaki adres: http://0.0.0.0:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
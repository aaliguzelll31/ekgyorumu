# ekg_v14.py — Sözlük ekg_ai_pro.py'den import edilir (aynı klasörde dursun)
import os, io, re, json, base64, logging, time, concurrent.futures
from functools import partial
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageOps

try:
    from ekg_ai_pro import TEDAVI_ALGORITMALARI, tedavi_algoritmasi_bul, MANUEL_ISIMLER
    print("✓ Sözlük ekg_ai_pro.py'den yüklendi")
except Exception as e:
    raise SystemExit("❌ ekg_ai_pro.py aynı klasörde olmalı! " + str(e))

GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY"); GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"); ANTHROPIC_API_KEY=os.environ.get("ANTHROPIC_API_KEY")
MISTRAL_API_KEY=os.environ.get("MISTRAL_API_KEY"); XAI_API_KEY=os.environ.get("XAI_API_KEY")
COHERE_API_KEY=os.environ.get("COHERE_API_KEY"); OPENROUTER_API_KEY=os.environ.get("OPENROUTER_API_KEY")
if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)

EMERGENCY={"ANTERIOR_MI","INFERIOR_MI","LATERAL_MI","POSTERIOR_MI","SAG_V_MI","YAYGIN_ANTERIOR_MI","VT","VF","ASISTOLI"}
KEY2CODE={"ANTERIOR_MI":"SB-ASH-Y-02 (AKS)","INFERIOR_MI":"SB-ASH-Y-02 (AKS)","LATERAL_MI":"SB-ASH-Y-02 (AKS)","POSTERIOR_MI":"SB-ASH-Y-02 (AKS)","SAG_V_MI":"SB-ASH-Y-02 (AKS)","YAYGIN_ANTERIOR_MI":"SB-ASH-Y-02 (AKS)","NSTEMI":"SB-ASH-Y-02 (AKS)","AF":"SB-ASH-Y-08","SVT":"SB-ASH-Y-08","VT":"SB-ASH-Y-08","VF":"SB-ASH-Y-11","ASISTOLI":"SB-ASH-Y-10","BRADIKARDI":"SB-ASH-Y-07","AV_BLOK":"SB-ASH-Y-07","NORMAL":"SB-ASH-Y-02","GENEL":"SB-ASH-Y-01/02"}
SAHA_NOT={"INFERIOR_MI":["V4R çek (Sağ V MI ekarte et)","TA<90 ise nitrogliserin VERME","Sıvı bolus hazır"],"SAG_V_MI":["NİTROGLİSERİN YASAK","250-500 ml SF bolus","V4R ST elevasyonu ara"],"POSTERIOR_MI":["V7-V9 arka derivasyon çek","V1-V3 ayna görüntüsünü doğrula"],"NSTEMI":["Seri EKG: 5-10 dk'da bir tekrarla","STEMI'ye dönüşebilir"],"NORMAL":["Semptom varsa seri EKG + troponin","Normal EKG ≠ sorun yok"],"AV_BLOK":["Pacing hazır","Mobitz II/tam blokta atropin genelde etkisiz"],"BRADIKARDI":["Pacing hazır","İnferior MI + ilaç öyküsü düşün"],"VT":["Nabız kontrolü: yoksa SB-ASH-Y-11'e geç"],"VF":["Şoklanabilir ritim: CPR+defibrilasyon"],"ASISTOLI":["Şoklanmaz ritim: CPR+adrenalin, şok YOK"]}

# ================= ATLAS + SİSTEMATİK PROMPT =================
ATLAS="""KLASİK EKG ÖRNEKLERİ (referans atlas):
Anterior STEMI: V1-V4 ST elevasyonu + hiperakut T; resiprokal inferior depresyon.
Yaygın Anterior: + V5-V6, I, aVL elevasyonu (LAD proksimal).
Inferior: II,III,aVF elevasyonu; resiprokal I,aVL depresyonu; V4R kontrol.
Lateral: I,aVL,V5-V6 elevasyonu. Posterior: V1-V3 ST depresyonu + belirgin R + dik T (ayna); V7-V9 elevasyon.
Sağ V MI: inferior MI + V4R ≥1mm elevasyon + hipotansiyon + temiz akciğer.
de Winter: V1-V6 upsloping ST depresyonu + dik simetrik T = LAD oklüzyon eşdeğeri.
Wellens: V2-V3 bifazik/derin negatif T = kritik LAD.
NSTEMI: ST depresyonu/T inversiyonu, elevasyon YOK.
AF: düzensiz dar kompleks, P yok, fibrilatuar dalga. SVT: düzenli dar 150-250, P gömülü.
VT: düzenli geniş >100/dk, AV disosiasyon, füzyon/yakalama atımı. VF: kaotik, QRS yok.
Blok: 1° PR>200; Mobitz I uzayarak düşen; Mobitz II sabit oranlı düşen; 3° AV disosiasyon. Asistoli: düz hat."""
DETAYLI_PROMPT="""Sana İKİ görüntü verildi: orijinal ve otomatik iyileştirilmiş (kontrast/keskinlik/gri). Hangisi daha netse ONU kullan.
Sen kıdemli kardiyologsun. 12 derivasyon EKG'yi SİSTEMATİK oku: 1)Kalite/lead 2)Hız 3)Düzen 4)P 5)PR 6)QRS 7)QTc 8)ST: inferior(II,III,aVF), anterior(V1-V4), lateral(I,aVL,V5-V6), aVR 9)Blok.
"""+ATLAS+"""
TANI LİSTESİ (tahmin'e TAM birini yaz): "Anterior STEMI","Yaygın Anterior MI","Inferior STEMI","Lateral STEMI","Posterior STEMI","Sağ Ventrikül MI","NSTEMI","AF","SVT","VT","VF","Bradikardi","AV Blok","Asistoli","Normal Sinüs Ritmi".
KURALLAR: ST elevasyonu komşu ≥2 derivasyonda; de Winter/Wellens/aVR elevasyonu/yeni LBBB = STEMI eşdeğeri. Dar+düzenli→SVT, dar+düzensiz→AF, geniş düzenli→VT. Bulanık/lead eksikse kalite="kötü".
Sadece JSON: {"kalite":"iyi|orta|kötü","tum_leadler":true|false,"hiz":int|null,"duzenli":true|false|null,"p":"var|yok|belirsiz","pr_ms":int|null,"qrs_ms":int|null,"qtc_ms":int|null,"st":{"inferior":"elevasyon|depresyon|normal","anterior":"elevasyon|depresyon|normal","lateral":"elevasyon|depresyon|normal","avr":"elevasyon|normal"},"blok":"yok|1.derece|mobitz1|mobitz2|tam|lbbb|rbbb","tahmin":"<listeden>","stemi":true|false,"guven":0-100,"detay":"3-5 cümle Türkçe, derivasyon belirterek"}"""
SEMA_HAKEM='''Sadece JSON: {"tahmin":"<listeden>","stemi":true|false,"guven":0-100,"olcumler":{"hiz":int|null,"pr_ms":int|null,"qrs_ms":int|null,"qtc_ms":int|null},"detay":"4-6 cümle Türkçe","nihai_gerekce":"2-3 cümle"}'''
SEMA_VER='''Sadece JSON: {"kritik_bulgu":true|false,"oneri_tahmin":"<listeden>","guven":0-100,"gerekce":"1-2 cümle"}'''

# ================= GÖRÜNTÜ DÜZELTME (2 VARYANT) =================
def _prep(img, gray=False, smin=2000, con=1.15, shp=1.5):
    if gray: img=ImageOps.grayscale(img).convert("RGB")
    w,h=img.size
    if w<smin:
        f=smin/w; img=img.resize((int(w*f),int(h*f)),Image.LANCZOS)
    img=ImageOps.autocontrast(img,cutoff=1)
    img=ImageEnhance.Contrast(img).enhance(con); img=ImageEnhance.Sharpness(img).enhance(shp)
    b=io.BytesIO(); img.save(b,"PNG"); return b.getvalue()
def varyantlar(raw):
    base=Image.open(io.BytesIO(raw)).convert("RGB")
    return (_prep(base), _prep(base,gray=True))

# ================= OKUYUCULAR =================
def _b64(b): return base64.b64encode(b).decode()
def _tj(t):
    t=re.sub(r"```(?:json)?","",t).strip(); return json.loads(t[t.find("{"):t.rfind("}")+1])
def _retry(fn,*a,tries=2,**kw):
    s=None
    for _ in range(tries):
        try: return fn(*a,**kw)
        except Exception as e: s=e; time.sleep(1)
    raise s
def _oc(base,key,model,iv,prompt=None,eh=None,timeout=90):
    h={"Authorization":f"Bearer {key}"}
    if eh:h.update(eh)
    imgs=[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{_b64(x)}"}} for x in iv]
    pl={"model":model,"messages":[{"role":"user","content":[{"type":"text","text":prompt or DETAYLI_PROMPT}]+imgs}]}
    if model.startswith(("o1","o3","o4")): pl["max_completion_tokens"]=1200
    else: pl["temperature"]=0.0; pl["max_tokens"]=1200
    r=requests.post(base.rstrip("/")+"/chat/completions",headers=h,timeout=timeout,json=pl)
    r.raise_for_status(); return _tj(r.json()["choices"][0]["message"]["content"])
def gemini_oku(iv,model="gemini-2.5-flash",prompt=None):
    ims=[Image.open(io.BytesIO(x)) for x in iv]
    m=genai.GenerativeModel(model,generation_config={"temperature":0.0,"response_mime_type":"application/json"})
    return _tj(m.generate_content([prompt or DETAYLI_PROMPT]+ims).text)
def claude_oku(iv,prompt=None):
    blk=[{"type":"image","source":{"type":"base64","media_type":"image/png","data":_b64(x)}} for x in iv]
    r=requests.post("https://api.anthropic.com/v1/messages",headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01"},timeout=90,
        json={"model":"claude-sonnet-4-5","max_tokens":1200,"temperature":0.0,"messages":[{"role":"user","content":blk+[{"type":"text","text":prompt or DETAYLI_PROMPT}]}]})
    r.raise_for_status(); return _tj(r.json()["content"][0]["text"])
def groq_oku(iv,prompt=None):
    s=None
    for m in ["llama-3.2-90b-vision-preview","meta-llama/llama-4-scout-17b-16e-instruct"]:
        try: return _oc("https://api.groq.com/openai/v1",GROQ_API_KEY,m,iv,prompt)
        except Exception as e: s=e
    raise s
def arastir(tahmin):
    if not GEMINI_API_KEY: return None
    try:
        m=genai.GenerativeModel("gemini-2.5-flash",tools=[{"google_search":{}}])
        r=m.generate_content(f"'{tahmin}' EKG tanısının klasik kriterlerini 2-3 cümle Türkçe, derivasyon belirterek özetle.")
        return (r.text or "")[:600]
    except Exception as e:
        logging.warning("grounding: %s",e); return None

REG={}
def _add(n,f):
    if f and n not in REG: REG[n]=f
if GEMINI_API_KEY:
    _add("gemini-flash",partial(gemini_oku,model="gemini-2.5-flash")); _add("gemini-pro",partial(gemini_oku,model="gemini-2.5-pro"))
if GROQ_API_KEY: _add("groq",groq_oku)
if OPENAI_API_KEY:
    _add("gpt-4o",partial(_oc,"https://api.openai.com/v1",OPENAI_API_KEY,"gpt-4o"))
    _add("o3",partial(_oc,"https://api.openai.com/v1",OPENAI_API_KEY,"o3",timeout=150))
if ANTHROPIC_API_KEY: _add("claude",claude_oku)
if MISTRAL_API_KEY: _add("pixtral",partial(_oc,"https://api.mistral.ai/v1",MISTRAL_API_KEY,"pixtral-large-latest"))
if XAI_API_KEY: _add("grok",partial(_oc,"https://api.x.ai/v1",XAI_API_KEY,"grok-2-vision-1212"))
if COHERE_API_KEY: _add("aya-vision",partial(_oc,"https://api.cohere.com/v2",COHERE_API_KEY,"aya-vision-32b"))
if OPENROUTER_API_KEY:
    OR={"HTTP-Referer":"https://ekg.local","X-Title":"Paramedik EKG"}
    for n,s in [("gemini-pro","google/gemini-2.5-pro"),("gpt-4o","openai/gpt-4o"),("claude","anthropic/claude-sonnet-4"),("qwen-vl","qwen/qwen-2.5-vl-72b-instruct"),("llama-vl","meta-llama/llama-3.2-90b-vision-instruct")]:
        _add(n,partial(_oc,"https://openrouter.ai/api/v1",OPENROUTER_API_KEY,s,eh=OR))
READERS=list(REG.items())
HAKEM=[(n,REG[n]) for n in ["o3","gpt-4o","gemini-pro","claude","gemini-flash","groq"] if n in REG]
VERIF=[(n,REG[n]) for n in ["claude","grok","pixtral","qwen-vl","gemini-pro","gpt-4o","groq"] if n in REG]
print(f"✓ {len(READERS)} AI: {[a for a,_ in READERS]} | Hakem: {HAKEM[0][0] if HAKEM else 'YOK'} | İnternet doğrulama: {'AÇIK' if GEMINI_API_KEY else 'kapalı'}")

def _med(l):
    l=[x for x in l if isinstance(x,(int,float))]; return int(sorted(l)[len(l)//2]) if l else None
def key_of(t):
    a=tedavi_algoritmasi_bul(t)
    for k,v in TEDAVI_ALGORITMALARI.items():
        if v is a: return k
    return "GENEL"

# ================= 5 AŞAMALI ANALİZ =================
def ensemble_analyze(raw):
    if not READERS: raise Exception("API anahtarı yok")
    iv=varyantlar(raw); t0=time.time(); oylar={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(READERS),12)) as ex:
        fs={ex.submit(_retry,fn,iv):ad for ad,fn in READERS}
        for f in concurrent.futures.as_completed(fs,timeout=160):
            try:
                r=f.result(); r["_k"]=key_of(r.get("tahmin")); oylar[fs[f]]=r
            except Exception as e: logging.warning("[%s] %s",fs[f],e)
    if not oylar: raise Exception("Hiçbir AI yanıt vermedi")
    say={}
    for v in oylar.values(): say[v["_k"]]=say.get(v["_k"],0)+1
    cog=sorted(say.items(),key=lambda kv:(-kv[1],-(kv[0] in EMERGENCY)))[0][0]
    kal=[v.get("kalite") for v in oylar.values()]
    kalite=max(set(kal),key=kal.count) if kal else "orta"
    lead_ok=sum(1 for v in oylar.values() if v.get("tum_leadler"))>=len(oylar)/2
    stb={}
    for bol in ["inferior","anterior","lateral","avr"]:
        vals=[(v.get("st") or {}).get(bol) for v in oylar.values()]
        stb[bol]="elevasyon" if "elevasyon" in vals else ("depresyon" if "depresyon" in vals else "normal")
    # HAKEM (muhakeme modeli öncelikli)
    hk_ad,hk=None,None
    ot=json.dumps([{"ai":k,"tahmin":v.get("tahmin"),"key":v["_k],"kalite":v.get("kalite"),"st":v.get("st"),"qrs":v.get("qrs_ms"),"detay":(v.get("detay") or "")[:160]} for k,v in oylar.items()],ensure_ascii=False)
    hp="BAŞ HAKEMSİN. Oylar:\n"+ot+"\nİki görüntüyü de incele; atlas kriterleriyle KENDİN karar ver. Çoğunluğa körü körüne uyma.\n"+DETAYLI_PROMPT+"\n"+SEMA_HAKEM
    for ad,fn in HAKEM:
        try:
            h=_retry(fn,iv,prompt=hp); h["_k"]=key_of(h.get("tahmin")); hk_ad,hk=ad,h; break
        except Exception as e: logging.warning("Hakem %s %s",ad,e)
    ref=hk["_k"] if hk else cog
    # KIRMIZI EKİP + İNTERNET DOĞRULAMA (paralel)
    v_ad,ver,arast=None,None,None
    vp="KIRMIZI EKİPSİN. Önceki karar: "+ref+". Kaçırılmış STEMI/VF/asistoli var mı? ST'yi derivasyon derivasyon tara. kritik_bulgu sadece gerçek hayatî bulguda true.\n"+DETAYLI_PROMPT+"\n"+SEMA_VER
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex2:
        fv=None
        for ad,fn in VERIF:
            if ad==hk_ad: continue
            fv=ex2.submit(_retry,fn,iv,prompt=vp); v_ad=ad; break
        fa=ex2.submit(arastir,MANUEL_ISIMLER.get(ref,ref))
        if fv:
            try:
                v=fv.result(); v["_k"]=key_of(v.get("oneri_tahmin")); ver=v
            except Exception as e: logging.warning("Verif %s %s",v_ad,e)
        try: arast=fa.result()
        except Exception: arast=None
    # NİHAİ + GÜVENLİK
    duz=None; final=hk["_k"] if hk else cog
    if ver and ver.get("kritik_bulgu") and final not in EMERGENCY:
        final=ver["_k"] if ver["_k"] in EMERGENCY else "ANTERIOR_MI"; duz=f"Doğrulayıcı ({v_ad}) kritik bulgu → {final}'ye yükseltildi"
    elif final=="NORMAL" and cog in EMERGENCY:
        final=cog; duz="Hakem NORMAL dedi, çoğunluk acil → güvenlik önceliği"
    kalite_uyarisi="⚠️ EKG kalitesi yetersiz/lead eksik. Standartlara uygun yeniden çek (25 mm/s, 10 mm/mV) ve tekrar analiz et." if (kalite=="kötü" or not lead_ok) else None
    hm=(hk.get("olcumler") or {}) if hk else {}
    olc={a:(hm.get(a) or _med([v.get(a) for v in oylar.values()])) for a in ["hiz","pr_ms","qrs_ms","qtc_ms"]}
    guven=round(100*say.get(final,0)/len(oylar))
    if hk and hk["_k"]==final: guven=max(guven,70)
    if ver and (not ver.get("kritik_bulgu") or ver["_k"]==final): guven=min(100,guven+5)
    if arast: guven=min(100,guven+3)
    if kalite_uyarisi: guven=min(guven,50)
    bay=[]
    qrs=olc.get("qrs_ms") or 0; qtc=olc.get("qtc_ms") or 0
    if final=="SVT" and qrs>=120: bay.append("QRS geniş: VT olasılığını gözden geçir (SB-ASH-Y-08)")
    if final=="AF" and all(v.get("duzenli") for v in oylar.values() if v.get("duzenli") is not None): bay.append("Ritim düzenli görünüyor: AF'yi doğrula")
    if qtc>=500: bay.append("QTc ≥500 ms: Torsades riski")
    if final not in EMERGENCY and stb["avr"]=="elevasyon": bay.append("aVR ST elevasyonu: sol ana koroner/üç damar düşün")
    temsilci=(hk if hk and hk["_k"]==final else None) or next((v for v in oylar.values() if v["_k"]==final),next(iter(oylar.values())))
    return {"tahmin":temsilci.get("tahmin") or MANUEL_ISIMLER.get(final,final),"anahtar":final,"resmi_protokol":KEY2CODE.get(final,"-"),
        "risk_seviyesi":"Yüksek" if final in EMERGENCY else ("Orta" if final in {"NSTEMI","AF","SVT","BRADIKARDI","AV_BLOK"} else "Düşük"),
        "acil_mudahale":final in EMERGENCY or final in {"NSTEMI","AF","SVT","BRADIKARDI","AV_BLOK"},
        "detay":(hk or {}).get("detay") or temsilci.get("detay") or "","guven":guven,"hakem":hk_ad,
        "dogrulayici":{"model":v_ad,"kritik":bool(ver and ver.get("kritik_bulgu")),"duzeltme":duz},
        "oylama":{k:v.get("tahmin") for k,v in oylar.items()},"olcumler":olc,"st_bolgeler":stb,
        "kalite":kalite,"kalite_uyarisi":kalite_uyarisi,"tutarlilik":bay,"saha_notlari":SAHA_NOT.get(final,[]),
        "arastirma":arast,"nihai_gerekce":(hk or {}).get("nihai_gerekce",""),"sure_sn":round(time.time()-t0,1),
        "uyari":"Klinik onay zorunludur; nihai değerlendirme hekime aittir.","algoritma":TEDAVI_ALGORITMALARI[final]}

# ================= HTML (IZGARA) =================
HTML=r"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Paramedik EKG Asistanı v14</title><style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#c31432,#240b36);min-height:100vh;padding:20px}
.container{max-width:1150px;margin:0 auto}.header{background:#fff;padding:22px 28px;border-radius:12px 12px 0 0;border-bottom:4px solid #dc3545;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
.header h1{color:#2c3e50;font-size:24px}.badge{background:#dc3545;color:#fff;padding:2px 8px;border-radius:4px;font-size:10px}
.ai-chips{font-size:11px;color:#27ae60;align-self:center}.warning-box{background:#fff3cd;border-left:4px solid #ffc107;padding:12px 15px;font-size:13px;color:#856404}
.main-content{background:#fff;padding:28px}.upload-zone{border:2px dashed #dc3545;border-radius:8px;padding:32px;text-align:center;background:#fff5f5}
input[type=file]{display:none}.upload-options{display:flex;gap:10px;margin-top:14px;justify-content:center;flex-wrap:wrap}
.upload-btn{padding:12px 22px;background:#dc3545;color:#fff;border:none;border-radius:6px;cursor:pointer;font-weight:600}.upload-btn.camera{background:#007bff}
.button-group{display:flex;gap:10px;margin-top:18px;justify-content:center}button{padding:12px 24px;border:none;border-radius:6px;font-weight:600;cursor:pointer}
.btn-analyze{background:#27ae60;color:#fff}.btn-analyze:disabled{background:#95a5a6}.btn-reset{background:#95a5a6;color:#fff}
.preview-img{max-width:100%;max-height:260px;border-radius:6px;margin:14px 0;display:none}
.result-section{margin-top:22px;display:none}.result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:14px}
.gcard{background:#f8f9fa;border:1px solid #eee;border-left:4px solid #dc3545;border-radius:8px;padding:13px}
.gcard.safe{border-left-color:#27ae60}.gcard .lbl{font-size:10px;color:#7f8c8d;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:5px}
.gcard .val{font-size:17px;font-weight:800;color:#2c3e50}.gcard .sub{font-size:12px;color:#555;margin-top:4px}.full{grid-column:1/-1}
.oy{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:6px;padding:3px 8px;margin:3px;font-size:11px}
.not{display:inline-block;background:#e7f3ff;border:1px solid #b6d4fe;border-radius:6px;padding:4px 9px;margin:3px;font-size:12px;color:#004085}
.bay{display:inline-block;background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:4px 9px;margin:3px;font-size:12px;color:#856404}
.analysis-text{font-size:14px;color:#34495e;line-height:1.7;white-space:pre-wrap}
.algorithm-box{background:#fff5f5;padding:20px;border-radius:8px;border-left:4px solid #dc3545;margin-top:14px}
.algorithm-title{font-size:16px;font-weight:700;color:#dc3545;margin-bottom:8px}
.algorithm-urgency{display:inline-block;padding:6px 12px;border-radius:6px;font-size:13px;font-weight:600;background:#fff;margin-bottom:10px}
.algorithm-content{font-size:15px;color:#2c3e50;line-height:1.9;white-space:pre-wrap}
.error-message{background:#fdf2f2;border-left:4px solid #e74c3c;color:#c0392b;padding:15px;border-radius:4px}
.footer{background:#fff;padding:16px;border-radius:0 0 12px 12px;font-size:12px;color:#7f8c8d;text-align:center}
.loading{text-align:center;padding:20px}.spinner{border:4px solid #f3f3f3;border-top:4px solid #dc3545;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}</style></head><body><div class="container">
<div class="header"><h1>🚑 Paramedik EKG Asistanı <span class="badge">v14 MULTI-AI + ATLAS</span></h1><div class="ai-chips" id="aiChips"></div></div>
<div class="warning-box">⚠️ <strong>Klinik Uyarı:</strong> 112 paramedik yetkileri çerçevesinde karar desteğidir. Doktor orderı gereken ilaçlar komuta merkezi onayı ile verilir.</div>
<div class="main-content">
<div class="upload-zone"><div style="font-size:46px">📸</div><div style="font-size:16px;font-weight:500;color:#2c3e50">EKG Fotoğrafını Yükleyin</div>
<div style="font-size:12px;color:#7f8c8d;margin-top:5px">Kötü fotoğraf otomatik düzeltilir · Çoklu AI + atlas → Muhakeme Hakemi → Kırmızı Ekip → İnternet doğrulaması</div>
<div class="upload-options"><button class="upload-btn" onclick="document.getElementById('fg').click()">📁 Galeriden Seç</button>
<button class="upload-btn camera" onclick="document.getElementById('fc').click()">📷 Kamera ile Çek</button></div>
<input type="file" id="fg" accept="image/*"><input type="file" id="fc" accept="image/*" capture="environment"></div>
<img id="previewImg" class="preview-img" alt="">
<div class="button-group" id="buttonGroup" style="display:none"><button class="btn-analyze" id="analyzeBtn">🧠 EKG'yi Analiz Et</button><button class="btn-reset" id="resetBtn">↻ Temizle</button></div>
<div class="result-section" id="resultSection"><div id="resultContent"></div></div></div>
<div class="footer">v14 Multi-AI | Prm. Ali GÜZEL | ASHGM Hastane Öncesi Tanı ve Tedavi Algoritmaları tabanlıdır</div></div>
<script>
const fg=document.getElementById('fg'),fc=document.getElementById('fc'),pv=document.getElementById('previewImg'),bg=document.getElementById('buttonGroup'),ab=document.getElementById('analyzeBtn'),rb=document.getElementById('resetBtn'),rs=document.getElementById('resultSection'),rc=document.getElementById('resultContent');
let sel=null;
fetch('/health').then(r=>r.json()).then(h=>{document.getElementById('aiChips').textContent='🟢 '+(h.okuyucular||[]).join(', ')}).catch(()=>{});
function hf(e){const f=e.target.files[0];if(!f)return;sel=f;const r=new FileReader();r.onload=ev=>{pv.src=ev.target.result;pv.style.display='block';bg.style.display='flex';rs.style.display='none'};r.readAsDataURL(f);}
fg.addEventListener('change',hf);fc.addEventListener('change',hf);
rb.addEventListener('click',()=>{fg.value='';fc.value='';sel=null;pv.style.display='none';bg.style.display='none';rs.style.display='none';});
ab.addEventListener('click',async()=>{
 if(!sel)return alert('Dosya seçin!');
 ab.disabled=true;ab.textContent='🧠 Çoklu AI + atlas + hakem çalışıyor...';rs.style.display='block';
 rc.innerHTML='<div class="loading"><div class="spinner"></div><p>Görüntü düzeltiliyor → AI okuyucular → Muhakeme Hakemi → Kırmızı Ekip + İnternet (30-90 sn)...</p></div>';
 const fd=new FormData();fd.append('file',sel);
 try{const res=await fetch('/api/analyze',{method:'POST',body:fd});const r=await res.json();
  if(r.status==='success')show(r.prediction);else rc.innerHTML='<div class="error-message"><strong>Hata:</strong> '+r.message+'</div>';
 }catch(e){rc.innerHTML='<div class="error-message"><strong>Bağlantı Hatası:</strong> '+e.message+'</div>';}
 finally{ab.disabled=false;ab.textContent="🧠 EKG'yi Analiz Et";}});
function show(p){
 const o=p.olcumler||{},dg=p.dogrulayici||{},st=p.st_bolgeler||{};
 let g='<div class="result-grid">'
 +'<div class="gcard '+(p.acil_mudahale?'':'safe')+'"><div class="lbl">🎯 Tanı + Resmi Protokol</div><div class="val">'+p.tahmin+'</div><div class="sub">📋 '+p.resmi_protokol+' · '+p.risk_seviyesi+' risk</div></div>'
 +'<div class="gcard '+(p.acil_mudahale?'':'safe')+'"><div class="lbl">📊 Aciliyet / Güven</div><div class="val">'+(p.acil_mudahale?'🚨 ACİL':'✅ STABİL')+'</div><div class="sub">Oy birliği %'+p.guven+' · '+p.sure_sn+' sn</div></div>'
 +'<div class="gcard"><div class="lbl">⚖️ Denetim</div><div class="val" style="font-size:14px">Hakem: '+(p.hakem||'yok')+'</div><div class="sub">Kırmızı Ekip: '+(dg.model||'yok')+(dg.kritik?' · 🚨':'')+'</div>'+(dg.duzeltme?'<div class="sub" style="color:#c0392b">⚖ '+dg.duzeltme+'</div>':'')+'</div>'
 +'<div class="gcard"><div class="lbl">📐 Ölçümler</div><div class="val" style="font-size:14px">Hız '+(o.hiz||'-')+' · PR '+(o.pr_ms||'-')+'</div><div class="sub">QRS '+(o.qrs_ms||'-')+' · QTc '+(o.qtc_ms||'-')+' ms</div></div>'
 +'<div class="gcard"><div class="lbl">🗺️ ST Bölgeleri</div><div class="sub">İnferior: <b>'+st.inferior+'</b><br>Anterior: <b>'+st.anterior+'</b><br>Lateral: <b>'+st.lateral+'</b><br>aVR: <b>'+st.avr+'</b></div></div>'
 +'<div class="gcard"><div class="lbl">🖼️ Kalite / Düzeltme</div><div class="val" style="font-size:14px">'+(p.kalite||'-')+'</div><div class="sub">Renkli+gri varyant otomatik iyileştirildi</div>'+(p.kalite_uyarisi?'<div class="sub" style="color:#c0392b">'+p.kalite_uyarisi+'</div>':'')+'</div>'
 +(p.arastirma?'<div class="gcard full"><div class="lbl">🔎 İnternet + Atlas Doğrulaması</div><div class="sub">'+p.arastirma+'</div></div>':'')
 +'<div class="gcard full"><div class="lbl">🗳️ AI Oyları</div><div>';
 for(const k in (p.oylama||{}))g+='<span class="oy">'+k+' → <b>'+p.oylama[k]+'</b></span>';
 g+='</div>'+(p.nihai_gerekce?'<div class="sub" style="margin-top:6px">'+p.nihai_gerekce+'</div>':'')+'</div>';
 if((p.saha_notlari&&p.saha_notlari.length)||(p.tutarlilik&&p.tutarlilik.length)){
  g+='<div class="gcard full"><div class="lbl">🚑 Saha Notları / Tutarlılık</div><div>';
  (p.saha_notlari||[]).forEach(n=>g+='<span class="not">📌 '+n+'</span>');
  (p.tutarlilik||[]).forEach(n=>g+='<span class="bay">⚠ '+n+'</span>');
  g+='</div></div>';}
 g+='<div class="gcard full"><div class="lbl">📝 EKG Değerlendirmesi</div><div class="analysis-text">'+(p.detay||'')+'</div></div></div>';
 g+='<div class="algorithm-box"><div class="algorithm-title">🚑 PARAMEDİK MÜDAHALE PROTOKOLÜ</div><div class="algorithm-urgency">'+(p.algoritma&&p.algoritma.aciliyeti||'')+'</div><div class="algorithm-content">'+(p.algoritma&&p.algoritma.algoritma||'')+'</div><div style="margin-top:18px;padding-top:14px;border-top:1px solid #f5c6cb;font-size:12px;color:#555">📚 Kaynak: T.C. Sağlık Bakanlığı ASHGM · ⚠️ '+p.uyari+'</div></div>';
 rc.innerHTML=g;}
</script></body></html>"""

logging.basicConfig(level=logging.INFO)
app=FastAPI(title="Paramedik EKG v14",version="14.0")
@app.get("/",response_class=HTMLResponse)
async def home(): return HTMLResponse(HTML)
@app.get("/health")
async def health(): return {"okuyucular":[a for a,_ in READERS],"hakem":HAKEM[0][0] if HAKEM else None}
@app.post("/api/analyze")
async def analyze(file: UploadFile=File(...)):
    try:
        c=await file.read()
        if not c: raise HTTPException(400,"Dosya boş")
        p=ensemble_analyze(c)
        print(f"✓ {p['tahmin']} | {p['resmi_protokol']} | %{p['guven']} | hakem:{p['hakem']}")
        return {"status":"success","prediction":p}
    except Exception as e:
        return JSONResponse(status_code=500,content={"status":"error","message":str(e)})
@app.get("/api/manual/{kod}")
async def manual(kod:str):
    if kod not in TEDAVI_ALGORITMALARI: return JSONResponse(status_code=404,content={"status":"error","message":"Bulunamadı"})
    a=TEDAVI_ALGORITMALARI[kod]
    return {"status":"success","prediction":{"tahmin":MANUEL_ISIMLER.get(kod,kod),"risk_seviyesi":"Manuel","acil_mudahale":"ACİL" in a["aciliyeti"] or "KIRMIZI" in a["aciliyeti"] or "ARREST" in a["aciliyeti"],"detay":"","uyari":"Öğrenme modu.","resmi_protokol":KEY2CODE.get(kod,"-"),"olcumler":{},"st_bolgeler":{},"saha_notlari":SAHA_NOT.get(kod,[]),"tutarlilik":[],"kalite":"-","arastirma":None,"algoritma":a}}
if __name__=="__main__":
    import uvicorn
    print("📍 http://localhost:8000"); uvicorn.run(app,host="0.0.0.0",port=8000)
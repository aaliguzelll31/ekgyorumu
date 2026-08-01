# ekg_v16.py — mm bazlı ST ölçümü + kural motoru + hakem
# Sözlük ekg_ai_pro.py'den otomatik okunur (aynı klasörde dursun)
import os
import io
import re
import json
import base64
import logging
import time
import concurrent.futures
from functools import partial
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageOps

try:
    from ekg_ai_pro import (
        TEDAVI_ALGORITMALARI,
        tedavi_algoritmasi_bul,
        MANUEL_ISIMLER,
    )
    print("✓ Sözlük ekg_ai_pro.py'den yüklendi")
except Exception as e:
    raise SystemExit("❌ ekg_ai_pro.py aynı klasörde olmalı! " + str(e))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

EMERGENCY = {
    "ANTERIOR_MI", "INFERIOR_MI", "LATERAL_MI", "POSTERIOR_MI",
    "SAG_V_MI", "YAYGIN_ANTERIOR_MI", "VT", "VF", "ASISTOLI",
}
ORTA = {"NSTEMI", "AF", "SVT", "BRADIKARDI", "AV_BLOK"}
KEY2CODE = {
    "ANTERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "INFERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "LATERAL_MI": "SB-ASH-Y-02 (AKS)",
    "POSTERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "SAG_V_MI": "SB-ASH-Y-02 (AKS)",
    "YAYGIN_ANTERIOR_MI": "SB-ASH-Y-02 (AKS)",
    "NSTEMI": "SB-ASH-Y-02 (AKS)",
    "AF": "SB-ASH-Y-08",
    "SVT": "SB-ASH-Y-08",
    "VT": "SB-ASH-Y-08",
    "VF": "SB-ASH-Y-11",
    "ASISTOLI": "SB-ASH-Y-10",
    "BRADIKARDI": "SB-ASH-Y-07",
    "AV_BLOK": "SB-ASH-Y-07",
    "NORMAL": "SB-ASH-Y-02",
    "GENEL": "SB-ASH-Y-01/02",
}
SAHA_NOT = {
    "INFERIOR_MI": [
        "V4R çek (Sağ V MI ekarte et)",
        "TA<90 ise nitrogliserin VERME",
        "Sıvı bolus hazır",
    ],
    "SAG_V_MI": [
        "NİTROGLİSERİN YASAK",
        "250-500 ml SF bolus",
        "V4R ST elevasyonu ara",
    ],
    "POSTERIOR_MI": [
        "V7-V9 arka derivasyon çek",
        "V1-V3 ayna görüntüsünü doğrula",
    ],
    "NSTEMI": [
        "Seri EKG: 5-10 dk'da bir tekrarla",
        "STEMI'ye dönüşebilir",
    ],
    "NORMAL": [
        "Semptom varsa seri EKG + troponin",
        "Normal EKG ≠ sorun yok",
    ],
    "AV_BLOK": [
        "Pacing hazır",
        "Mobitz II/tam blokta atropin genelde etkisiz",
    ],
    "BRADIKARDI": [
        "Pacing hazır",
        "İnferior MI + ilaç öyküsü düşün",
    ],
    "VT": ["Nabız kontrolü: yoksa SB-ASH-Y-11'e geç"],
    "VF": ["Şoklanabilir ritim: CPR+defibrilasyon"],
    "ASISTOLI": ["Şoklanmaz ritim: CPR+adrenalin, şok YOK"],
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
CANON = {l.upper(): l for l in LEADS}

def temizle_algoritma(m):
    m = re.sub(
        r"\n\d️⃣ HASTANE TRANSFERİ:.*?(?=\n[━─]{6,}|\n⚠️|\n⚡|\n|\Z)",
        "", m, flags=re.S)
    m = re.sub(
        r"[━─]{6,}\n📚 HASTANEDE.*?(?=\n⚠️|\n⚡|\n|\Z)",
        "", m, flags=re.S)
    m = re.sub(
        r"[━─]{6,}\n🚑 HASTANEYE TRANSPORT.*?(?=\n⚡|\n|\Z)",
        "", m, flags=re.S)
    m = re.sub(r"\n{3,}", "\n\n", m)
    return m.strip()

def temiz_algo(k):
    a = TEDAVI_ALGORITMALARI[k]
    return {
        "aciliyeti": a["aciliyeti"],
        "algoritma": temizle_algoritma(a["algoritma"]),
    }

ATLAS = """ATLAS:
Anterior STEMI: V1-V4 elev + hiperakut T, resiprokal inferior dep.
Yaygın Anterior: + V5-V6, I, aVL elev (LAD proksimal).
Inferior: II, III, aVF elev, resiprokal I, aVL dep, V4R kontrol.
Lateral: I, aVL, V5-V6 elev.
Posterior: V1-V3 dep + belirgin R + dik T (ayna), V7-V9 elev.
Sağ V MI: inferior + V4R >=1mm + hipotansiyon + temiz akciğer.
de Winter: V1-V6 upsloping dep 1-3mm + dik simetrik T = LAD eşd.
Wellens: V2-V3 bifazik / derin negatif T.
NSTEMI: dep / T inversiyonu, elevasyon YOK.
AF: düzensiz dar kompleks, P yok.
SVT: düzenli dar 150-250/dk.
VT: düzenli geniş, AV disosiasyon.
VF: kaotik, QRS yok.
Blok: 1° PR>200; Mobitz I uzayarak düşen; Mobitz II sabit düşen;
3° AV disosiasyon. Asistoli: düz hat."""

DETAYLI_PROMPT = """Sana İKİ görüntü verildi: orijinal ve iyileştirilmiş.
Net olanı kullan. Kıdemli kardiyologsun.
ÖNCE ölç, SONRA tanı koy. Önce st_mm alanını doldur:
12 derivasyonun HER biri için ST sapması (mm, +elev/-dep, 0.5 hassas).
Sonra tahmin ver.
""" + ATLAS + """
TANI LİSTESİ (tahmin'e TAM birini yaz):
"Anterior STEMI","Yaygın Anterior MI","Inferior STEMI",
"Lateral STEMI","Posterior STEMI","Sağ Ventrikül MI","NSTEMI",
"AF","SVT","VT","VF","Bradikardi","AV Blok","Asistoli",
"Normal Sinüs Ritmi".
KURALLAR: komşu >=2 derivasyonda >=1mm elev (V2-V3 >=1.5-2mm) = STEMI.
Resiprokal depresyon tanıyı güçlendirir.
V1-V3 dep + belirgin R + elev YOK = Posterior.
V1-V6 upsloping dep + dik T = de Winter (STEMI eşdeğeri).
aVR elev + yaygın dep = sol ana koroner.
dar+düzenli -> SVT, dar+düzensiz -> AF, geniş düzenli -> VT.
Sadece JSON:
{"kalite":"iyi|orta|kötü","tum_leadler":true|false,
"st_mm":{"I":0,"II":0,"III":0,"aVR":0,"aVL":0,"aVF":0,
"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0},
"hiz":int|null,"duzenli":true|false|null,
"pr_ms":int|null,"qrs_ms":int|null,"qtc_ms":int|null,
"tahmin":"<listeden>","stemi":true|false,"guven":0-100,
"detay":"3-5 cümle Türkçe, derivasyon+mm belirterek"}"""

SEMA_HAKEM = """Sadece JSON:
{"tahmin":"<listeden>","stemi":true|false,"guven":0-100,
"st_mm":{12 lead mm},
"olcumler":{"hiz":int|null,"pr_ms":int|null,
"qrs_ms":int|null,"qtc_ms":int|null},
"detay":"4-6 cümle Türkçe","nihai_gerekce":"2-3 cümle"}"""

SEMA_VER = """Sadece JSON:
{"kritik_bulgu":true|false,"oneri_tahmin":"<listeden>",
"guven":0-100,"gerekce":"1-2 cümle"}"""

def _prep(img, gray=False):
    if gray:
        img = ImageOps.grayscale(img).convert("RGB")
    w, h = img.size
    if w < 2000:
        f = 2000 / w
        img = img.resize((int(w * f), int(h * f)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()

def varyantlar(raw):
    base = Image.open(io.BytesIO(raw)).convert("RGB")
    return (_prep(base), _prep(base, gray=True))

def _b64(b):
    return base64.b64encode(b).decode()

def _tj(t):
    t = re.sub(r"```(?:json)?", "", t).strip()
    return json.loads(t[t.find("{"):t.rfind("}") + 1])

def _retry(fn, *a, tries=2, **kw):
    s = None
    for _ in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            s = e
            time.sleep(1)
    raise s

def _oc(base, key, model, iv, prompt=None, eh=None, timeout=90):
    h = {"Authorization": f"Bearer {key}"}
    if eh:
        h.update(eh)
    imgs = []
    for x in iv:
        imgs.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(x)}"},
        })
    content = [{"type": "text", "text": prompt or DETAYLI_PROMPT}]
    content += imgs
    pl = {"model": model,
          "messages": [{"role": "user", "content": content}]}
    if model.startswith(("o1", "o3", "o4")):
        pl["max_completion_tokens"] = 1400
    else:
        pl["temperature"] = 0.0
        pl["max_tokens"] = 1400
    url = base.rstrip("/") + "/chat/completions"
    r = requests.post(url, headers=h, timeout=timeout, json=pl)
    r.raise_for_status()
    return _tj(r.json()["choices"][0]["message"]["content"])

def gemini_oku(iv, model="gemini-2.5-flash", prompt=None):
    ims = [Image.open(io.BytesIO(x)) for x in iv]
    m = genai.GenerativeModel(
        model,
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        })
    txt = m.generate_content([prompt or DETAYLI_PROMPT] + ims).text
    return _tj(txt)

def claude_oku(iv, prompt=None):
    blk = []
    for x in iv:
        blk.append({
            "type": "image",
            "source": {"type": "base64",
                       "media_type": "image/png",
                       "data": _b64(x)},
        })
    blk.append({"type": "text", "text": prompt or DETAYLI_PROMPT})
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY,
                 "anthropic-version": "2023-06-01"},
        timeout=90,
        json={"model": "claude-sonnet-4-5", "max_tokens": 1400,
              "temperature": 0.0,
              "messages": [{"role": "user", "content": blk}]})
    r.raise_for_status()
    return _tj(r.json()["content"][0]["text"])

def groq_oku(iv, prompt=None):
    s = None
    for m in ["llama-3.2-90b-vision-preview",
              "meta-llama/llama-4-scout-17b-16e-instruct"]:
        try:
            return _oc("https://api.groq.com/openai/v1",
                       GROQ_API_KEY, m, iv, prompt)
        except Exception as e:
            s = e
    raise s

def arastir(t):
    if not GEMINI_API_KEY:
        return None
    try:
        m = genai.GenerativeModel(
            "gemini-2.5-flash", tools=[{"google_search": {}}])
        q = f"'{t}' EKG tanısının klasik kriterlerini 2-3 cümle Türkçe özetle."
        return (m.generate_content(q).text or "")[:500]
    except Exception:
        return None

REG = {}
def _add(n, f):
    if f and n not in REG:
        REG[n] = f

if GEMINI_API_KEY:
    _add("gemini-flash",
         partial(gemini_oku, model="gemini-2.5-flash"))
    _add("gemini-pro",
         partial(gemini_oku, model="gemini-2.5-pro"))
if GROQ_API_KEY:
    _add("groq", groq_oku)
if OPENAI_API_KEY:
    _add("gpt-4o", partial(_oc,
         "https://api.openai.com/v1", OPENAI_API_KEY, "gpt-4o"))
    _add("o3", partial(_oc,
         "https://api.openai.com/v1", OPENAI_API_KEY, "o3",
         timeout=150))
if ANTHROPIC_API_KEY:
    _add("claude", claude_oku)
if MISTRAL_API_KEY:
    _add("pixtral", partial(_oc,
         "https://api.mistral.ai/v1", MISTRAL_API_KEY,
         "pixtral-large-latest"))
if XAI_API_KEY:
    _add("grok", partial(_oc,
         "https://api.x.ai/v1", XAI_API_KEY, "grok-2-vision-1212"))
if COHERE_API_KEY:
    _add("aya-vision", partial(_oc,
         "https://api.cohere.com/v2", COHERE_API_KEY,
         "aya-vision-32b"))
if OPENROUTER_API_KEY:
    OR = {"HTTP-Referer": "https://ekg.local",
          "X-Title": "Paramedik EKG"}
    OR_MODELS = [
        ("gemini-pro", "google/gemini-2.5-pro"),
        ("gpt-4o", "openai/gpt-4o"),
        ("claude", "anthropic/claude-sonnet-4"),
        ("qwen-vl", "qwen/qwen-2.5-vl-72b-instruct"),
        ("llama-vl", "meta-llama/llama-3.2-90b-vision-instruct"),
    ]
    for n, s in OR_MODELS:
        _add(n, partial(_oc,
             "https://openrouter.ai/api/v1",
             OPENROUTER_API_KEY, s, eh=OR))

READERS = list(REG.items())
HAKEM = [(n, REG[n]) for n in
         ["o3", "gpt-4o", "gemini-pro", "claude",
          "gemini-flash", "groq"] if n in REG]
VERIF = [(n, REG[n]) for n in
         ["claude", "grok", "pixtral", "qwen-vl",
          "gemini-pro", "gpt-4o", "groq"] if n in REG]
print(f"✓ {len(READERS)} AI | Hakem: "
      f"{HAKEM[0][0] if HAKEM else 'YOK'}")

def _med(l):
    l = [x for x in l if isinstance(x, (int, float))]
    return round(sorted(l)[len(l) // 2], 1) if l else None

def key_of(t):
    a = tedavi_algoritmasi_bul(t)
    for k, v in TEDAVI_ALGORITMALARI.items():
        if v is a:
            return k
    return "GENEL"

def norm_st(d):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        c = CANON.get(str(k).strip().upper())
        if c and isinstance(v, (int, float)):
            out[c] = float(v)
    return out

def medyan_st(liste):
    out = {}
    for L in LEADS:
        vals = [d.get(L) for d in liste if d.get(L) is not None]
        if vals:
            out[L] = _med(vals)
    return out

def st_karar(st):
    if not st:
        return []
    def elev(leads, thr=1.0):
        return [L for L in leads if (st.get(L) or 0) >= thr]
    def dep(leads, thr=0.5):
        return [L for L in leads if (st.get(L) or 0) <= -thr]
    inf_e = elev(["II", "III", "aVF"])
    ant_e = elev(["V1", "V2", "V3", "V4"])
    lat_e = elev(["I", "aVL", "V5", "V6"])
    ant_d = dep(["V1", "V2", "V3"])
    lat_d = dep(["I", "aVL", "V5", "V6"])
    inf_d = dep(["II", "III", "aVF"])
    avr = st.get("aVR") or 0
    bul = []
    if len(ant_e) >= 2:
        yay_l = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "aVL"]
        yay = len([L for L in yay_l
                   if (st.get(L) or 0) >= 1]) >= 6
        bolge = "YAYGIN_ANTERIOR_MI" if yay else "ANTERIOR_MI"
        bul.append({"bolge": bolge, "leadler": ant_e,
                    "tip": "elevasyon",
                    "resiprokal": bool(inf_d)})
    if len(inf_e) >= 2:
        bul.append({"bolge": "INFERIOR_MI", "leadler": inf_e,
                    "tip": "elevasyon",
                    "resiprokal": bool(lat_d)})
    if len(lat_e) >= 2:
        bul.append({"bolge": "LATERAL_MI", "leadler": lat_e,
                    "tip": "elevasyon",
                    "resiprokal": bool(inf_d)})
    if len(ant_d) >= 2 and not ant_e:
        bul.append({"bolge": "POSTERIOR_MI", "leadler": ant_d,
                    "tip": "ayna-depresyon", "resiprokal": False})
    vdep = dep(["V2", "V3", "V4", "V5", "V6"], 1.0)
    if len(vdep) >= 4 and not ant_e and not inf_e:
        bul.append({"bolge": "ANTERIOR_MI", "leadler": vdep,
                    "tip": "de-Winter eşdeğeri",
                    "resiprokal": False})
    yay_dep = len(ant_d) + len(inf_d) + len(lat_d)
    if avr >= 1 and yay_dep >= 4:
        bul.append({"bolge": "SOL_ANA_KORONER",
                    "leadler": ["aVR"],
                    "tip": "aVR elev+yaygın dep",
                    "resiprokal": False})
    return bul

def ensemble_analyze(raw):
    if not READERS:
        raise Exception("API anahtarı yok")
    iv = varyantlar(raw)
    t0 = time.time()
    oylar = {}
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(READERS), 12)) as ex:
        fs = {ex.submit(_retry, fn, iv): ad for ad, fn in READERS}
        for f in concurrent.futures.as_completed(fs, timeout=160):
            try:
                r = f.result()
                r["_k"] = key_of(r.get("tahmin"))
                r["_st"] = norm_st(r.get("st_mm"))
                oylar[fs[f]] = r
            except Exception as e:
                logging.warning("[%s] %s", fs[f], e)
    if not oylar:
        raise Exception("Hiçbir AI yanıt vermedi")
    st_mm = medyan_st([v["_st"] for v in oylar.values()])
    karar = st_karar(st_mm)
    st_stemi = [k["bolge"] for k in karar
                if k["bolge"] in EMERGENCY]
    say = {}
    for v in oylar.values():
        say[v["_k"]] = say.get(v["_k"], 0) + 1
    cog = sorted(say.items(),
                 key=lambda kv: (-kv[1], -(kv[0] in EMERGENCY)))[0][0]
    kal = [v.get("kalite") for v in oylar.values()]
    kalite = max(set(kal), key=kal.count) if kal else "orta"
    lead_ok = sum(1 for v in oylar.values()
                  if v.get("tum_leadler")) >= len(oylar) / 2
    hk_ad, hk = None, None
    ot_list = []
    for k, v in oylar.items():
        ot_list.append({
            "ai": k,
            "tahmin": v.get("tahmin"),
            "key": v["_k"],
            "st_mm": v["_st"],
            "qrs": v.get("qrs_ms"),
            "detay": (v.get("detay") or "")[:150],
        })
    ot = json.dumps(ot_list, ensure_ascii=False)
    hp = ("BAŞ HAKEMSİN. OBJEKTİF ST ÖLÇÜMÜ (AI medyanı, mm):\n"
          + json.dumps(st_mm)
          + "\nKURAL MOTORU KARARI:\n"
          + json.dumps(karar, ensure_ascii=False)
          + "\nOYLAR:\n" + ot
          + "\nKURAL: kural motoru komşu elevasyon veya"
          + " ayna-depresyon bulduysa ve görüntü doğruluyorsa"
          + " o STEMI'dir; çoğunluk NORMAL/NSTEMI dese bile."
          + " İki görüntüyü incele, KENDİN karar ver.\n"
          + DETAYLI_PROMPT + "\n" + SEMA_HAKEM)
    for ad, fn in HAKEM:
        try:
            h = _retry(fn, iv, prompt=hp)
            h["_k"] = key_of(h.get("tahmin"))
            h["_st"] = norm_st(h.get("st_mm"))
            hk_ad, hk = ad, h
            break
        except Exception as e:
            logging.warning("Hakem %s %s", ad, e)
    if hk and hk["_st"]:
        st_mm = medyan_st([st_mm, hk["_st"]])
        karar = st_karar(st_mm)
        st_stemi = [k["bolge"] for k in karar
                    if k["bolge"] in EMERGENCY]
    ref = hk["_k"] if hk else cog
    v_ad, ver, arast = None, None, None
    vp = ("KIRMIZI EKİPSİN. Önceki karar: " + ref
          + ". OBJEKTİF ST: " + json.dumps(st_mm)
          + " KURAL: " + json.dumps(karar, ensure_ascii=False)
          + ". Kaçırılmış STEMI/VF/asistoli var mı?"
          + " kritik_bulgu sadece gerçek hayatî bulguda true.\n"
          + DETAYLI_PROMPT + "\n" + SEMA_VER)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex2:
        fv = None
        for ad, fn in VERIF:
            if ad == hk_ad:
                continue
            fv = ex2.submit(_retry, fn, iv, prompt=vp)
            v_ad = ad
            break
        fa = ex2.submit(arastir, MANUEL_ISIMLER.get(ref, ref))
        if fv:
            try:
                v = fv.result()
                v["_k"] = key_of(v.get("oneri_tahmin"))
                ver = v
            except Exception as e:
                logging.warning("Verif %s %s", v_ad, e)
        try:
            arast = fa.result()
        except Exception:
            arast = None
    duz = None
    final = hk["_k"] if hk else cog
    if st_stemi and final not in EMERGENCY:
        final = st_stemi[0]
        duz = (f"Objektif ST kural motoru {st_stemi[0]} buldu"
               f" → karar buna yükseltildi")
    elif ver and ver.get("kritik_bulgu") and final not in EMERGENCY:
        final = ver["_k"] if ver["_k"] in EMERGENCY else "ANTERIOR_MI"
        duz = f"Doğrulayıcı ({v_ad}) kritik bulgu → {final}"
    elif final == "NORMAL" and cog in EMERGENCY:
        final = cog
        duz = "Hakem NORMAL dedi, çoğunluk acil → güvenlik önceliği"
    kalite_uyarisi = None
    if kalite == "kötü" or not lead_ok:
        kalite_uyarisi = ("⚠️ EKG kalitesi yetersiz/lead eksik."
                          " Standartlara uygun yeniden çek"
                          " ve tekrar analiz et.")
    hm = (hk.get("olcumler") or {}) if hk else {}
    olc = {}
    for a in ["hiz", "pr_ms", "qrs_ms", "qtc_ms"]:
        vals = [v.get(a) for v in oylar.values()]
        olc[a] = hm.get(a) or _med(vals)
    guven = round(100 * say.get(final, 0) / len(oylar))
    if hk and hk["_k"] == final:
        guven = max(guven, 70)
    if st_stemi and final in st_stemi:
        guven = min(100, guven + 10)
    if ver and (not ver.get("kritik_bulgu")
                or ver["_k"] == final):
        guven = min(100, guven + 5)
    if kalite_uyarisi:
        guven = min(guven, 50)
    bay = []
    qrs = olc.get("qrs_ms") or 0
    qtc = olc.get("qtc_ms") or 0
    if final == "SVT" and qrs >= 120:
        bay.append("QRS geniş: VT olasılığını gözden geçir")
    if final == "AF" and all(
            v.get("duzenli") for v in oylar.values()
            if v.get("duzenli") is not None):
        bay.append("Ritim düzenli: AF'yi doğrula")
    if qtc and qtc >= 500:
        bay.append("QTc >=500 ms: Torsades riski")
    if final not in EMERGENCY and (st_mm.get("aVR") or 0) >= 1:
        bay.append("aVR elevasyonu: sol ana koroner düşün")
    if final == "NSTEMI" and st_stemi:
        bay.append("DİKKAT: objektif ST elevasyonu var,"
                   " NSTEMI değil STEMI olabilir")
    temsilci = None
    if hk and hk["_k"] == final:
        temsilci = hk
    if not temsilci:
        for v in oylar.values():
            if v["_k"] == final:
                temsilci = v
                break
    if not temsilci:
        temsilci = next(iter(oylar.values()))
    return {
        "tahmin": temsilci.get("tahmin")
                  or MANUEL_ISIMLER.get(final, final),
        "anahtar": final,
        "resmi_protokol": KEY2CODE.get(final, "-"),
        "risk_seviyesi": "Yüksek" if final in EMERGENCY
                         else ("Orta" if final in ORTA else "Düşük"),
        "acil_mudahale": final in EMERGENCY or final in ORTA,
        "detay": (hk or {}).get("detay")
                 or temsilci.get("detay") or "",
        "guven": guven,
        "hakem": hk_ad,
        "dogrulayici": {
            "model": v_ad,
            "kritik": bool(ver and ver.get("kritik_bulgu")),
            "duzeltme": duz,
        },
        "oylama": {k: v.get("tahmin") for k, v in oylar.items()},
        "olcumler": olc,
        "st_mm": st_mm,
        "st_karar": karar,
        "kalite": kalite,
        "kalite_uyarisi": kalite_uyarisi,
        "tutarlilik": bay,
        "saha_notlari": SAHA_NOT.get(final, []),
        "arastirma": arast,
        "nihai_gerekce": (hk or {}).get("nihai_gerekce", ""),
        "sure_sn": round(time.time() - t0, 1),
        "uyari": "Klinik onay zorunludur;"
                 " nihai değerlendirme hekime aittir.",
        "algoritma": temiz_algo(final),
    }

HTML = r"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Paramedik EKG Asistanı</title><style>
:root{--bg:#0b1220;--card:rgba(255,255,255,.05);
--line:rgba(255,255,255,.09);--txt:#e5e7eb;--mut:#94a3b8;
--red:#ef4444;--amb:#f59e0b;--grn:#22c55e}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
background-image:radial-gradient(900px 400px at 15% -10%,
rgba(220,38,38,.18),transparent),
radial-gradient(800px 400px at 90% 0%,rgba(37,99,235,.15),transparent);
min-height:100vh;padding:22px;color:var(--txt)}
.app{max-width:1180px;margin:0 auto;display:flex;
flex-direction:column;gap:14px}
.glass{background:var(--card);border:1px solid var(--line);
border-radius:16px;backdrop-filter:blur(10px)}
.topbar{display:flex;align-items:center;gap:14px;padding:16px 22px}
.logo{width:46px;height:46px;border-radius:14px;
background:linear-gradient(135deg,#dc2626,#7f1d1d);
display:grid;place-items:center;font-size:24px;
box-shadow:0 6px 18px rgba(220,38,38,.4)}
.topbar h1{font-size:20px;font-weight:800}
.topbar .sub{font-size:11px;color:var(--mut)}
.chips{margin-left:auto;font-size:11px;color:#4ade80;
text-align:right;max-width:55%}
.warn{padding:12px 18px;font-size:12.5px;color:#fde68a;
background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3)}
.main{padding:24px}
.tabs{display:flex;gap:6px;margin-bottom:18px}
.tab{flex:1;padding:12px;border:1px solid var(--line);
background:transparent;color:var(--mut);border-radius:10px;
font-weight:700;font-size:13px;cursor:pointer}
.tab.active{background:linear-gradient(135deg,#dc2626,#991b1b);
color:#fff;border-color:transparent}
.tcontent{display:none}.tcontent.active{display:block}
.upload{border:2px dashed rgba(239,68,68,.45);border-radius:14px;
padding:30px;text-align:center;background:rgba(239,68,68,.05)}
input[type=file]{display:none}
.urow{display:flex;gap:10px;justify-content:center;
margin-top:14px;flex-wrap:wrap}
.btn{padding:12px 22px;border:none;border-radius:10px;
font-weight:700;font-size:13px;cursor:pointer}
.btn.red{background:linear-gradient(135deg,#dc2626,#991b1b);color:#fff}
.btn.blue{background:linear-gradient(135deg,#2563eb,#1e40af);color:#fff}
.btn.green{background:linear-gradient(135deg,#16a34a,#166534);color:#fff;
font-size:15px;padding:14px 30px}
.btn.gray{background:rgba(255,255,255,.1);color:var(--txt)}
.btn:disabled{opacity:.5;cursor:not-allowed}
select{width:100%;padding:13px;border-radius:10px;background:#111a2e;
color:var(--txt);border:1px solid var(--line);font-size:14px}
.preview-img{max-width:100%;max-height:260px;border-radius:12px;
margin:14px auto 0;display:none;border:1px solid var(--line)}
.bgroup{display:flex;gap:10px;justify-content:center;margin-top:16px}
.result{display:none}
.ribbon{border-radius:16px;padding:20px 24px;color:#fff;display:flex;
flex-wrap:wrap;gap:18px;align-items:center;
justify-content:space-between;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.ribbon.r-red{background:linear-gradient(135deg,#7f1d1d,#dc2626);
animation:pulse 1.6s infinite}
.ribbon.r-amb{background:linear-gradient(135deg,#78350f,#d97706)}
.ribbon.r-grn{background:linear-gradient(135deg,#14532d,#16a34a)}
@keyframes pulse{0%,100%{box-shadow:0 10px 30px rgba(220,38,38,.35)}
50%{box-shadow:0 10px 44px rgba(220,38,38,.7)}}
.ribbon .dx{font-size:25px;font-weight:900}
.ribbon .meta{font-size:12.5px;opacity:.92;margin-top:4px}
.ribbon .right{text-align:right;min-width:190px}
.meter{height:8px;background:rgba(255,255,255,.28);border-radius:6px;
margin-top:8px;overflow:hidden}
.meter>div{height:100%;background:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,
minmax(230px,1fr));gap:12px;margin-top:14px}
.card{padding:16px}
.card .lbl{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;
color:var(--mut);font-weight:700;margin-bottom:9px}
.card .val{font-size:16px;font-weight:800}
.card .sub{font-size:12px;color:var(--mut);margin-top:5px;line-height:1.55}
.full{grid-column:1/-1}
.mini{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.mini div{background:rgba(255,255,255,.05);border-radius:8px;
padding:7px;text-align:center}
.mini b{display:block;font-size:9px;color:var(--mut);
text-transform:uppercase}
.lead{display:inline-block;border-radius:8px;padding:5px 8px;
margin:3px;font-size:11px;font-weight:700;border:1px solid}
.l-elev{background:rgba(239,68,68,.18);border-color:var(--red);
color:#fca5a5}
.l-dep{background:rgba(245,158,11,.15);border-color:var(--amb);
color:#fde68a}
.l-norm{background:rgba(34,197,94,.1);border-color:var(--grn);
color:#86efac}
.karar{display:inline-block;background:rgba(239,68,68,.18);
border:1px solid var(--red);border-radius:8px;padding:5px 10px;
margin:3px;font-size:12px;color:#fca5a5;font-weight:700}
.oy{display:inline-block;background:rgba(255,255,255,.07);
border:1px solid var(--line);border-radius:8px;padding:4px 9px;
margin:3px;font-size:11px}
.not{display:inline-block;background:rgba(37,99,235,.15);
border:1px solid rgba(37,99,235,.4);border-radius:8px;
padding:5px 10px;margin:3px;font-size:12px;color:#bfdbfe}
.bay{display:inline-block;background:rgba(245,158,11,.15);
border:1px solid rgba(245,158,11,.4);border-radius:8px;
padding:5px 10px;margin:3px;font-size:12px;color:#fde68a}
.analysis{font-size:13.5px;line-height:1.7;white-space:pre-wrap;
color:#cbd5e1}
.proto{padding:22px;border-left:4px solid var(--red)}
.proto h3{color:#fca5a5;font-size:15px;margin-bottom:8px}
.proto .urg{display:inline-block;background:rgba(255,255,255,.08);
border:1px solid var(--line);padding:6px 12px;border-radius:8px;
font-size:12.5px;font-weight:700;margin-bottom:12px}
.proto pre{font-family:inherit;font-size:14px;line-height:1.85;
white-space:pre-wrap}
.proto .src{margin-top:16px;padding-top:12px;
border-top:1px solid var(--line);font-size:11.5px;color:var(--mut)}
.footer{text-align:center;font-size:11px;color:var(--mut);padding:8px}
.loading{text-align:center;padding:30px}
.spinner{width:44px;height:44px;border:4px solid rgba(255,255,255,.12);
border-top-color:var(--red);border-radius:50%;
animation:spin 1s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
.fade{animation:fade .4s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}
to{opacity:1;transform:none}}
</style></head><body><div class="app">
<div class="glass topbar"><div class="logo">🫀</div>
<div><h1>Paramedik EKG Asistanı</h1>
<div class="sub">mm Bazlı ST Ölçümü · Kural Motoru · Muhakeme Hakemi · Kırmızı Ekip</div></div>
<div class="chips" id="aiChips"></div></div>
<div class="glass warn">⚠️ <b>Klinik Uyarı:</b> 112 paramedik yetkileri
çerçevesinde karar desteğidir. Doktor orderı gereken ilaçlar
komuta merkezi onayı ile verilir.</div>
<div class="glass main">
<div class="tabs">
<button class="tab active" id="t1" onclick="stab(1)">📸 Fotoğraf Analizi</button>
<button class="tab" id="t2" onclick="stab(2)">📚 Manuel Seçim (Öğren)</button>
</div>
<div class="tcontent active" id="c1">
<div class="upload"><div style="font-size:42px">📸</div>
<div style="font-size:16px;font-weight:600;margin-top:6px">EKG Fotoğrafını Yükleyin</div>
<div style="font-size:12px;color:var(--mut);margin-top:4px">Kötü fotoğraf otomatik düzeltilir · AI'lar önce mm ölçer, sonra tanı koyar</div>
<div class="urow">
<button class="btn red" onclick="fg.click()">📁 Galeriden Seç</button>
<button class="btn blue" onclick="fc.click()">📷 Kamera ile Çek</button>
</div>
<input type="file" id="fg" accept="image/*">
<input type="file" id="fc" accept="image/*" capture="environment"></div>
<img id="pv" class="preview-img" alt="">
<div class="bgroup" id="bg" style="display:none">
<button class="btn green" id="ab">🧠 EKG'yi Analiz Et</button>
<button class="btn gray" id="rb">↻ Temizle</button></div>
</div>
<div class="tcontent" id="c2">
<div class="card glass" style="margin-bottom:14px">
<div class="lbl">🎯 EKG Ritmi / Tanısı Seçin</div>
<select id="ritmSelect"><option value="">-- Bir ritim seçin --</option>
<optgroup label="🚨 Kalp Krizleri">
<option value="ANTERIOR_MI">Anterior STEMI</option>
<option value="INFERIOR_MI">Inferior STEMI</option>
<option value="LATERAL_MI">Lateral STEMI</option>
<option value="POSTERIOR_MI">Posterior STEMI</option>
<option value="SAG_V_MI">Sağ Ventrikül MI</option>
<option value="YAYGIN_ANTERIOR_MI">Yaygın Anterior MI</option>
<option value="NSTEMI">NSTEMI / USAP</option></optgroup>
<optgroup label="⚠️ Aritmiler">
<option value="AF">AF</option><option value="SVT">SVT</option>
<option value="VT">VT</option><option value="VF">VF (Arrest)</option>
<option value="BRADIKARDI">Bradikardi</option></optgroup>
<optgroup label="🔌 Bloklar / Diğer">
<option value="AV_BLOK">AV Blok</option>
<option value="ASISTOLI">Asistoli (Arrest)</option>
<option value="NORMAL">Normal Sinüs Ritmi</option></optgroup>
</select></div>
<div class="bgroup">
<button class="btn green" id="mb">📖 Paramedik Protokolünü Göster</button>
</div></div>
<div class="result" id="rs"><div id="rc"></div></div></div>
<div class="footer">Kaynak: T.C. Sağlık Bakanlığı ASHGM (SB-ASH-Y) · Prm. Ali GÜZEL</div></div>
<script>
var fg=document.getElementById('fg'),fc=document.getElementById('fc'),
pv=document.getElementById('pv'),bg=document.getElementById('bg'),
ab=document.getElementById('ab'),rb=document.getElementById('rb'),
rs=document.getElementById('rs'),rc=document.getElementById('rc'),
mb=document.getElementById('mb');
var sel=null;
fetch('/health').then(function(r){return r.json();})
.then(function(h){
document.getElementById('aiChips').innerHTML='🟢 '+(h.okuyucular||[]).join(' · ');
}).catch(function(){});
function stab(n){
document.getElementById('t1').classList.toggle('active',n==1);
document.getElementById('t2').classList.toggle('active',n==2);
document.getElementById('c1').classList.toggle('active',n==1);
document.getElementById('c2').classList.toggle('active',n==2);
rs.style.display='none';}
function hf(e){
var f=e.target.files[0]; if(!f)return; sel=f;
var r=new FileReader();
r.onload=function(ev){pv.src=ev.target.result;
pv.style.display='block';bg.style.display='flex';
rs.style.display='none';};
r.readAsDataURL(f);}
fg.onchange=hf; fc.onchange=hf;
rb.onclick=function(){fg.value='';fc.value='';sel=null;
pv.style.display='none';bg.style.display='none';
rs.style.display='none';};
ab.onclick=function(){run('/api/analyze',function(){
var fd=new FormData(); fd.append('file',sel); return fd;});};
mb.onclick=function(){
var s=document.getElementById('ritmSelect').value;
if(!s)return alert('Ritim seçin!');
run('/api/manual/'+s,null);};
function run(url,mkfd){
ab.disabled=true; mb.disabled=true; rs.style.display='block';
rc.innerHTML='<div class="loading"><div class="spinner"></div>'
+'<p style="color:var(--mut)">AI\'lar mm ölçüyor → Kural motoru'
+' → Hakem → Kırmızı Ekip...</p></div>';
var opt=mkfd?{method:'POST',body:mkfd()}:{};
fetch(url,opt).then(function(res){return res.json();})
.then(function(r){
if(r.status==='success')show(r.prediction);
else rc.innerHTML='<div class="glass card" style="border-left:4px solid var(--red)"><b>Hata:</b> '+r.message+'</div>';
}).catch(function(e){
rc.innerHTML='<div class="glass card" style="border-left:4px solid var(--red)"><b>Hata:</b> '+e.message+'</div>';
}).finally(function(){ab.disabled=false;mb.disabled=false;});}
function leadCls(v){
if(v>=1)return 'l-elev';
if(v<=-0.5)return 'l-dep';
return 'l-norm';}
function fmt(v){
if(v==null)return '-';
return (v>0?'+':'')+v;}
function show(p){
var o=p.olcumler||{},dg=p.dogrulayici||{},sm=p.st_mm||{};
var hasOyl=Object.keys(p.oylama||{}).length>0;
var rc_=p.risk_seviyesi==='Yüksek'?'r-red':
((p.risk_seviyesi==='Orta'||p.acil_mudahale)?'r-amb':'r-grn');
var h='<div class="fade"><div class="ribbon '+rc_+'"><div>'
+'<div class="dx">'+(p.acil_mudahale?'🚨':'✅')+' '+p.tahmin+'</div>'
+'<div class="meta">📋 '+p.resmi_protokol+' · '+p.risk_seviyesi
+' risk'+(p.sure_sn?' · '+p.sure_sn+' sn':'')
+(p.hakem?' · Hakem: '+p.hakem:'')
+(dg.model?' · Kırmızı Ekip: '+dg.model+(dg.kritik?' 🚨':''):'')
+'</div>'
+(dg.duzeltme?'<div class="meta">⚖ '+dg.duzeltme+'</div>':'')
+'</div><div class="right"><div style="font-size:12px">GÜVEN SKORU</div>'
+'<div style="font-size:24px;font-weight:900">%'+(p.guven||0)+'</div>'
+'<div class="meter"><div style="width:'+(p.guven||0)+'%"></div></div>'
+'</div></div>';
h+='<div class="grid">';
h+='<div class="glass card full"><div class="lbl">🧮 Objektif ST Haritası (AI medyanı, mm) — 🟥 elev ≥1 · 🟨 dep ≥0.5</div><div>';
["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
.forEach(function(L){
var v=sm[L];
h+='<span class="lead '+(v==null?'l-norm':leadCls(v))+'">'+L+' '+fmt(v)+'</span>';});
h+='</div>';
if(p.st_karar&&p.st_karar.length){
h+='<div style="margin-top:8px">';
p.st_karar.forEach(function(k){
h+='<span class="karar">⚡ '+k.bolge.replace('_',' ')+' · '
+k.leadler.join(',')+' · '+k.tip
+(k.resiprokal?' +resiprokal':'')+'</span>';});
h+='</div>';}
else h+='<div class="sub" style="margin-top:6px">Kural motoru: ST kriteri karşılanmadı</div>';
h+='</div>';
h+='<div class="glass card"><div class="lbl">📐 Ölçümler</div><div class="mini">'
+'<div><b>Hız</b>'+(o.hiz||'-')+'</div>'
+'<div><b>PR</b>'+(o.pr_ms||'-')+'</div>'
+'<div><b>QRS</b>'+(o.qrs_ms||'-')+'</div>'
+'<div><b>QTc</b>'+(o.qtc_ms||'-')+'</div></div></div>';
h+='<div class="glass card"><div class="lbl">🖼️ Kalite</div>'
+'<div class="val">'+(p.kalite||'-')+'</div>'
+'<div class="sub">Renkli + gri varyant otomatik iyileştirildi</div>'
+(p.kalite_uyarisi?'<div class="sub" style="color:#fca5a5">'+p.kalite_uyarisi+'</div>':'')
+'</div>';
if(hasOyl){
h+='<div class="glass card"><div class="lbl">🗳️ AI Oyları</div><div>';
for(var k in p.oylama)
h+='<span class="oy">'+k+' → <b>'+p.oylama[k]+'</b></span>';
h+='</div></div>';}
if((p.saha_notlari&&p.saha_notlari.length)||(p.tutarlilik&&p.tutarlilik.length)){
h+='<div class="glass card full"><div class="lbl">🚑 Saha Notları / Tutarlılık</div><div>';
(p.saha_notlari||[]).forEach(function(n){
h+='<span class="not">📌 '+n+'</span>';});
(p.tutarlilik||[]).forEach(function(n){
h+='<span class="bay">⚠ '+n+'</span>';});
h+='</div></div>';}
if(p.arastirma)
h+='<div class="glass card full"><div class="lbl">🔎 Atlas + İnternet Doğrulaması</div><div class="sub">'+p.arastirma+'</div></div>';
if(p.detay)
h+='<div class="glass card full"><div class="lbl">📝 EKG Değerlendirmesi</div><div class="analysis">'+p.detay+'</div>'
+(p.nihai_gerekce?'<div class="sub" style="margin-top:8px">⚖ '+p.nihai_gerekce+'</div>':'')
+'</div>';
h+='</div>';
h+='<div class="glass proto" style="margin-top:14px">'
+'<h3>🚑 PARAMEDİK MÜDAHALE PROTOKOLÜ</h3>'
+'<div class="urg">'+(p.algoritma&&p.algoritma.aciliyeti||'')+'</div>'
+'<pre>'+(p.algoritma&&p.algoritma.algoritma||'')+'</pre>'
+'<div class="src">📚 Kaynak: ASHGM (SB-ASH-Y) · ⚠ '+(p.uyari||'')+'</div></div></div>';
rc.innerHTML=h;}
</script></body></html>"""

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Paramedik EKG", version="16.0")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)

@app.get("/health")
async def health():
    return {"okuyucular": [a for a, _ in READERS],
            "hakem": HAKEM[0][0] if HAKEM else None}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        c = await file.read()
        if not c:
            raise HTTPException(400, "Dosya boş")
        p = ensemble_analyze(c)
        print(f"✓ {p['tahmin']} | {p['resmi_protokol']}"
              f" | %{p['guven']} | ST-karar:{len(p['st_karar'])}")
        return {"status": "success", "prediction": p}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"status": "error",
                                     "message": str(e)})

@app.get("/api/manual/{kod}")
async def manual(kod: str):
    if kod not in TEDAVI_ALGORITMALARI:
        return JSONResponse(status_code=404,
                            content={"status": "error",
                                     "message": "Bulunamadı"})
    a = temiz_algo(kod)
    return {"status": "success", "prediction": {
        "tahmin": MANUEL_ISIMLER.get(kod, kod),
        "risk_seviyesi": "Manuel",
        "acil_mudahale": "ACİL" in a["aciliyeti"]
                         or "KIRMIZI" in a["aciliyeti"]
                         or "ARREST" in a["aciliyeti"],
        "detay": "", "uyari": "Öğrenme modu.",
        "resmi_protokol": KEY2CODE.get(kod, "-"),
        "olcumler": {}, "st_mm": {}, "st_karar": [],
        "saha_notlari": SAHA_NOT.get(kod, []),
        "tutarlilik": [], "kalite": "-",
        "arastirma": None, "nihai_gerekce": "",
        "sure_sn": 0, "dogrulayici": {}, "oylama": {},
        "guven": 100, "algoritma": a}}

if __name__ == "__main__":
    import uvicorn
    print("📍 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
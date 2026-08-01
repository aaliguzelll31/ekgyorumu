# ekg_ai_pro.py — Paramedik EKG Tedavi Algoritmalari

MANUEL_ISIMLER = {
    "ANTERIOR_MI": "Anterior STEMI",
    "INFERIOR_MI": "Inferior STEMI",
    "LATERAL_MI": "Lateral STEMI",
    "POSTERIOR_MI": "Posterior STEMI",
    "SAG_V_MI": "Sag Ventrikul MI",
    "YAYGIN_ANTERIOR_MI": "Yaygin Anterior MI",
    "NSTEMI": "NSTEMI / USAP",
    "AF": "Atriyal Fibrilasyon",
    "SVT": "Supraventrikuler Tasikardi",
    "VT": "Ventrikuler Tasikardi",
    "VF": "Ventrikuler Fibrilasyon",
    "BRADIKARDI": "Semptomatik Bradikardi",
    "AV_BLOK": "AV Blok",
    "ASISTOLI": "Asistoli",
    "NORMAL": "Normal Sinuz Ritmi",
    "GENEL": "Genel EKG Degerlendirmesi",
    "SOL_ANA_KORONER": "Sol Ana Koroner Darlik / Yaygin Iskemi",
}

TEDAVI_ALGORITMALARI = {
    "ANTERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Bilinc, solunum, cilt, TA, nabiz, SpO2 degerlendir.\n"
            "   • 12 derivasyon EKG cek, V1-V4 ST elevasyonu ara.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin 325 mg cigneme (alerji yoksa).\n"
            "   • Nitrogliserin sublingual (sistolik TA > 90 mmHg).\n"
            "   • Morfin 2-5 mg IV gerekirse (agri/dispne).\n"
            "   • Oksijen sadece SpO2 < 90 veya dispne varsa.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Hedef kapiya kadinik sure < 90 dk.\n"
            "   • PCI merkezine direkt transport, hastaneye yakin degilse helikopter.\n\n"
            "---\n"
            "📚 HASTANEDE: Heparin, P2Y12 yukleme, koroner anjiyografi."
        ),
    },
    "INFERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • II, III, aVF ST elevasyonu var mi?\n"
            "   • Mutlaka V4R cek, sag ventrikul MI ekarte et.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin 325 mg cigneme.\n"
            "   • Inferior MI'de preload dusurucu ilaclara dikkat.\n"
            "   • TA < 90 mmHg ise NITROGLISERIN KONTRENDIKE.\n"
            "   • Sag V MI varsa SF 250-500 ml bolus.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • PCI merkezine hizli transport.\n\n"
            "---\n"
            "📚 HASTANEDE: Sag kateterizasyon, koroner anjiyografi."
        ),
    },
    "LATERAL_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • I, aVL, V5-V6 ST elevasyonu.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin, nitrat (TA uygunsa), oksijen endikasyonuna gore.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • PCI merkezine direkt transport."
        ),
    },
    "POSTERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • V1-V3 yatay ST depresyonu, belirgin R, dik T.\n"
            "   • V7-V9 cek, elevasyon dogrula.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin, nitrat dikkatli.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • PCI merkezine transport."
        ),
    },
    "SAG_V_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Inferior MI + V4R ST elevasyonu >= 1 mm.\n"
            "   • Hipotansiyon, JVD, temiz akciger.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • NITROGLISERIN KESINLIKLE YASAK.\n"
            "   • Morfin yavas ve dikkatli.\n"
            "   • SF 250-500 ml bolus, hipotansiyon devam ederse tekrarla.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Hizli transport, sag ventrikul disfonksiyonu riski."
        ),
    },
    "YAYGIN_ANTERIOR_MI": {
        "aciliyeti": "KIRMIZI ALARM - ACIL (Yuksek Risk)",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • V1-V6 + I, aVL yaygin ST elevasyonu.\n"
            "   • Proksimal LAD tutulumu, kardiogenik sok olabilir.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin, nitrat dikkatli (TA duserse kes).\n"
            "   • Hipotansiyon varsa sivi, inotrop dusun.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Acil PCI, kardiyojenik sok ekibi haberli."
        ),
    },
    "NSTEMI": {
        "aciliyeti": "SARI ALARM - YAKIN TAKIP",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • ST elevasyonu yok, depresyon/T inversiyonu olabilir.\n"
            "   • Agrisi olan, instabil hasta yuksek risk.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin, nitrat (TA uygunsa).\n"
            "   • Seri EKG 5-10 dk'de bir.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Uygun merkeze transport, troponin takibi."
        ),
    },
    "AF": {
        "aciliyeti": "SARI ALARM - Hemodinami belirler",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Duzensiz dar QRS, P dalgasi yok.\n"
            "   • Hiz ve hemodinami degerlendir.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Hemodinamik bozukluk varsa sedelektrik kardiyoversiyon.\n"
            "   • Stabilse hiz kontrolu (diltiazem/amiadoron).\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Antikoagulasyon karari hastanede."
        ),
    },
    "SVT": {
        "aciliyeti": "SARI ALARM - Hemodinami belirler",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Duzenli dar QRS tasikardi 150-250/dk.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Vagal manevra.\n"
            "   • Adenozin 6 mg hizli bolus, sonra 12 mg.\n"
            "   • Genis QRS ise VT dusun.\n"
            "   • Hemodinamik bozukluk varsa senkronize kardiyoversiyon.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • EKG ile transport."
        ),
    },
    "VT": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Genis QRS, duzenli ritim.\n"
            "   • Nabiz var mi kontrol et.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Nabiz yoksa defibrilasyon + CPR (SB-ASH-Y-11).\n"
            "   • Nabiz varsa amiodaron 150-300 mg IV.\n"
            "   • Hemodinamik bozukluk varsa senkronize kardiyoversiyon.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Surekli izlem, defibrilator hazir."
        ),
    },
    "VF": {
        "aciliyeti": "KIRMIZI ALARM - ARREST",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Kaotik ritim, QRS yok.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Hemen baslangic CPR.\n"
            "   • Defibrilatoru hazirla, sokla.\n"
            "   • Adrenalin 1 mg IV/IO her 3-5 dk.\n"
            "   • Reversible nedenleri dusun (H/T).\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • ROSC sonrasi post-arrest bakim merkezine transport."
        ),
    },
    "BRADIKARDI": {
        "aciliyeti": "SARI ALARM - Semptom varsa KIRMIZI",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Hiz < 60/dk ve semptom (hipotansiyon, bilinc bulanikligi, sok).\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Atropin 0.5 mg IV her 3-5 dk (maks 3 mg).\n"
            "   • Transkutan pacing hazir.\n"
            "   • Atropin yetersizse dopamin/epinefrin infuzyonu.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Transkutan pacing devam ederken transport."
        ),
    },
    "AV_BLOK": {
        "aciliyeti": "SARI ALARM - Mobitz II/tam blok KIRMIZI",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • PR uzama, Mobitz I/II, tam blok.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Mobitz II veya tam blokta transkutan pacing hazir.\n"
            "   • Atropin tam blokta genelde etkisiz.\n"
            "   • Hipotansiyon varsa atropin + pacing.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Hemen transport, transvenous pacing gerekebilir."
        ),
    },
    "ASISTOLI": {
        "aciliyeti": "KIRMIZI ALARM - ARREST",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Duz hat, elektriksel aktivite yok.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Hemen baslangic CPR.\n"
            "   • Adrenalin 1 mg IV/IO her 3-5 dk.\n"
            "   • SOK YOK.\n"
            "   • H/T nedenlerini dusun (hipovolemi, hipoksia, hidrojen iyonu, hiper/hipokalemi, hipotermi, tabletler, kardiak tamponat, tansiyon pnomoni, tromboemboli).\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • ROSC saglanirsa post-arrest merkezi."
        ),
    },
    "NORMAL": {
        "aciliyeti": "YESIL - Acil degil",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Normal sinuz ritmi, hiz 60-100/dk.\n"
            "   • ST/T patolojik degisiklik yok.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Semptom varsa seri EKG + troponin planla.\n"
            "   • Hastaya gore transport karari."
        ),
    },
    "SOL_ANA_KORONER": {
        "aciliyeti": "KIRMIZI ALARM - ACIL",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • aVR elevasyonu + yaygin ST depresyonu.\n"
            "   • Sol ana koroner veya proksimal LAD kritik darlik.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Aspirin, sivi, inotrop destegi.\n"
            "   • Nitrat dikkatli, TA duserse kes.\n\n"
            "3️⃣ HASTANE TRANSFERI:\n"
            "   • Acil PCI merkezine transport."
        ),
    },
    "GENEL": {
        "aciliyeti": "SARI ALARM - Degerlendirme gerekli",
        "algoritma": (
            "1️⃣ GENEL DEGERLENDIRME:\n"
            "   • Ritim, hiz, duzenlilik, P-QRS-T sistematik incele.\n"
            "   • EKG kalitesi yetersizse tekrar cek.\n\n"
            "2️⃣ OLAY YERI MUDAHALESI:\n"
            "   • Semptom ve hemodinamiye gore mudahale.\n"
            "   • Supheli ise acil kabul et."
        ),
    },
}


def tedavi_algoritmasi_bul(tani_metni: str):
    """Tani metninden en uygun algoritmayi bulur."""
    if not tani_metni:
        return TEDAVI_ALGORITMALARI["GENEL"]

    t = tani_metni.upper()

    if "VF" in t or "FIBRILASYON" in t:
        return TEDAVI_ALGORITMALARI["VF"]
    if "ASISTOL" in t or "DUZ HAT" in t:
        return TEDAVI_ALGORITMALARI["ASISTOLI"]
    if "SAG VENTRIKUL" in t or "SAG V" in t or "V4R" in t:
        return TEDAVI_ALGORITMALARI["SAG_V_MI"]
    if "YAYGIN ANTERIOR" in t:
        return TEDAVI_ALGORITMALARI["YAYGIN_ANTERIOR_MI"]
    if "POSTERIOR" in t or "POSTERIYOR" in t:
        return TEDAVI_ALGORITMALARI["POSTERIOR_MI"]
    if "ANTERIOR" in t:
        return TEDAVI_ALGORITMALARI["ANTERIOR_MI"]
    if "INFERIOR" in t:
        return TEDAVI_ALGORITMALARI["INFERIOR_MI"]
    if "LATERAL" in t:
        return TEDAVI_ALGORITMALARI["LATERAL_MI"]
    if "NSTEMI" in t or "USAP" in t or "NON-ST" in t:
        return TEDAVI_ALGORITMALARI["NSTEMI"]
    if "SOL ANA" in t or ("AVR" in t and "YAYGIN" in t):
        return TEDAVI_ALGORITMALARI["SOL_ANA_KORONER"]
    if "ATRIYAL FIBRILASYON" in t or ("AF " in t or t.endswith("AF")):
        return TEDAVI_ALGORITMALARI["AF"]
    if "SVT" in t or "SUPRAVENTRIKULER" in t:
        return TEDAVI_ALGORITMALARI["SVT"]
    if "VENTRIKULER TASIKARDI" in t or " VT" in t or t.endswith("VT"):
        return TEDAVI_ALGORITMALARI["VT"]
    if "BRADIKARDI" in t or "YAVAS RITIM" in t:
        return TEDAVI_ALGORITMALARI["BRADIKARDI"]
    if "AV BLOK" in t or "BLOK" in t:
        return TEDAVI_ALGORITMALARI["AV_BLOK"]
    if "NORMAL" in t or "SINUZ" in t:
        return TEDAVI_ALGORITMALARI["NORMAL"]

    return TEDAVI_ALGORITMALARI["GENEL"]
import streamlit as st
import librosa
import numpy as np
import io
import warnings
from scipy.signal import butter, lfilter   # ← ДОБАВИТЬ ЭТУ СТРОКУ
warnings.filterwarnings('ignore')

# ==========================================
# 1. КОНСТАНТЫ И СПРАВОЧНЫЕ ДАННЫЕ
# ==========================================
STANDARD_TUNING = {
    6: 82.41,   # E2
    5: 110.00,  # A2
    4: 146.83,  # D3
    3: 196.00,  # G3
    2: 246.94,  # B3
    1: 329.63   # E4
}

# ==========================================
# 2. АЛГОРИТМИЧЕСКАЯ ЧАСТЬ (Бизнес-логика)
# ==========================================

def highpass_filter(y, sr, cutoff=70):
    """Фильтр высоких частот для отсечения фонового шума (Спринт 1)"""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(N=5, Wn=normal_cutoff, btype='high', analog=False)  # ← было librosa.filters.butter
    return lfilter(b, a, y)                                            # ← было librosa.lfilter

def extract_pitch(audio_bytes, sr=22050):
    """Извлечение F0 с использованием pyin (или crepe)"""
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    y = highpass_filter(y, sr)
    
    # Убираем атаку и затухание (берем среднюю часть ноты для стабильности)
    # Атака может давать артефакты питча
    center = len(y) // 2
    start = max(0, center - int(sr * 0.2))
    end = min(len(y), center + int(sr * 0.2))
    y_stable = y[start:end]
    
    # Основной питч-детектор (librosa pyin)
    f0, voiced_flag, voiced_probs = librosa.pyin(y_stable, fmin=70, fmax=1000, sr=sr)
    f0 = f0[~np.isnan(f0)]
    
    # Альтернатива: crepe (раскомментируйте, если установлен tensorflow)
    # import crepe
    # time, frequency, confidence, activation = crepe.predict(y_stable, sr, viterbi=True)
    # f0 = frequency[confidence > 0.6]

    if len(f0) == 0:
        return None, None, None
        
    f0_median = np.median(f0)
    rms = np.sqrt(np.mean(y**2))
    return f0_median, rms, y

def calculate_cents(f_measured, f_target):
    """Расчет отклонения в центах"""
    if f_measured is None or f_target is None:
        return 0
    return 1200 * np.log2(f_measured / f_target)

def detect_buzz(y, sr, f0):
    """Детектор дребезга (Спринт 3)"""
    # Берем первые 100 мс (атака)
    attack_frames = int(sr * 0.1)
    y_attack = y[:attack_frames]
    
    if len(y_attack) < 512:
        return False, 0, 0

    # Спектральный центроид
    centroid = librosa.feature.spectral_centroid(y=y_attack, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    # ZCR (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y_attack)[0]
    mean_zcr = np.mean(zcr)
    
    # Энергия высоких частот (>2000 Гц) vs F0
    S = np.abs(librosa.stft(y_attack))
    freqs = librosa.fft_frequencies(sr=sr)
    
    high_freq_mask = freqs > 2000
    high_energy = np.sum(S[high_freq_mask, :])
    
    # Энергия вокруг F0
    f0_mask = (freqs > f0 * 0.9) & (freqs < f0 * 1.1)
    f0_energy = np.sum(S[f0_mask, :])
    
    ratio = high_energy / (f0_energy + 1e-6)
    
    # Эвристика: дребезг дает высокий центроид, высокий ZCR и много "верхов"
    is_buzz = (mean_centroid > 1500) and (mean_zcr > 0.1) and (ratio > 0.5)
    
    return is_buzz, mean_centroid, ratio

# ==========================================
# 3. МАТРИЦА ПРИНЯТИЯ РЕШЕНИЙ (Rule Engine)
# ==========================================
def get_recommendation(state, string_num, cents_delta, is_buzz, buzz_frets):
    recs = []
    
    if state == 2 and cents_delta is not None:
        if cents_delta > 3:
            recs.append(f"⚠️ Мензура слишком короткая (завышение на {cents_delta:.1f} центов). "
                        "Сдвиньте седло 6-й струны назад (к концу гитары) на 1-2 мм.")
        elif cents_delta < -3:
            recs.append(f"⚠️ Мензура слишком длинная (занижение на {abs(cents_delta):.1f} центов). "
                        "Сдвиньте седло вперед (к грифу).")
        else:
            recs.append("✅ Мензура в норме.")

    if state == 3 and is_buzz:
        if 1 in buzz_frets or 5 in buzz_frets:
            recs.append("⚠️ Дребезг у головы грифа. Ослабьте анкер (добавьте прогиб) или проверьте верхний порожек.")
        elif 7 in buzz_frets or 9 in buzz_frets:
            recs.append("⚠️ Дребезг в середине грифа. Подтяните анкер или поднимите седла на бридже.")
            
    if not recs:
        recs.append("✅ Проблем не обнаружено (или данные недостаточны).")
        
    return recs

# ==========================================
# 4. UI / FRONTEND (Streamlit)
# ==========================================
st.set_page_config(page_title="Guitar Setup Analyzer", layout="wide")
st.title("🎸 Guitar Setup Analyzer (MVP)")

with st.sidebar:
    st.header("Настройки")
    string_num = st.selectbox("Выберите струну", [6, 5, 4, 3, 2, 1], format_func=lambda x: f"Струна {x}")
    fresh_strings = st.checkbox("Струны свежие (< 2 недель)", value=True)
    if not fresh_strings:
        st.warning("⚠️ На старых струнах точность питч-детектора падает из-за негармоничных обертонов.")

# Инициализация состояния сессии
if 'f0_open' not in st.session_state: st.session_state.f0_open = None
if 'f0_harm' not in st.session_state: st.session_state.f0_harm = None
if 'f0_fretted' not in st.session_state: st.session_state.f0_fretted = None
if 'buzz_data' not in st.session_state: st.session_state.buzz_data = []

target_freq = STANDARD_TUNING[string_num]

tab1, tab2, tab3 = st.tabs(["Шаг 1: Калибровка (Строй)", "Шаг 2: Мензура", "Шаг 3: Action & Relief"])

# --- ШАГ 1: КАЛИБРОВКА ---
with tab1:
    st.subheader(f"Сыграйте открытую {string_num}-ю струну")
    audio_open = st.audio_input("Запись открытой струны", key="rec_open")
    
    if audio_open:
        f0, rms, _ = extract_pitch(audio_open.getvalue())
        st.session_state.f0_open = f0
        
        if f0:
            cents = calculate_cents(f0, target_freq)
            col1, col2 = st.columns(2)
            col1.metric("Целевая частота", f"{target_freq:.2f} Гц")
            col2.metric("Ваша частота", f"{f0:.2f} Гц", f"{cents:+.1f} центов")
            
            if abs(cents) > 5:
                st.error(f"❌ Отклонение {abs(cents):.1f} центов. Подстройте колки! Переход к Шагу 2 заблокирован.")
            else:
                st.success("✅ Струна настроена. Переходите к Шагу 2.")
        else:
            st.warning("Не удалось распознать ноту. Сыграйте громче и чище.")

# --- ШАГ 2: МЕНЗУРА ---
with tab2:
    if st.session_state.f0_open is None or abs(calculate_cents(st.session_state.f0_open, target_freq)) > 5:
        st.warning("🔒 Сначала точно настройте открытую струну в Шаге 1.")
    else:
        st.subheader("Диагностика мензуры (12-й лад)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**1. Флажолет на 12 ладу**")
            audio_harm = st.audio_input("Запись флажолета", key="rec_harm")
            if audio_harm:
                f0_h, _, _ = extract_pitch(audio_harm.getvalue())
                st.session_state.f0_harm = f0_h
                if f0_h: st.metric("F0 флажолета", f"{f0_h:.2f} Гц")

        with col2:
            st.markdown("**2. Зажатая нота на 12 ладу**")
            audio_fret = st.audio_input("Запись зажатой ноты", key="rec_fret")
            if audio_fret:
                f0_f, rms_f, _ = extract_pitch(audio_fret.getvalue())
                st.session_state.f0_fretted = f0_f
                if f0_f:
                    st.metric("F0 зажатой", f"{f0_f:.2f} Гц")
                    if rms_f > 0.15:
                        st.warning("⚠️ Слишком сильная атака! Сыграйте мягче, иначе питч будет завышен физически.")

        if st.session_state.f0_harm and st.session_state.f0_fretted:
            delta = calculate_cents(st.session_state.f0_fretted, st.session_state.f0_harm)
            st.divider()
            st.markdown(f"### Дельта мензуры: `{delta:+.2f}` центов")
            
            recs = get_recommendation(2, string_num, delta, False, [])
            for r in recs: st.info(r)

# --- ШАГ 3: ACTION & RELIEF ---
with tab3:
    st.subheader("Диагностика высоты струн и анкера")
    st.markdown("Сыграйте по очереди на **1, 5, 7 и 9 ладах**. Система проанализирует атаку на наличие дребезга.")
    
    frets_to_check = [1, 5, 7, 9]
    
    if st.button("Начать анализ дребезга", type="primary"):
        st.session_state.buzz_data = []

    for fret in frets_to_check:
        audio_buzz = st.audio_input(f"Запись {fret}-го лада", key=f"rec_buzz_{fret}")
        if audio_buzz:
            f0_b, _, y_b = extract_pitch(audio_buzz.getvalue())
            if f0_b:
                is_buzz, cent, ratio = detect_buzz(y_b, 22050, f0_b)
                st.session_state.buzz_data.append({
                    'fret': fret, 
                    'is_buzz': is_buzz, 
                    'centroid': cent, 
                    'ratio': ratio
                })
                
                status = "🔴 ДРЕБЕЗГ" if is_buzz else "🟢 Чисто"
                st.caption(f"Лад {fret}: {status} (Центроид: {cent:.0f}, Ratio: {ratio:.2f})")

    if st.session_state.buzz_data:
        st.divider()
        buzz_frets = [d['fret'] for d in st.session_state.buzz_data if d['is_buzz']]
        any_buzz = len(buzz_frets) > 0
        
        recs = get_recommendation(3, string_num, None, any_buzz, buzz_frets)
        for r in recs:
            if "Дребезг" in r: st.error(r)
            else: st.success(r)
            
        # Визуализация тепло карты (Спринт 4)
        st.markdown("**Тепловая карта ладов:**")
        cols = st.columns(len(frets_to_check))
        for i, fret in enumerate(frets_to_check):
            is_buzzed = fret in buzz_frets
            cols[i].metric(f"Лад {fret}", "BUZZ" if is_buzzed else "OK")
            if is_buzzed: cols[i].error("⚠️")
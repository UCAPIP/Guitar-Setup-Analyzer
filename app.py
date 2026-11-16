import math
import streamlit as st
import numpy as np
import time, warnings
from collections import deque
from scipy.signal import butter, lfilter
import librosa
warnings.filterwarnings('ignore')
try:
    import sounddevice as sd
except ImportError:
    sd = None

# ============ КОНСТАНТЫ ============
SR, BLOCK, BUFFER_SEC = 44100, 2048, 4
RMS_QUIET, RMS_LOUD = 0.003, 0.30
RMS_HARM_QUIET = 0.001
TUNE_TOL = 5.0          # ТЗ: блок шага 2 при отклонении > 5 центов
INTON_TOL = 3.0         # ТЗ: порог мензуры, центы
JUMP_THR, HF_THR, ZJUMP_THR = 1.5, 0.5, 1.5   # базовые пороги дребезга (до масштаба по калибру)
STANDARD_TUNING = {6: 82.41, 5: 110.00, 4: 146.83, 3: 196.00, 2: 246.94, 1: 329.63}
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Наборы калибров (дюймы), струны 1..6
GAUGE_SETS = {
    "009–042 (Extra Light)":  [.009, .011, .016, .024, .032, .042],
    "010–046 (Regular Light)": [.010, .013, .017, .026, .036, .046],
    "011–049 (Medium Light)": [.011, .014, .018, .028, .038, .049],
    "012–053 (Medium)":       [.012, .016, .020, .032, .042, .053],
    "013–056 (Heavy)":        [.013, .017, .026, .036, .046, .056],
}
REF_GAUGES = dict(zip(range(1, 7), GAUGE_SETS["010–046 (Regular Light)"]))
TUNINGS = {
    "Стандарт (E A D G B E)": {s: 0 for s in range(1, 7)},
    "Полтона вниз (Eb)": {s: -1 for s in range(1, 7)},
    "Тон вниз (D)": {s: -2 for s in range(1, 7)},
    "Drop D": {**{s: 0 for s in range(1, 7)}, 6: -2},
}
SCALES = {
    '25.5" (648 мм) Fender': 648.0,
    '24.75" (629 мм) Gibson': 629.0,
    '25" (635 мм) PRS': 635.0,
    '24" (610 мм) короткая': 610.0,
    '27" (686 мм) баритон': 686.0,
}

# ============ ПРОФИЛЬ СТРУНЫ (калибр -> пороги) ============
def is_wound(s, gauge):
    return s >= 4 or (s == 3 and gauge >= 0.020)

def string_profile(s, gauge, f_target, scale_mm):
    """Эвристическая модель: натяжение, жёсткость, масштаб порогов. Требует калибровки на реальных записях."""
    L = scale_mm / 1000.0
    D = gauge * 0.0254
    rho = 7850.0
    wound = is_wound(s, gauge)
    if wound:
        mu = 0.66 * rho * math.pi * D ** 2 / 4      # обмотка легче сплошной стали
        d_core = 0.43 * D
    else:
        mu = rho * math.pi * D ** 2 / 4
        d_core = D
    T = mu * (2 * L * f_target) ** 2                 # Н
    B = math.pi ** 3 * 2.0e11 * d_core ** 4 / (64 * T * L ** 2)   # коэф. инharmоничности
    inh_cents = 1200 * math.log2(math.sqrt((1 + 4 * B) / (1 + B)))  # 2-я гармоника vs 2*F0 (справочно)
    stretch = float(np.clip(60.0 / T, 0.8, 1.5))     # тонкие/слабонатянутые тянутся сильнее
    ref = REF_GAUGES[s]
    buzz_scale = (1.3 if wound else 1.0) * float(np.clip(1 - 10 * (gauge - ref), 0.85, 1.15))
    return {
        'gauge': gauge, 'wound': wound, 'T': T, 'B': B, 'inh_cents': inh_cents,
        'sharp_thr': 10.0 * stretch, 'buzz_scale': buzz_scale, 'scale_mm': scale_mm,
    }

def evaluate_buzz(m, prof):
    if not m or m.get('jump') is None:
        return False
    k = prof['buzz_scale']
    votes = int(m['jump'] > JUMP_THR * k) + int(m['hf'] > HF_THR * k) + int(m['zjump'] > ZJUMP_THR * k)
    return votes >= 2

def saddle_mm(delta_cents, scale_mm):
    """На сколько мм сместить седло, чтобы убрать дельту (длина зажатой части на 12 ладу = L/2)."""
    return (scale_mm / 2) * (2 ** (abs(delta_cents) / 1200) - 1)

# ============ АУДИО-ДВИЖОК ============
class MicRecorder:
    def __init__(self, device=None, channel=0):
        self.channel = channel
        self.sr = SR
        self.stream = None
        self.buf = deque(maxlen=int(SR * BUFFER_SEC / BLOCK) + 4)
        for sr, ch in [(SR, 1), (SR, 2), (48000, 1), (48000, 2), (96000, 1)]:
            try:
                self.stream = sd.InputStream(samplerate=sr, channels=ch, dtype='float32',
                                             blocksize=BLOCK, device=device, callback=self._cb)
                self.sr, self.channels = sr, ch
                break
            except Exception:
                continue
        if self.stream is None:
            raise RuntimeError("Не удалось открыть аудиоустройство")
        self.buf = deque(maxlen=int(self.sr * BUFFER_SEC / BLOCK) + 4)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        ch = min(self.channel, indata.shape[1] - 1)
        self.buf.append(indata[:, ch].copy())

    def stop(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass

    def window(self, seconds=1.5):
        blocks = list(self.buf)
        data = np.concatenate(blocks) if blocks else np.zeros(1, np.float32)
        n = int(self.sr * seconds)
        return data[-n:] if len(data) >= n else np.pad(data, (n - len(data), 0))

    def filled_seconds(self):
        return sum(len(b) for b in list(self.buf)) / self.sr


@st.cache_resource
def get_recorder(device):
    return MicRecorder(device=device, channel=0)

# ============ DSP ============
def highpass(y, sr, cutoff=70):
    b, a = butter(4, cutoff / (sr / 2), 'high')
    return lfilter(b, a, y)

def get_pitch(y, sr, fmin=65, fmax=1200, frame_length=4096):
    """y — уже вырезанный кусок сигнала. Возвращает (F0, confidence)."""
    if len(y) < frame_length + 1024:
        return None, 0.0
    y = highpass(y, sr)[int(sr * 0.03):]   # отбрасываем переходный процесс фильтра
    if len(y) < frame_length + 512 or np.max(np.abs(y)) < 1e-5:
        return None, 0.0
    try:
        f0, _, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                 frame_length=frame_length, hop_length=512)
    except Exception:
        return None, 0.0
    ok = ~np.isnan(f0)
    if not ok.any():
        return None, 0.0
    return float(np.median(f0[ok])), float(np.nanmean(vp[ok]))

def cents(f_meas, f_target):
    if not f_meas or not f_target:
        return 0.0
    return 1200 * np.log2(f_meas / f_target)

def freq_to_note(f):
    if not f or f < 20:
        return None, 0, 0
    midi = 69 + 12 * np.log2(f / 440.0)
    mr = int(round(midi))
    return NOTE_NAMES[mr % 12], (mr // 12) - 1, 440.0 * 2 ** ((mr - 69) / 12)

def find_onset(y, sr):
    k = max(1, int(sr * 0.005))
    env = np.convolve(np.abs(y), np.ones(k) / k, mode='same')
    if env.max() < 1e-5:
        return None
    idx = np.where(env > 0.3 * env.max())[0]
    return int(idx[0]) if len(idx) else None

def frame_feats(y, sr, f0, n=2048):
    """Признаки по кадрам (hop 10 мс): центроид, ZCR, HF/F0, энергия F0."""
    hop = int(sr * 0.01)
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    out = []
    for s0 in range(0, len(y) - n, hop):
        fr = y[s0:s0 + n]
        S = np.abs(np.fft.rfft(fr * win))
        tot = S.sum() + 1e-9
        cent = float((freqs * S).sum() / tot)
        zcr = float(np.mean(np.abs(np.diff(np.sign(fr))) > 0))
        hf = float(S[freqs > 2000].sum())
        fe = float(S[(freqs > f0 * 0.85) & (freqs < f0 * 1.15)].sum()) if f0 else tot
        out.append((cent, zcr, hf / (fe + 1e-9), fe))
    return np.array(out)

def buzz_metrics(y, f0, sr):
    """Сравнивает атаку (первые ~90 мс после onset) с сустейном. Возвращает dict метрик."""
    on = find_onset(y, sr)
    if on is None:
        return None
    atk_seg = y[on:on + int(sr * 0.14)]
    sus_seg = y[on + int(sr * 0.25):]
    if len(atk_seg) < 2048 + 441 or len(sus_seg) < 2048 + 441:
        return None
    A = frame_feats(atk_seg, sr, f0)[:5]
    S = frame_feats(sus_seg, sr, f0)
    if len(A) == 0 or len(S) == 0:
        return None
    return {
        'jump': float(A[:, 0].mean() / (S[:, 0].mean() + 1e-9)),
        'zjump': float(A[:, 1].mean() / (S[:, 1].mean() + 1e-9)),
        'hf': float(A[:, 2].mean()),
        'centroid': float(A[:, 0].mean()),
        'f0_drop': float(A[:, 3].mean() / (S[:, 3].mean() + 1e-9)),
        'onset': on,
        'rms_atk': float(np.sqrt(np.mean(atk_seg[:int(sr * 0.1)] ** 2))),
    }

# ============ LIVE-ЗАХВАТ ============
def live_capture(rec, title, f_target=None, timeout=25, is_harmonic=False, conf_min=0.4):
    st.subheader(title)
    ph = st.empty()
    stable, t0 = [], time.time()
    quiet = RMS_HARM_QUIET if is_harmonic else RMS_QUIET
    sr = rec.sr
    while time.time() - t0 < timeout:
        w = rec.window(1.0)
        rms = float(np.sqrt(np.mean(w[-int(sr * 0.3):] ** 2)))
        fmin = max(60, (f_target or 82) * 0.8)
        fmax = (f_target or 330) * 2.5
        f0, conf = get_pitch(w[-int(sr * 0.6):], sr, fmin=fmin, fmax=fmax)
        c = cents(f0, f_target) if (f0 and f_target) else None
        msg = (f"**F0:** {f'{f0:.1f}' if f0 else '—'} Гц | **Цент:** {f'{c:+.1f}' if c is not None else '—'} | "
               f"**Conf:** {conf:.2f} | **RMS:** {rms:.4f}")
        if rms > RMS_LOUD: msg += "\n\n⚠️ Слишком сильная атака — играйте мягче!"
        elif rms < quiet: msg += f"\n\n🔇 Не слышно. Играйте громче (порог: {quiet:.4f})."
        elif f0 is None: msg += "\n\n🎵 Ищу питч... тяните ноту длиннее."
        ph.markdown(msg)

        if f0 and conf > conf_min and rms > quiet:
            stable.append(c if c is not None else 0)
            if len(stable) >= 6 and np.std(stable[-6:]) < 5:
                ph.markdown(f"✅ **Замер зафиксирован: {f0:.1f} Гц**")
                return {'f0': f0, 'cents': c, 'rms': rms}
        time.sleep(0.2)
    ph.markdown("⏱️ Таймаут. Сыграйте ровнее и попробуйте ещё раз.")
    return None

def capture_fret_buzz(rec, fret, open_f0, prof, timeout=25):
    st.caption(f"Лад {fret}: зажмите струну и ударьте умеренно")
    ph = st.empty()
    bar = st.empty()
    sr = rec.sr
    f_exp = open_f0 * 2 ** (fret / 12)
    prev, t0 = 0.0, time.time()
    while time.time() - t0 < timeout:
        rms = float(np.sqrt(np.mean(rec.window(0.1) ** 2)))
        ph.markdown(f"Лад {fret}: ждём атаку... RMS {rms:.4f}")
        if rms > RMS_QUIET and prev < RMS_QUIET and (time.time() - t0) > 0.5:
            time.sleep(0.9)
            w = rec.window(1.3)
            m = buzz_metrics(w, f_exp, sr)
            if m is None:
                ph.markdown("⚠️ Не удалось выделить атаку. Ударьте ещё раз.")
                prev = 0.0
                time.sleep(1.0)
                continue
            on = m['onset']
            f_sus, _ = get_pitch(w[on + int(sr * 0.25):on + int(sr * 0.75)], sr,
                                 fmin=f_exp * 0.8, fmax=f_exp * 1.25)
            f_atk, _ = get_pitch(w[on:on + int(sr * 0.14)], sr, fmin=f_exp * 0.8,
                                 fmax=f_exp * 1.25, frame_length=2048)
            m['f0'] = f_sus
            m['sharp'] = cents(f_sus, f_exp) if f_sus else 0.0      # завышение vs ожидаемая нота лада
            m['bend'] = cents(f_atk, f_sus) if (f_atk and f_sus) else 0.0
            m['fret'] = fret
            m['too_loud'] = m['rms_atk'] > RMS_LOUD
            buzz = evaluate_buzz(m, prof)
            bar.progress(min(1.0, m['rms_atk'] / RMS_LOUD), text=f"Сила атаки: {m['rms_atk']:.3f}")
            warn = "  \n⚠️ Слишком сильная атака — результат может быть недостоверным." if m['too_loud'] else ""
            ph.markdown(f"Лад {fret}: {'🔴 Дребезг' if buzz else '🟢 Чисто'} "
                        f"(скачок центроида ×{m['jump']:.2f}, HF {m['hf']:.2f}, "
                        f"завышение {m['sharp']:+.1f} ц){warn}")
            return m
        prev = rms
        time.sleep(0.05)
    return {'fret': fret, 'jump': None, 'zjump': None, 'hf': None, 'f0': None,
            'sharp': 0.0, 'bend': 0.0, 'centroid': 0.0, 'too_loud': False}

# ============ ТЮНЕР ============
def render_tuner(rec, conf_min=0.3):
    sr = rec.sr
    w = rec.window(0.8)
    rms = float(np.sqrt(np.mean(w[-int(sr * 0.3):] ** 2)))
    f0, conf = get_pitch(w[-int(sr * 0.5):], sr, fmin=65, fmax=1200)

    if f0 and conf > conf_min and rms > 0.001:
        note, octave, center = freq_to_note(f0)
        c = cents(f0, center)
        if abs(c) < 2: color, status = "#00ff00", "✅ В ноте"
        elif abs(c) < 10: color, status = "#ffaa00", "⚠️ Близко"
        else: color, status = "#ff4444", "❌ Далеко"
        bar_pos = 50.0 + max(-1.0, min(1.0, c / 50)) * 50.0
        st.markdown(f"""
        <div style="background:#1e1e1e;padding:20px;border-radius:10px;text-align:center;">
            <div style="font-size:48px;font-weight:bold;color:{color};">{note}{octave}</div>
            <div style="margin-top:10px;font-size:16px;color:#aaa;">
                {f0:.2f} Гц | <span style="color:{color};font-weight:bold;">{c:+.2f} центов</span> | {status}
            </div>
            <div style="margin-top:20px;height:10px;background:#2a2a2a;border-radius:5px;position:relative;border:1px solid #444;">
                <div style="position:absolute;left:50%;top:-5px;width:2px;height:20px;background:#fff;z-index:2;"></div>
                <div style="position:absolute;left:{bar_pos}%;top:-7px;width:14px;height:24px;background:{color};border-radius:3px;transform:translateX(-50%);box-shadow:0 0 8px {color};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:11px;color:#666;">
                <span>-50¢</span><span>0</span><span>+50¢</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return True
    debug = f"RMS: {rms:.4f} | Conf: {conf:.2f} | SR: {sr} | буфер: {rec.filled_seconds():.1f} с"
    if f0: debug += f" | F0: {f0:.1f} Гц"
    st.markdown(f"""
    <div style="background:#1e1e1e;padding:20px;border-radius:10px;text-align:center;">
        <div style="font-size:48px;font-weight:bold;color:#666;">—</div>
        <div style="margin-top:10px;font-size:16px;color:#888;">Сыграйте ноту для отображения</div>
        <div style="margin-top:10px;font-size:12px;color:#555;">{debug}</div>
    </div>
    """, unsafe_allow_html=True)
    return False

# ============ ВЫБОР СТРУНЫ ============
def get_string_status(s, results, profiles):
    res = results.get(s, {'buzz': {}})
    if 'open' not in res:
        return None
    if 'intonation' not in res or len(res['buzz']) < 4:
        return 'partial'
    err = abs(res['intonation']) > INTON_TOL or abs(res['open'].get('cents') or 0) > TUNE_TOL
    for f in (1, 5, 7, 9):
        if evaluate_buzz(res['buzz'].get(f), profiles[s]):
            err = True
            break
    return 'error' if err else 'success'

def pick_string(current, results, profiles):
    st.markdown("**Выберите струну:**")
    cols = st.columns(6)
    for i in range(6):
        s = 6 - i
        status = get_string_status(s, results, profiles)
        label = {'success': f"✅ Струна {s}", 'error': f"⚠️ Струна {s}",
                 'partial': f"🔧 Струна {s}"}.get(status, f"Струна {s}")
        if cols[i].button(label, key=f"str_{s}", type="primary" if s == current else "secondary",
                          use_container_width=True):
            return s
    return current

# ============ МАТРИЦА РЕШЕНИЙ ============
def recommendations(res, prof):
    out = []
    d = res.get('intonation')
    if d is not None:
        mm = saddle_mm(d, prof['scale_mm'])
        if d > INTON_TOL:
            out.append(f"⚠️ Мензура короткая ({d:+.1f} ц). Сдвиньте седло назад (от грифа) ≈ на {mm:.1f} мм.")
        elif d < -INTON_TOL:
            out.append(f"⚠️ Мензура длинная ({d:+.1f} ц). Сдвиньте седло вперёд (к грифу) ≈ на {mm:.1f} мм.")
        else:
            out.append("✅ Мензура в норме.")
    buzz = res['buzz']
    flags = {f: evaluate_buzz(buzz.get(f), prof) for f in (1, 5, 7, 9)}
    low = [f for f in (1, 5) if flags[f]]
    mid = [f for f in (7, 9) if flags[f]]
    if len(low) == 2 and len(mid) == 2:
        out.append("⚠️ Дребезг на всех ладах: струны слишком низко. Поднимите седла на бридже.")
    elif low and not mid:
        out.append("⚠️ Дребезг у головы грифа: слегка ослабьте анкер (добавить прогиб) или проверьте высоту верхнего порожка.")
    elif mid and not low:
        out.append("⚠️ Дребезг в середине: проверьте прогиб (возможно, подтянуть анкер) или поднимите седла.")
    sharp = [v['sharp'] for v in buzz.values() if v.get('sharp') is not None and v.get('f0')]
    if len(sharp) >= 3 and sum(1 for x in sharp if x > prof['sharp_thr']) >= 3:
        out.append(f"⚠️ Завышение питча при нажатии > {prof['sharp_thr']:.0f} ц на нескольких ладах: "
                   "слишком высокая action. Опустите седла.")
    if any(v.get('too_loud') for v in buzz.values()):
        out.append("ℹ️ На некоторых ладах атака была слишком сильной — перемерьте их мягче.")
    if len(out) == 1 and "норме" in out[0]:
        out.append("✅ Проблем по action/анкеру не обнаружено.")
    return out

# ============ UI / МАСТЕР ============
st.set_page_config(page_title="Guitar Setup Analyzer", layout="wide")
st.title("🎸 Guitar Setup Analyzer (MVP)")

if sd is None:
    st.error("Не найден sounddevice. Выполните: py -m pip install sounddevice")
    st.stop()

for k, v in (('results', {}), ('step', 0), ('string', 6), ('current_fret', None)):
    if k not in st.session_state:
        st.session_state[k] = v

all_devs = sd.query_devices()
input_devs = [(i, d) for i, d in enumerate(all_devs) if d['max_input_channels'] > 0]
if not input_devs:
    st.error("❌ Нет устройств ввода.")
    st.stop()

def dev_label(idx):
    i, d = input_devs[idx]
    try:
        host = sd.query_hostapi_info(d['hostapi'])['name']
    except Exception:
        host = f"host{d['hostapi']}"
    return f"[{i}] {d['name'][:40]} — {host}"

default_pos = 0
for pos, (i, d) in enumerate(input_devs):
    try:
        h = sd.query_hostapi_info(d['hostapi'])['name']
        if 'WASAPI' in h and 'BEHRINGER' in d['name'].upper():
            default_pos = pos
            break
    except Exception:
        pass

# ---- Сайдбар: устройство ----
sel = st.sidebar.selectbox("Микрофон/вход", range(len(input_devs)), index=default_pos, format_func=dev_label)
device_index = input_devs[sel][0]
channel = st.sidebar.radio("Канал", [0, 1], index=0, help="0 = INSTRUMENT/Line 1, 1 = Line 2")
st.sidebar.caption(f"Устройство №{device_index}: {input_devs[sel][1]['name']}")
if st.sidebar.button("♻️ Перезапустить аудио"):
    st.cache_resource.clear()
    st.rerun()

# ---- Сайдбар: инструмент ----
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎸 Инструмент")
tuning_name = st.sidebar.selectbox("Строй", list(TUNINGS.keys()))
scale_name = st.sidebar.selectbox("Мензура", list(SCALES.keys()))
scale_mm = SCALES[scale_name]
set_name = st.sidebar.selectbox("Набор струн", list(GAUGE_SETS.keys()), index=1)
gauges = dict(zip(range(1, 7), GAUGE_SETS[set_name]))
if st.sidebar.checkbox("Задать калибры вручную"):
    for s in range(1, 7):
        gauges[s] = st.sidebar.number_input(f"Струна {s} (дюйм)", 0.007, 0.080, float(gauges[s]),
                                            step=0.001, format="%.3f", key=f"g_{s}")
fresh = st.sidebar.checkbox("Струны свежие (менее 2 недель)", value=True)
if not fresh:
    st.sidebar.warning("Старые струны дают негармоничные обертоны: точность питча и детектора дребезга снижена.")
conf_min = 0.4 if fresh else 0.3

targets = {s: STANDARD_TUNING[s] * 2 ** (TUNINGS[tuning_name][s] / 12) for s in range(1, 7)}
profiles = {s: string_profile(s, gauges[s], targets[s], scale_mm) for s in range(1, 7)}

# ---- Рекордер ----
try:
    rec = get_recorder(device_index)
    rec.channel = channel
except Exception as e:
    st.error(f"Не удалось открыть устройство: {e}")
    st.stop()

if rec.filled_seconds() < 0.8:
    time.sleep(0.9)

try:
    st.markdown("### 🎵 Онлайн-тюнер")
    st.caption("Сыграйте любую ноту — тюнер покажет частоту и отклонение." if st.session_state.step == 0
               else "Тюнер показывает последнее значение. Настройте гитару при необходимости.")
    render_tuner(rec, conf_min=0.3)

    if st.session_state.step == 0:
        st.markdown("---")
        st.markdown("**Готовы приступить к диагностике?**")
        st.caption("Убедитесь, что гитара настроена. Мастер проведёт вас через проверку настройки, мензуры и дребезга.")
        if st.button("🎸 Приступить к диагностике", type="primary"):
            st.session_state.step = 1
            st.rerun()
        time.sleep(0.3)
        st.rerun()

    else:
        st.markdown("---")
        string_num = pick_string(st.session_state.string, st.session_state.results, profiles)
        if string_num != st.session_state.string:
            st.session_state.string = string_num
            st.session_state.step = 1
            st.session_state.current_fret = None
            st.session_state.results.setdefault(string_num, {'buzz': {}})
            st.rerun()

        target = targets[string_num]
        prof = profiles[string_num]
        res = st.session_state.results.setdefault(string_num, {'buzz': {}})
        st.caption(f"Струна {string_num}: калибр {prof['gauge']:.3f}\" "
                   f"({'обмоточная' if prof['wound'] else 'простая'}), цель {target:.2f} Гц, "
                   f"расчётное натяжение ≈ {prof['T']:.0f} Н, порог завышения {prof['sharp_thr']:.0f} ц")

        # ---------- ШАГ 1: строй ----------
        if st.session_state.step == 1:
            if 'open' not in res:
                r = live_capture(rec, f"Шаг 1. Сыграйте открытую {string_num}-ю струну", target, conf_min=conf_min)
                if r:
                    res['open'] = r
                    st.rerun()
            else:
                c0 = res['open'].get('cents') or 0
                if abs(c0) <= TUNE_TOL:
                    st.success(f"✅ Строй в норме: {res['open']['f0']:.2f} Гц ({c0:+.1f} ц)")
                    st.markdown("---")
                    st.markdown("**Готовы перейти к Шагу 2 (диагностика мензуры)?**")
                    st.caption("Сравним флажолет на 12 ладу с зажатой нотой на 12 ладу.")
                    if st.button("✅ Да, перейти к Шагу 2 →", type="primary"):
                        st.session_state.step = 2
                        st.rerun()
                else:
                    if c0 < 0:
                        hint = "Струна ниже нужной ноты — подтяните колок (увеличьте натяжение)."
                    else:
                        hint = "Струна выше нужной ноты — ослабьте колок (уменьшите натяжение)."
                    st.error(f"❌ Отклонение {c0:+.1f} ц (допуск ±{TUNE_TOL:.0f} ц). {hint}")
                    st.caption("Переход к Шагу 2 заблокирован до настройки. Подстройте и перемерьте.")
                    if st.button("🔄 Перемерить после настройки", type="primary"):
                        res.pop('open', None)
                        st.rerun()

        # ---------- ШАГ 2а: флажолет ----------
        elif st.session_state.step == 2:
            if 'open' not in res:
                st.session_state.step = 1
                st.rerun()
            open_f0 = res['open']['f0']
            st.markdown(f"📏 Референс открытой: **{open_f0:.2f} Гц** ({(res['open'].get('cents') or 0):+.1f} ц)")
            st.caption("Абсолютная точность не важна — сравниваем флажолет и зажатую ноту между собой.")
            if 'harm' not in res:
                r1 = live_capture(rec, "Шаг 2а. Флажолет на 12 ладу", open_f0 * 2, is_harmonic=True, conf_min=conf_min)
                if r1:
                    res['harm'] = r1
                    st.rerun()
            else:
                st.success(f"✅ Флажолет зафиксирован: **{res['harm']['f0']:.2f} Гц**")
                st.markdown("---")
                st.markdown("**Готовы перейти к Шагу 2б (зажатая нота на 12 ладу)?**")
                st.caption("Зажмите струну на 12 ладу и сыграйте. Не давите слишком сильно!")
                if st.button("✅ Да, перейти к Шагу 2б →", type="primary"):
                    st.session_state.step = 2.5
                    st.rerun()

        # ---------- ШАГ 2б: зажатая ----------
        elif st.session_state.step == 2.5:
            open_f0 = res['open']['f0']
            if 'harm' not in res:
                r1 = live_capture(rec, "Шаг 2а. Флажолет на 12 ладу", open_f0 * 2, is_harmonic=True, conf_min=conf_min)
                if r1:
                    res['harm'] = r1
                    st.rerun()
            else:
                st.markdown(f"✅ Флажолет: **{res['harm']['f0']:.2f} Гц**")
                if 'fretted' not in res:
                    r2 = live_capture(rec, "Шаг 2б. Зажатая нота на 12 ладу", open_f0 * 2, conf_min=conf_min)
                    if r2:
                        res['fretted'] = r2
                        res['intonation'] = cents(r2['f0'], res['harm']['f0'])
                        st.rerun()
                else:
                    st.markdown("### Результаты Шага 2")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Флажолет F0", f"{res['harm']['f0']:.2f} Гц")
                    c2.metric("Зажатая F0", f"{res['fretted']['f0']:.2f} Гц")
                    c3.metric("Дельта мензуры", f"{res['intonation']:+.1f} центов")
                    if abs(res['intonation']) > INTON_TOL:
                        st.info(f"Ориентир: сместить седло ≈ на {saddle_mm(res['intonation'], scale_mm):.1f} мм "
                                f"({'назад' if res['intonation'] > 0 else 'вперёд'}).")
                    if abs(res['intonation']) < 0.5:
                        st.warning("⚠️ Дельта ≈ 0. Подозрительно — возможно, обе ноты записались одинаково. Перемерьте.")
                    else:
                        st.success("✅ Шаг 2 завершён.")
                    if st.button("🔄 Перемерить Шаг 2"):
                        for k in ('harm', 'fretted', 'intonation'):
                            res.pop(k, None)
                        st.rerun()
                    st.markdown("---")
                    st.markdown("**Готовы перейти к Шагу 3 (диагностика дребезга)?**")
                    st.caption("Проверим лады 1, 5, 7, 9.")
                    if st.button("✅ Да, перейти к Шагу 3 →", type="primary"):
                        st.session_state.step = 3
                        st.rerun()

        # ---------- ШАГ 3: дребезг ----------
        elif st.session_state.step == 3:
            st.markdown("### Шаг 3. Диагностика дребезга")
            st.caption("Играйте с умеренным нажатием. Нажимайте кнопку только когда готовы сыграть.")
            frets_sequence = [1, 5, 7, 9]
            open_f0 = res['open']['f0']

            if st.session_state.current_fret is None:
                st.markdown("---")
                st.markdown("**Готовы начать диагностику дребезга?**")
                if st.button("▶️ Начать с лада 1", type="primary"):
                    st.session_state.current_fret = 1
                    st.rerun()
            else:
                cf = st.session_state.current_fret
                if cf not in res['buzz']:
                    b = capture_fret_buzz(rec, cf, open_f0, prof)
                    if b:
                        res['buzz'][cf] = b
                        st.rerun()
                else:
                    b = res['buzz'][cf]
                    buzz = evaluate_buzz(b, prof)
                    st.markdown(f"**Лад {cf}:** {'🔴 ДРЕБЕЗГ' if buzz else '🟢 Чисто'}")
                    if b.get('jump') is None:
                        st.caption("Атака не была зафиксирована — перемерьте.")
                    else:
                        st.caption(f"Скачок центроида ×{b['jump']:.2f} | HF/F0 {b['hf']:.2f} | "
                                   f"ZCR ×{b['zjump']:.2f} | завышение {b.get('sharp', 0):+.1f} ц | "
                                   f"bend {b.get('bend', 0):+.1f} ц")
                    if b.get('too_loud'):
                        st.warning("Слишком сильная атака — сыграйте мягче и перемерьте.")
                    if st.button(f"🔄 Перемерить лад {cf}"):
                        del res['buzz'][cf]
                        st.rerun()
                    st.markdown("---")
                    next_fret = next((f for f in frets_sequence if f not in res['buzz']), None)
                    if next_fret is not None:
                        st.markdown(f"**Готовы перейти к ладу {next_fret}?**")
                        if st.button(f"▶️ Перейти к ладу {next_fret}", type="primary"):
                            st.session_state.current_fret = next_fret
                            st.rerun()
                    else:
                        st.success("✅ Все лады измерены!")
                        if st.button("📊 Показать итоги", type="primary"):
                            st.session_state.step = 4
                            st.session_state.current_fret = None
                            st.rerun()

        # ---------- ШАГ 4: итоги ----------
        elif st.session_state.step == 4:
            st.markdown("### Итоги диагностики")
            if 'open' in res:
                st.metric("Строй", f"{(res['open'].get('cents') or 0):+.1f} центов")
            if 'intonation' in res:
                st.metric("Дельта мензуры", f"{res['intonation']:+.1f} центов")
            if len(res['buzz']) >= 4:
                st.markdown("**Тепловая карта:**")
                cols = st.columns(4)
                for i, f in enumerate([1, 5, 7, 9]):
                    b = res['buzz'].get(f)
                    cols[i].markdown(f"### Лад {f}")
                    cols[i].markdown("🔴 ДРЕБЕЗГ" if evaluate_buzz(b, prof) else "🟢 Чисто")
                st.markdown("### Рекомендации")
                for r in recommendations(res, prof):
                    st.info(r)
            if st.button("🔄 Начать заново для этой струны"):
                st.session_state.results.pop(string_num, None)
                st.session_state.step = 1
                st.session_state.current_fret = None
                st.rerun()

except Exception as e:
    if type(e).__name__ in ("RerunException", "StopException"):
        raise
    import traceback
    st.error(f"Ошибка захвата: {e}")
    st.code(traceback.format_exc())
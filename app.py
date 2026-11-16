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
RMS_HARM_QUIET = 0.001  # флажолет тише обычной ноты
STANDARD_TUNING = {6: 82.41, 5: 110.00, 4: 146.83, 3: 196.00, 2: 246.94, 1: 329.63}

# ============ АУДИО-ДВИЖОК ============
class MicRecorder:
    def __init__(self, device=None, channel=0):
        self.channel = channel
        self.buf = deque(maxlen=int(SR * BUFFER_SEC))
        configs = [(SR, 1), (SR, 2), (48000, 1), (48000, 2), (96000, 1)]
        for sr, ch in configs:
            try:
                self.stream = sd.InputStream(samplerate=sr, channels=ch, dtype='float32',
                                             blocksize=BLOCK, device=device, callback=self._cb)
                self.sr = sr
                self.channels = ch
                break
            except sd.PortAudioError:
                continue
        else:
            raise RuntimeError("Не удалось открыть аудиоустройство")

    def _cb(self, indata, frames, t, status):
        if indata.shape[1] > 1:
            ch = min(self.channel, indata.shape[1] - 1)
            self.buf.append(indata[:, ch].astype(np.float32))
        else:
            self.buf.append(indata[:, 0].copy())

    def start(self): self.stream.start()
    def stop(self):
        try: self.stream.stop(); self.stream.close()
        except Exception: pass

    def window(self, seconds=1.5):
        data = np.concatenate(list(self.buf)) if self.buf else np.zeros(1, np.float32)
        n = int(self.sr * seconds)
        return data[-n:] if len(data) >= n else np.pad(data, (0, n - len(data)))

# ============ DSP ============
def highpass(y, cutoff=70):
    b, a = butter(5, cutoff / (SR / 2), 'high')
    return lfilter(b, a, y)

def get_pitch(y, fmin=65, fmax=1200):
    y = highpass(y)
    start = int(SR * 0.06)
    seg = y[start:start + int(SR * 0.45)]
    if len(seg) < SR * 0.2:
        return None, 0.0
    f0, _, vp = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=SR, frame_length=2048)
    if np.all(np.isnan(f0)):
        return None, 0.0
    return float(np.nanmedian(f0)), float(np.nanmean(vp[~np.isnan(vp)]))

def cents(f_meas, f_target):
    if not f_meas or not f_target: return 0.0
    return 1200 * np.log2(f_meas / f_target)

def buzz_metrics(y, f0):
    atk = y[:int(SR * 0.12)]
    cent = float(librosa.feature.spectral_centroid(atk, sr=SR)[0].mean())
    zcr = float(librosa.feature.zero_crossing_rate(atk)[0].mean())
    S = np.abs(librosa.stft(atk))
    fr = librosa.fft_frequencies(sr=SR)
    he = float(S[fr > 2000].sum())
    fe = float(S[(fr > f0 * 0.9) & (fr < f0 * 1.1)].sum()) if f0 else 1.0
    ratio = he / (fe + 1e-6)
    is_buzz = cent > 1500 and zcr > 0.1 and ratio > 0.5
    return is_buzz, cent, zcr, ratio

# ============ LIVE-ЗАХВАТ ============
def live_capture(rec, title, f_target=None, timeout=25, is_harmonic=False):
    """is_harmonic=True снижает порог RMS для флажолетов."""
    st.subheader(title)
    ph = st.empty()
    stable, t0 = [], time.time()
    quiet_threshold = RMS_HARM_QUIET if is_harmonic else RMS_QUIET
    
    while time.time() - t0 < timeout:
        w = rec.window(1.5)
        rms = float(np.sqrt(np.mean(w ** 2)))
        fmin = max(65, (f_target or 82) * 0.8)
        f0, conf = get_pitch(w, fmin=fmin, fmax=(f_target or 330) * 3)
        c = cents(f0, f_target) if (f0 and f_target) else None
        f0_str = f"{f0:.1f}" if f0 is not None else "—"
        c_str = f"{c:+.1f}" if c is not None else "—"
        msg = f"**F0:** {f0_str} Гц | **Цент:** {c_str} | **Conf:** {conf:.2f} | **RMS:** {rms:.4f}"
        if rms > RMS_LOUD: msg += "\n\n⚠️ Слишком сильная атака — играйте мягче!"
        elif rms < quiet_threshold: msg += f"\n\n🔇 Не слышно. Играйте громче (порог: {quiet_threshold:.4f})."
        elif f0 is None: msg += "\n\n🎵 Ищу питч... тяните ноту длиннее."
        ph.markdown(msg)
        
        if f0 and conf > 0.4 and rms > quiet_threshold:
            stable.append(c if c is not None else 0)
            if len(stable) >= 6 and np.std(stable[-6:]) < 5:
                ph.markdown(f"✅ **Замер зафиксирован: {f0:.1f} Гц**")
                return {'f0': f0, 'cents': c, 'rms': rms}
        time.sleep(0.25)
    ph.markdown("⏱️ Таймаут. Сыграйте ровнее и попробуйте ещё раз.")
    return None

def capture_fret_buzz(rec, fret, timeout=25):
    st.caption(f"Лад {fret}: ударьте по струне")
    ph = st.empty()
    prev, t0 = 0.0, time.time()
    while time.time() - t0 < timeout:
        w = rec.window(1.5)
        rms = float(np.sqrt(np.mean(w ** 2)))
        ph.markdown(f"Лад {fret}: ждём атаку... RMS {rms:.4f}")
        if rms > RMS_QUIET and prev < RMS_QUIET:
            time.sleep(0.5)
            w2 = rec.window(0.8)
            f0, _ = get_pitch(w2)
            is_buzz, cent, zcr, ratio = buzz_metrics(w2, f0)
            bend = 0.0
            f_atk, _ = get_pitch(w2[:int(SR * 0.08)])
            if f0 and f_atk: bend = cents(f_atk, f0)
            ph.markdown(f"Лад {fret}: {'🔴 Дребезг' if is_buzz else '🟢 Чисто'} "
                        f"(центроид {cent:.0f}, ratio {ratio:.2f}, bend {bend:+.1f})")
            return {'fret': fret, 'is_buzz': is_buzz, 'bend': bend, 'f0': f0}
        prev = rms
        time.sleep(0.1)
    return {'fret': fret, 'is_buzz': False, 'bend': 0.0, 'f0': None}

# ============ ВЫБОР СТРУНЫ С ПОДСВЕТКОЙ ============
def pick_string(current):
    st.markdown("**Выберите струну:**")
    cols = st.columns(6)
    for i in range(6):
        s = 6 - i
        btn_type = "primary" if s == current else "secondary"
        if cols[i].button(f"Струна {s}", key=f"str_{s}", type=btn_type, use_container_width=True):
            return s
    return current

# ============ МАТРИЦА РЕШЕНИЙ ============
def recommendations(res):
    out = []
    d = res.get('intonation')
    if d is not None:
        if d > 3: out.append(f"⚠️ Мензура короткая (+{d:.1f} ц). Сдвиньте седло назад на 1-2 мм.")
        elif d < -3: out.append(f"⚠️ Мензура длинная ({d:.1f} ц). Сдвиньте седло вперёд.")
        else: out.append("✅ Мензура в норме.")
    buzz_low = [f for f in (1, 5) if res['buzz'].get(f, {}).get('is_buzz')]
    buzz_mid = [f for f in (7, 9) if res['buzz'].get(f, {}).get('is_buzz')]
    if buzz_low and not buzz_mid:
        out.append("⚠️ Дребезг у головы: ослабьте анкер или проверьте верхний порожек.")
    if buzz_mid and not buzz_low:
        out.append("⚠️ Дребезг в середине: подтяните анкер или поднимите седла.")
    bends = [v['bend'] for v in res['buzz'].values() if v.get('bend', 0) > 10]
    if len(bends) >= 3:
        out.append("⚠️ Завышение питча при нажатии: слишком высокая action.")
    if len(out) == 1 and "норме" in out[0]:
        out.append("✅ Проблем по action/анкеру не обнаружено.")
    return out

# ============ UI / МАСТЕР ============
st.set_page_config(page_title="Guitar Setup Analyzer", layout="wide")
st.title("🎸 Guitar Setup Analyzer (MVP)")

if sd is None:
    st.error("Не найден sounddevice. Выполните: py -m pip install sounddevice")
    st.stop()

# Инициализация состояния
if 'results' not in st.session_state: st.session_state.results = {}
if 'step' not in st.session_state: st.session_state.step = 1
if 'string' not in st.session_state: st.session_state.string = 6
if 'substep' not in st.session_state: st.session_state.substep = 'a'  # для шага 2

# ============ ВЫБОР УСТРОЙСТВА ============
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
            default_pos = pos; break
    except Exception: pass

sel = st.sidebar.selectbox("Микрофон/вход", range(len(input_devs)),
                           index=default_pos, format_func=dev_label)
device_index = input_devs[sel][0]
channel = st.sidebar.radio("Канал", [0, 1], index=0,
                           help="0 = INSTRUMENT/Line 1, 1 = Line 2")
st.sidebar.caption(f"Устройство №{device_index}: {input_devs[sel][1]['name']}")

# ============ ВЫБОР СТРУНЫ (всегда виден) ============
string_num = pick_string(st.session_state.string)
if string_num != st.session_state.string:
    st.session_state.string = string_num
    # Сброс при смене струны
    st.session_state.step = 1
    st.session_state.substep = 'a'
    if string_num not in st.session_state.results:
        st.session_state.results[string_num] = {'buzz': {}}

target = STANDARD_TUNING[string_num]
res = st.session_state.results.setdefault(string_num, {'buzz': {}})
fresh = st.sidebar.checkbox("Струны свежие (<2 недель)", value=True)
if not fresh:
    st.sidebar.warning("⚠️ На старых струнах точность падает.")

# ============ ОТЛАДОЧНАЯ ПАНЕЛЬ ============
with st.sidebar.expander("🔧 Отладка", expanded=False):
    st.write(f"**Текущая струна:** {string_num}")
    st.write(f"**Шаг:** {st.session_state.step}")
    st.write(f"**Подшаг:** {st.session_state.substep}")
    st.write(f"**res keys:** {list(res.keys())}")
    if 'open' in res: st.write(f"  open.f0 = {res['open']['f0']:.2f}")
    if 'harm' in res: st.write(f"  harm.f0 = {res['harm']['f0']:.2f}")
    if 'fretted' in res: st.write(f"  fretted.f0 = {res['fretted']['f0']:.2f}")
    if 'intonation' in res: st.write(f"  intonation = {res['intonation']:+.1f} ц")
    st.write(f"**buzz:** {list(res['buzz'].keys())}")
    if st.button("🔄 Сбросить состояние"):
        st.session_state.step = 1
        st.session_state.substep = 'a'
        if string_num in st.session_state.results:
            del st.session_state.results[string_num]
        st.rerun()

# ============ ЗАПУСК АУДИО ============
rec = MicRecorder(device=device_index, channel=channel)
rec.start()

try:
    # ШАГ 1: Открытая струна
    if st.session_state.step == 1:
        if 'open' not in res:
            r = live_capture(rec, f"Шаг 1. Сыграйте открытую {string_num}-ю струну", target)
            if r:
                res['open'] = r
                st.session_state.step = 2
                st.session_state.substep = 'a'
                st.rerun()
        else:
            st.success(f"✅ Открытая струна: {res['open']['f0']:.2f} Гц")
            if st.button("Перейти к Шагу 2 →"):
                st.session_state.step = 2
                st.session_state.substep = 'a'
                st.rerun()

    # ШАГ 2: Мензура
    elif st.session_state.step == 2:
        open_f0 = res['open']['f0']
        open_cents = res['open'].get('cents', 0) or 0
        st.markdown(f"📏 Референс открытой: **{open_f0:.2f} Гц** ({open_cents:+.1f} центов)")
        st.caption("Абсолютная точность не важна — сравниваем флажолет и зажатую ноту между собой.")
        
        # 2а: Флажолет
        if st.session_state.substep == 'a':
            if 'harm' not in res:
                r1 = live_capture(rec, "Шаг 2а. Флажолет на 12 ладу", open_f0 * 2, is_harmonic=True)
                if r1:
                    res['harm'] = r1
                    st.session_state.substep = 'b'
                    st.rerun()
            else:
                st.success(f"✅ Флажолет: {res['harm']['f0']:.2f} Гц")
                if st.button("Перейти к Шагу 2б →"):
                    st.session_state.substep = 'b'
                    st.rerun()
        
        # 2б: Зажатая нота
        elif st.session_state.substep == 'b':
            st.markdown(f"✅ Флажолет зафиксирован: **{res['harm']['f0']:.2f} Гц**")
            if 'fretted' not in res:
                r2 = live_capture(rec, "Шаг 2б. Зажатая нота на 12 ладу", open_f0 * 2)
                if r2:
                    res['fretted'] = r2
                    res['intonation'] = cents(r2['f0'], res['harm']['f0'])
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.success(f"✅ Зажатая нота: {res['fretted']['f0']:.2f} Гц")
                st.markdown(f"**Дельта мензуры:** {res['intonation']:+.1f} центов")
                if st.button("Перейти к Шагу 3 →"):
                    st.session_state.step = 3
                    st.rerun()

    # ШАГ 3: Дребезг
    elif st.session_state.step == 3:
        st.markdown("### Шаг 3. Диагностика дребезга")
        st.markdown("Играйте по очереди на ладах **1, 5, 7, 9** с умеренным нажатием.")
        
        for f in [1, 5, 7, 9]:
            if f not in res['buzz']:
                b = capture_fret_buzz(rec, f)
                if b:
                    res['buzz'][f] = b
                    st.rerun()
        
        # Все 4 лада измерены
        st.session_state.step = 4
        st.rerun()

except Exception as e:
    st.error(f"Ошибка захвата: {e}")
    import traceback
    st.code(traceback.format_exc())
finally:
    rec.stop()

# ============ ИТОГИ ============
if 'open' in res:
    open_c = res['open'].get('cents', 0) or 0
    st.metric("Строй", f"{open_c:+.1f} центов")
if 'intonation' in res:
    st.metric("Дельта мензуры", f"{res['intonation']:+.1f} центов")
if len(res['buzz']) >= 4:
    st.markdown("**Тепловая карта:**")
    cols = st.columns(4)
    for i, f in enumerate([1, 5, 7, 9]):
        b = res['buzz'].get(f, {})
        cols[i].markdown(f"### Лад {f}")
        cols[i].markdown("🔴 ДРЕБЕЗГ" if b.get('is_buzz') else "🟢 Чисто")
    st.markdown("### Рекомендации")
    for r in recommendations(res):
        st.info(r)
    if st.button("🔄 Начать заново для этой струны"):
        del st.session_state.results[string_num]
        st.session_state.step = 1
        st.session_state.substep = 'a'
        st.rerun()
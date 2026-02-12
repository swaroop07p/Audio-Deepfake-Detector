import librosa
import numpy as np
import os
import scipy.stats
import logging
import gc 

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- BASELINE STATISTICS ---
HUMAN_BASELINE = {
    "pitch_jitter": (0.012, 0.007),
    "silence_ratio": (0.14, 0.11),
    "mfcc_consistency": (850, 320),
    "cepstral_peak": (15.5, 4.5),
    "spectral_entropy": (4.5, 1.6),
}

def calculate_anomaly_score(value, baseline_mean, baseline_std):
    try:
        z_score = abs(value - baseline_mean) / (baseline_std + 1e-6)
        probability = (scipy.stats.norm.cdf(z_score) - 0.5) * 200
        return min(max(probability, 0), 99)
    except:
        return 0

def calculate_cepstral_peak(y, sr):
    try:
        S = np.abs(librosa.stft(y))
        cepstrum = np.fft.ifft(np.log(S + 1e-6), axis=0).real
        quefrency_axis = np.fft.fftfreq(cepstrum.shape[0], d=1/sr)
        valid_mask = (quefrency_axis > 0.002) & (quefrency_axis < 0.015)
        if not np.any(valid_mask): return 0
        peak_val = np.max(np.abs(cepstrum[valid_mask, :]))
        return peak_val * 1000
    except:
        return 0

async def analyze_audio_forensics(file_upload, filename: str):
    temp_filename = f"temp_{filename}"
    
    try:
        # 1. Save File
        content = await file_upload.read()
        with open(temp_filename, "wb") as f:
            f.write(content)
        
        # Free memory
        del content
        gc.collect()

        # 2. ULTRA-LIGHT LOAD
        # Load only 5 seconds. This drastically reduces the memory footprint.
        # AI artifacts are usually present throughout the file, so 5s is enough.
        duration_limit = 5
        y, sr = librosa.load(temp_filename, sr=16000, duration=duration_limit)
        y = librosa.util.normalize(y)

        # --- FEATURE 1: PITCH JITTER (The RAM Killer) ---
        # Optimization: Analyze ONLY the center 1.0 second for pitch.
        # This prevents the transition matrix from eating all RAM.
        mid_point = len(y) // 2
        slice_len = int(1.0 * sr) # 1 second
        start = max(0, mid_point - slice_len // 2)
        end = min(len(y), mid_point + slice_len // 2)
        y_slice = y[start:end]

        f0, _, _ = librosa.pyin(y_slice, fmin=60, fmax=500, sr=sr)
        
        pitch_jitter = 0.0
        if f0 is not None:
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 5: # Relaxed count check
                pitch_jitter = (np.mean(np.abs(np.diff(f0))) / np.mean(f0))

        # Cleanup immediately
        del f0, y_slice
        gc.collect()

        # --- FEATURE 2: SPECTRAL & CEPSTRAL ---
        cpp_val = calculate_cepstral_peak(y, sr)

        S = np.abs(librosa.stft(y))
        psd = np.mean(S**2, axis=1)
        psd_norm = psd / (np.sum(psd) + 1e-6)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfcc, axis=1))

        # --- FEATURE 3: SILENCE ---
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        non_silent_dur = sum(end - start for start, end in non_silent_intervals) / sr
        total_dur = librosa.get_duration(y=y, sr=sr)
        
        silence_ratio = 0.0
        if total_dur > 0:
            silence_ratio = (total_dur - non_silent_dur) / total_dur

        # --- SCORING ---
        scores = {}
        for feature_name, value in [
            ("pitch_jitter", pitch_jitter),
            ("cepstral_peak", cpp_val),
            ("spectral_entropy", spectral_entropy),
            ("silence_ratio", silence_ratio),
            ("mfcc_consistency", mfcc_var)
        ]:
            baseline_mean, baseline_std = HUMAN_BASELINE[feature_name]
            scores[feature_name] = calculate_anomaly_score(value, baseline_mean, baseline_std)

        # Weighted Calculation
        final_fake_prob = (
            (scores["pitch_jitter"] * 0.20) +
            (scores["cepstral_peak"] * 0.25) +
            (scores["spectral_entropy"] * 0.20) +
            (scores["silence_ratio"] * 0.15) +
            (scores["mfcc_consistency"] * 0.20)
        )

        # Logic Adjustments
        if pitch_jitter < 0.002: final_fake_prob += 10
        if cpp_val < 3.0: final_fake_prob += 10
        
        final_fake_prob = min(max(final_fake_prob, 2), 98)

        if final_fake_prob > 60:
            verdict = "AI/Synthetic"
        else:
            verdict = "Real Human"

        reasons = []
        if verdict == "AI/Synthetic":
            if scores["pitch_jitter"] > 50: reasons.append("Unnatural pitch stability.")
            if scores["cepstral_peak"] > 50: reasons.append("Weak vocal resonance.")
            if scores["mfcc_consistency"] > 50: reasons.append("Flat vocal texture.")
        else:
            reasons = ["Organic pitch fluctuations.", "Natural harmonic structure."]

        # FINAL CLEANUP
        del y, S, mfcc
        gc.collect()

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        return {
            "verdict": verdict,
            "confidence_score": float(round(final_fake_prob, 2)),
            "reasons": reasons,
            "features": {
                "jitter": float(round(pitch_jitter, 5)),
                "cepstral_peak": float(round(cpp_val, 2)),
                "spectral_entropy": float(round(spectral_entropy, 3)),
                "silence_ratio": float(round(silence_ratio, 3))
            },
            "metadata": {"sample_rate": int(sr), "duration": float(round(total_dur, 2))}
        }

    except Exception as e:
        logger.error(f"Forensics Error: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return {
            "verdict": "Error",
            "confidence_score": 0.0,
            "reasons": ["Analysis failed (Server Busy)"],
            "features": {},
            "metadata": {}
        }
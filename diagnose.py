"""Diagnose audio quality before deciding on preprocessing.

Computes:
  - Basic file info (duration, sample rate, bitrate) via ffprobe
  - RMS energy and peak amplitude
  - Silence ratio (fraction of file below a noise floor threshold)
  - SNR estimate (ratio of loud frames to quiet frames)
  - Spectral centroid mean (proxy for speech vs. noise content)
  - Saves waveform + mel spectrogram as a PNG for visual inspection

Run this on a problematic file AND a good file for comparison.
The output tells you which preprocessing step to apply.
"""
import json
import os
import subprocess


# ── 1. ffprobe ──────────────────────────────────────────────────────────────

def ffprobe_info(audio_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    info = json.loads(result.stdout)
    audio_stream = next(
        (s for s in info.get("streams", []) if s["codec_type"] == "audio"), {}
    )
    fmt = info.get("format", {})
    return {
        "duration_s":   float(fmt.get("duration", 0)),
        "bitrate_kbps": int(fmt.get("bit_rate", 0)) // 1000,
        "codec":        audio_stream.get("codec_name", "unknown"),
        "sample_rate":  int(audio_stream.get("sample_rate", 0)),
        "channels":     int(audio_stream.get("channels", 0)),
    }


# ── 2. Signal metrics ────────────────────────────────────────────────────────

def signal_metrics(audio_path: str, sr: int = 16000,
                   offset: float = 0.0,
                   max_duration: float = 300.0) -> dict:
    """Load max_duration seconds starting at offset and compute diagnostics."""
    import numpy as np
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, mono=True,
                        offset=offset, duration=max_duration)

    # frame-level RMS (25ms frames, 10ms hop — standard speech analysis window)
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]

    # silence threshold: frames below 1% of max RMS are "silent"
    threshold = rms_frames.max() * 0.01
    silence_ratio = (rms_frames < threshold).mean()

    # SNR estimate: compare top 10% loudest frames (speech) to bottom 10% (noise)
    sorted_rms = np.sort(rms_frames)
    n = len(sorted_rms)
    noise_floor = sorted_rms[:max(1, n // 10)].mean()
    speech_level = sorted_rms[-(n // 10):].mean()
    snr_db = 20 * np.log10(speech_level / (noise_floor + 1e-10))

    # spectral centroid: higher = more high-frequency content (speech-like)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]

    return {
        "rms_mean":        float(rms_frames.mean()),
        "rms_max":         float(rms_frames.max()),
        "peak_amplitude":  float(abs(y).max()),
        "silence_ratio":   float(silence_ratio),
        "snr_db_est":      float(snr_db),
        "spectral_centroid_mean_hz": float(centroid.mean()),
        "_y": y,
        "_sr": sr,
        "_rms_frames": rms_frames,
        "_hop_len": hop_len,
    }


# ── 3. Plots ─────────────────────────────────────────────────────────────────

def save_plots(metrics: dict, audio_path: str, out_dir: str = ".audiorag") -> str:
    import numpy as np
    import librosa
    import matplotlib
    matplotlib.use("Agg")  # no display needed — saves to file
    import matplotlib.pyplot as plt

    y    = metrics["_y"]
    sr   = metrics["_sr"]
    hop  = metrics["_hop_len"]
    name = os.path.splitext(os.path.basename(audio_path))[0]
    out  = os.path.join(out_dir, f"diagnose_{name}.png")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Audio diagnostics: {os.path.basename(audio_path)}", fontsize=12)

    # waveform
    times = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(times, y, linewidth=0.3, color="steelblue")
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop,
                                        n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                                    x_axis="time", y_axis="mel",
                                    fmax=8000, ax=axes[1])
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Mel spectrogram")

    # frame-level RMS over time
    rms = metrics["_rms_frames"]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop
    )
    axes[2].plot(rms_times, rms, linewidth=0.5, color="darkorange")
    axes[2].axhline(rms.max() * 0.01, color="red", linestyle="--",
                     linewidth=0.8, label="silence threshold (1% peak)")
    axes[2].set_title("Frame-level RMS energy")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("RMS")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close(fig)
    return out


# ── 4. Interpret and recommend ───────────────────────────────────────────────

def interpret(info: dict, metrics: dict) -> list[str]:
    """Return a list of human-readable observations and recommendations."""
    notes = []

    if info["sample_rate"] != 16000:
        notes.append(
            f"⚠  Sample rate is {info['sample_rate']} Hz (Whisper wants 16000). "
            f"Resample with --enhance."
        )

    if metrics["peak_amplitude"] < 0.1:
        notes.append(
            f"⚠  Very quiet: peak amplitude {metrics['peak_amplitude']:.3f}. "
            f"Loudness normalisation (--enhance) likely to help."
        )
    elif metrics["rms_mean"] < 0.01:
        notes.append(
            f"⚠  Low average RMS ({metrics['rms_mean']:.4f}). "
            f"Audio is quiet overall — use --enhance."
        )

    if metrics["silence_ratio"] > 0.6:
        notes.append(
            f"⚠  {metrics['silence_ratio']*100:.0f}% of frames are near-silent. "
            f"High silence content — check if audio was captured correctly. "
            f"VAD filter should help but consider --enhance first."
        )

    if metrics["snr_db_est"] < 10:
        notes.append(
            f"⚠  Estimated SNR ~{metrics['snr_db_est']:.1f} dB (low). "
            f"Significant background noise — try --denoise."
        )
    elif metrics["snr_db_est"] > 30:
        notes.append(
            f"✓  Estimated SNR ~{metrics['snr_db_est']:.1f} dB (good). "
            f"Noise is unlikely to be the problem."
        )

    if metrics["spectral_centroid_mean_hz"] < 500:
        notes.append(
            f"⚠  Very low spectral centroid ({metrics['spectral_centroid_mean_hz']:.0f} Hz). "
            f"Audio may be mostly low-frequency noise rather than speech."
        )

    if not notes:
        notes.append(
            "✓  Signal looks reasonable. Issue may be Whisper model size — "
            "try --model-size small or medium."
        )

    return notes


# ── 5. Entry point ───────────────────────────────────────────────────────────

def diagnose(audio_path: str, persist_dir: str = ".audiorag",
             offset: float = 1200.0, duration: float = 300.0) -> None:
    print(f"\n{'─'*60}")
    print(f"  {os.path.basename(audio_path)}")
    print(f"{'─'*60}")

    print("\n[1/3] File info (ffprobe)...")
    info = ffprobe_info(audio_path)
    for k, v in info.items():
        print(f"  {k:<25} {v}")

    print("\n[2/3] Signal metrics (loading audio)...")
    print(f"  analysing {duration:.0f}s starting at {offset:.0f}s ({offset/60:.1f} min in)")
    metrics = signal_metrics(audio_path, offset=offset, max_duration=duration)
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")

    print("\n[3/3] Saving plots...")
    os.makedirs(persist_dir, exist_ok=True)
    plot_path = save_plots(metrics, audio_path, out_dir=persist_dir)
    print(f"  Saved → {plot_path}")

    print("\n── Observations ──")
    for note in interpret(info, metrics):
        print(f"  {note}")
    print()
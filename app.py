import io, os, cv2, numpy as np
from PIL import Image, ImageOps, ImageChops
import streamlit as st
from skimage import color
from scipy.fft import fft2, fftshift

# (Optional) HEIC support if installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ---------------- helpers ----------------
def sample_video_frames(video_bytes: bytes, max_frames: int = 24):
    tmp_path = "tmp_upload_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(tmp_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        idxs = np.linspace(0, max(0, total-1), num=min(max_frames, total)).astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, fr = cap.read()
            if ret: frames.append(fr)
    else:
        i = 0
        while cap.isOpened() and i < max_frames:
            ret, fr = cap.read()
            if not ret: break
            frames.append(fr); i += 1
    cap.release()
    try: os.remove(tmp_path)
    except: pass
    return frames

def to_rgb(np_bgr): return np_bgr[:, :, ::-1]

def resize_for_display(img, max_side=720):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

# --------------- forensics ----------------
def error_level_analysis(pil_img: Image.Image, quality: int = 95):
    """ELA: re-encode as JPEG, diff with original → highlight edits."""
    if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
    buf = io.BytesIO(); pil_img.save(buf, "JPEG", quality=quality, optimize=True); buf.seek(0)
    comp = Image.open(buf).convert("RGB")
    try:
        diff = ImageChops.difference(pil_img, comp)
        d = np.asarray(diff).astype(np.float32)
    except Exception:
        a = np.asarray(pil_img, dtype=np.int16); b = np.asarray(comp, dtype=np.int16)
        d = np.abs(a - b).astype(np.float32)
    d_norm = d / (d.max() + 1e-6)
    d_vis = (d_norm ** 0.5) * 255.0
    d_vis_uint8 = d_vis.clip(0,255).astype(np.uint8)
    d_gray = color.rgb2gray(d_vis_uint8)  # 0..1
    return d_vis_uint8, d_gray

def high_freq_energy(rgb_np):
    """FFT-based high-frequency ratio."""
    gray = color.rgb2gray(rgb_np)
    H, W = gray.shape
    H -= H % 2; W -= W % 2
    gray = cv2.resize((gray*255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    F = fftshift(np.abs(fft2(gray))); F /= (F.max() + 1e-6)
    cy, cx = H//2, W//2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y-cy)**2 + (X-cx)**2)
    r_inner, r_outer = min(H,W)*0.10, min(H,W)*0.45
    hi_mask = (R >= r_inner) & (R <= r_outer)
    hi_energy = F[hi_mask].sum()
    return float(hi_energy / (F.sum() + 1e-6))

def block_artifact_score(rgb_np):
    """JPEG blockiness across 8x8 boundaries."""
    gray = (color.rgb2gray(rgb_np)*255.0).astype(np.uint8)
    h, w = gray.shape
    v_edges = np.arange(8, w, 8); h_edges = np.arange(8, h, 8)
    v_score = sum(np.mean(np.abs(gray[:, x-1].astype(np.int16) - gray[:, x].astype(np.int16))) for x in v_edges)
    h_score = sum(np.mean(np.abs(gray[y-1, :].astype(np.int16) - gray[y, :].astype(np.int16))) for y in h_edges)
    denom = max(len(v_edges) + len(h_edges), 1)
    return float((v_score + h_score) / denom)

def combined_suspicion(rgb_np):
    """Fuse ELA + high-freq + blockiness into 0..1 score (demo calibration)."""
    pil = Image.fromarray(rgb_np)
    ela_vis, ela_gray = error_level_analysis(pil, quality=95)
    ela_score = float(np.clip(ela_gray.mean() * 2.5, 0, 1))
    hfe = high_freq_energy(rgb_np)
    hfe_score = float(np.clip((hfe - 0.18) / 0.25, 0, 1))
    bas_raw = block_artifact_score(rgb_np)
    bas_score = float(np.clip((bas_raw - 2.0) / 6.0, 0, 1))
    score = float(np.clip(0.5*ela_score + 0.35*hfe_score + 0.15*bas_score, 0, 1))
    # NumPy 2.0 fix: use np.ptp instead of arr.ptp()
    heatmap = (ela_gray - ela_gray.min()) / (np.ptp(ela_gray) + 1e-6)
    return score, (ela_vis, heatmap)

def overlay_heatmap(rgb, heatmap01):
    heat = (heatmap01 * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    return cv2.addWeighted(rgb, 0.6, heat_color, 0.4, 0)

# ---------------- UI ----------------
st.set_page_config(page_title="DeepShield — Deepfake Demo", layout="centered")
st.title("🛡️ DeepShield — Deepfake Demo (V0)")
st.write("Upload an **image** or a **short video (≤15s)**. This prototype returns a **Fake Likelihood** and a **heatmap** using classic forensic signals (ELA, frequency, block artifacts).")

mode = st.radio("Choose upload type:", ["Image", "Video"], horizontal=True)

if mode == "Image":
    f = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp","heic","heif"])
    if f:
        try:
            pil = Image.open(io.BytesIO(f.read()))
            if pil.mode not in ("RGB","RGBA"): pil = pil.convert("RGB")
            rgb = np.array(pil)
            rgb_small = resize_for_display(rgb, max_side=720)
            with st.spinner("Analyzing…"):
                score, (ela_vis, ela_heat) = combined_suspicion(rgb_small)
                vis = overlay_heatmap(rgb_small, ela_heat)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original"); st.image(rgb_small, use_column_width=True)
            with col2:
                st.subheader("Heatmap (ELA-based)"); st.image(vis, use_column_width=True)
            st.metric("Fake Likelihood (demo)", f"{score*100:.1f}%")
            st.caption("Prototype only. Production adds trained detectors (CNN/ViT + temporal), provenance (C2PA), and takedown flow.")
        except Exception as e:
            st.error(f"Could not analyze this image: {e}")
            st.info("Try a JPG/PNG/WEBP or install HEIC support with: pip install pillow-heif")

else:
    f = st.file_uploader("Upload short video (≤15s)", type=["mp4","mov","avi","mkv"])
    if f:
        data = f.read()
        with st.spinner("Sampling frames…"):
            frames_bgr = sample_video_frames(data, max_frames=16)
        if not frames_bgr:
            st.error("Could not read frames. Try another file.")
        else:
            frame_scores, previews = [], []
            for i, fr_bgr in enumerate(frames_bgr):
                rgb = to_rgb(fr_bgr)
                rgb_small = resize_for_display(rgb, max_side=512)
                s, (ela_vis, ela_heat) = combined_suspicion(rgb_small)
                frame_scores.append(s)
                if i < 6: previews.append(overlay_heatmap(rgb_small, ela_heat))
            avg_score = float(np.mean(frame_scores))
            st.metric("Fake Likelihood (avg across frames)", f"{avg_score*100:.1f}%")
            st.write(f"Frames analyzed: {len(frame_scores)}")
            if previews:
                st.subheader("Sample frames (with heatmaps)")
                for img in previews: st.image(img, use_column_width=True)

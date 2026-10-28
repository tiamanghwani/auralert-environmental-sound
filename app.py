import json, os, tempfile
import joblib, librosa, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Auralert | Environmental AI", page_icon="🔊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background:radial-gradient(circle at 10% 0%,#172554 0,#0b1020 40%,#080b14 100%);color:#f8fafc}
.block-container{padding-top:2rem;max-width:1200px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif}
.hero{padding:1.6rem 1.8rem;border:1px solid #334155;border-radius:24px;background:linear-gradient(135deg,#1e293b,#0f172a);margin-bottom:1.2rem}
.kicker{color:#67e8f9;letter-spacing:.18em;font-size:.75rem;font-weight:700}
.hero h1{font-size:clamp(2rem,4vw,3.2rem);margin:.35rem 0}
.hero p,.note{color:#cbd5e1}
.metric-card{background:#0f172acc;border:1px solid #334155;padding:1rem;border-radius:16px}
.metric-label{color:#94a3b8;font-size:.82rem}
.metric-value{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")

def extract_features(path):
    audio, sr = librosa.load(path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_summary = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral = np.array([centroid.mean(), centroid.std(), zcr.mean(), zcr.std(), rolloff.mean(), rolloff.std()])
    return np.concatenate([mfcc_summary, spectral]).reshape(1,-1), audio, sr

try:
    bundle = load_bundle()
except Exception:
    st.error("Model file missing. Run the notebook, download model_bundle.joblib, and put it beside app.py.")
    st.stop()

st.markdown("""<div class="hero"><div class="kicker">AUDIO INTELLIGENCE LAB · MACHINE LEARNING</div>
<h1>🔊 Auralert</h1><p>Hear the world. Detect the unexpected. Upload an environmental recording and classify it across 50 sound categories.</p></div>""", unsafe_allow_html=True)

tab_demo, tab_metrics, tab_about = st.tabs(["🎧 Sound Lab", "📊 Model Performance", "🧠 How it works"])
with tab_demo:
    left, right = st.columns([1,1], gap="large")
    with left:
        st.subheader("Try a sound")
        st.write("Upload a short WAV, MP3, OGG, FLAC or M4A recording.")
        uploaded = st.file_uploader("Choose an audio file", type=["wav","mp3","ogg","flac","m4a"])
        if uploaded:
            st.audio(uploaded)
            if st.button("✨ Analyze this sound", type="primary", use_container_width=True):
                temp_path = None
                try:
                    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer()); temp_path = tmp.name
                    with st.spinner("Listening to the audio features..."):
                        feats, audio, sr = extract_features(temp_path)
                        model = bundle["model"]
                        pred = model.predict(feats)[0]
                        scores = model.predict_proba(feats)[0] if hasattr(model,"predict_proba") else None
                        st.session_state["prediction"] = {"name":str(pred), "filename":uploaded.name, "duration":len(audio)/sr,
                                                          "scores":scores, "classes":bundle["classes"]}
                except Exception as e:
                    st.error(f"Could not analyze this recording: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path): os.remove(temp_path)
    with right:
        st.subheader("Prediction")
        result = st.session_state.get("prediction")
        if result:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">PREDICTED SOUND</div>
            <div class="metric-value">🔎 {result['name'].replace('_',' ').title()}</div>
            <div class="note">File: {result['filename']} · {result['duration']:.1f} seconds</div></div>""", unsafe_allow_html=True)
            if result["scores"] is not None:
                ix = np.argsort(result["scores"])[::-1][:5]
                top = pd.DataFrame({"Category":[str(result["classes"][i]).replace("_"," ").title() for i in ix],
                                    "Model score":[float(result["scores"][i]) for i in ix]})
                st.write("Top 5 candidate categories")
                st.bar_chart(top.set_index("Category"))
                st.caption("Scores are model estimates and are not guarantees of real-world correctness.")
        else:
            st.info("Upload a recording and click Analyze to see a prediction.")
            st.write("Demo ideas: dog bark, rain, siren, keyboard typing, helicopter, or footsteps.")

with tab_metrics:
    st.subheader("Measured model performance")
    st.write("Metrics below are from five-fold out-of-fold evaluation in the notebook.")
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", encoding="utf-8") as f: info = json.load(f)
        st.caption(f"Dataset: {info['dataset_rows']} recordings · {info['class_count']} classes · Best model: {info['best_model']}")
        frame = pd.DataFrame(info["metrics"])
        for _, row in frame.iterrows():
            st.markdown("### " + row["Model"])
            cols = st.columns(4)
            for col, key in zip(cols, ["Accuracy","Macro Precision","Macro Recall","Macro F1"]):
                with col:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>{key}</div><div class='metric-value'>{row[key]:.1%}</div></div>", unsafe_allow_html=True)
            st.write("")
        st.dataframe(frame.style.format({c:"{:.4f}" for c in frame.columns if c != "Model"}), use_container_width=True)
        if os.path.exists("model_metrics.csv"):
            st.download_button("Download metrics CSV", open("model_metrics.csv","rb"), file_name="model_metrics.csv", mime="text/csv")
    else:
        st.warning("Add model_metrics.json and model_metrics.csv generated by the notebook.")

with tab_about:
    st.subheader("How it works")
    st.markdown("""
    1. **Audio input:** upload an environmental sound.
    2. **Feature extraction:** calculate MFCC and spectral features.
    3. **Inference:** the saved classifier predicts one of 50 known categories.
    4. **Evaluation:** view accuracy, macro precision, macro recall and macro F1 from five-fold testing.

    **Models compared:** SVM, KNN and Random Forest.  
    **Dataset:** ESC-50, 2,000 labelled five-second recordings.  
    **Limitations:** the model always chooses among known classes and may misclassify unfamiliar, noisy, or mixed sounds. It is not a safety-critical alarm detector.
    """)
st.markdown("---")
st.caption("Auralert · Environmental Sound Classification · Python + Librosa + Scikit-learn + Streamlit")

from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------
# ไฟล์ที่ต้องอยู่โฟลเดอร์เดียวกับ app.py
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "mobile_price_classification.keras"
SCALER_PATH = BASE_DIR / "scaler.joblib"  # ไม่บังคับ: ใส่ถ้าตอนเทรนมีการ scale ข้อมูล

# ---------------------------------------------------------------
# โมเดลนี้รับ input 20 ตัว ตรงกับ "Mobile Price Classification" dataset
# ที่นิยมใช้กันบน Kaggle พอดี (20 คอลัมน์ + price_range เป็น target)
# ผมจึงใส่ชื่อคอลัมน์มาตรฐานของ dataset นี้ให้แล้ว
# *** แต่ยังต้องเช็ก 1 เรื่อง: ลำดับคอลัมน์ต้องตรงกับตอนเทรนเป๊ะ ๆ ***
# เช็กด้วย: print(X_train.columns.tolist())  แล้วเทียบกับลิสต์นี้
# ถ้า dataset ของคุณเป็นคนละชุด ให้แก้ name/label/min/max ให้ตรงของจริง
# ---------------------------------------------------------------
FEATURES = [
    {"name": "battery_power", "label": "ความจุแบตเตอรี่ (mAh)", "kind": "number",
     "min": 500, "max": 6000, "default": 1500, "step": 50},
    {"name": "blue", "label": "มีบลูทูธ", "kind": "bool"},
    {"name": "clock_speed", "label": "ความเร็วสัญญาณนาฬิกา (GHz)", "kind": "number",
     "min": 0.5, "max": 3.0, "default": 1.5, "step": 0.1},
    {"name": "dual_sim", "label": "รองรับ Dual SIM", "kind": "bool"},
    {"name": "fc", "label": "กล้องหน้า (ล้านพิกเซล)", "kind": "number",
     "min": 0, "max": 20, "default": 5, "step": 1},
    {"name": "four_g", "label": "รองรับ 4G", "kind": "bool"},
    {"name": "int_memory", "label": "หน่วยความจำภายใน (GB)", "kind": "number",
     "min": 2, "max": 256, "default": 32, "step": 2},
    {"name": "m_dep", "label": "ความหนาเครื่อง (ซม.)", "kind": "number",
     "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.1},
    {"name": "mobile_wt", "label": "น้ำหนักเครื่อง (กรัม)", "kind": "number",
     "min": 80, "max": 250, "default": 150, "step": 5},
    {"name": "n_cores", "label": "จำนวนคอร์ CPU", "kind": "number",
     "min": 1, "max": 8, "default": 4, "step": 1},
    {"name": "pc", "label": "กล้องหลัง (ล้านพิกเซล)", "kind": "number",
     "min": 0, "max": 20, "default": 10, "step": 1},
    {"name": "px_height", "label": "ความละเอียดจอ - สูง (พิกเซล)", "kind": "number",
     "min": 0, "max": 2000, "default": 800, "step": 10},
    {"name": "px_width", "label": "ความละเอียดจอ - กว้าง (พิกเซล)", "kind": "number",
     "min": 0, "max": 2000, "default": 1200, "step": 10},
    {"name": "ram", "label": "แรม (MB)", "kind": "number",
     "min": 256, "max": 8000, "default": 2000, "step": 64},
    {"name": "sc_h", "label": "ความสูงหน้าจอ (ซม.)", "kind": "number",
     "min": 5, "max": 20, "default": 12, "step": 1},
    {"name": "sc_w", "label": "ความกว้างหน้าจอ (ซม.)", "kind": "number",
     "min": 0, "max": 18, "default": 7, "step": 1},
    {"name": "talk_time", "label": "เวลาคุยโทรศัพท์ต่อเนื่อง (ชม.)", "kind": "number",
     "min": 2, "max": 20, "default": 10, "step": 1},
    {"name": "three_g", "label": "รองรับ 3G", "kind": "bool"},
    {"name": "touch_screen", "label": "หน้าจอสัมผัส", "kind": "bool"},
    {"name": "wifi", "label": "มี WiFi", "kind": "bool"},
]

# ป้ายกำกับคลาสผลลัพธ์ (โมเดลนี้ทำนาย 4 ระดับ: 0,1,2,3)
CLASS_LABELS = {
    0: "ราคาถูก (Low)",
    1: "ราคาปานกลาง (Medium)",
    2: "ราคาสูง (High)",
    3: "ราคาสูงมาก (Very High)",
}

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")


@st.cache_resource
def load_model():
    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists():
        return None
    import joblib

    return joblib.load(SCALER_PATH)


st.title("📱 ทำนายระดับราคามือถือ")
st.caption("โมเดล MLP (Keras) — กรอกสเปกมือถือแล้วกดทำนาย")

# ---- โหลดโมเดล ----
if not MODEL_PATH.exists():
    st.error(f"ไม่พบไฟล์ {MODEL_PATH.name} ในโฟลเดอร์ {BASE_DIR}")
    st.stop()

try:
    model = load_model()
except ModuleNotFoundError:
    st.error("ยังไม่ได้ติดตั้ง TensorFlow — รัน `pip install tensorflow` ใน venv แล้วเปิดแอปใหม่")
    st.stop()
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

scaler = load_scaler()
n_inputs = model.input_shape[-1]

# ---- ตรวจว่า FEATURES ตรงกับโมเดล ----
if len(FEATURES) != n_inputs:
    st.error(
        f"โมเดลนี้รับ input {n_inputs} ตัว แต่ใน app.py กำหนด FEATURES ไว้ {len(FEATURES)} ตัว "
        "— แก้รายการ FEATURES ให้ตรงกับคอลัมน์ที่ใช้เทรน"
    )
    st.stop()


def fallback_scale(raw_values: list[float]) -> np.ndarray:
    """ไม่มี scaler.joblib จริง: ประมาณค่าด้วย min-max ของแต่ละฟีเจอร์ (จากช่วงในฟอร์ม)
    แล้ว map ไปเป็นช่วงประมาณ -2 ถึง 2 ซึ่งใกล้เคียงช่วงที่โมเดลประเภทนี้มักถูกเทรนด้วย
    เป็นการประมาณคร่าว ๆ เท่านั้น ไม่ใช่ scaler ตัวจริงที่ใช้ตอนเทรน"""
    out = []
    for v, f in zip(raw_values, FEATURES):
        if f["kind"] == "bool":
            out.append((v - 0.5) * 4)  # 0 -> -2, 1 -> 2
        else:
            lo, hi = f["min"], f["max"]
            span = (hi - lo) or 1
            out.append(((v - lo) / span) * 4 - 2)  # lo -> -2, hi -> 2
    return np.array([out], dtype="float32")


with st.sidebar:
    st.subheader("ข้อมูลโมเดล")
    st.write(f"จำนวน input: **{n_inputs}**")
    if scaler is not None:
        st.write("Scaler: ✅ ใช้ scaler.joblib (ตัวจริง)")
    else:
        st.write("Scaler: ⚠️ ไม่พบไฟล์ — ใช้การประมาณค่าอัตโนมัติแทน")
        st.caption(
            "ผลทำนายจะเปลี่ยนตามค่าที่กรอกได้ปกติ แต่ความแม่นยำอาจคลาดเคลื่อนจากตอนเทรนจริง "
            "ถ้ามี scaler.joblib ตัวจริง ให้วางไว้ข้าง app.py แล้วรีสตาร์ทแอป"
        )

# ---- ฟอร์มกรอกข้อมูล ----
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    values = []
    for i, f in enumerate(FEATURES):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if f["kind"] == "bool":
                v = st.checkbox(f["label"], key=f["name"])
                v = 1 if v else 0
            else:
                v = st.number_input(
                    f["label"],
                    min_value=f["min"],
                    max_value=f["max"],
                    value=f["default"],
                    step=f["step"],
                    key=f["name"],
                )
        values.append(v)
    submitted = st.form_submit_button("ทำนาย", type="primary")

# ---- ทำนาย ----
if submitted:
    x_raw = np.array([values], dtype="float32")
    if scaler is not None:
        x = scaler.transform(x_raw).astype("float32")
    else:
        x = fallback_scale(values)

    proba = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(proba))
    label = CLASS_LABELS.get(pred, str(pred))

    st.subheader("ผลการทำนาย")
    st.success(f"ระดับราคาที่ทำนายได้: **{label}**")

    st.write("ความน่าจะเป็นแต่ละระดับ:")
    st.bar_chart(
        {CLASS_LABELS.get(i, str(i)): float(p) for i, p in enumerate(proba)}
    )

    with st.expander("ดูค่าที่ส่งเข้าโมเดล"):
        st.write("ค่าดิบที่กรอก")
        st.write(dict(zip([f["name"] for f in FEATURES], values)))
        if scaler is not None:
            st.write("ค่าหลัง scale (ที่ส่งเข้าโมเดลจริง)")
            st.write(x.tolist())

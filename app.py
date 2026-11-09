import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

DEFAULT_WEIGHTS = "best.pt"   # วางไฟล์ best.pt ไว้โฟลเดอร์เดียวกับ app.py

# ---------- หน้าเพจ + สไตล์ ----------
st.set_page_config(page_title="ตรวจจับใบหน้าคนเมา", page_icon="🫧", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=IBM+Plex+Sans+Thai:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, label, button, input {
    font-family: 'IBM Plex Sans Thai', sans-serif;
}
h1, h2, h3, .title-xl {
    font-family: 'Bricolage Grotesque', 'IBM Plex Sans Thai', sans-serif;
    letter-spacing: -0.02em;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }

.title-xl { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0 0 .35rem 0; color: #1B2340; }
.subtitle { color: #4A5275; font-size: 1.02rem; max-width: 46rem; margin-bottom: 1.4rem; }

/* การ์ดผลลัพธ์: สีเปลี่ยนตามผล ไม่ใช่แค่ตกแต่ง */
.verdict { border-radius: 14px; padding: 1.1rem 1.3rem; margin: .4rem 0 1rem 0; border-left: 8px solid; }
.verdict .big { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1.1; }
.verdict .small { font-size: .95rem; margin-top: .25rem; }
.verdict.hit  { background: #F6E4EE; border-color: #7B2D5B; color: #4D1536; }
.verdict.miss { background: #DCEFEA; border-color: #1F7A6D; color: #124A41; }
.verdict.none { background: #FFFFFF; border-color: #B8BFD6; color: #4A5275; }

.legend span { display: inline-block; width: .8rem; height: .8rem; border-radius: 3px; margin-right: .4rem; vertical-align: -1px; }
.note { color: #4A5275; font-size: .88rem; }

[data-testid="stSidebar"] { border-right: 1px solid #D7DCE8; }
button:focus-visible { outline: 3px solid #7B2D5B !important; outline-offset: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

COLOR_DRUNK = (123, 45, 91)
COLOR_OTHER = (31, 122, 109)


# ---------- โมเดล ----------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def load_model(path: str):
    return YOLO(path)


def draw_boxes(img: Image.Image, result, drunk_id: int, names: dict, min_conf: float):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 250)
    rows = []
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        cid = int(box.cls[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        is_drunk = cid == drunk_id
        color = COLOR_DRUNK if is_drunk else COLOR_OTHER
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{'drunk' if is_drunk else 'sober'} {conf:.2f}"
        tw = draw.textlength(label)
        draw.rectangle([x1, max(0, y1 - 16), x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill="white")
        rows.append(
            {
                "ผล": "น่าจะเมา" if is_drunk else "ปกติ",
                "ความมั่นใจ": round(conf, 3),
                "คลาสในโมเดล": names.get(cid, str(cid)),
                "ตำแหน่ง (x1,y1,x2,y2)": f"{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}",
            }
        )
    return img, rows


# ---------- แถบด้านข้าง ----------
with st.sidebar:
    st.header("ตั้งค่า")

    weights_file = st.file_uploader("ไฟล์โมเดล (.pt)", type=["pt"], help="ถ้าไม่อัปโหลด จะใช้ best.pt ในโฟลเดอร์แอป")
    if weights_file is not None:
        tmp = Path("uploaded_best.pt")
        tmp.write_bytes(weights_file.getvalue())
        weights_path = str(tmp)
    else:
        weights_path = DEFAULT_WEIGHTS

    conf = st.slider("ความมั่นใจขั้นต่ำ", 0.05, 0.95, 0.40, 0.05,
                     help="ยิ่งสูง ยิ่งตัดผลที่ไม่แน่ใจทิ้ง")
    drunk_id = st.radio(
        "คลาสไหนในโมเดลคือ “คนเมา”?",
        options=[0, 1],
        index=1,
        format_func=lambda i: f"คลาส {i}",
        help="Dataset ตั้งชื่อคลาสเป็น 0 กับ 1 เปิดดูใน Roboflow แล้วเลือกให้ตรง",
    )
    st.markdown(
        '<p class="note">แอปนี้เป็นงานทดลองจาก dataset ขนาดเล็ก ใช้ประกอบการเรียนรู้เท่านั้น '
        "ไม่ควรใช้ตัดสินใจเกี่ยวกับตัวบุคคล</p>",
        unsafe_allow_html=True,
    )

# ---------- ส่วนหัว ----------
st.markdown('<div class="title-xl">ตรวจจับใบหน้าคนเมา</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ใส่รูปหรือถ่ายภาพ โมเดล YOLOv8 จะวาดกรอบรอบใบหน้าและบอกว่าแต่ละคนน่าจะเมาหรือไม่</div>',
    unsafe_allow_html=True,
)

if not Path(weights_path).exists():
    st.error("ยังไม่พบไฟล์โมเดล: วาง best.pt ไว้ข้าง app.py หรืออัปโหลดที่แถบด้านข้าง")
    st.stop()

model = load_model(weights_path)

# ---------- รับภาพ ----------
tab_upload, tab_camera = st.tabs(["อัปโหลดรูป", "ถ่ายด้วยกล้อง"])
source = None
with tab_upload:
    up = st.file_uploader("เลือกรูป JPG หรือ PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if up:
        source = Image.open(io.BytesIO(up.getvalue()))
with tab_camera:
    cam = st.camera_input("ถ่ายภาพ", label_visibility="collapsed")
    if cam:
        source = Image.open(io.BytesIO(cam.getvalue()))

# ---------- ประมวลผล ----------
if source is None:
    st.markdown(
        '<div class="verdict none"><div class="big">รอรูปภาพ</div>'
        '<div class="small">อัปโหลดรูปหรือถ่ายภาพที่แท็บด้านบนเพื่อเริ่มตรวจ</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("กำลังตรวจ…"):
    result = model.predict(source, conf=0.05, verbose=False)[0]
    annotated, rows = draw_boxes(source, result, drunk_id, model.names, conf)

n_drunk = sum(r["ผล"] == "น่าจะเมา" for r in rows)
n_ok = len(rows) - n_drunk

left, right = st.columns([3, 2], gap="large")

with left:
    st.image(annotated, use_container_width=True)
    st.markdown(
        '<div class="legend note"><span style="background:#7B2D5B"></span>น่าจะเมา'
        '&nbsp;&nbsp;&nbsp;<span style="background:#1F7A6D"></span>ปกติ</div>',
        unsafe_allow_html=True,
    )

with right:
    if not rows:
        st.markdown(
            '<div class="verdict none"><div class="big">ไม่พบใบหน้า</div>'
            '<div class="small">ลองลดค่าความมั่นใจขั้นต่ำที่แถบด้านข้าง หรือใช้รูปที่เห็นใบหน้าชัดขึ้น</div></div>',
            unsafe_allow_html=True,
        )
    elif n_drunk:
        st.markdown(
            f'<div class="verdict hit"><div class="big">พบคนน่าจะเมา {n_drunk} คน</div>'
            f'<div class="small">จากใบหน้าทั้งหมด {len(rows)} ใบหน้าในรูป</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict miss"><div class="big">ไม่พบคนเมา</div>'
            f'<div class="small">พบใบหน้าปกติ {n_ok} ใบหน้า</div></div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    c1.metric("น่าจะเมา", n_drunk)
    c2.metric("ปกติ", n_ok)

    if rows:
        df = pd.DataFrame(rows).sort_values("ความมั่นใจ", ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button("บันทึกรูปที่มีกรอบ", buf.getvalue(), file_name="result.png", mime="image/png")

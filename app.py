from io import BytesIO
from urllib.request import urlopen

import streamlit as st
from PIL import Image
from ultralytics import YOLO


SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

st.set_page_config(page_title="YOLO 이미지 객체 탐지", page_icon="🔎", layout="wide")
st.title("🔎 YOLO 이미지 객체 탐지")
st.caption("이미지를 업로드하거나 예제 이미지를 선택해 객체를 탐지합니다.")


@st.cache_resource
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


@st.cache_data
def load_sample_image() -> Image.Image:
    with urlopen(SAMPLE_IMAGE_URL, timeout=15) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


with st.sidebar:
    st.header("⚙️ 추론 설정")
    weights = st.text_input("모델 가중치 경로", value="yolo26n.pt")
    confidence = st.slider("최소 신뢰도", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    iou = st.slider("NMS IoU 임계값", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    st.caption("공식 가중치 이름 또는 사용자 학습 가중치 파일 경로를 입력하세요.")

source_type = st.radio("이미지 선택", ["파일 업로드", "예제 이미지"], horizontal=True)
uploaded_file = None
if source_type == "파일 업로드":
    uploaded_file = st.file_uploader("탐지할 이미지", type=["jpg", "jpeg", "png", "bmp", "webp"])
else:
    st.image(SAMPLE_IMAGE_URL, caption="Ultralytics 예제 이미지", width=500)

if st.button("객체 탐지 실행", type="primary"):
    if source_type == "파일 업로드" and uploaded_file is None:
        st.warning("먼저 이미지 파일을 업로드해 주세요.")
    elif not weights.strip():
        st.warning("모델 가중치 경로를 입력해 주세요.")
    else:
        try:
            if source_type == "파일 업로드":
                image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
            else:
                image = load_sample_image()

            with st.spinner("모델을 불러오고 객체를 탐지하는 중입니다..."):
                model = load_model(weights.strip())
                results = model.predict(source=image, conf=confidence, iou=iou, verbose=False)

            result = results[0]
            annotated = result.plot()
            image_col, result_col = st.columns([3, 2])
            with image_col:
                st.subheader("탐지 결과")
                st.image(annotated, channels="BGR", use_container_width=True)
                output = BytesIO()
                Image.fromarray(annotated[:, :, ::-1]).save(output, format="PNG")
                st.download_button(
                    "결과 이미지 다운로드",
                    data=output.getvalue(),
                    file_name="yolo_result.png",
                    mime="image/png",
                )

            with result_col:
                st.subheader("탐지 정보")
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    st.info("객체를 탐지하지 못했습니다.")
                else:
                    xywh = boxes.xywh.cpu().tolist()
                    xywhn = boxes.xywhn.cpu().tolist()
                    xyxy = boxes.xyxy.cpu().tolist()
                    xyxyn = boxes.xyxyn.cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    detections = [
                        {
                            "클래스": result.names[class_id],
                            "신뢰도": confidence_score,
                            "xywh": box_xywh,
                            "xywhn": box_xywhn,
                            "xyxy": box_xyxy,
                            "xyxyn": box_xyxyn,
                        }
                        for class_id, confidence_score, box_xywh, box_xywhn, box_xyxy, box_xyxyn in zip(
                            classes, confidences, xywh, xywhn, xyxy, xyxyn
                        )
                    ]
                    st.metric("탐지 개수", len(detections))
                    st.dataframe(detections, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error("이미지 처리 또는 추론 중 오류가 발생했습니다.")
            st.exception(exc)
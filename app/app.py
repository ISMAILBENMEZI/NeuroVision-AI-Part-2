import streamlit as st
from PIL import Image

from src.prediction import load_prediction_model, predict_image


st.set_page_config(
    page_title="NeuroVision AI",
    layout="wide",
)

st.title("NeuroVision AI")
st.subheader("Brain MRI Classification")
st.write(
    "Upload a brain MRI image to get a model prediction for one of the four classes."
)


@st.cache_resource
def get_model():
    return load_prediction_model()


uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=350,
    )

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            model = get_model()
            result = predict_image(model, uploaded_file)

        st.success(f"Predicted class: {result['predicted_class']}")
        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%",
        )

        st.subheader("Class Probabilities")

        st.bar_chart(result["probabilities"])

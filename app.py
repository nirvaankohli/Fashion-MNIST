import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import FashionMNIST
from PIL import Image
import pandas as pd
import os

# 1) Page config & custom CSS

st.set_page_config(

    page_title="Fashion MNIST with EfficientNet",

    layout="wide",

    initial_sidebar_state="expanded",

)

st.markdown(

    """
    
    <style>
    .main > div:first-child {padding-top: 1rem;}
    footer {visibility: hidden;}
    </style>
    
    """,

    unsafe_allow_html=True,

)

# 2) Download one sample per class (if needed)

SAMPLES_DIR = "./samples"

class_names = [

    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"

]

@st.cache_resource

def prepare_samples():

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # Download test set and pick the first occurrence of each label
    
    ds = FashionMNIST(
        
        root="./data", 
        
        download=True, 
        
        train=False
        
        )  # download=True pulls from the internet :contentReference[oaicite:2]{index=2}
    
    found = {}
    
    for img, label in ds:

        name = class_names[label]

        if name not in found:

            found[name] = img

        if len(found) == len(class_names):

            break

    # Save each one as PNG

    for name, img in found.items():

        fname = name.replace("/", "_").replace(" ", "_") + ".png"
        
        path = os.path.join(SAMPLES_DIR, fname)
        
        if not os.path.exists(path):
        
            img.resize((64,64)).convert("RGB").save(path)

    return {name: os.path.join(SAMPLES_DIR, name.replace("/", "_").replace(" ", "_") + ".png")
            for name in class_names}

sample_paths = prepare_samples()


# 3) Main area: title, selector & gallery

st.title("🤖 Fashion MNIST Classification")
st.write(
    """
    This demo uses **EfficientNet-B0** (with MixUp, CutMix & SWA)
    to classify 28×28 grayscale fashion items into 10 categories with ~93% test accuracy.
    """
)

st.header("🔍 Select Input Image")
options = ["Upload your own"] + class_names
choice = st.selectbox("Or pick a sample:", options)

# File-like for downstream
uploaded = None
if choice == "Upload your own":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])
else:
    # load the pre-downloaded sample
    uploaded = open(sample_paths[choice], "rb")

# Show carousel/gallery of all samples
st.subheader("Sample Images Gallery")
st.image(
    [sample_paths[name] for name in class_names],
    caption=class_names,
    width=80,
    use_container_width=False  # displays as a horizontal gallery :contentReference[oaicite:3]{index=3}
)

# ———————————————————————————————————————————————
# 4) Classification (unchanged)
# ———————————————————————————————————————————————
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    @st.cache_resource
    def load_model(
        version: int = 3,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        device: str = 'cpu'
    ):
        model = models.efficientnet_b0(pretrained=True)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_feat, num_classes),
        )
        ckpt = torch.load(f"best_V{version}_model.pth", map_location='cpu')
        sd = {k.replace('module.', ''): v for k,v in ckpt.get('model', ckpt).items()}
        model.load_state_dict(sd)
        model.eval()
        return model.to(device)

    MODEL_VERSION = 3
    with st.spinner(f"Loading model v{MODEL_VERSION}…"):
        model = load_model(MODEL_VERSION)
    st.success("Model loaded!")

    preprocess = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    inp = preprocess(img).unsqueeze(0)

    with st.spinner("Classifying…"):
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).squeeze()
            topk = torch.topk(probs, 5)
    st.success("Done!")

    labels = topk.indices.tolist()
    confs  = topk.values.tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Top-1", class_names[labels[0]])
    c2.metric("🎯 Confidence", f"{confs[0]*100:.1f}%")
    c3.metric("🥈 2nd-Best", class_names[labels[1]])

    df = pd.DataFrame({
        "Rank":       list(range(1,6)),
        "Class":      [class_names[i] for i in labels],
        "Confidence": [f"{p*100:.1f}%" for p in confs]
    })
    st.table(df)

    st.markdown("**Was our prediction correct?**")
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None
    y, n = st.columns(2)
    if y.button("👍 Yes", use_container_width=True):
        st.session_state.feedback = "yes"
    if n.button("👎 No", use_container_width=True):
        st.session_state.feedback = "no"
    if st.button("Submit Feedback", use_container_width=True):
        if st.session_state.feedback:
            st.success("✔️ Thank you for your feedback!")
        else:
            st.warning("Please choose 👍 or 👎 first.")

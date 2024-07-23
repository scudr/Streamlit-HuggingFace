import streamlit as st
from transformers import CLIPTextModel, AutoTokenizer, CLIPFeatureExtractor, pipeline
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from PIL import Image
import requests
from io import BytesIO

def load_image_gen_model():
    model_name = "stabilityai/stable-diffusion-2-1"
    
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # Attempt to load tokenizer, default to None if not found
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    except Exception as e:
        st.warning(f"Warning loading tokenizer: {e}")
        tokenizer = None
    
    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name, subfolder="feature_extractor")

    # Exclude the safety checker for now and set it to None
    safety_checker = None

    return StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        tokenizer=tokenizer,
        safety_checker=safety_checker,  # Explicitly set safety_checker to None
        feature_extractor=feature_extractor
    )
# Función para cargar el modelo de clasificación de imágenes
def load_image_class_model():
    model_name = "microsoft/resnet-50"
    return pipeline("image-classification", model=model_name)

# Cargar los modelos
image_gen_model = load_image_gen_model()
image_class_model = load_image_class_model()

# Interfaz de usuario
st.title("Aplicación Web con Modelos de HuggingFace")

# Sección de Generación de Imágenes
st.header("Generación de Imágenes")
image_prompt = st.text_input("Ingrese el texto para generar la imagen:")
if st.button("Generar Imagen"):
    if image_prompt:
        generated_images = image_gen_model(image_prompt)
        st.image(generated_images[0]["image"], caption="Imagen Generada")

# Sección de Clasificación de Imágenes
st.header("Clasificación de Imágenes")
uploaded_file = st.file_uploader("Suba una imagen para clasificar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen para Clasificar")
    predictions = image_class_model(image)
    for prediction in predictions:
        st.write(f"{prediction['label']}: {prediction['score']:.2f}")


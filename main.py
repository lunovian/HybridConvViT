import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from models.cwgan import CWGAN

def load_models(generator_path, critic_path):
    cwgan = CWGAN(in_channels=1, out_channels=2)
    cwgan.load_model(generator_path, critic_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwgan.generator.to(device)
    return cwgan

def enhance_image(image):
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply histogram equalization to the L channel
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_channel = cv2.equalizeHist(l_channel)
    lab_image = cv2.merge((l_channel, a_channel, b_channel))
    
    # Convert back to RGB color space
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    
    # Increase saturation
    hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.add(s, 10)  # Increase saturation
    s = np.clip(s, 0, 255)
    hsv_image = cv2.merge((h, s, v))
    
    # Convert back to RGB color space
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return enhanced_image

def preprocess_image(image):
    print("Preprocessing image")
    original_size = image.shape[:2]
    image = Image.fromarray(image).convert('RGB')
    lab_image = rgb2lab(np.array(image))
    L = lab_image[:, :, 0] / 100.0
    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float()
    print(f"Preprocessed L shape: {L.shape}")
    return L, original_size

def postprocess_image(L, AB, original_size):
    print("Postprocessing image")
    L = L.squeeze(0).squeeze(0).cpu().numpy() * 100  # Shape: (height, width)
    AB = (AB.squeeze(0).permute(1, 2, 0).cpu().numpy() - 0.5) * 256  # Shape: (height, width, 2)
    
    lab_image = np.zeros((L.shape[0], L.shape[1], 3))  # Shape: (height, width, 3)
    lab_image[:, :, 0] = L
    lab_image[:, :, 1:] = AB
    
    rgb_image = lab2rgb(lab_image)
    print(f"Postprocessed image shape: {rgb_image.shape}")

    # Resize back to original size
    rgb_image = Image.fromarray((rgb_image * 255).astype(np.uint8)).resize((original_size[1], original_size[0]), Image.LANCZOS)
    enhanced_image = enhance_image(np.array(rgb_image))
    return enhanced_image

def colorize_image(cwgan, image):
    print("Colorizing image")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, original_size = preprocess_image(image)
    L = L.to(device)
    with torch.no_grad():
        output = cwgan.predict(L)
    colorized_image = postprocess_image(L, output, original_size)
    return colorized_image

cwgan = load_models('models/ResUnet_latest.pt', 'models/PatchGAN_latest.pt')

def process_image(image):
    colorized_image = colorize_image(cwgan, image)
    return colorized_image

with gr.Blocks() as demo:
    gr.Markdown("# REColor")
    with gr.Row():
        inp = gr.Image(type='numpy', label="Upload Black and White Image")
        out = gr.Image(type="numpy", label="Colorized Image")
    btn_submit = gr.Button("Submit")
    btn_cancel = gr.Button("Cancel")

    btn_submit.click(fn=process_image, inputs=inp, outputs=out)
    btn_cancel.click(fn=lambda: None, inputs=None, outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)

import streamlit as st
import cv2
import numpy as np
import os
import random
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
import time
import pytesseract 

DATASET_PATH = "TrashType_Image_Dataset/"
WASTE_CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def remove_background(frame):
    """Applies background removal and replaces it with a white background."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to detect the foreground
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Create a white background
    white_bg = np.full(frame.shape, 255, dtype=np.uint8)
    
    # Replace the background with white where mask is 0
    no_bg = np.where(mask[:, :, None] == 0, white_bg, frame)

    return no_bg

def detect_text_regions(image):
    """Detects potential text regions in an image using contour analysis and extracts text-like patterns."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding to enhance text regions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours (potential text regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_region_count = 0
    detected_text = ""

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        # Filter contours that resemble text (small, elongated)
        if 50 < area < 5000 and 0.2 < aspect_ratio < 5:
            text_region_count += 1

            # Extract ROI and resize for better visibility
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (w*2, h*2))

            # Simple method to recognize patterns (match against a stored font template)
            avg_pixel_value = np.mean(roi)
            if avg_pixel_value < 200:  # Thresholding assumption for text-like structure
                detected_text += "#"

    return detected_text if detected_text else None


def augment_image(image):
    """Applies random flip, rotation, and blur to an image."""
    img_array = np.array(image)
    if random.random() > 0.5:
        img_array = cv2.flip(img_array, 1)
    angle = random.uniform(-30, 30)
    h, w = img_array.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img_array = cv2.warpAffine(img_array, M, (w, h))
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        img_array = cv2.GaussianBlur(img_array, (ksize, ksize), 0)
    return Image.fromarray(img_array)

def compute_ssim(image1, image2):
    gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
    gray1 = cv2.resize(gray1, (128, 128))
    gray2 = cv2.resize(gray2, (128, 128))
    return ssim(gray1, gray2)


def extract_lbp(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist / np.sum(hist)  # Normalize

def chi_square_distance(hist1, hist2):
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))

def load_dataset():
    with st.spinner("Loading "):
        dataset_images = []
        dataset_labels = []
        dataset_features = []
        for category in WASTE_CATEGORIES:
            folder_path = os.path.join(DATASET_PATH, category)
            if not os.path.exists(folder_path):
                continue
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (128, 128))
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                dataset_images.append(img_pil)
                dataset_labels.append(category)
                dataset_features.append(extract_lbp(img_pil))

# Apply augmentation only to non-text-heavy categories
                if category not in ["paper", "cardboard"]:
                    augmented_img = augment_image(img_pil)
                    dataset_images.append(augmented_img)
                    dataset_labels.append(category)
                    dataset_features.append(extract_lbp(augmented_img))

    return dataset_images, dataset_labels, dataset_features

def classify_waste(image, dataset_images, dataset_labels, dataset_features):
    st.write("Processing image... Please wait.")
    
    best_match = "Unknown"
    best_score = float("inf")
    image_feature = extract_lbp(image)

    min_chi, max_chi = float("inf"), float("-inf")
    min_ssim, max_ssim = float("inf"), float("-inf")
    
    chi_scores = []
    ssim_scores = []

    # Compute SSIM and Chi-Square distances for all dataset images
    for i, dataset_feature in enumerate(dataset_features):
        chi_dist = chi_square_distance(image_feature, dataset_feature)
        ssim_score = compute_ssim(image, dataset_images[i])

        chi_scores.append(chi_dist)
        ssim_scores.append(ssim_score)

        min_chi = min(min_chi, chi_dist)
        max_chi = max(max_chi, chi_dist)
        min_ssim = min(min_ssim, ssim_score)
        max_ssim = max(max_ssim, ssim_score)

    # Normalize and compute total score
    for i in range(len(dataset_features)):
        norm_chi_dist = (chi_scores[i] - min_chi) / (max(1e-10, max_chi - min_chi))  # Normalize Chi-square
        norm_ssim = (ssim_scores[i] - min_ssim) / (max(1e-10, max_ssim - min_ssim))  # Normalize SSIM


        total_score = 0.5 * (1 - norm_ssim) + 0.5 * norm_chi_dist  # Balanced weightage

        if total_score < best_score:
            best_score = total_score
            best_match = dataset_labels[i]

    # Improve text detection classification
    detected_text = detect_text_regions(image)
    if detected_text and len(detected_text) > 10:
       if best_match in ["cardboard", "paper"]:
          best_score = min(best_score, 0.85)  # Strengthen biodegradable classification
       else:
          best_score += 0.1  # Slight penalty to non-paper classifications


  
    st.write("Image processed successfully!")
    return best_match, 1 - best_score



dataset_images, dataset_labels, dataset_features = load_dataset()

st.title("Waste Segregation using OpenCV & SSIM with Data Augmentation")
option = st.selectbox("Choose Classification Mode", ["Upload Image", "Real-time Classification"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((128, 128))
        waste_type, confidence = classify_waste(image, dataset_images, dataset_labels, dataset_features)
        if waste_type in ["cardboard", "paper"]:
            st.write("♻️ **BioDegradable Waste**")
        else:
            st.write("🚮 **Non-BioDegradable Waste**")

        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write(f"**Predicted Waste Category:** {waste_type}")
        st.write(f"**Similarity Score:** {confidence:.2f}")
elif option == "Real-time Classification":
    st.write("📷 **Camera ready! Click 'Capture Photo' to proceed.**")

    if st.button("Capture Photo"):
        countdown_text = st.empty()

        for i in range(3, 0, -1):  # Reduced countdown time
            countdown_text.write(f"⏳ **Capturing in {i} seconds...**")
            time.sleep(1)

        countdown_text.write("📸 **Capturing image now...**")

        cap = cv2.VideoCapture(0)  # Removed cv2.CAP_DSHOW for better compatibility
        time.sleep(2)

        ret, frame = cap.read()
        attempts = 0
        while not ret and attempts < 5:
            ret, frame = cap.read()
            attempts += 1
            time.sleep(0.5)

        cap.release()

        if ret:
            st.write("✅ **Image Captured!**")

            # Convert to PIL Image (Original)
            original_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Apply background removal
            frame_with_white_bg = remove_background(frame)
            processed_pil = Image.fromarray(cv2.cvtColor(frame_with_white_bg, cv2.COLOR_BGR2RGB))

            # Show processing progress
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

            # Waste classification
            waste_type, confidence = classify_waste(processed_pil, dataset_images, dataset_labels, dataset_features)

            # Display classification result with color-coded message
            if waste_type in ["cardboard", "paper"]:
                st.success("♻️ **BioDegradable Waste**")
            else:
                st.error("🚮 **Non-BioDegradable Waste**")

            # Show processed image with classification result
            st.image(original_pil, caption=f"🎯 **Type: {waste_type} | Confidence: {confidence:.2f}**", use_container_width=True)

        else:
            st.error(" **Failed to capture image. Try again!**")

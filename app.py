"""
Waste Classifier web app (Gradio)

Run:   python app.py
Open:  http://127.0.0.1:7860

Needs in the same folder: waste_classifier.keras, class_names.json
"""

import json

import gradio as gr
import numpy as np
import tensorflow as tf

MODEL_PATH = "waste_classifier.keras"
CLASS_PATH = "class_names.json"
CONFIDENCE_THRESHOLD = 0.50

model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.load(open(CLASS_PATH))
img_size = tuple(model.input_shape[1:3])  # read from the trained model

DISPOSAL_TIPS = {
    "cardboard": "Flatten the box and keep it dry. Recycle with paper and cardboard.",
    "glass": "Rinse and place in the glass recycling bin. Do not mix with ceramics.",
    "metal": "Rinse cans and place in the metal recycling bin.",
    "paper": "Keep it clean and dry. Greasy or wet paper cannot be recycled.",
    "plastic": "Rinse and check the recycling number. Remove caps if required locally.",
    "trash": "General waste. Cannot be recycled with the other categories.",
}


def classify(image):
    if image is None:
        return {}, "Upload an image to get a prediction."

    x = tf.image.resize(image, img_size).numpy()  # raw 0-255, model rescales internally
    probs = model.predict(x[None, ...], verbose=0)[0]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top = int(np.argmax(probs))
    label, conf = class_names[top], float(probs[top])

    if conf < CONFIDENCE_THRESHOLD:
        message = (
            f"**Low confidence ({conf * 100:.1f}%).** Best guess: {label}. "
            "Try a clearer photo with one item on a plain background."
        )
    else:
        message = (
            f"**{label.capitalize()}** ({conf * 100:.1f}% confidence)\n\n"
            f"{DISPOSAL_TIPS.get(label, '')}"
        )
    return scores, message


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Upload a waste image"),
    outputs=[
        gr.Label(num_top_classes=len(class_names), label="Class probabilities"),
        gr.Markdown(label="Result"),
    ],
    title="Image-Based Waste Classification (CNN)",
    description=(
        "Upload a photo of a single waste item. The CNN predicts one of: "
        + ", ".join(class_names)
        + "."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

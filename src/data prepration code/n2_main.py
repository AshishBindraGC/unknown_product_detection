import os
import base64
from tqdm import tqdm
from openai import OpenAI
import re

# =========================
# CONFIG
# =========================

IMAGE_FOLDER = "/mnt/data/ashish/storage/output/ecpl_prdict_sap/images"
OUTPUT_FILE = "correct_detection_soap.txt"

API_KEY = ""
BASE_URL = "https://integrate.api.nvidia.com/v1"

MODEL = "mistralai/mistral-large-3-675b-instruct-2512"


# =====================
# CLIENT
# =====================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# =====================
# IMAGE ENCODER
# =====================


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =====================
# ASK MODEL
# =====================


def check_missing_products(image_path):

    img_base64 = encode_image(image_path)

    prompt = """
This is a retail shelf image with bounding boxes showing detected products.

Count how many products DO NOT have bounding boxes.

Return ONLY a number.

Example:
0
1
2
3
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
        ],
        max_tokens=5,
    )

    text = response.choices[0].message.content.strip()

    match = re.search(r"\d+", text)

    if match:
        return int(match.group())

    return 999


# =====================
# MAIN
# =====================

accepted = []

images = [
    f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for img in tqdm(images):
    path = os.path.join(IMAGE_FOLDER, img)

    try:
        missing = check_missing_products(path)

        if missing <= 2:
            accepted.append(img)

    except Exception as e:
        print("Error:", img, e)

# =====================
# SAVE TXT
# =====================

with open(OUTPUT_FILE, "w") as f:
    for name in accepted:
        f.write(name + "\n")

print(f"Accepted {len(accepted)} images out of {len(images)}")

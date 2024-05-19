import io
import base64
from PIL import Image
from typing import Optional
import numpy as np
import torch
import cv2
import uvicorn
from fastapi import FastAPI, Path, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# import os
from inference.model_prediction import FashionMnist

# from segmentation import SegmentationModel


# FastAPI is a Python class that provides all the functionality for your API.
app = FastAPI(
    title="Image Segmentation API",
    description="Processes uploaded images and returns segmented versions.",
    version="0.0.1",
    openapi_tags=[{"name": "Predict", "description": "API endpoints that return predicted label"}]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

template = Jinja2Templates(directory='templates')

# dir_path = os.path.dirname(
#     os.path.realpath(os.path.join(os.getcwd()))
# )

@app.get("/", tags=["Home page"])
def root(req: Request):
    """Home page."""
    return template.TemplateResponse(
        name='index.html',
        context={'request': req}
    )

# @app.get("/items/{id}", response_class=HTMLResponse)
# async def read_item(request: Request, id: str):
#     return template.TemplateResponse(
#         request=request, name="item.html", context={"id": id}
#     )
# async def read_item(req: Request, id: str):
#     return template.TemplateResponse(
#         name='item.html',
#         context={'request': req, 'id': id}
#     )

# model = SegmentationModel()


@app.post("/predict-label/", tags=["Predict"])
async def predict_label(
        file: UploadFile = File(description="A required image file for segmentation.")
):
    """Receives an image file and segments it using a predefined model, returning the segmented
    image as a base64-encoded PNG string.

    Args:
    - **file** (UploadFile): The image file to segment. Must be in a valid image format (e.g., "image/png").

    Returns:
    - **dict**: A dictionary containing the base64-encoded string of the segmented image.
    """
    # Ensure that the uploaded file is an image.
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    # Read image file as PIL Image.
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    img_array = np.array(image)
    fmnist = FashionMnist("model/saved_model.pth")

    label_number, label_name = fmnist.predict(img_array)
    # Process the image through the segmentation model.
    # mask = model(image)

    # Convert the output mask to a byte stream to return to user.
    # byte_arr = io.BytesIO()
    # mask.save(byte_arr, format="PNG")

    # Encode the byte stream in base64 to send as JSON.
    # JSON is a text-based format and can only directly handle textual data.
    # Therefore, binary data (like images) cannot be included directly in JSON
    # because it may contain bytes that could interfere with the proper parsing
    # of the JSON structure, such as bytes corresponding to curly braces, brackets,
    # quotation marks, and control characters like newline or tab.
    # Base64 is a method used to encode binary data into a string of ASCII characters,
    # which are safe to use in text-based formats including JSON. This encoding converts
    # each set of three bytes into four ASCII characters, ensuring that the binary data does
    # not corrupt the surrounding textual data in a JSON object.
    # result_image_base64 = base64.b64encode(byte_arr.getvalue()).decode("ascii")
    result_image_base64 = base64.b64encode(image_data).decode("ascii")

    return {"filename": file.filename, "mask_image": result_image_base64, "label_number": label_number, "label_name": label_name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
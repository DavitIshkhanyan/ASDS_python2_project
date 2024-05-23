import io
import base64
from PIL import Image
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from inference.model_prediction import FashionMnist


# FastAPI is a Python class that provides all the functionality for your API.
app = FastAPI(
    title="Fashion label prediction API",
    description="Processes uploaded images and returns predicted labels.",
    version="0.0.1",
    openapi_tags=[{"name": "Predict", "description": "API endpoints that return predicted label"}]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

template = Jinja2Templates(directory='templates')

@app.get("/", tags=["Home page"])
def root(req: Request):
    """Home page."""
    return template.TemplateResponse(
        name='index.html',
        context={'request': req}
    )

class Item(BaseModel):
    filename: str
    mask_image: str
    label_number: int
    label_name: str

@app.post("/predict-label/", tags=["Predict"], response_model=Item)
async def predict_label(
        file: UploadFile = File(description="A required image file for label prediction.")
):
    """Receives an image file and predicts label using a predefined model, returning the label, label number and
    image as a base64-encoded JPEG string.

    Args:
    - **file** (UploadFile): The image file to predict label. Must be in a valid image format (e.g., "image/jpeg").

    Returns:
    - **dict**: A dictionary containing the base64-encoded string of the predicted image, it's label and label number.
    """
    # Ensure that the uploaded file is an image.
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    img_array = np.array(image)
    fmnist = FashionMnist("model/saved_model.pth")

    label_number, label_name = fmnist.predict(img_array)

    result_image_base64 = base64.b64encode(image_data).decode("ascii")

    return {"filename": file.filename, "mask_image": result_image_base64, "label_number": label_number, "label_name": label_name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
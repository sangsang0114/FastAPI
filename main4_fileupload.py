##########  Post Method 예제 (파일 업로드 )
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = FastAPI()


@app.post("/files/")
async def create_file(file: UploadFile):
    content = await file.read()

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    print(img_tensor.shape)
    from tensorflow.keras.models import load_model

    model = load_model("k_model.h5")

    y = model.predict(img_tensor)
    print(y)

    if y < 0.5:
        result = "고양이"
    else:
        result = "강아지"
    return {"filename": file.filename, "result": result}

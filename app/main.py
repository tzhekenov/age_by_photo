# app/main.py
#uvicorn app.main:app --reload

import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from PIL import Image
import shutil
from app.inference import AgeClassifier


app = FastAPI()

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Настройка шаблонов
templates = Jinja2Templates(directory="app/templates")

# Инициализация модели
MODEL_PATH = 'models/best_model.pth'  # Убедитесь, что этот путь правильный относительно корня проекта
NUM_CLASSES = 4  # Настройте в зависимости от вашего набора данных
CLASS_LABELS = {
    0: 'Ребенок (0)',    # Пример меток; настройте в соответствии с вашим набором данных
    1: 'Молодой (1)',
    2: 'Взрослый (2) ',
    3: 'Пожилой (3)'
}

# Создание экземпляра AgeClassifier
classifier = AgeClassifier(model_path=MODEL_PATH, num_classes=NUM_CLASSES)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Отображает главную страницу с формой загрузки изображения.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    Обрабатывает загрузку изображения, предсказание и отображает страницу с результатом.
    """
    # Определение директории загрузки
    upload_dir = "app/static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Сохранение загруженного файла
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Открытие изображения с помощью PIL
    try:
        pil_image = Image.open(file_location).convert('RGB')
    except Exception as e:
        return HTMLResponse(content=f"Ошибка обработки изображения: {e}", status_code=400)

    # Выполнение предсказания
    prediction = classifier.predict_pil_image(pil_image)
    predicted_class = CLASS_LABELS.get(prediction, "Неизвестно")

    # Подготовка контекста для шаблона
    context = {
        "request": request,
        "filename": file.filename,
        "predicted_class": predicted_class
    }

    return templates.TemplateResponse("result.html", context)

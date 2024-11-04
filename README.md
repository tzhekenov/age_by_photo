# Классификация Возраста по Изображению Лица

Проект по определению возраста человека на основе фотографии лица с использованием глубокого обучения.

## Описание проекта

Этот проект предназначен для определения возраста человека по фотографии лица с использованием модели глубокого обучения на основе архитектуры ResNet50 и методов переноса обучения. Реализованы методы повышения качества входных изображений для улучшения точности классификации.

## Технологии

- **Python 3.10.11**
- **PyTorch** — построение и обучение модели
- **FastAPI** — создание веб-приложения
- **Uvicorn** — ASGI-сервер для запуска FastAPI
- **Poetry** — управление зависимостями и виртуальным окружением
- **OpenCV** — обработка изображений
- **Albumentations** — аугментация данных
- **MTCNN** — детекция лиц
- **Jinja2** — шаблоны HTML

## Структура проекта

<pre>
age-by-photo/
├── pyproject.toml
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── inference.py
│   ├── static/
│   │   └── uploads/
│   └── templates/
│       ├── index.html
│       └── result.html
├── scripts/
│   ├── check_augmented.py
│   ├── evalute_model.py
│   └── train.py
├── models/
│   └── best_model.pth
├── data/
│   ├── test/
│   ├── train/
│   ├── train_augmented.csv
│   ├── test.csv
│   └── train_augmented/
├── logs/
│   ├── train_model.log
│   ├── install_kernel.log
│   ├── augment_data.log
│   └── create_train_augmented_csv.log
└── .gitignore
</pre>

## Установка

1. **Установка Poetry**:
    Если у вас не установлен Poetry, вы можете установить его, следуя инструкциям на [сайте Poetry](https://python-poetry.org/docs/#installation).

    Например, используя `curl`:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Или используя `pip`:
    ```sh
    pip install poetry
    ```

2. **Клонирование репозитория**:
    ```sh
    git clone https://github.com/tzhekenov/age_by_photo.git
    cd age_by_photo
    ```

3. **Инициализация окружения Poetry**:
    ```sh
    poetry install
    ```

4. **Активация окружения Poetry**:
    ```sh
    poetry shell
    ```

Это активирует виртуальное окружение, созданное Poetry.

5. **Добавление окружения в Jupyter kernel**:
    ```sh
    python -m ipykernel install --user --name age-by-photo-py3.11
    ```

Это установит все зависимости и настроит Jupyter kernel для проекта.

## Запуск приложения

6. **Для запуска веб-приложения используется Uvicorn:**:
    ```sh
    poetry run uvicorn app.main:app --reload
    ```

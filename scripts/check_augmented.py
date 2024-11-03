# import os
# import pandas as pd

# # Путь к директории с аугментированными изображениями
# augmented_dir = 'data/train_augmented'

# data = []

# for filename in os.listdir(augmented_dir):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         parts = filename.split('_')
#         if len(parts) >= 4:
#             # Извлечение информации из имени файла
#             augmentation_type = f"{parts[0]}_{parts[1]}"
#             class_label = parts[2]
#             original_image_name = parts[3].split('.')[0]
#             filepath = os.path.join(augmented_dir, filename)
            
#             data.append({
#                 'filepath': filepath,
#                 'class': class_label
#             })

# # Создание DataFrame и сохранение в CSV
# df = pd.DataFrame(data)
# df.to_csv('data/train_augmented.csv', index=False)

# print("Файл 'train_augmented.csv' успешно создан.")

# scripts/create_test_csv.py
# scripts/create_test_csv.py

import os
import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("create_test_csv.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """
    Script to create 'test.csv' from test images.
    Assumes that test images are in a flat directory with class labels in filenames.
    """
    test_dir = r'C:\Users\temir\Documents\GitHub\age_by_photo\data\test'
    print(f"Looking for images in: {test_dir}")
    
    if not os.path.exists(test_dir):
        logging.error(f"Test directory '{test_dir}' does not exist.")
        sys.exit(1)
    
    data = []
    
    images = os.listdir(test_dir)
    print(f"Found {len(images)} files in test directory.")
    for img_name in images:
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract class label from filename
            parts = img_name.split('_')
            if len(parts) >= 2:
                class_label_str = parts[0]
                # Validate if class_label_str is a digit
                if class_label_str.isdigit():
                    class_label = int(class_label_str)
                    img_path = os.path.join(test_dir, img_name).replace('\\', '/')
                    data.append({
                        'filepath': img_path,
                        'class': class_label
                    })
                    print(f"Added: {img_path} with class {class_label}")
                else:
                    logging.warning(f"Invalid class label in filename: {img_name}")
            else:
                logging.warning(f"Filename does not conform to expected format: {img_name}")
        else:
            logging.warning(f"Skipped non-image file: {img_name}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = 'data/test.csv'
    df.to_csv(csv_path, index=False)
    
    logging.info(f"'test.csv' created successfully with {len(df)} records.")

if __name__ == "__main__":
    main()

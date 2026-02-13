import os
from PIL import Image

def create_compressed_gif(input_folder, output_name, step=10, scale_factor=0.5, duration=200):
    images = []
    
    # Получаем список файлов и сортируем их
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])
    
    # Отбираем каждое n-е изображение
    selected_files = files[::step]
    
    if not selected_files:
        print("Изображения не найдены!")
        return

    print(f"Обработка {len(selected_files)} кадров...")

    for filename in selected_files:
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # Преобразуем в RGB (на случай если это PNG с прозрачностью)
            img = img.convert("RGB")
            
            # Уменьшаем разрешение
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
            
            # Конвертируем в адаптивную палитру для уменьшения веса (P mode)
            img = img.convert("P", palette=Image.ADAPTIVE)
            
            images.append(img)

    # Сохраняем GIF
    if images:
        images[0].save(
            output_name,
            save_all=True,
            append_images=images[1:],
            optimize=True, # Включает сжатие палитры
            duration=duration, # Длительность кадра в мс
            loop=0 # Бесконечный цикл
        )
        print(f"Готово! GIF сохранен как {output_name}")

# Настройки
create_compressed_gif(
    input_folder='/home/lemul/carla_garage/logs/longest6_route1_12_17_17_00_27',      # Папка с картинками (текущая)
    output_name='result.gif', 
    step=10,               # Каждый 10-й кадр
    scale_factor=0.5,      # Уменьшить размер в 2 раза (0.5)
    duration=100           # Скорость анимации (100 мс = 10 кадров в сек)
)
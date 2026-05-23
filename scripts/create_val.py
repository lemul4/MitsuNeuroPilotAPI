from pathlib import Path
import shutil
import re

SRC_ROOT = Path("/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2_dual_cameras")
DST_ROOT = Path("/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2_dual_cameras_val")

SRC_DATA = SRC_ROOT / "data"
DST_DATA = DST_ROOT / "data"

SRC_RESULTS = SRC_ROOT / "results"
DST_RESULTS = DST_ROOT / "results"

# Регулярка для извлечения номера маршрута из названия папки
# пример: Town13_Rep0_4_1_route0_05_18_18_08_05 -> 4_1
ROUTE_RE = re.compile(r"Town13_Rep\d+_(\d+_\d+)_route")

moved_routes = set()

for subdir in SRC_DATA.rglob("*"):
    if subdir.is_dir() and "Town13" in subdir.name:
        # Найти родительскую категорию (например Accident)
        # Структура: data/<Category>/<Town13_...>
        try:
            category = subdir.relative_to(SRC_DATA).parts[0]
        except ValueError:
            continue

        # Переместить папку Town13_... в dst, сохраняя структуру
        dst_category_dir = DST_DATA / category
        dst_category_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_category_dir / subdir.name

        # Перемещение всей папки
        shutil.move(str(subdir), str(dst_path))

        # Извлечь номер маршрута и запомнить
        m = ROUTE_RE.search(subdir.name)
        if m:
            moved_routes.add((category, m.group(1)))

# Переместить соответствующие result-файлы
for category, route_id in moved_routes:
    src_result = SRC_RESULTS / category / f"{route_id}_result.json"
    if src_result.exists():
        dst_result_dir = DST_RESULTS / category
        dst_result_dir.mkdir(parents=True, exist_ok=True)
        dst_result = dst_result_dir / src_result.name
        shutil.move(str(src_result), str(dst_result))

print(f"Готово. Перемещено маршрутов: {len(moved_routes)}")
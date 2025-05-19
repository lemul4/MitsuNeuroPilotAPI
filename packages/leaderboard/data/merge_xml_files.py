import argparse
import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom


def merge_xml_files(input_files, output_file):
    # Создаем корневой элемент для объединенного XML
    merged_root = ET.Element("routes")

    current_id = 0  # Начинаем нумерацию с 0

    for file_pattern in input_files:
        # Раскрываем шаблон в список файлов
        matched_files = glob.glob(file_pattern)
        if not matched_files:
            print(f"Предупреждение: нет файлов, соответствующих шаблону {file_pattern}")
            continue

        for file_path in matched_files:
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                for route in root.findall("route"):
                    # Обновляем id route
                    route.set("id", str(current_id))
                    current_id += 1

                    # Добавляем route в объединенный XML
                    merged_root.append(route)

            except ET.ParseError as e:
                print(f"Ошибка при обработке файла {file_path}: {e}")

    # Создаем строку XML
    rough_string = ET.tostring(merged_root, 'utf-8')

    # Парсим и форматируем XML без лишних пустых строк
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="   ", encoding="UTF-8")

    # Удаляем лишние пустые строки
    pretty_xml = b'\n'.join(
        line for line in pretty_xml.splitlines()
        if line.strip() or line == b''
    )

    # Записываем в файл
    with open(output_file, 'wb') as f:
        f.write(pretty_xml)

    print(f"Объединенный файл сохранен как {output_file}. Всего маршрутов: {current_id - 1}")


def main():
    parser = argparse.ArgumentParser(description="Объединяет XML файлы с маршрутами, перенумеровывая route id")
    parser.add_argument("input_files", nargs="+",
                        help="Список XML файлов или шаблонов (например, *.xml) для объединения")
    parser.add_argument("-o", "--output", default="merged_routes.xml", help="Имя выходного файла")

    args = parser.parse_args()

    merge_xml_files(args.input_files, args.output)


if __name__ == "__main__":
    main()
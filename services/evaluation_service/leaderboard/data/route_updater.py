import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def transform_routes(input_file: str,):
    # Парсим старый XML
    tree = ET.parse(input_file)
    old_root = tree.getroot()

    # Создаем новый корень
    new_root = ET.Element('routes')

    # Обрабатываем каждый маршрут
    for route in old_root.findall('route'):
        # Создаем новую ноду route с атрибутами id и town (оставляем старое значение town)
        new_route = ET.SubElement(new_root, 'route', id=route.get('id'), town=route.get('town'))

        # Вставляем комментарий внутри <route>
        comment = ET.Comment(' Urbanization route focused on junction crossing vehicles ')
        new_route.append(comment)

        # Блок weathers с двумя погодными условиями
        weathers_el = ET.SubElement(new_route, 'weathers')
        ET.SubElement(weathers_el, 'weather', route_percentage="0",
                      cloudiness="0.0", precipitation="0.0", precipitation_deposits="0.0", wetness="0.0",
                      wind_intensity="0.0", sun_azimuth_angle="90.0", sun_altitude_angle="90.0", fog_density="0.0")

        ET.SubElement(weathers_el, 'weather', route_percentage="100",
                      cloudiness="0.0", precipitation="0.0", precipitation_deposits="0.0", wetness="0.0",
                      wind_intensity="0.0", sun_azimuth_angle="90.0", sun_altitude_angle="90.0", fog_density="0.0")

        # Блок waypoints с преобразованными координатами
        waypoints_el = ET.SubElement(new_route, 'waypoints')
        for wp in route.findall('waypoint'):
            # Берем только x, y, z
            attrs = {k: wp.get(k) for k in ('x', 'y', 'z')}
            ET.SubElement(waypoints_el, 'position', **attrs)

        # Пустой блок scenarios
        ET.SubElement(new_route, 'scenarios')

    # Преобразуем дерево в prettified строку
    rough_string = ET.tostring(new_root, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="   ", encoding='UTF-8')

    # Записываем в файл
    with open("updated_" + input_file, 'wb') as f:
        f.write(pretty_xml)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform old route XML to new format')
    parser.add_argument('input', help='Input old XML file path')
    args = parser.parse_args()

    transform_routes(args.input)

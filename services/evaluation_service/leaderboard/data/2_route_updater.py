import xml.etree.ElementTree as ET

# Заменяемые элементы
NEW_WEATHERS = '''
<weathers>
    <weather route_percentage="0" cloudiness="0.0" precipitation="0.0"
             precipitation_deposits="0.0" wetness="0.0"
             wind_intensity="0.0" sun_azimuth_angle="90.0"
             sun_altitude_angle="90.0" fog_density="0.0"/>
    <weather route_percentage="100" cloudiness="0.0" precipitation="0.0"
             precipitation_deposits="0.0" wetness="0.0"
             wind_intensity="0.0" sun_azimuth_angle="90.0"
             sun_altitude_angle="90.0" fog_density="0.0"/>
</weathers>
'''

def update_xml(input_path: str):
    tree = ET.parse(input_path)
    root = tree.getroot()

    for route in root.findall('route'):
        # Удаляем старый <weathers> и добавляем новый
        old_weathers = route.find('weathers')
        if old_weathers is not None:
            route.remove(old_weathers)
        new_weathers = ET.fromstring(NEW_WEATHERS)
        route.insert(1, new_weathers)  # вставим после комментария (обычно index 1)

        # Очищаем <scenarios>
        scenarios = route.find('scenarios')
        if scenarios is not None:
            scenarios.clear()

    tree.write("updated_" + input_path, encoding='utf-8', xml_declaration=True)

# Пример использования
update_xml('routes_training.xml')

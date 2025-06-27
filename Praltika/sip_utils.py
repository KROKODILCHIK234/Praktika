import requests
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime, timedelta
import gzip
import shutil
from dataclasses import dataclass
import logging
import h5py
import tempfile
import json

# Проверка доступности библиотеки coordinates
try:
    from coordinates import satellite_xyz
    COORDINATES_AVAILABLE = True
except ImportError:
    COORDINATES_AVAILABLE = False

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Константы ---
RE = 6378000.0
TIME_STEP_SECONDS = 30
HEIGHT_OF_THIN_IONOSPHERE = 300000

# --- GNSS спутники ---
GNSS_SATS = []
GNSS_SATS.extend(['G' + str(i).zfill(2) for i in range(1, 33)])
GNSS_SATS.extend(['R' + str(i).zfill(2) for i in range(1, 25)])
GNSS_SATS.extend(['E' + str(i).zfill(2) for i in range(1, 37)])
GNSS_SATS.extend(['C' + str(i).zfill(2) for i in range(1, 41)])

# --- Загрузка навигационного файла ---
def load_nav_file(epoch: datetime, tempdir: str = "./") -> Path:
    """
    Загружает nav-файл с SIMuRG для указанной даты
    
    Args:
        epoch: дата для загрузки nav-файла
        tempdir: временная директория для сохранения файлов
    
    Returns:
        Path: путь к загруженному nav-файлу
    """
    try:
        yday = str(epoch.timetuple().tm_yday).zfill(3)
        file_name = f"BRDC00IGS_R_{epoch.year}{yday}0000_01D_MN.rnx"
        
        # Попробуем несколько вариантов URL для разных лет
        possible_urls = [
            f"https://simurg.space/files2/{epoch.year}/{yday}/nav/{file_name}.gz",
            f"https://simurg.space/files/{epoch.year}/{yday}/nav/{file_name}.gz"
        ]
        
        gziped_file = Path(tempdir) / (file_name + ".gz")
        local_file = Path(tempdir) / file_name
        
        print(f"📡 Загрузка nav-файла для {epoch.strftime('%Y-%m-%d')} (день {yday})")
        
        # Пробуем загрузить с разных источников
        for i, url in enumerate(possible_urls):
            try:
                print(f"🔄 Попытка {i+1}: {url}")
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    print(f"✅ Успешно получен nav-файл с источника {i+1}")
                    
                    # Сохраняем файл
                    with open(gziped_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Распаковываем gzip файл
                    print(f"📦 Распаковка файла {gziped_file.name}")
                    with gzip.open(gziped_file, 'rb') as f_in:
                        with open(local_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Проверяем размер файла
                    file_size = local_file.stat().st_size
                    if file_size > 1000:  # Минимальный размер nav-файла
                        print(f"✅ Nav-файл успешно загружен: {local_file} ({file_size} байт)")
                        return local_file
                    else:
                        print(f"⚠️ Файл слишком мал ({file_size} байт), возможно поврежден")
                        continue
                        
                elif response.status_code == 404:
                    print(f"❌ Файл не найден (404): {url}")
                    continue
                else:
                    print(f"❌ HTTP ошибка {response.status_code}: {url}")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ Таймаут при загрузке с источника {i+1}")
                continue
            except requests.exceptions.ConnectionError:
                print(f"🌐 Ошибка подключения к источнику {i+1}")
                continue
            except Exception as e:
                print(f"❌ Ошибка при загрузке с источника {i+1}: {e}")
                continue
        
        # Если все источники не сработали
        raise ValueError(f"Не удалось загрузить nav-файл для даты {epoch.strftime('%Y-%m-%d')} ни с одного источника")
        
    except Exception as e:
        print(f"❌ Критическая ошибка при загрузке nav-файла: {e}")
        raise

# --- Получение координат спутников ---
def get_sat_xyz(nav_file: Path, start: datetime, end: datetime, sats: list = GNSS_SATS, timestep: int = TIME_STEP_SECONDS):
    """
    Получение координат спутников из навигационного файла.
    
    Args:
        nav_file: путь к навигационному файлу
        start: начальное время
        end: конечное время
        sats: список спутников
        timestep: временной шаг в секундах
    
    Returns:
        tuple: (словарь координат спутников, список времен)
    """
    if not COORDINATES_AVAILABLE:
        print("❌ Библиотека coordinates не установлена")
        print("💡 Установите её командой: pip install git+https://github.com/gnss-lab/coordinates.git#egg=coordinates")
        return {}, []
    
    try:
        # Проверяем существование nav-файла
        if not nav_file.exists():
            print(f"❌ Nav-файл не найден: {nav_file}")
            return {}, []
        
        file_size = nav_file.stat().st_size
        print(f"📁 Обработка nav-файла: {nav_file} ({file_size} байт)")
        
        # Создаем временные метки
        times = []
        current_time = start
        while current_time <= end:
            times.append(current_time)
            current_time += timedelta(seconds=timestep)
        
        print(f"⏰ Временной диапазон: {len(times)} точек с {start} до {end}")
        
        # Получаем координаты для всех спутников
        sats_xyz = {}
        successful_sats = 0
        
        for sat_idx, sat in enumerate(sats):
            try:
                # Показываем прогресс
                if sat_idx % 10 == 0:
                    print(f"🛰️ Обработка спутника {sat_idx+1}/{len(sats)}: {sat}")
                
                # Получаем координаты для каждого спутника
                xyz_data = []
                valid_points = 0
                
                for t in times:
                    try:
                        # Используем правильный формат для библиотеки coordinates
                        # Формат: satellite_xyz(nav_file_path, constellation, prn, epoch)
                        constellation = sat[0]  # G, R, E, C
                        prn = int(sat[1:])      # номер спутника
                        
                        xyz = satellite_xyz(str(nav_file), constellation, prn, t)
                        
                        if xyz is not None and len(xyz) == 3 and not all(x == 0 for x in xyz):
                            xyz_data.append(xyz)
                            valid_points += 1
                        else:
                            # Если данных нет, пропускаем точку
                            xyz_data.append([0.0, 0.0, 0.0])
                            
                    except Exception as e:
                        # Молча пропускаем ошибки для отдельных эпох
                        xyz_data.append([0.0, 0.0, 0.0])
                        continue
                
                # Добавляем спутник только если есть достаточно валидных точек
                if valid_points > len(times) * 0.1:  # Минимум 10% валидных точек
                    sats_xyz[sat] = np.array(xyz_data)
                    successful_sats += 1
                    if sat_idx < 5:  # Показываем детали для первых спутников
                        print(f"  ✅ {sat}: {valid_points}/{len(times)} валидных точек")
                
            except Exception as e:
                # Пропускаем спутники с ошибками
                if sat_idx < 5:  # Показываем ошибки только для первых спутников
                    print(f"  ❌ {sat}: ошибка - {str(e)[:50]}")
                continue
        
        print(f"✅ Успешно обработано {successful_sats} спутников из {len(sats)}")
        
        if successful_sats == 0:
            print("❌ Не удалось получить координаты ни для одного спутника")
            print("💡 Возможные причины:")
            print("   - Nav-файл поврежден или неполный")
            print("   - Неправильная дата или формат времени")
            print("   - Проблемы с библиотекой coordinates")
        
        return sats_xyz, times
        
    except Exception as e:
        print(f"❌ Критическая ошибка в get_sat_xyz: {e}")
        return {}, []

# --- Преобразование XYZ в элевейшн/азимут ---
def xyz_to_el_az(xyz_site: tuple, xyz_sat: NDArray, earth_radius=RE):
    def cartesian_to_latlon(x, y, z, earth_radius=earth_radius):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z / r)
        return lat, lon, r - earth_radius

    (x_0, y_0, z_0) = xyz_site
    (x_s, y_s, z_s) = xyz_sat[:, 0], xyz_sat[:, 1], xyz_sat[:, 2]
    (b_0, l_0, h_0) = cartesian_to_latlon(*xyz_site)
    (b_s, l_s, h_s) = cartesian_to_latlon(x_s, y_s, z_s)
    r_k = np.sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2)
    sigma = np.arctan2(
        np.sqrt(1 - (np.sin(b_0) * np.sin(b_s) + np.cos(b_0) * np.cos(b_s) * np.cos(l_s - l_0)) ** 2),
        (np.sin(b_0) * np.sin(b_s) + np.cos(b_0) * np.cos(b_s) * np.cos(l_s - l_0))
    )
    x_t = -(x_s - x_0) * np.sin(l_0) + (y_s - y_0) * np.cos(l_0)
    y_t = (
        -1 * (x_s - x_0) * np.cos(l_0) * np.sin(b_0) -
        (y_s - y_0) * np.sin(l_0) * np.sin(b_0) +
        (z_s - z_0) * np.cos(b_0)
    )
    el = np.arctan2((np.cos(sigma) - earth_radius / r_k), np.sin(sigma))
    az = np.arctan2(x_t, y_t)
    az = np.where(az < 0, az + 2 * np.pi, az)
    return np.concatenate([[el], [az]]).T

def get_sat_elevation_azimuth(site_location_xyz: tuple, sats_xyz: dict):
    elaz = {}
    for sat, sat_xyz in sats_xyz.items():
        elaz[sat] = xyz_to_el_az(site_location_xyz, sat_xyz)
    return elaz

# --- Расчёт SIP ---
def calculate_sips(site_lat, site_lon, elevation, azimuth, ionospheric_height=HEIGHT_OF_THIN_IONOSPHERE, earth_radius=RE):
    psi = (
        (np.pi / 2 - elevation) -
        np.arcsin(np.cos(elevation) * earth_radius / (earth_radius + ionospheric_height))
    )
    lat = np.arcsin(
        np.sin(site_lat) * np.cos(psi) +
        np.cos(site_lat) * np.sin(psi) * np.cos(azimuth)
    )
    lon = site_lon + np.arcsin(np.sin(psi) * np.sin(azimuth) / np.cos(site_lat))
    lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
    lon = np.where(lon < -np.pi, lon + 2 * np.pi, lon)
    return np.concatenate([[np.degrees(lat)], [np.degrees(lon)]]).T

def get_sat_sips(site_latlon: tuple, sats_elaz: dict):
    sip_latlon = {}
    site_lat, site_lon = site_latlon
    for sat, elaz in sats_elaz.items():
        sips = calculate_sips(np.radians(site_lat), np.radians(site_lon), elaz[:, 0], elaz[:, 1])
        sip_latlon[sat] = sips
    return sip_latlon

# --- Great circle distance ---
def calculate_great_circle_distance(late, lone, latp, lonp, radius=RE):
    lone = np.where(lone < 0, lone + 2 * np.pi, lone)
    lonp = np.where(lonp < 0, lonp + 2 * np.pi, lonp)
    dlon = lonp - lone
    inds = np.where((dlon > 0) & (dlon > np.pi))
    dlon[inds] = 2 * np.pi - dlon[inds]
    dlon = np.where((dlon < 0) & (dlon < -np.pi), dlon + 2 * np.pi, dlon)
    dlon = np.where((dlon < 0) & (dlon < -np.pi), -dlon, dlon)
    cosgamma = np.sin(late) * np.sin(latp) + np.cos(late) * np.cos(latp) * np.cos(dlon)
    return radius * np.arccos(cosgamma)

def get_event_sat_gcd(event_latlon: tuple, sat_sips_latlon: dict):
    sat_event_gcd = {}
    event_lat, event_lon = event_latlon
    for sat, sip_latlon in sat_sips_latlon.items():
        elat_array = np.full_like(sip_latlon[:, 0], event_lat)
        elon_array = np.full_like(sip_latlon[:, 0], event_lon)
        gcd = calculate_great_circle_distance(
            np.radians(elat_array), np.radians(elon_array),
            np.radians(sip_latlon[:, 0]), np.radians(sip_latlon[:, 1])
        )
        sat_event_gcd[sat] = gcd
    return sat_event_gcd

# --- Маска по расстоянию ---
def calculte_sats_mask(sat_event_gcd: dict, max_distance: float):
    sat_mask = {}
    for sat, gcd in sat_event_gcd.items():
        mask = np.where(gcd < max_distance)[0]
        sat_mask[sat] = mask
    return sat_mask

@dataclass
class PointOfInterest:
    site: str
    sat: str
    time: datetime
    distance: float
    latitude: float
    longitude: float
    height: float

def get_point_of_interest(site: str, height: float, sat_mask: dict, times: list, sat_event_gcd: dict, sat_sips: dict):
    points = {}
    for sat, mask in sat_mask.items():
        points[sat] = []
        for i in mask:
            sip_lat = sat_sips[sat][i, 0]
            sip_lon = sat_sips[sat][i, 1]
            point = PointOfInterest(site, sat, times[i], sat_event_gcd[sat][i], sip_lat, sip_lon, height)
            points[sat].append(point)
    return points

# --- Функции для работы с API станций ---

def get_all_stations(limit=15000):
    """
    Получает список всех GNSS станций из API simurg.space
    
    Args:
        limit (int): Максимальное количество станций для загрузки (по умолчанию 15000)
    
    Returns:
        list: Список кортежей (station_id, lat, lon) или пустой список при ошибке
    """
    try:
        # Получаем список всех станций
        response = requests.get("https://api.simurg.space/sites/", timeout=30)
        response.raise_for_status()
        
        all_stations_data = response.json()
        total_stations = len(all_stations_data)
        
        # Берем все доступные станции или лимит
        max_stations = min(limit, total_stations)
        step = max(1, total_stations // max_stations)
        
        stations = []
        processed = 0
        valid_stations = 0
        error_stations = 0
        
        # Загружаем детали станций
        for i in range(0, total_stations, step):
            if len(stations) >= max_stations:
                break
                
            station_id = all_stations_data[i]
            
            try:
                # Получаем детали станции
                station_url = f"https://api.simurg.space/sites/{station_id}"
                station_response = requests.get(station_url, timeout=10)
                station_response.raise_for_status()
                
                station_data = station_response.json()
                
                # Извлекаем координаты
                if 'location' in station_data and 'lat' in station_data['location'] and 'lon' in station_data['location']:
                    lat = float(station_data['location']['lat'])
                    lon = float(station_data['location']['lon'])
                    
                    # Проверяем валидность координат
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        stations.append((station_id, lat, lon))
                        valid_stations += 1
                    else:
                        error_stations += 1
                else:
                    error_stations += 1
                    
            except Exception as e:
                error_stations += 1
                continue
            
            processed += 1
        
        return stations
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка сети при получении списка станций: {e}")
        return []
    except Exception as e:
        print(f"❌ Неожиданная ошибка при получении станций: {e}")
        return []

def find_stations_in_polygon(polygon_points):
    """
    Находит все станции внутри заданного полигона
    
    Args:
        polygon_points (list): Список точек полигона [(lat, lon), ...]
    
    Returns:
        dict: Словарь станций в полигоне {station_id: {'lat': lat, 'lon': lon, 'name': name, 'height': height}}
    """
    print(f"🔍 Поиск станций в полигоне с {len(polygon_points)} точками...")
    
    # Получаем все доступные станции
    all_stations_list = get_all_stations(limit=200)  # Увеличиваем лимит для лучшего покрытия
    
    if not all_stations_list:
        print("❌ Не удалось получить список станций")
        return {}
    
    print(f"📊 Проверяю {len(all_stations_list)} станций на попадание в полигон...")
    
    stations_in_polygon = {}
    
    for station_id, lat, lon in all_stations_list:
        if is_point_in_polygon(lat, lon, polygon_points):
            stations_in_polygon[station_id.lower()] = {
                'lat': lat,
                'lon': lon,
                'name': station_id.upper(),
                'height': 0.0  # Высота по умолчанию, можно получить из API если нужно
            }
    
    print(f"✅ Найдено {len(stations_in_polygon)} станций в полигоне")
    
    if stations_in_polygon:
        print("📍 Найденные станции:")
        for i, (station_id, station_info) in enumerate(list(stations_in_polygon.items())[:10]):
            print(f"  {i+1}. {station_id.upper()}: ({station_info['lat']:.2f}°, {station_info['lon']:.2f}°)")
        if len(stations_in_polygon) > 10:
            print(f"  ... и еще {len(stations_in_polygon) - 10} станций")
    
    return stations_in_polygon

def auto_download_nav_file_for_date(date):
    """
    Автоматически загружает NAV файл для указанной даты и получает станции из API
    
    Args:
        date: datetime.date объект
    
    Returns:
        dict: Информация о загруженном файле и найденных станциях
    """
    import tempfile
    from datetime import datetime
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Загружаем nav-файл
            nav_file_path = load_nav_file(datetime.combine(date, datetime.min.time()), temp_dir)
            
            # Получаем ВСЕ станции из API (убираем лимит для показа всех станций)
            stations_list = get_all_stations(limit=15000)  # Увеличиваем лимит для получения всех станций
            
            if not stations_list:
                raise Exception("Не удалось получить список станций из API")
            
            # Преобразуем список станций в нужный формат
            stations = {}
            for station_id, lat, lon in stations_list:
                stations[station_id.lower()] = {
                    'lat': lat,
                    'lon': lon,
                    'name': station_id.upper(),
                    'height': 0.0  # Высота по умолчанию
                }
            
            # Проверяем размер NAV файла
            nav_file_size = nav_file_path.stat().st_size
            lines_processed = 0
            
            # Считаем количество строк в NAV файле для статистики
            try:
                with open(nav_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines_processed = sum(1 for _ in f)
            except:
                lines_processed = 0
            
            return {
                'success': True,
                'stations': stations,
                'file_path': str(nav_file_path),
                'file_size': nav_file_size,
                'date': date,
                'stations_count': len(stations),
                'lines_processed': lines_processed,
                'source': 'API + NAV_file'
            }
            
    except Exception as e:
        error_msg = f"Ошибка загрузки данных для {date}: {str(e)}"
        return {
            'success': False,
            'error': error_msg,
            'date': date,
            'stations': {},
            'suggestions': [
                'Проверьте подключение к интернету',
                'Убедитесь, что дата корректна',
                f'NAV файл для {date} может быть недоступен на SIMuRG',
                'API simurg.space может быть временно недоступен',
                'Попробуйте выбрать другую дату'
            ]
        }

def request_ionosphere_data(date, structure_type, polygon_points, station_code=None, preloaded_nav_info=None):
    """
    Загружает nav-файл для указанной даты, находит все станции в полигоне,
    рассчитывает SIP траектории для всех найденных станций
    и возвращает только те траектории, которые пересекаются с полигоном.
    
    Args:
        date: datetime.date объект
        structure_type: тип структуры (не используется в новой версии)
        polygon_points: список точек полигона [(lat, lon), ...]
        station_code: код конкретной станции (если None, ищет все станции в полигоне)
        preloaded_nav_info: информация о предзагруженном nav-файле из session state
    
    Returns:
        dict: структурированные данные с SIP траекториями или сообщение об ошибке
    """
    # Если указана конкретная станция, используем новую функцию
    if station_code:
        print(f"🎯 Обработка конкретной станции: {station_code.upper()}")
        result = process_station_sips(station_code, date, polygon_points)
        
        if result['success']:
            # Преобразуем результат в формат, ожидаемый приложением
            return {
                'points': result['trajectory_points'],
                'metadata': {
                    'stations_processed': 1,
                    'stations_with_intersections': 1 if result['intersection_points'] > 0 else 0,
                    'total_intersection_points': result['intersection_points'],
                    'satellites_processed': result['satellites_total'],
                    'satellites_with_intersections': result['satellites_with_intersections'],
                    'time_range': result['time_range'],
                    'polygon_points_count': result['polygon_points_count'],
                    'date': result['date'],
                    'source': 'single_station_processing',
                    'priority': 'HIGH',
                    'processed_stations': [{
                        'code': result['station']['code'],
                        'name': result['station']['name'],
                        'lat': result['station']['lat'],
                        'lon': result['station']['lon'],
                        'satellites_with_intersections': result['satellites_with_intersections'],
                        'intersection_points': result['intersection_points']
                    }]
                }
            }
        else:
            return {
                'points': [],
                'metadata': {
                    'error': result['error'],
                    'station': result['station'],
                    'date': str(date)
                }
            }
    
    # Если станция не указана, используем старую логику для поиска всех станций в полигоне
    import requests
    import gzip
    import io
    import tempfile
    from datetime import datetime, date as date_class, timedelta
    
    # Используем реальный год из даты
    current_year = date.year
    day_of_year = date.timetuple().tm_yday
    
    print(f"🔍 Обработка данных для {current_year} года, день: {day_of_year}")
    
    # Ищем все станции в полигоне
    stations_to_process = find_stations_in_polygon(polygon_points)
    
    if not stations_to_process:
        return {
            'points': [],
            'metadata': {
                'error': 'В полигоне не найдено ни одной станции',
                'polygon_points_count': len(polygon_points),
                'date': str(date)
            }
        }
    
    print(f"📡 Будут обработаны станции: {list(stations_to_process.keys())}")
    
    # Загружаем навигационный файл
    try:
        # Проверяем, есть ли предзагруженный nav-файл для этой даты
        if (preloaded_nav_info and 
            preloaded_nav_info.get('loaded') and 
            preloaded_nav_info.get('date') == date):
            
            print(f"📡 Используем предзагруженный nav-файл для {date}")
            nav_file_path = preloaded_nav_info['path']
            temp_dir = preloaded_nav_info['temp_dir']
            print(f"✅ Навигационный файл: {nav_file_path}")
            
            # Определяем временной диапазон (весь день)
            start_time = datetime.combine(date, datetime.min.time())
            end_time = datetime.combine(date, datetime.max.time().replace(microsecond=0))
            
            print(f"⏰ Временной диапазон: {start_time} - {end_time}")
            
            # Получаем координаты спутников
            print(f"🛰️ Получение координат спутников из nav-файла...")
            sats_xyz, times = get_sat_xyz(nav_file_path, start_time, end_time, GNSS_SATS, TIME_STEP_SECONDS)
            
        else:
            # Загружаем nav-файл заново
            print(f"📡 Загрузка навигационного файла для {current_year}-{day_of_year:03d}...")
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Используем функцию load_nav_file для загрузки nav-файла
                    nav_file_path = load_nav_file(datetime.combine(date, datetime.min.time()), temp_dir)
                    print(f"✅ Навигационный файл загружен: {nav_file_path}")
                except ValueError as e:
                    # Ошибка загрузки nav-файла
                    error_msg = f"Не удалось загрузить nav-файл для {date}: {str(e)}"
                    print(f"❌ {error_msg}")
                    return {
                        'points': [],
                        'metadata': {
                            'error': error_msg,
                            'date': str(date),
                            'year': current_year,
                            'day_of_year': day_of_year,
                            'suggestions': [
                                'Проверьте подключение к интернету',
                                'Убедитесь, что дата корректна',
                                f'Nav-файл для {date} может быть недоступен на SIMuRG',
                                'Попробуйте выбрать другую дату'
                            ]
                        }
                    }
                except Exception as e:
                    error_msg = f"Неожиданная ошибка при загрузке nav-файла: {str(e)}"
                    print(f"❌ {error_msg}")
                    return {
                        'points': [],
                        'metadata': {
                            'error': error_msg,
                            'date': str(date)
                        }
                    }
                
                # Определяем временной диапазон (весь день)
                start_time = datetime.combine(date, datetime.min.time())
                end_time = datetime.combine(date, datetime.max.time().replace(microsecond=0))
                
                print(f"⏰ Временной диапазон: {start_time} - {end_time}")
                
                # Получаем координаты спутников
                print(f"🛰️ Получение координат спутников из nav-файла...")
                sats_xyz, times = get_sat_xyz(nav_file_path, start_time, end_time, GNSS_SATS, TIME_STEP_SECONDS)
        
        if not sats_xyz:
            error_msg = "Не удалось получить координаты спутников из nav-файла"
            print(f"❌ {error_msg}")
            return {
                'points': [],
                'metadata': {
                    'error': error_msg,
                    'date': str(date),
                    'nav_file': str(nav_file_path) if 'nav_file_path' in locals() else 'unknown',
                    'suggestions': [
                        'Проверьте, что библиотека coordinates установлена',
                        'Nav-файл может быть поврежден',
                        'Попробуйте другую дату',
                        'Установите coordinates: pip install git+https://github.com/gnss-lab/coordinates.git'
                    ]
                }
            }
        
        print(f"✅ Получены координаты для {len(sats_xyz)} спутников")
        
        # Проверяем качество данных спутников
        valid_satellites = []
        for sat, xyz_data in sats_xyz.items():
            # Проверяем, есть ли валидные (не нулевые) координаты
            non_zero_points = np.count_nonzero(np.any(xyz_data != 0, axis=1))
            if non_zero_points > len(times) * 0.1:  # Минимум 10% валидных точек
                valid_satellites.append(sat)
        
        print(f"📊 Спутников с достаточным количеством данных: {len(valid_satellites)}")
        
        if len(valid_satellites) < 4:  # Минимум для нормальной работы
            print("⚠️ Предупреждение: Мало спутников с валидными данными")
            print("💡 Это может повлиять на качество расчета SIP траекторий")
        
        # Обрабатываем каждую станцию
        all_result_points = []
        stations_processed = []
        total_intersections = 0
        
        for station_code, station in stations_to_process.items():
            print(f"\n🔄 Обработка станции {station_code.upper()}: {station['name']}")
            
            try:
                # Проверяем корректность данных станции
                if not all(key in station for key in ['lat', 'lon', 'height', 'name']):
                    print(f"  ❌ Некорректные данные станции {station_code.upper()}: отсутствуют обязательные поля")
                    continue
                
                # Проверяем валидность координат
                if not (-90 <= station['lat'] <= 90):
                    print(f"  ❌ Некорректная широта станции {station_code.upper()}: {station['lat']}")
                    continue
                
                if not (-180 <= station['lon'] <= 180):
                    print(f"  ❌ Некорректная долгота станции {station_code.upper()}: {station['lon']}")
                    continue
                
                # Координаты станции в XYZ
                station_lat_rad = np.radians(station['lat'])
                station_lon_rad = np.radians(station['lon'])
                station_height = station['height']
                
                # Приблизительное преобразование в ECEF координаты
                N = RE / np.sqrt(1 - (2*0.003353 - 0.003353**2) * np.sin(station_lat_rad)**2)
                station_xyz = (
                    (N + station_height) * np.cos(station_lat_rad) * np.cos(station_lon_rad),
                    (N + station_height) * np.cos(station_lat_rad) * np.sin(station_lon_rad),
                    (N * (1 - 0.003353**2) + station_height) * np.sin(station_lat_rad)
                )
                
                print(f"  📍 Координаты станции: {station['lat']:.4f}°, {station['lon']:.4f}°, {station_height}м")
                
                # Рассчитываем элевейшн/азимут для всех спутников
                try:
                    sats_elaz = get_sat_elevation_azimuth(station_xyz, sats_xyz)
                    if not sats_elaz:
                        print(f"  ⚠️ Не удалось рассчитать углы для станции {station_code.upper()}")
                        continue
                    print(f"  📐 Рассчитаны углы для {len(sats_elaz)} спутников")
                except Exception as e:
                    print(f"  ❌ Ошибка расчета углов для станции {station_code.upper()}: {e}")
                    continue
                
                # Рассчитываем SIP точки для всех спутников
                try:
                    sat_sips = get_sat_sips((station['lat'], station['lon']), sats_elaz)
                    if not sat_sips:
                        print(f"  ⚠️ Не удалось рассчитать SIP точки для станции {station_code.upper()}")
                        continue
                    print(f"  🎯 Рассчитаны SIP точки для {len(sat_sips)} спутников")
                except Exception as e:
                    print(f"  ❌ Ошибка расчета SIP точек для станции {station_code.upper()}: {e}")
                    continue
                
                # Фильтруем SIP траектории по полигону
                try:
                    filtered_sips = filter_sips_by_polygon(sat_sips, polygon_points, times)
                    print(f"  🔍 Фильтрация по полигону завершена")
                except Exception as e:
                    print(f"  ❌ Ошибка фильтрации по полигону для станции {station_code.upper()}: {e}")
                    continue
                
                # Подсчитываем количество точек пересечения для этой станции
                station_intersections = 0
                trajectory_info = {}
                
                for sat, trajectory in filtered_sips.items():
                    if trajectory and len(trajectory) > 0:
                        station_intersections += len(trajectory)
                        trajectory_info[sat] = {
                            'points_count': len(trajectory),
                            'trajectory': trajectory
                        }
                
                if station_intersections > 0:
                    print(f"  ✅ Станция {station_code.upper()}: {len(trajectory_info)} спутников, {station_intersections} пересечений")
                    
                    # Формируем результат для этой станции
                    try:
                        for sat, info in trajectory_info.items():
                            for i, point in enumerate(info['trajectory']):
                                # Проверяем валидность данных точки
                                if not isinstance(point, dict) or 'lat' not in point or 'lon' not in point:
                                    continue
                                
                                # Получаем углы для данного времени
                                elevation = float(sats_elaz[sat][i, 0]) if sat in sats_elaz and i < len(sats_elaz[sat]) else 0.0
                                azimuth = float(sats_elaz[sat][i, 1]) if sat in sats_elaz and i < len(sats_elaz[sat]) else 0.0
                                
                                all_result_points.append({
                                    'satellite': sat,
                                    'latitude': point['lat'],
                                    'longitude': point['lon'],
                                    'time': point['time'].isoformat() if 'time' in point else times[i].isoformat() if i < len(times) else '',
                                    'elevation': elevation,
                                    'azimuth': azimuth,
                                    'station': station_code.upper(),
                                    'station_name': station['name'],
                                    'station_lat': station['lat'],
                                    'station_lon': station['lon']
                                })
                    except Exception as e:
                        print(f"  ❌ Ошибка формирования результатов для станции {station_code.upper()}: {e}")
                        continue
                    
                    stations_processed.append({
                        'code': station_code.upper(),
                        'name': station['name'],
                        'lat': station['lat'],
                        'lon': station['lon'],
                        'satellites_with_intersections': len(trajectory_info),
                        'intersection_points': station_intersections
                    })
                    
                    total_intersections += station_intersections
                else:
                    print(f"  ⚪ Станция {station_code.upper()}: нет пересечений с полигоном")
                
            except Exception as e:
                print(f"  ❌ Критическая ошибка обработки станции {station_code.upper()}: {e}")
                print(f"  💡 Возможные причины: некорректные координаты, проблемы с nav-файлом, ошибки расчета")
                continue
        
        print(f"\n🎉 Обработка завершена!")
        print(f"📊 Всего обработано станций: {len(stations_processed)}")
        print(f"📊 Общее количество точек пересечения: {total_intersections}")
        
        return {
            'points': all_result_points,
            'metadata': {
                'stations_in_polygon': len(stations_to_process),
                'stations_processed': len(stations_processed),
                'stations_with_intersections': len(stations_processed),
                'total_intersection_points': total_intersections,
                'satellites_processed': len(sats_xyz),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'polygon_points_count': len(polygon_points),
                'nav_file': str(nav_file_path),
                'date': str(date),
                'year': current_year,
                'day_of_year': day_of_year,
                'source': 'nav_file + SIP_calculation + API_stations',
                'priority': 'HIGH',
                'processed_stations': stations_processed
            }
        }
        
    except Exception as e:
        print(f"❌ Ошибка при обработке данных: {e}")
        return {
            'points': [],
            'metadata': {
                'error': f'Ошибка при обработке данных: {str(e)}',
                'date': str(date)
            }
        }

# --- Функции парсинга различных форматов данных ---

def parse_text_ionosphere_content(content, polygon_points, structure_type):
    """Парсинг текстовых файлов с данными ионосферы"""
    try:
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        lines = content.strip().split('\n')
        points = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        tec = float(parts[2])
                        
                        if is_point_in_polygon(lat, lon, polygon_points):
                            points.append({
                                'latitude': lat,
                                'longitude': lon,
                                'tec': tec,
                                'index': 0.0,
                                'timestamp': 'real_data',
                                'source': 'simurg_text',
                                'quality': 'real'
                            })
                    except ValueError:
                        continue
        
        return {
            'points': points,
            'metadata': {
                'source': 'text_file',
                'file_type': 'txt',
                'parsed_lines': len(lines),
                'valid_points': len(points)
            }
        }
    except Exception as e:
        print(f"Ошибка парсинга текстового файла: {e}")
        return None

def parse_hdf5_content(content, polygon_points, structure_type):
    """Парсинг HDF5 файлов"""
    try:
        import tempfile
        import h5py
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            points = []
            with h5py.File(tmp_file.name, 'r') as f:
                # Пытаемся найти стандартные датасеты
                if 'latitude' in f and 'longitude' in f and 'tec' in f:
                    lats = f['latitude'][:]
                    lons = f['longitude'][:]
                    tecs = f['tec'][:]
                    
                    for lat, lon, tec in zip(lats, lons, tecs):
                        if is_point_in_polygon(lat, lon, polygon_points):
                            points.append({
                                'latitude': float(lat),
                                'longitude': float(lon),
                                'tec': float(tec),
                                'index': 0.0,
                                'timestamp': 'real_data',
                                'source': 'simurg_hdf5',
                                'quality': 'real'
                            })
            
            return {
                'points': points,
                'metadata': {
                    'source': 'hdf5_file',
                    'file_type': 'h5',
                    'valid_points': len(points)
                }
            }
    except Exception as e:
        print(f"Ошибка парсинга HDF5 файла: {e}")
        return None

def parse_rinex_content(content, polygon_points, structure_type):
    """Парсинг RINEX файлов"""
    # Заглушка для RINEX файлов
    return {
        'points': [],
        'metadata': {
            'source': 'rinex_file',
            'file_type': 'rinex',
            'note': 'RINEX parsing not implemented'
        }
    }

def parse_gim_content(content, polygon_points, structure_type):
    """Парсинг GIM файлов"""
    # Заглушка для GIM файлов
    return {
        'points': [],
        'metadata': {
            'source': 'gim_file',
            'file_type': 'gim',
            'note': 'GIM parsing not implemented'
        }
    }

def parse_tec_dat_file(content, polygon_points, structure_type):
    """Парсинг .dat файлов с TEC данными"""
    return parse_text_ionosphere_content(content, polygon_points, structure_type)

def parse_gim_file(gim_path):
    """
    Обработка GIM (Global Ionosphere Map) файла
    
    Args:
        gim_path: str - путь к GIM файлу
    
    Returns:
        dict - структурированные данные ионосферы
    """
    # Функция заменена на parse_gim_file_content для работы с содержимым файлов
    logger.warning("parse_gim_file устарела, используйте parse_gim_file_content")
    return None

def parse_text_ionosphere_file(file_path):
    """
    Обработка текстового файла с данными ионосферы
    
    Args:
        file_path: str - путь к текстовому файлу
    
    Returns:
        dict - структурированные данные ионосферы
    """
    # Функция заменена на parse_text_ionosphere_content для работы с содержимым файлов
    logger.warning("parse_text_ionosphere_file устарела, используйте parse_text_ionosphere_content")
    return None

def filter_data_by_polygon(data, polygon_points):
    """
    Фильтрация данных ионосферы по полигону
    
    Args:
        data: dict - данные ионосферы
        polygon_points: list - точки полигона
    
    Returns:
        dict - отфильтрованные данные
    """
    if not data or not polygon_points or len(polygon_points) < 3:
        return data
    
    filtered_data = {
        'points': [],
        'metadata': data.get('metadata', {})
    }
    
    # Фильтруем точки, которые находятся внутри полигона
    for point in data.get('points', []):
        lat = point.get('latitude', 0)
        lon = point.get('longitude', 0)
        if is_point_in_polygon(lat, lon, polygon_points):
            filtered_data['points'].append(point)
    
    logger.info(f"Отфильтровано {len(filtered_data['points'])} точек из {len(data.get('points', []))} по полигону")
    return filtered_data 

def is_point_in_polygon(lat, lon, polygon):
    """
    Проверяет, находится ли точка внутри полигона
    Использует алгоритм ray casting
    
    Args:
        lat (float): Широта точки
        lon (float): Долгота точки  
        polygon (list): Список точек полигона [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
        bool: True если точка внутри полигона
    """
    if len(polygon) < 3:
        return False
    
    x, y = lon, lat
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][1], polygon[i][0]  # lon, lat
        xj, yj = polygon[j][1], polygon[j][0]  # lon, lat
        
        # Проверяем пересечение луча с ребром полигона
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside

def filter_sips_by_polygon(sat_sips, polygon_coords, times):
    """
    Фильтрует SIP точки по полигону.
    
    Args:
        sat_sips: словарь SIP координат для каждого спутника
        polygon_coords: координаты полигона [(lat, lon), ...]
        times: список временных меток
    
    Returns:
        dict: отфильтрованные траектории спутников
    """
    filtered_trajectories = {}
    
    for sat, sips in sat_sips.items():
        trajectory = []
        
        for i, sip in enumerate(sips):
            lat, lon = sip[0], sip[1]
            
            # Проверяем, находится ли точка внутри полигона
            if is_point_in_polygon(lat, lon, polygon_coords):
                time_obj = times[i] if i < len(times) else None
                trajectory.append({
                    'lat': lat,
                    'lon': lon,
                    'time': time_obj
                })
        
        if trajectory:
            filtered_trajectories[sat] = trajectory
    
    return filtered_trajectories 

def process_station_sips(station_code, date, polygon_points):
    """
    Обрабатывает конкретную станцию - получает все SIP траектории, пересекающие полигон
    
    Args:
        station_code (str): Код станции (например, 'ERKG')
        date (datetime.date): Дата для обработки
        polygon_points (list): Список точек полигона [(lat, lon), ...]
    
    Returns:
        dict: Результат обработки с SIP траекториями или ошибкой
    """
    import tempfile
    from datetime import datetime, timedelta
    
    print(f"🔄 Обработка станции {station_code.upper()} для даты {date}")
    
    try:
        # 1. Получаем информацию о станции из API
        print(f"📡 Получение информации о станции {station_code.upper()}...")
        try:
            station_url = f"https://api.simurg.space/sites/{station_code.lower()}"
            station_response = requests.get(station_url, timeout=10)
            station_response.raise_for_status()
            
            station_data = station_response.json()
            
            if 'location' not in station_data or 'lat' not in station_data['location'] or 'lon' not in station_data['location']:
                return {
                    'success': False,
                    'error': f'Некорректные данные станции {station_code.upper()}: отсутствуют координаты',
                    'station': station_code.upper()
                }
            
            station_info = {
                'lat': float(station_data['location']['lat']),
                'lon': float(station_data['location']['lon']),
                'name': station_code.upper(),
                'height': float(station_data['location'].get('height', 0.0))
            }
            
            print(f"✅ Станция найдена: {station_info['name']} ({station_info['lat']:.4f}°, {station_info['lon']:.4f}°, {station_info['height']:.1f}м)")
            
        except requests.exceptions.RequestException:
            return {
                'success': False,
                'error': f'Станция {station_code.upper()} не найдена в API',
                'station': station_code.upper()
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка получения данных станции {station_code.upper()}: {str(e)}',
                'station': station_code.upper()
            }
        
        # 2. Загружаем навигационный файл
        print(f"📁 Загрузка навигационного файла для {date}...")
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                nav_file_path = load_nav_file(datetime.combine(date, datetime.min.time()), temp_dir)
                print(f"✅ Навигационный файл загружен: {nav_file_path}")
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Не удалось загрузить nav-файл для {date}: {str(e)}',
                    'station': station_code.upper(),
                    'date': str(date)
                }
            
            # 3. Определяем временной диапазон (весь день)
            start_time = datetime.combine(date, datetime.min.time())
            end_time = datetime.combine(date, datetime.max.time().replace(microsecond=0))
            
            print(f"⏰ Временной диапазон: {start_time} - {end_time}")
            
            # 4. Получаем координаты всех спутников из nav-файла
            print(f"🛰️ Получение координат спутников из nav-файла...")
            sats_xyz, times = get_sat_xyz(nav_file_path, start_time, end_time, GNSS_SATS, TIME_STEP_SECONDS)
            
            if not sats_xyz:
                return {
                    'success': False,
                    'error': 'Не удалось получить координаты спутников из nav-файла',
                    'station': station_code.upper(),
                    'date': str(date)
                }
            
            print(f"✅ Получены координаты для {len(sats_xyz)} спутников")
            
            # 5. Преобразуем координаты станции в XYZ (ECEF)
            station_lat_rad = np.radians(station_info['lat'])
            station_lon_rad = np.radians(station_info['lon'])
            station_height = station_info['height']
            
            # Преобразование в ECEF координаты
            N = RE / np.sqrt(1 - (2*0.003353 - 0.003353**2) * np.sin(station_lat_rad)**2)
            station_xyz = (
                (N + station_height) * np.cos(station_lat_rad) * np.cos(station_lon_rad),
                (N + station_height) * np.cos(station_lat_rad) * np.sin(station_lon_rad),
                (N * (1 - 0.003353**2) + station_height) * np.sin(station_lat_rad)
            )
            
            print(f"📍 Координаты станции XYZ: ({station_xyz[0]:.0f}, {station_xyz[1]:.0f}, {station_xyz[2]:.0f})")
            
            # 6. Рассчитываем элевейшн/азимут для всех спутников
            print(f"📐 Расчет углов элевейшн/азимут для всех спутников...")
            sats_elaz = get_sat_elevation_azimuth(station_xyz, sats_xyz)
            
            if not sats_elaz:
                return {
                    'success': False,
                    'error': f'Не удалось рассчитать углы для станции {station_code.upper()}',
                    'station': station_code.upper()
                }
            
            print(f"✅ Рассчитаны углы для {len(sats_elaz)} спутников")
            
            # 7. Рассчитываем SIP точки для всех спутников
            print(f"🎯 Расчет SIP точек для всех спутников...")
            sat_sips = get_sat_sips((station_info['lat'], station_info['lon']), sats_elaz)
            
            if not sat_sips:
                return {
                    'success': False,
                    'error': f'Не удалось рассчитать SIP точки для станции {station_code.upper()}',
                    'station': station_code.upper()
                }
            
            print(f"✅ Рассчитаны SIP точки для {len(sat_sips)} спутников")
            
            # 8. Фильтруем SIP траектории по полигону
            print(f"🔍 Фильтрация SIP траекторий по полигону...")
            filtered_sips = filter_sips_by_polygon(sat_sips, polygon_points, times)
            
            # 9. Подсчитываем результаты
            total_intersections = 0
            satellites_with_intersections = []
            all_trajectory_points = []
            
            for sat, trajectory in filtered_sips.items():
                if trajectory and len(trajectory) > 0:
                    total_intersections += len(trajectory)
                    satellites_with_intersections.append(sat)
                    
                    # Добавляем точки траектории в общий список
                    for i, point in enumerate(trajectory):
                        # Получаем углы для данного времени
                        elevation = float(sats_elaz[sat][i, 0]) if sat in sats_elaz and i < len(sats_elaz[sat]) else 0.0
                        azimuth = float(sats_elaz[sat][i, 1]) if sat in sats_elaz and i < len(sats_elaz[sat]) else 0.0
                        
                        all_trajectory_points.append({
                            'satellite': sat,
                            'latitude': point['lat'],
                            'longitude': point['lon'],
                            'time': point['time'].isoformat() if 'time' in point else times[i].isoformat() if i < len(times) else '',
                            'elevation': elevation,
                            'azimuth': azimuth,
                            'station': station_code.upper(),
                            'station_name': station_info['name'],
                            'station_lat': station_info['lat'],
                            'station_lon': station_info['lon']
                        })
            
            print(f"🎉 Обработка завершена!")
            print(f"📊 Спутников с пересечениями: {len(satellites_with_intersections)} из {len(sats_xyz)}")
            print(f"📊 Общее количество точек пересечения: {total_intersections}")
            
            if satellites_with_intersections:
                print(f"🛰️ Спутники с пересечениями: {', '.join(satellites_with_intersections[:10])}")
                if len(satellites_with_intersections) > 10:
                    print(f"   ... и еще {len(satellites_with_intersections) - 10} спутников")
            
            return {
                'success': True,
                'station': {
                    'code': station_code.upper(),
                    'name': station_info['name'],
                    'lat': station_info['lat'],
                    'lon': station_info['lon'],
                    'height': station_info['height']
                },
                'date': str(date),
                'satellites_total': len(sats_xyz),
                'satellites_with_intersections': len(satellites_with_intersections),
                'intersection_points': total_intersections,
                'polygon_points_count': len(polygon_points),
                'trajectory_points': all_trajectory_points,
                'satellites_list': satellites_with_intersections,
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'metadata': {
                    'nav_file': str(nav_file_path),
                    'source': 'nav_file + SIP_calculation',
                    'processing_type': 'single_station'
                }
            }
            
    except Exception as e:
        print(f"❌ Критическая ошибка обработки станции {station_code.upper()}: {e}")
        return {
            'success': False,
            'error': f'Критическая ошибка: {str(e)}',
            'station': station_code.upper()
        } 
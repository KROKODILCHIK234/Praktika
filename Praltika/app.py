import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from sip_utils import *  
import pandas as pd
import json
from math import radians, sin, cos, sqrt, atan2
import re
from streamlit_plotly_events import plotly_events  # Добавляем импорт библиотеки для обработки кликов
import h5py  # Для работы с HDF файлами
import requests  # Для загрузки данных
import sys  # Для прогресс бара
from pathlib import Path as PathLib  # Для работы с путями
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from collections import defaultdict
from dateutil import tz  # Для работы с временными зонами
import pickle  # Для сериализации данных
import os  # Для работы с файловой системой
from typing import Union
from plotly.subplots import make_subplots
import tempfile

st.set_page_config(page_title="Локализация SIP", layout="wide")

# ==================== PICKLE COMPATIBILITY FIX ====================

# Регистрируем классы в глобальном пространстве имен для pickle совместимости
def register_classes_for_pickle():
    """Регистрирует классы в sys.modules['__main__'] для pickle совместимости"""
    try:
        import sys
        current_module = sys.modules[__name__]
        main_module = sys.modules.get('__main__')
        
        if main_module and hasattr(current_module, 'GnssSite'):
            # Копируем классы в модуль __main__
            main_module.GnssSite = current_module.GnssSite
            main_module.GnssSat = current_module.GnssSat
            main_module.DataProduct = current_module.DataProduct
            main_module.ColorLimits = current_module.ColorLimits
            main_module.DataProducts = current_module.DataProducts
    except Exception as e:
        print(f"Ошибка регистрации классов: {e}")

# ==================== END PICKLE COMPATIBILITY FIX ====================

# ==================== PERSISTENT DATA STORAGE ====================

# Папка для сохранения данных
DATA_DIR = PathLib("app_data")
DATA_DIR.mkdir(exist_ok=True)

# Папка для HDF файлов
HDF_DIR = DATA_DIR / "hdf_files"
HDF_DIR.mkdir(exist_ok=True)

def save_session_data():
    """Сохраняет важные данные сессии в файл"""
    try:
        # Данные для сохранения
        persistent_data = {
            'polygon_points': st.session_state.get('polygon_points', []),
            'polygon_completed': st.session_state.get('polygon_completed', False),
            'polygon_mode': st.session_state.get('polygon_mode', False),
            'last_selected_structure': st.session_state.get('last_selected_structure', 'equatorial anomaly'),
            'ionosphere_data': st.session_state.get('ionosphere_data', None),
            'hdf_file_path': str(st.session_state.get('hdf_file_path', '')) if st.session_state.get('hdf_file_path') else '',
            'hdf_date': st.session_state.get('hdf_date', None),
            'site_sat_data_available': 'site_sat_data' in st.session_state and st.session_state['site_sat_data'] is not None,
            'selected_sites_count': len(st.session_state.get('selected_sites', [])),
            'nav_file_stations': st.session_state.get('nav_file_stations', None),
            'nav_date_loaded': st.session_state.get('nav_date_loaded', False),
            'nav_date_stations': st.session_state.get('nav_date_stations', None),
            'last_selected_date': st.session_state.get('last_selected_date', None)
        }
        
        # Сохраняем в файл
        with open(DATA_DIR / "session_data.pkl", "wb") as f:
            pickle.dump(persistent_data, f)
            
    except Exception as e:
        # Не показываем ошибку пользователю, просто логируем
        print(f"Ошибка при сохранении данных: {e}")

def load_session_data():
    """Загружает сохраненные данные сессии"""
    try:
        # Проверяем, существует ли директория для данных
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(exist_ok=True)
            return False
            
        session_file = DATA_DIR / "session_data.pkl"
        if session_file.exists():
            with open(session_file, "rb") as f:
                persistent_data = pickle.load(f)
            
            # Восстанавливаем данные в session_state
            for key, value in persistent_data.items():
                if key == 'hdf_file_path' and value:
                    # Проверяем, что HDF файл еще существует
                    hdf_path = PathLib(value)
                    if hdf_path.exists() and hdf_path.is_file():
                        st.session_state['hdf_file_path'] = hdf_path
                elif key == 'site_sat_data_available':
                    # Пропускаем этот ключ, данные будут загружены отдельно
                    continue
                elif key == 'selected_sites_count':
                    # Пропускаем этот ключ, данные будут загружены отдельно
                    continue
                else:
                    st.session_state[key] = value
            
            return True
    except (pickle.PickleError, ImportError, AttributeError, ModuleNotFoundError) as e:
        st.warning(f"⚠️ Не удалось загрузить основные сохраненные данные: {e}")
        # Удаляем проблемный файл
        try:
            session_file = DATA_DIR / "session_data.pkl"
            if session_file.exists():
                session_file.unlink()
        except:
            pass
        return False
    except Exception as e:
        st.warning(f"⚠️ Не удалось загрузить сохраненные данные: {e}")
        return False
    return False

def save_hdf_data():
    """Сохраняет HDF данные отдельно (они большие)"""
    try:
        if 'site_sat_data' in st.session_state and st.session_state['site_sat_data']:
            # Сохраняем site_sat_data
            with open(DATA_DIR / "site_sat_data.pkl", "wb") as f:
                pickle.dump(st.session_state['site_sat_data'], f)
        
        if 'sat_data' in st.session_state and st.session_state['sat_data']:
            # Сохраняем sat_data
            with open(DATA_DIR / "sat_data.pkl", "wb") as f:
                pickle.dump(st.session_state['sat_data'], f)
                
        if 'selected_sites' in st.session_state and st.session_state['selected_sites']:
            # Сохраняем selected_sites
            with open(DATA_DIR / "selected_sites.pkl", "wb") as f:
                pickle.dump(st.session_state['selected_sites'], f)
                
    except Exception as e:
        # Не показываем ошибку пользователю, просто логируем
        print(f"Ошибка при сохранении HDF данных: {e}")

def load_hdf_data():
    """Загружает HDF данные"""
    try:
        # Проверяем, существует ли директория для данных
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(exist_ok=True)
            return False
            
        # Загружаем site_sat_data
        site_sat_file = DATA_DIR / "site_sat_data.pkl"
        if site_sat_file.exists():
            with open(site_sat_file, "rb") as f:
                st.session_state['site_sat_data'] = pickle.load(f)
        
        # Загружаем sat_data
        sat_data_file = DATA_DIR / "sat_data.pkl"
        if sat_data_file.exists():
            with open(sat_data_file, "rb") as f:
                st.session_state['sat_data'] = pickle.load(f)
                
        # Загружаем selected_sites
        selected_sites_file = DATA_DIR / "selected_sites.pkl"
        if selected_sites_file.exists():
            with open(selected_sites_file, "rb") as f:
                st.session_state['selected_sites'] = pickle.load(f)
                
        return True
    except (pickle.PickleError, ImportError, AttributeError, ModuleNotFoundError) as e:
        st.warning(f"⚠️ Не удалось загрузить HDF данные: {e}")
        # Удаляем проблемные файлы
        try:
            for file_name in ["site_sat_data.pkl", "sat_data.pkl", "selected_sites.pkl"]:
                file_path = DATA_DIR / file_name
                if file_path.exists():
                    file_path.unlink()
        except:
            pass
        return False
    except Exception as e:
        st.warning(f"⚠️ Не удалось загрузить HDF данные: {e}")
        return False

def clear_all_data():
    """Очищает все сохраненные данные"""
    try:
        # Создаем папку если её нет
        DATA_DIR.mkdir(exist_ok=True)
        
        # Удаляем файлы
        files_to_clear = ["session_data.pkl", "site_sat_data.pkl", "sat_data.pkl", "selected_sites.pkl"]
        cleared_files = []
        
        for file_name in files_to_clear:
            file_path = DATA_DIR / file_name
            if file_path.exists():
                file_path.unlink()
                cleared_files.append(file_name)
        
        # Очищаем HDF файлы
        if HDF_DIR.exists() and HDF_DIR.is_dir():
            hdf_files_count = 0
            for hdf_file in HDF_DIR.glob("*.h5"):
                hdf_file.unlink()
                hdf_files_count += 1
            if hdf_files_count > 0:
                cleared_files.append(f"{hdf_files_count} HDF файлов")
        
        # Очищаем session_state
        keys_to_clear = [
            'polygon_points', 'polygon_completed', 'polygon_mode', 'last_selected_structure',
            'ionosphere_data', 'hdf_file_path', 'hdf_date', 'site_sat_data', 'sat_data', 
            'selected_sites', 'nav_file_stations', 'nav_date_loaded', 'nav_date_stations'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        if cleared_files:
            st.success(f"✅ Очищены файлы: {', '.join(cleared_files)}")
        else:
            st.info("ℹ️ Нет данных для очистки")
                
    except Exception as e:
        st.error(f"❌ Ошибка при очистке данных: {e}")

# Инициализируем session_state с базовыми значениями
def init_session_state():
    """Инициализирует session_state с базовыми значениями"""
    defaults = {
        'polygon_points': [],
        'polygon_completed': False,
        'polygon_mode': False,
        'last_selected_structure': 'equatorial anomaly',
        'map_loaded': False,
        'coords_from_map': False,
        'auto_add_points': True,
        'nav_file_stations': None,
        'nav_date_loaded': False,
        'nav_date_stations': None,
        'last_selected_date': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def validate_hdf_path(path):
    """Проверяет и исправляет путь к HDF файлу"""
    if path is None:
        return None
        
    # Конвертируем в PathLib если это строка
    if isinstance(path, str):
        path = PathLib(path)
    
    # Проверяем существование файла
    if not path.exists():
        return None
    
    # Проверяем, что это файл, а не директория
    if not path.is_file():
        return None
    
    return path

def validate_and_fix_hdf_path(path):
    """Проверяет и исправляет путь к HDF файлу, обновляя session_state"""
    if path is None:
        if 'hdf_file_path' in st.session_state:
            del st.session_state['hdf_file_path']
        return None
        
    # Конвертируем в PathLib если это строка
    if isinstance(path, str):
        path = PathLib(path)
    
    # Проверяем существование файла
    if not path.exists():
        st.warning(f"⚠️ Файл {path} не существует")
        if 'hdf_file_path' in st.session_state:
            del st.session_state['hdf_file_path']
        return None
    
    # Проверяем, что это файл, а не директория
    if not path.is_file():
        st.warning(f"⚠️ Путь {path} указывает на директорию, а не файл")
        if 'hdf_file_path' in st.session_state:
            del st.session_state['hdf_file_path']
        return None
    
    # Обновляем путь в session_state
    st.session_state['hdf_file_path'] = path
    return path

# Загружаем сохраненные данные при старте приложения
if 'data_loaded' not in st.session_state:
    init_session_state()
    load_session_data()
    load_hdf_data()
    
    # Проверяем путь к HDF файлу
    if 'hdf_file_path' in st.session_state:
        validate_and_fix_hdf_path(st.session_state['hdf_file_path'])
    
    st.session_state['data_loaded'] = True

# ==================== END PERSISTENT DATA STORAGE ====================

# ==================== HDF DATA CLASSES AND FUNCTIONS ====================

@dataclass
class ColorLimits():
    min: float
    max: float
    units: str

@dataclass(frozen=True)
class DataProduct():
    long_name: str
    hdf_name: str
    color_limits: ColorLimits

@dataclass
class SimurgDataFile():
    url: str
    local_path: PathLib

@dataclass(frozen=True)
class GnssSite:
    name: str
    lat: float
    lon: float

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, GnssSite):
            return NotImplemented
        return self.name == other.name

@dataclass
class GnssSat:
    name: str
    system: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, GnssSat):
            return NotImplemented
        return self.name == other.name

class DataProducts(Enum):
    roti = DataProduct("ROTI", "roti", ColorLimits(0, 0.5, 'TECU/min'))
    dtec_2_10 = DataProduct("2-10 minute TEC variations", "dtec_2_10", ColorLimits(-0.4, 0.4, 'TECU'))
    dtec_10_20 = DataProduct("10-20 minute TEC variations", "dtec_10_20", ColorLimits(-0.6, 0.6, 'TECU'))
    dtec_20_60 = DataProduct("20-60 minute TEC variations", "dtec_20_60", ColorLimits(-0.8, 0.8, 'TECU'))
    atec = DataProduct("Vertical TEC adjusted using GIM", "tec_adjusted", ColorLimits(0, 50, 'TECU'))
    elevation = DataProduct("Elevation angle", "elevation", ColorLimits(0, 90, 'Degrees'))
    azimuth = DataProduct("Azimuth angle", "azimuth", ColorLimits(0, 360, 'Degrees'))
    timestamp = DataProduct("Timestamp", "timestamp", None)
    time = DataProduct("Time", None, None)

# Регистрируем классы для pickle совместимости после их определения
register_classes_for_pickle()

def load_hdf_data(url: str, local_file: PathLib, override: bool = False) -> None:
    """Загружает HDF файл с SIMuRG"""
    if local_file.exists() and not override:
        st.info(f"📁 Файл {local_file.name} уже существует")
        return

    # Проверяем, что путь не указывает на директорию
    if local_file.is_dir():
        st.error(f"❌ Ошибка: {local_file} является директорией, а не файлом")
        return

    with st.spinner(f"📥 Загрузка {local_file.name} с {url}..."):
        try:
            # Создаем родительскую директорию, если она не существует
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_file, "wb") as f:
                response = requests.get(url, stream=True)
                
                # Проверяем успешность запроса
                if response.status_code != 200:
                    st.error(f"❌ Ошибка при загрузке файла: HTTP {response.status_code}")
                    return
                    
                total_length = response.headers.get('content-length')

                if total_length is None:
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    progress_bar = st.progress(0)
                    for chunk in response.iter_content(chunk_size=4096):
                        dl += len(chunk)
                        f.write(chunk)
                        done = int(100 * dl / total_length)
                        progress_bar.progress(done)
                    progress_bar.empty()
            
            st.success(f"✅ Файл {local_file.name} успешно загружен в директорию приложения")
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")
            # Удаляем частично загруженный файл при ошибке
            if local_file.exists():
                try:
                    local_file.unlink()
                except:
                    pass

def get_sites_from_hdf(local_file: Union[str, PathLib], min_lat: float = -90, max_lat: float = 90, 
                      min_lon: float = -180, max_lon: float = 180) -> list[GnssSite]:
    """Извлекает станции из HDF файла"""
    try:
        # Проверяем, что путь указывает на файл, а не директорию
        path = local_file if isinstance(local_file, PathLib) else PathLib(local_file)
        if not path.is_file():
            st.error(f"❌ Ошибка: {path} не является файлом или не существует")
            return []
            
        with h5py.File(local_file, 'r') as f:
            sites_names = list(f.keys())
            sites = []
            for site_name in sites_names:
                if site_name in f:
                    site_info = f[site_name].attrs
                    site_lat = np.degrees(site_info['lat'])
                    site_lon = np.degrees(site_info['lon'])
                    if min_lat < site_lat < max_lat and min_lon < site_lon < max_lon:
                        sites.append(GnssSite(site_name, site_lat, site_lon))
            return sites
    except Exception as e:
        st.error(f"❌ Ошибка при чтении HDF файла: {e}")
        return []

def retrieve_visible_sats_data(local_file: Union[str, PathLib], sites: list[GnssSite]) -> dict[GnssSite, dict[GnssSat, dict[DataProduct, NDArray]]]:
    """Извлекает данные спутников для заданных станций"""
    from datetime import datetime
    from dateutil import tz
    
    _UTC = tz.gettz('UTC')
    
    try:
        # Проверяем, что путь указывает на файл, а не директорию
        path = local_file if isinstance(local_file, PathLib) else PathLib(local_file)
        if not path.is_file():
            st.error(f"❌ Ошибка: {path} не является файлом или не существует")
            return {}
            
        with h5py.File(local_file, 'r') as f:
            data = {}
            for site in sites:
                if site.name not in f:
                    continue
                data[site] = {}
                sats = f[site.name].keys()
                for sat_name in sats:
                    sat = GnssSat(sat_name, sat_name[0])
                    try:
                        # Проверяем наличие данных timestamp
                        if DataProducts.timestamp.value.hdf_name not in f[site.name][sat.name]:
                            continue
                            
                        timestamps = f[site.name][sat.name][DataProducts.timestamp.value.hdf_name][:]
                        times = [datetime.fromtimestamp(t).replace(tzinfo=_UTC) for t in timestamps]
                        
                        # Создаем словарь данных для спутника
                        data[site][sat] = {DataProducts.time.value: np.array(times)}
                        
                        # Загружаем все остальные продукты данных
                        for data_product in DataProducts:
                            if data_product.value.hdf_name is None:
                                continue
                            if data_product.value.hdf_name in f[site.name][sat.name]:
                                data[site][sat][data_product] = f[site.name][sat.name][data_product.value.hdf_name][:]
                                
                    except Exception as e:
                        st.warning(f"⚠️ Ошибка при обработке спутника {sat_name} для станции {site.name}: {e}")
                        continue
            return data
    except Exception as e:
        st.error(f"❌ Ошибка при извлечении данных: {e}")
        return {}

def reorder_data_by_sat(data: dict[GnssSite, dict[GnssSat, dict[DataProduct, NDArray]]]) -> dict[GnssSat, dict[GnssSite, dict[DataProduct, NDArray]]]:
    """Переупорядочивает данные по спутникам"""
    _data = defaultdict(dict)
    for site in data:
        for sat in data[site]:
            _data[sat][site] = data[site][sat]
    return _data

# ==================== END HDF FUNCTIONS ====================

# Подключаем JavaScript для обработки кликов на карте
with open("map_click_handler.js", "r", encoding='utf-8') as js_file:
    js_code = js_file.read()
    
st.components.v1.html(
    f"""
    <script>
    {js_code}
    </script>
    """,
    height=0
)

def calculate_polygon_area(polygon_points):
    """Расчет площади полигона в км² по формуле сферической геометрии"""
    if len(polygon_points) < 3:
        return 0
    
    # Радиус Земли в км
    R = 6371.0
    
    # Преобразуем координаты в радианы
    coords = [(radians(p['lat']), radians(p['lon'])) for p in polygon_points]
    
    # Формула для расчета площади сферического полигона
    area = 0
    n = len(coords)
    
    for i in range(n):
        j = (i + 1) % n
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        
        area += (lon2 - lon1) * (2 + sin(lat1) + sin(lat2))
    
    area = abs(area) * R * R / 2
    return area

def calculate_polygon_center(polygon_points):
    """Расчет центра масс полигона"""
    if not polygon_points:
        return None
    
    total_lat = sum(p['lat'] for p in polygon_points)
    total_lon = sum(p['lon'] for p in polygon_points)
    
    center_lat = total_lat / len(polygon_points)
    center_lon = total_lon / len(polygon_points)
    
    return {'lat': center_lat, 'lon': center_lon}

def calculate_polygon_bounds(polygon_points):
    """Расчет границ полигона (север, юг, восток, запад)"""
    if not polygon_points:
        return None
    
    lats = [p['lat'] for p in polygon_points]
    lons = [p['lon'] for p in polygon_points]
    
    return {
        'north': max(lats),
        'south': min(lats),
        'east': max(lons),
        'west': min(lons),
        'lat_span': max(lats) - min(lats),
        'lon_span': max(lons) - min(lons)
    }

def parse_nav_file(nav_file_content):
    """Парсинг NAV файла для извлечения станций и их координат"""
    try:
        lines = nav_file_content.decode('utf-8').split('\n')
        stations = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Поиск строк с координатами станций (различные форматы RINEX)
            # Формат 1: SITE LAT LON HEIGHT
            if re.match(r'^[A-Z0-9]{4}\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+', line):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        site_name = parts[0]
                        lat = float(parts[1])
                        lon = float(parts[2])
                        
                        # Проверяем валидность координат
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            stations.append({
                                'name': site_name,
                                'lat': lat,
                                'lon': lon,
                                'color': 'red'  # Цвет для станций из NAV файла
                            })
                    except (ValueError, IndexError):
                        continue
            
            # Формат 2: Поиск заголовков RINEX с APPROX POSITION XYZ
            elif 'APPROX POSITION XYZ' in line and len(line.split()) >= 4:
                try:
                    # Извлекаем XYZ координаты и конвертируем в lat/lon
                    parts = line.split()
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    
                    # Простая конверсия XYZ в lat/lon (приблизительная)
                    r = sqrt(x*x + y*y + z*z)
                    if r > 0:
                        lat = np.degrees(np.arcsin(z/r))
                        lon = np.degrees(np.arctan2(y, x))
                        
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            stations.append({
                                'name': f'STA_{len(stations)+1:03d}',
                                'lat': lat,
                                'lon': lon,
                                'color': 'red'
                            })
                except (ValueError, IndexError, ZeroDivisionError):
                    continue
            
            # Формат 3: Поиск строк с MARKER NAME
            elif 'MARKER NAME' in line:
                try:
                    # Извлекаем имя станции из заголовка RINEX
                    site_name = line.split()[0][:4]  # Первые 4 символа как имя станции
                    if site_name and len(site_name) == 4 and site_name.isalnum():
                        # Ищем координаты в следующих строках (это заглушка)
                        # В реальном RINEX файле координаты идут отдельно
                        pass
                except (ValueError, IndexError):
                    continue
                
        if stations:
            st.info(f"🔍 Обработано {len(lines)} строк, найдено {len(stations)} станций")
        
        return stations
        
    except UnicodeDecodeError:
        # Попробуем другую кодировку
        try:
            lines = nav_file_content.decode('latin-1').split('\n')
            st.warning("⚠️ Файл декодирован с кодировкой latin-1")
            # Повторяем логику парсинга...
            return []
        except Exception as e:
            st.error(f"Ошибка декодирования файла: {str(e)}")
            return []
    except Exception as e:
        st.error(f"Ошибка при парсинге NAV файла: {str(e)}")
        return []

# Инициализация состояния
if 'polygon_points' not in st.session_state:
    st.session_state['polygon_points'] = []
if 'polygon_mode' not in st.session_state:
    st.session_state['polygon_mode'] = False
if 'polygon_completed' not in st.session_state:
    st.session_state['polygon_completed'] = False
if 'ionosphere_data' not in st.session_state:
    st.session_state['ionosphere_data'] = None
if 'last_click_coords' not in st.session_state:
    st.session_state['last_click_coords'] = None
if 'nav_file_stations' not in st.session_state:
    st.session_state['nav_file_stations'] = None
if 'nav_date_stations' not in st.session_state:
    st.session_state['nav_date_stations'] = None
if 'nav_date_loaded' not in st.session_state:
    st.session_state['nav_date_loaded'] = False
if 'last_selected_date' not in st.session_state:
    st.session_state['last_selected_date'] = None
if 'nav_date_info' not in st.session_state:
    st.session_state['nav_date_info'] = {}

st.title("🛰️ Система локализации структур ионосферы (SIP)")

# Статус сохраненных данных
col_status1, col_status2, col_status3, col_status4, col_status5 = st.columns(5)

with col_status1:
    polygon_count = len(st.session_state.get('polygon_points', []))
    if polygon_count > 0:
        st.success(f"📍 Полигон: {polygon_count} точек")
    else:
        st.info("📍 Полигон: не создан")

with col_status2:
    if 'ionosphere_data' in st.session_state and st.session_state['ionosphere_data']:
        points_count = len(st.session_state['ionosphere_data'].get('points', []))
        st.success(f"🌐 Данные ионосферы: {points_count} точек")
    else:
        st.info("🌐 Данные ионосферы: нет")

with col_status3:
    if 'hdf_file_path' in st.session_state and st.session_state.get('hdf_file_path') and st.session_state['hdf_file_path'].exists():
        st.success(f"📊 HDF файл: загружен")
    else:
        st.info("📊 HDF файл: не загружен")

with col_status4:
    if 'site_sat_data' in st.session_state and st.session_state['site_sat_data']:
        sites_count = len(st.session_state['site_sat_data'])
        st.success(f"🛰️ Site-Sat: {sites_count} станций")
    else:
        st.info("🛰️ Site-Sat: нет данных")

with col_status5:
    # Диагностика сохраненных файлов
    data_files = ["session_data.pkl", "site_sat_data.pkl", "sat_data.pkl", "selected_sites.pkl"]
    existing_files = [f for f in data_files if (DATA_DIR / f).exists()]
    
    if existing_files:
        st.warning(f"💾 Файлов: {len(existing_files)}")
        if st.button("🧹 Очистить", help="Очистить все сохраненные данные", key="clear_status"):
            clear_all_data()
            st.rerun()
    else:
        st.info("💾 Нет файлов")

st.markdown("---")

# Выбор режима работы
mode = st.radio(
    "Выберите режим работы:",
    ["Анализ ионосферы", "Разметка (Tinder)"],
    horizontal=True
)

if mode == "Анализ ионосферы":
    # Оптимизированные GNSS станции для быстрой загрузки (уменьшенный набор)
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_global_stations():
        """Возвращает оптимизированный набор станций для быстрой загрузки"""
        return {
            "🌍 Глобальная карта": [
                {'name': 'BRAZ', 'lat': -15.950, 'lon': -47.877, 'color': 'blue'},   # Бразилия
                {'name': 'ALGO', 'lat': 45.956, 'lon': -78.073, 'color': 'green'},   # Канада
                {'name': 'ZIMM', 'lat': 46.877, 'lon': 7.465, 'color': 'purple'},    # Швейцария
                {'name': 'WUHN', 'lat': 30.532, 'lon': 114.357, 'color': 'orange'},  # Китай
                {'name': 'SYDN', 'lat': -33.779, 'lon': 151.150, 'color': 'brown'},  # Австралия
                {'name': 'HRAO', 'lat': -25.890, 'lon': 27.687, 'color': 'pink'},    # ЮАР
            ],
            "🇧🇷 Южная Америка": [
                {'name': 'AREQ', 'lat': -16.466, 'lon': -71.537, 'color': 'red'},
                {'name': 'BRAZ', 'lat': -15.950, 'lon': -47.877, 'color': 'blue'},
                {'name': 'RIOG', 'lat': -53.786, 'lon': -67.751, 'color': 'purple'},
                {'name': 'LPGS', 'lat': -34.907, 'lon': -57.932, 'color': 'orange'},
                {'name': 'BOGT', 'lat': 4.64, 'lon': -74.08, 'color': 'green'},
            ],
            "🇺🇸 Северная Америка": [
                {'name': 'ALGO', 'lat': 45.956, 'lon': -78.073, 'color': 'green'},
                {'name': 'FAIR', 'lat': 64.978, 'lon': -147.499, 'color': 'red'},
                {'name': 'CHUR', 'lat': 58.759, 'lon': -94.089, 'color': 'blue'},
            ],
            "🇪🇺 Европа": [
                {'name': 'ZIMM', 'lat': 46.877, 'lon': 7.465, 'color': 'purple'},
                {'name': 'GOPE', 'lat': 49.914, 'lon': 14.786, 'color': 'red'},
                {'name': 'POTS', 'lat': 52.379, 'lon': 13.066, 'color': 'orange'}
            ],
            "🇨🇳 Азия": [
                {'name': 'WUHN', 'lat': 30.532, 'lon': 114.357, 'color': 'orange'},
                {'name': 'BJFS', 'lat': 39.609, 'lon': 115.893, 'color': 'red'},
                {'name': 'LHAZ', 'lat': 29.657, 'lon': 91.104, 'color': 'blue'},
            ],
            "🇦🇺 Океания": [
                {'name': 'SYDN', 'lat': -33.779, 'lon': 151.150, 'color': 'brown'},
                {'name': 'ALIC', 'lat': -23.670, 'lon': 133.886, 'color': 'red'},
                {'name': 'DARW', 'lat': -12.844, 'lon': 131.133, 'color': 'blue'},
            ],
            "🇿🇦 Африка": [
                {'name': 'HRAO', 'lat': -25.890, 'lon': 27.687, 'color': 'pink'},
                {'name': 'SUTH', 'lat': -32.380, 'lon': 20.810, 'color': 'red'},
                {'name': 'NKLG', 'lat': 0.354, 'lon': 9.672, 'color': 'blue'},
            ],
            "🧊 Арктика": [
                {'name': 'NYA1', 'lat': 78.930, 'lon': 11.865, 'color': 'cyan'},
                {'name': 'KIRU', 'lat': 67.858, 'lon': 20.968, 'color': 'red'},
            ],
            "🐧 Антарктика": [
                {'name': 'SYOG', 'lat': -69.007, 'lon': 39.585, 'color': 'white'},
                {'name': 'MAWZ', 'lat': -67.605, 'lon': 62.871, 'color': 'red'},
            ]
        }
    
    # Получаем оптимизированные станции
    global_stations = get_global_stations()

    # Настройки отображения карты
    st.subheader("🗺️ Настройки карты и региона")
    
    # Быстрая загрузка - показываем основные элементы интерфейса сразу
    with st.spinner("⚡ Загрузка приложения..."):
        # Минимальная инициализация для быстрого отображения
        pass
    
    col_region, col_proj = st.columns(2)
    
    with col_region:
        # Выбор региона для анализа
        region_options = {
            "🌍 Глобальная карта": {
                "lat_range": [-90, 90], 
                "lon_range": [-180, 180],
                "description": "Весь мир"
            },
            "🇧🇷 Южная Америка": {
                "lat_range": [-40, 20], 
                "lon_range": [-90, -30],
                "description": "Бразилия, Аргентина, Перу, Чили"
            },
            "🇺🇸 Северная Америка": {
                "lat_range": [15, 75], 
                "lon_range": [-170, -50],
                "description": "США, Канада, Мексика"
            },
            "🇪🇺 Европа": {
                "lat_range": [35, 75], 
                "lon_range": [-15, 45],
                "description": "Европейский континент"
            },
            "🇨🇳 Азия": {
                "lat_range": [0, 70], 
                "lon_range": [60, 150],
                "description": "Китай, Россия, Индия, Япония"
            },
            "🇦🇺 Океания": {
                "lat_range": [-50, 10], 
                "lon_range": [110, 180],
                "description": "Австралия, Новая Зеландия"
            },
            "🇿🇦 Африка": {
                "lat_range": [-35, 40], 
                "lon_range": [-20, 55],
                "description": "Африканский континент"
            },
            "🧊 Арктика": {
                "lat_range": [60, 90], 
                "lon_range": [-180, 180],
                "description": "Северный полюс"
            },
            "🐧 Антарктика": {
                "lat_range": [-90, -60], 
                "lon_range": [-180, 180],
                "description": "Южный полюс"
            }
        }
        
        selected_region = st.selectbox(
            "🌍 Выберите регион для анализа",
            options=list(region_options.keys()),
            index=0,  # По умолчанию Глобальная карта
            help="Выберите географический регион для отображения и анализа данных ионосферы"
        )
        
        region_config = region_options[selected_region]
        st.info(f"📍 **Регион:** {region_config['description']}")
        
    with col_proj:
        # Выбор проекции карты
        projection_options = {
            "🌐 Ортографическая": "orthographic",
            "🗺️ Естественная Земля": "natural earth",
            "📐 Меркатор": "mercator",
            "🎯 Азимутальная": "azimuthal equal area",
            "📊 Эквидистантная": "equirectangular",
            "🌀 Стереографическая": "stereographic"
        }
        
        selected_projection = st.selectbox(
            "🗺️ Проекция карты",
            options=list(projection_options.keys()),
            index=0,
            help="Выберите проекцию для отображения карты"
        )
        
        projection_type = projection_options[selected_projection]

    # Выбираем станции для текущего региона (временно, будет переопределено в NAV секции)
    current_stations = global_stations.get(selected_region, global_stations["🌍 Глобальная карта"])
    site_names = [site['name'] for site in current_stations]

    # Основная карта и данные в самом начале страницы
    col1, col2 = st.columns([1, 1])
    
    # Карта SIP и станций 
    with col1:
        st.subheader("Карта станций и анализа ионосферы")
        
        # Показываем индикатор загрузки карты
        map_placeholder = st.empty()
        
        with map_placeholder.container():
            # Быстрая загрузка - показываем простую версию карты сначала
            if 'map_loaded' not in st.session_state:
                st.session_state['map_loaded'] = False
            
            if not st.session_state['map_loaded']:
                st.info("🗺️ Карта загружается...")
                # Создаем упрощенную версию карты для быстрого отображения
                fig = go.Figure()
                # Добавляем только первые 5 станций для быстрого рендеринга
                for i, site in enumerate(current_stations[:5]):
                    fig.add_trace(go.Scattergeo(
                        lon=[site['lon']], lat=[site['lat']], mode='markers',
                        marker=dict(color=site['color'], size=8),
                        name=site['name']
                    ))
                st.session_state['map_loaded'] = True
            else:
                # Полная версия карты
                fig = go.Figure()
                
                # Добавление всех станций для выбранного региона
                for site in current_stations:
                    fig.add_trace(go.Scattergeo(
                        lon=[site['lon']], lat=[site['lat']], mode='markers+text',
                        marker=dict(color=site['color'], size=10),
                        text=[site['name']], textposition="top center",
                        name=site['name']
                    ))

        # Добавление станций для выбранного региона
        for site in current_stations:
            fig.add_trace(go.Scattergeo(
                lon=[site['lon']], lat=[site['lat']], mode='markers+text',
                marker=dict(color=site['color'], size=10),
                text=[site['name']], textposition="top center",
                name=site['name']
            ))

        # Отображение полигона для выделения области
        if st.session_state['polygon_points']:
            polygon_lats = [p['lat'] for p in st.session_state['polygon_points']]
            polygon_lons = [p['lon'] for p in st.session_state['polygon_points']]
            
            # Определяем цвет полигона в зависимости от типа структуры
            structure_colors = {
                "equatorial anomaly": {"color": "orange", "rgba": "rgba(255, 165, 0, 0.3)"},
                "plasma bubbles": {"color": "purple", "rgba": "rgba(128, 0, 128, 0.3)"},
                "scintillation patches": {"color": "red", "rgba": "rgba(255, 0, 0, 0.3)"},
                "tec gradients": {"color": "blue", "rgba": "rgba(0, 0, 255, 0.3)"}
            }
            
            # Получаем текущий выбранный тип структуры
            current_structure = st.session_state.get('last_selected_structure', 'equatorial anomaly')
            polygon_style = structure_colors.get(current_structure, structure_colors["equatorial anomaly"])
            
            # Точки полигона с нумерацией
            for i, (lat, lon) in enumerate(zip(polygon_lats, polygon_lons)):
                fig.add_trace(go.Scattergeo(
                    lon=[lon], lat=[lat], 
                    mode='markers+text',
                    marker=dict(color=polygon_style["color"], size=12, symbol='circle'),
                    text=[str(i+1)],
                    textfont=dict(color='white', size=10),
                    textposition="middle center",
                    name=f'Точка {i+1}', 
                    showlegend=False,
                    hovertemplate=f'Точка {i+1}<br>Тип: {current_structure}<br>Широта: {lat}<br>Долгота: {lon}<extra></extra>'
                ))
            
            # Если полигон завершен, рисуем замкнутую область
            if st.session_state['polygon_completed'] and len(polygon_lats) >= 3:
                # Замыкаем полигон
                polygon_lats_closed = polygon_lats + [polygon_lats[0]]
                polygon_lons_closed = polygon_lons + [polygon_lons[0]]
                
                # Линии полигона
                fig.add_trace(go.Scattergeo(
                    lon=polygon_lons_closed, lat=polygon_lats_closed, mode='lines',
                    line=dict(color=polygon_style["color"], width=3),
                    name=f'Граница: {current_structure}', showlegend=True
                ))
                
                # Заливка полигона (полупрозрачная)
                fig.add_trace(go.Scattergeo(
                    lon=polygon_lons_closed, lat=polygon_lats_closed, 
                    mode='none',
                    fill='toself',
                    fillcolor=polygon_style["rgba"],
                    name=f'Область: {current_structure}', showlegend=True,
                    hovertemplate=f'Тип структуры: {current_structure}<br>Точек: {len(polygon_lats)}<extra></extra>'
                ))
                
                # Добавляем центр полигона
                center = calculate_polygon_center(st.session_state['polygon_points'])
                if center:
                    fig.add_trace(go.Scattergeo(
                        lon=[center['lon']], lat=[center['lat']], 
                        mode='markers+text',
                        marker=dict(color=polygon_style["color"], size=15, symbol='star', 
                                  line=dict(color='white', width=2)),
                        text=['C'],
                        textfont=dict(color='white', size=12, family='Arial Black'),
                        textposition="middle center",
                        name=f'Центр: {current_structure}', 
                        showlegend=True,
                        hovertemplate=f'Центр полигона<br>Тип: {current_structure}<br>Широта: {center["lat"]:.3f}<br>Долгота: {center["lon"]:.3f}<extra></extra>'
                    ))
            elif len(polygon_lats) >= 2:
                # Если полигон не завершен, но есть минимум 2 точки, показываем линии
                fig.add_trace(go.Scattergeo(
                    lon=polygon_lons, lat=polygon_lats, mode='lines',
                    line=dict(color=polygon_style["color"], width=2, dash='dash'),
                    name=f'Линии: {current_structure}', showlegend=False
                ))

        # Отображение данных ионосферы поверх основной карты
        if 'ionosphere_data' in st.session_state and st.session_state['ionosphere_data']:
            data = st.session_state['ionosphere_data']
            points = data.get('points', [])
            
            if points:
                # Добавляем точки данных на основную карту
                lats = [p['latitude'] for p in points]
                lons = [p['longitude'] for p in points]
                tec_values = [p.get('tec', 0) for p in points]
                
                fig.add_trace(go.Scattergeo(
                    lon=lons, 
                    lat=lats,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=tec_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="TEC", x=1.02),
                        opacity=0.8,
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"TEC: {tec}<br>Index: {p.get('index', 'N/A')}" for tec, p in zip(tec_values, points)],
                    hovertemplate='%{text}<extra></extra>',
                    name='Ионосферные данные'
                ))

        # Если режим полигона активен, добавляем невидимый слой для кликов
        if st.session_state['polygon_mode']:
            # Создаем невидимую область для ловли кликов по выбранному региону
            click_lats = []
            click_lons = []
            
            # Получаем границы выбранного региона
            lat_min, lat_max = region_config["lat_range"]
            lon_min, lon_max = region_config["lon_range"]
            
            # Создаем плотную сетку для ловли кликов в выбранном регионе
            lat_step = max(1, (lat_max - lat_min) // 50)  # Адаптивный шаг
            lon_step = max(1, (lon_max - lon_min) // 50)
            
            for lat in range(int(lat_min), int(lat_max) + 1, lat_step):
                for lon in range(int(lon_min), int(lon_max) + 1, lon_step):
                    click_lats.append(lat)
                    click_lons.append(lon)
            
            fig.add_trace(go.Scattergeo(
                lon=click_lons,
                lat=click_lats,
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(0,0,0,0)',  # Полностью прозрачные
                    line=dict(width=0)
                ),
                name='Область кликов',
                showlegend=False,
                hoverinfo='none',
                customdata=list(zip(click_lats, click_lons))
            ))

        # Обновление внешнего вида карты
        fig.update_geos(
            projection_type=projection_type,
            showcountries=True, showcoastlines=True, showland=True, landcolor="#f5f5f5",
            lataxis_range=region_config["lat_range"], 
            lonaxis_range=region_config["lon_range"],
            resolution=50, fitbounds="locations"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(bgcolor='rgba(0,0,0,0)'),
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0)
        )
        
        # Отображение основной карты (всегда интерактивная)
        if st.session_state['polygon_mode']:
            # Автоматическое добавление точек всегда включено
            st.info("🖱️ **Режим добавления точек АКТИВЕН.** Кликайте по карте или введите координаты!")
            
            # Добавляем простой ввод координат для надежности
            st.subheader("Добавить точку")
            col_lat, col_lon, col_add = st.columns([2, 2, 1])
            
            with col_lat:
                new_lat = st.number_input("Широта:", min_value=-90.0, max_value=90.0, value=0.0, step=1.0, key="new_lat")
            
            with col_lon:
                new_lon = st.number_input("Долгота:", min_value=-180.0, max_value=180.0, value=0.0, step=1.0, key="new_lon")
            
            with col_add:
                st.write("")  # Пустая строка для выравнивания
                if st.button("➕ Добавить", key="add_point_manual"):
                    new_point = {'lat': new_lat, 'lon': new_lon}
                    
                    # Проверяем дублирование
                    is_duplicate = any(
                        abs(p['lat'] - new_point['lat']) < 0.01 and abs(p['lon'] - new_point['lon']) < 0.01 
                        for p in st.session_state['polygon_points']
                    )
                    
                    if not is_duplicate:
                        st.session_state['polygon_points'].append(new_point)
                        save_session_data()  # Сохраняем данные
                        st.success(f"✅ Точка {len(st.session_state['polygon_points'])} добавлена: {new_lat:.4f}, {new_lon:.4f}")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Точка уже существует поблизости")
            
            # Используем streamlit_plotly_events для обработки кликов
            selected_points = plotly_events(
                fig, 
                click_event=True,
                override_height=600,
                override_width="100%",
                key="map_clicks_stable"  # Стабильный ключ
            )
            
            # Обрабатываем клик по карте
            if selected_points and len(selected_points) > 0:
                point = selected_points[0]
                clicked_lat = None
                clicked_lon = None
                
                # Отладочная информация
                st.write("DEBUG: Получен клик:", point)
                
                try:
                    # Пробуем получить координаты из разных форматов данных
                    if 'lat' in point and 'lon' in point:
                        clicked_lat = round(float(point['lat']), 4)
                        clicked_lon = round(float(point['lon']), 4)
                        st.write("DEBUG: Получены координаты lat/lon")
                    elif 'y' in point and 'x' in point:
                        clicked_lat = round(float(point['y']), 4)
                        clicked_lon = round(float(point['x']), 4)
                        st.write("DEBUG: Получены координаты x/y")
                    elif 'pointNumber' in point and 'curveNumber' in point:
                        # Если есть только pointNumber и curveNumber, попробуем получить координаты из данных фигуры
                        curve_num = point['curveNumber']
                        point_num = point['pointNumber']
                        st.write(f"DEBUG: pointNumber={point_num}, curveNumber={curve_num}")
                        
                        # Получаем данные из соответствующей кривой
                        if curve_num < len(fig.data) and hasattr(fig.data[curve_num], 'lat') and hasattr(fig.data[curve_num], 'lon'):
                            if point_num < len(fig.data[curve_num].lat):
                                clicked_lat = round(float(fig.data[curve_num].lat[point_num]), 4)
                                clicked_lon = round(float(fig.data[curve_num].lon[point_num]), 4)
                                st.write("DEBUG: Получены координаты из trace данных")
                    else:
                        st.write("DEBUG: Не удалось определить формат координат")
                    
                    if clicked_lat is not None and clicked_lon is not None:
                        # Проверяем валидность координат
                        if -90 <= clicked_lat <= 90 and -180 <= clicked_lon <= 180:
                            st.write(f"DEBUG: Валидные координаты: {clicked_lat}, {clicked_lon}")
                            
                            # Автоматически добавляем точку при клике
                            new_point = {'lat': clicked_lat, 'lon': clicked_lon}
                            
                            # Проверяем дублирование (в радиусе 0.01 градуса)
                            is_duplicate = any(
                                abs(p['lat'] - new_point['lat']) < 0.01 and abs(p['lon'] - new_point['lon']) < 0.01 
                                for p in st.session_state['polygon_points']
                            )
                            
                            if not is_duplicate:
                                st.session_state['polygon_points'].append(new_point)
                                save_session_data()  # Сохраняем данные
                                st.success(f"✅ Точка {len(st.session_state['polygon_points'])} добавлена: {clicked_lat:.4f}, {clicked_lon:.4f}")
                                st.rerun()
                            else:
                                st.warning(f"⚠️ Точка с координатами {clicked_lat:.4f}, {clicked_lon:.4f} уже существует поблизости")
                        else:
                            st.error(f"Недопустимые координаты: {clicked_lat}, {clicked_lon}")
                    else:
                        st.write("DEBUG: Координаты не извлечены")
                except Exception as e:
                    st.error(f"Ошибка при обработке клика: {str(e)}")
            

            
            # Отображаем текущие точки полигона в виде таблицы
            if st.session_state['polygon_points']:
                st.subheader("📍 Текущие точки полигона")
                
                # Создаем таблицу с точками и кнопками удаления
                for i, point in enumerate(st.session_state['polygon_points']):
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    with col1:
                        st.write(f"**{i+1}**")
                    with col2:
                        st.write(f"Широта: {point['lat']:.4f}")
                    with col3:
                        st.write(f"Долгота: {point['lon']:.4f}")
                    with col4:
                        if st.button("🗑️", key=f"delete_point_{i}"):
                            st.session_state['polygon_points'].pop(i)
                            save_session_data()  # Сохраняем данные
                            st.success(f"✅ Точка {i+1} удалена")
                            st.rerun()
                
                # Кнопка для завершения полигона
                if len(st.session_state['polygon_points']) >= 3:
                    if st.button("✅ Завершить полигон", key="complete_polygon_from_table"):
                        st.session_state['polygon_completed'] = True
                        st.session_state['polygon_mode'] = False
                        save_session_data()  # Сохраняем данные
                        st.success("✅ Полигон успешно завершен!")
                        st.rerun()
                else:
                    st.warning("⚠️ Для завершения полигона необходимо минимум 3 точки")
            

        else:
            # Обычное отображение без обработки кликов
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                config={"scrollZoom": True, "displayModeBar": True},
                key="main_interactive_map"
            )

    # Отображение данных анализа справа от карты
    with col2:
        st.subheader("📊 Оценка ионосферного эффекта")
        
        # Добавляем CSS для стилизации кнопок оценки
        st.markdown("""
        <style>
            .stButton > button {
                font-size: 20px !important;
                font-weight: bold !important;
                padding: 15px !important;
                border-radius: 10px !important;
                margin-top: 10px !important;
                margin-bottom: 10px !important;
            }
            .effect-button > button {
                background-color: #4CAF50 !important;
                color: white !important;
            }
            .no-effect-button > button {
                background-color: #f44336 !important;
                color: white !important;
            }
            .highlight-box {
                border: 2px solid black;
                padding: 5px;
                display: inline-block;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Проверяем наличие HDF данных
        if 'hdf_file' in st.session_state and st.session_state['hdf_file'] is not None:
            if 'hdf_data' not in st.session_state or not st.session_state['hdf_data']:
                # Если файл загружен, но данные не извлечены, предлагаем извлечь данные
                st.info("HDF файл загружен. Нажмите 'Извлечь данные' в разделе загрузки файла для отображения данных.")
            else:
                # Получаем данные из HDF файла
                hdf_data = st.session_state['hdf_data']
                
                # Создаем селектор для станций
                stations = list(hdf_data.keys())
                if stations:
                    selected_station = st.selectbox("Выберите станцию:", stations, key="station_selector_analysis")
                    
                    if selected_station in hdf_data:
                        station_data = hdf_data[selected_station]
                        satellites = list(station_data.keys())
                        
                        if satellites:
                            selected_satellite = st.selectbox("Выберите спутник:", satellites, key="satellite_selector_analysis")
                            
                            if selected_satellite in station_data:
                                sat_data = station_data[selected_satellite]
                                
                                # Создаем уникальный ключ для текущего набора данных
                                current_data_key = f"data_{selected_station}_{selected_satellite}"
                                
                                # Инициализируем словарь оценок, если его еще нет
                                if 'data_evaluations' not in st.session_state:
                                    st.session_state['data_evaluations'] = {}
                                
                                # Получаем данные TEC и ROTI
                                tec_data = sat_data.get(DataProducts.atec, [])
                                roti_data = sat_data.get(DataProducts.roti, [])
                                time_data = sat_data.get(DataProducts.timestamp, [])
                                elevation_data = sat_data.get(DataProducts.elevation, [])
                                
                                if len(tec_data) > 0 and len(roti_data) > 0 and len(time_data) > 0:
                                    # Преобразуем временные метки в datetime объекты
                                    time_objects = [datetime.fromtimestamp(t) for t in time_data]
                                    
                                    # Создаем фигуру для графика
                                    fig = make_subplots(rows=2, cols=1, 
                                                       shared_xaxes=True, 
                                                       vertical_spacing=0.1,
                                                       subplot_titles=("TEC", "ROTI"))
                                    
                                    # Определяем цвета для станций
                                    station_colors = {
                                        'AREQ': 'blue',
                                        'SCRZ': 'red',
                                        'BRAZ': 'green'
                                    }
                                    
                                    # Получаем цвет для текущей станции
                                    station_color = station_colors.get(selected_station, 'blue')
                                    
                                    # Добавляем TEC данные
                                    fig.add_trace(
                                        go.Scatter(
                                            x=time_objects, 
                                            y=tec_data, 
                                            mode='lines', 
                                            name='TEC',
                                            line=dict(color=station_color, width=2)
                                        ),
                                        row=1, col=1
                                    )
                                    
                                    # Добавляем ROTI данные
                                    fig.add_trace(
                                        go.Scatter(
                                            x=time_objects, 
                                            y=roti_data, 
                                            mode='lines', 
                                            name='ROTI',
                                            line=dict(color='red' if station_color != 'red' else 'orange', width=2)
                                        ),
                                        row=2, col=1
                                    )
                                    
                                    # Добавляем маску для выделения областей с эффектами
                                    # Простой алгоритм: выделяем области, где ROTI > 0.2
                                    effect_regions = []
                                    in_region = False
                                    start_idx = 0
                                    
                                    for i, roti_value in enumerate(roti_data):
                                        if roti_value > 0.2 and not in_region:
                                            in_region = True
                                            start_idx = i
                                        elif (roti_value <= 0.2 or i == len(roti_data) - 1) and in_region:
                                            in_region = False
                                            if i - start_idx > 5:  # Минимальная длина региона
                                                effect_regions.append((start_idx, i))
                                    
                                    # Добавляем выделение регионов с эффектами
                                    for start, end in effect_regions:
                                        # Выделение для TEC
                                        fig.add_shape(
                                            type="rect",
                                            xref="x",
                                            yref="paper",
                                            x0=time_objects[start],
                                            y0=0,
                                            x1=time_objects[end],
                                            y1=0.45,
                                            line=dict(width=0),
                                            fillcolor="rgba(0,0,0,0.1)",
                                            layer="below",
                                            row=1, col=1
                                        )
                                        
                                        # Выделение для ROTI
                                        fig.add_shape(
                                            type="rect",
                                            xref="x2",
                                            yref="paper",
                                            x0=time_objects[start],
                                            y0=0.55,
                                            x1=time_objects[end],
                                            y1=1,
                                            line=dict(width=0),
                                            fillcolor="rgba(0,0,0,0.1)",
                                            layer="below",
                                            row=2, col=1
                                        )
                                    
                                    # Настраиваем макет графика
                                    fig.update_layout(
                                        height=500,
                                        title=f"{selected_station} - {selected_satellite}",
                                        plot_bgcolor='rgb(20, 24, 35)',
                                        paper_bgcolor='rgb(20, 24, 35)',
                                        font=dict(color='white'),
                                        margin=dict(l=10, r=10, t=50, b=10),
                                        showlegend=True,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    
                                    # Добавляем сетку
                                    fig.update_xaxes(
                                        showgrid=True,
                                        gridwidth=1,
                                        gridcolor='rgba(255, 255, 255, 0.2)'
                                    )
                                    
                                    fig.update_yaxes(
                                        showgrid=True,
                                        gridwidth=1,
                                        gridcolor='rgba(255, 255, 255, 0.2)'
                                    )
                                    
                                    # Отображаем график
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Добавляем информацию о наличии эффекта
                                    if effect_regions:
                                        st.warning(f"⚠️ Обнаружено {len(effect_regions)} областей с возможным ионосферным эффектом")
                                    else:
                                        st.success("✅ Ионосферных эффектов не обнаружено")
                                    
                                    # Добавляем кнопки оценки в стиле Tinder
                                    st.markdown("### 📋 Оценка данных")
                                    
                                    col_effect, col_no_effect = st.columns(2)
                                    
                                    with col_effect:
                                        # Используем markdown для создания стилизованной кнопки
                                        st.markdown('<div class="effect-button">', unsafe_allow_html=True)
                                        if st.button("✅ ЕСТЬ ЭФФЕКТ", key=f"effect_{current_data_key}", use_container_width=True, help="Отметить наличие ионосферного эффекта"):
                                            st.session_state['data_evaluations'][current_data_key] = {
                                                "evaluation": "effect",
                                                "station": selected_station,
                                                "satellite": selected_satellite,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }
                                            save_session_data()  # Сохраняем данные
                                            st.success("✅ Отмечено наличие эффекта!")
                                            st.rerun()
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with col_no_effect:
                                        # Используем markdown для создания стилизованной кнопки
                                        st.markdown('<div class="no-effect-button">', unsafe_allow_html=True)
                                        if st.button("❌ НЕТ ЭФФЕКТА", key=f"no_effect_{current_data_key}", use_container_width=True, help="Отметить отсутствие ионосферного эффекта"):
                                            st.session_state['data_evaluations'][current_data_key] = {
                                                "evaluation": "no_effect",
                                                "station": selected_station,
                                                "satellite": selected_satellite,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }
                                            save_session_data()  # Сохраняем данные
                                            st.info("❌ Отмечено отсутствие эффекта.")
                                            st.rerun()
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Отображаем текущую оценку
                                    current_evaluation = st.session_state['data_evaluations'].get(current_data_key)
                                    if current_evaluation:
                                        if current_evaluation["evaluation"] == "effect":
                                            st.success(f"✅ Текущая оценка: ЕСТЬ ЭФФЕКТ (от {current_evaluation['timestamp']})")
                                        else:
                                            st.info(f"❌ Текущая оценка: НЕТ ЭФФЕКТА (от {current_evaluation['timestamp']})")
                                    
                                    # Отображаем историю оценок
                                    if st.session_state['data_evaluations']:
                                        with st.expander("📝 История оценок"):
                                            # Создаем DataFrame для отображения истории
                                            evaluation_data = []
                                            for data_key, eval_info in st.session_state['data_evaluations'].items():
                                                evaluation_data.append({
                                                    "Станция": eval_info.get("station", "Неизвестно"),
                                                    "Спутник": eval_info.get("satellite", "Неизвестно"),
                                                    "Оценка": "✅ ЕСТЬ ЭФФЕКТ" if eval_info.get("evaluation") == "effect" else "❌ НЕТ ЭФФЕКТА",
                                                    "Время оценки": eval_info.get("timestamp", "Неизвестно"),
                                                    "Тип": eval_info.get("evaluation", "unknown")
                                                })
                                            
                                            if evaluation_data:
                                                df = pd.DataFrame(evaluation_data)
                                                
                                                # Фильтры для истории
                                                col_filter1, col_filter2 = st.columns(2)
                                                
                                                with col_filter1:
                                                    filter_station = st.multiselect(
                                                        "Фильтр по станциям:",
                                                        options=sorted(df["Станция"].unique()),
                                                        default=[]
                                                    )
                                                
                                                with col_filter2:
                                                    filter_eval = st.multiselect(
                                                        "Фильтр по оценке:",
                                                        options=["✅ ЕСТЬ ЭФФЕКТ", "❌ НЕТ ЭФФЕКТА"],
                                                        default=[]
                                                    )
                                                
                                                # Применяем фильтры
                                                filtered_df = df.copy()
                                                if filter_station:
                                                    filtered_df = filtered_df[filtered_df["Станция"].isin(filter_station)]
                                                
                                                if filter_eval:
                                                    filtered_df = filtered_df[filtered_df["Оценка"].isin(filter_eval)]
                                                
                                                # Отображаем таблицу с историей
                                                st.dataframe(
                                                    filtered_df[["Станция", "Спутник", "Оценка", "Время оценки"]], 
                                                    use_container_width=True,
                                                    hide_index=True
                                                )
                                                
                                                # Статистика оценок
                                                st.markdown("#### 📊 Статистика оценок")
                                                
                                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                                
                                                with col_stat1:
                                                    total_evals = len(filtered_df)
                                                    st.metric("Всего оценок", total_evals)
                                                
                                                with col_stat2:
                                                    effect_count = len(filtered_df[filtered_df["Тип"] == "effect"])
                                                    effect_percent = (effect_count / total_evals * 100) if total_evals > 0 else 0
                                                    st.metric("Есть эффект", f"{effect_count} ({effect_percent:.1f}%)")
                                                
                                                with col_stat3:
                                                    no_effect_count = len(filtered_df[filtered_df["Тип"] == "no_effect"])
                                                    no_effect_percent = (no_effect_count / total_evals * 100) if total_evals > 0 else 0
                                                    st.metric("Нет эффекта", f"{no_effect_count} ({no_effect_percent:.1f}%)")
                                                
                                                # Кнопка для экспорта данных
                                                if st.button("📥 Экспортировать оценки в CSV", use_container_width=True):
                                                    csv = filtered_df.to_csv(index=False)
                                                    st.download_button(
                                                        label="📥 Скачать CSV файл",
                                                        data=csv,
                                                        file_name=f"ionosphere_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                        mime="text/csv",
                                                        use_container_width=True
                                                    )
                                else:
                                    st.warning("Недостаточно данных для отображения графиков TEC и ROTI")
                            else:
                                st.warning(f"Данные для спутника {selected_satellite} не найдены")
                        else:
                            st.warning("Нет доступных спутников для выбранной станции")
                    else:
                        st.warning(f"Данные для станции {selected_station} не найдены")
                else:
                    st.warning("Нет доступных станций в загруженных данных")
        else:
            st.info("Загрузите HDF файл для отображения данных и оценки ионосферных эффектов")

    # Анализ ионосферных структур
    st.markdown("---")
    st.markdown("### 🌐 Анализ ионосферных структур")
    
    # Все настройки структур в одном горизонтальном ряду
    col_struct_date, col_struct_type, col_polygon_btn, col_mode_btn, col_clear_btn, col_complete_btn = st.columns([2, 2, 1, 1, 1, 1])
    
    with col_struct_date:
        selected_date = st.date_input(
            "📅 Дата анализа", 
            value=date(2025, 6, 15),  # Используем фиксированную дату по умолчанию
            help="Дата для загрузки данных ионосферы с simurg.space"
        )
        
        with col_struct_type:
            ionosphere_structures = [
                "equatorial anomaly",
                "plasma bubbles", 
                "scintillation patches",
                "tec gradients"
            ]
            selected_structure = st.selectbox(
                "🌊 Тип структуры",
                ionosphere_structures,
                help="Выберите тип ионосферной структуры для анализа",
                key="structure_selector"
            )
    
    with col_polygon_btn:
        if st.button("🎯 Полигон", help="Создать типичный полигон для выбранной структуры"):
            # Создание ГЛОБАЛЬНЫХ полигонов для выбранной структуры
            structure_polygons = {
                "equatorial anomaly": [
                    # Южная Америка (экваториальная аномалия)
                    {'lat': -20.0, 'lon': -70.0}, {'lat': -10.0, 'lon': -75.0}, {'lat': 5.0, 'lon': -80.0},
                    {'lat': 15.0, 'lon': -85.0}, {'lat': 20.0, 'lon': -90.0}, {'lat': 10.0, 'lon': -95.0},
                    # Африка (экваториальная зона)
                    {'lat': 15.0, 'lon': 0.0}, {'lat': 10.0, 'lon': 10.0}, {'lat': 0.0, 'lon': 15.0},
                    {'lat': -10.0, 'lon': 20.0}, {'lat': -15.0, 'lon': 25.0}, {'lat': -5.0, 'lon': 30.0},
                    # Юго-Восточная Азия (экваториальная аномалия)
                    {'lat': 20.0, 'lon': 100.0}, {'lat': 15.0, 'lon': 110.0}, {'lat': 5.0, 'lon': 120.0},
                    {'lat': -5.0, 'lon': 125.0}, {'lat': -15.0, 'lon': 130.0}, {'lat': -10.0, 'lon': 135.0},
                    # Тихий океан (экваториальная зона)
                    {'lat': 10.0, 'lon': 160.0}, {'lat': 0.0, 'lon': 170.0}, {'lat': -10.0, 'lon': 180.0},
                    {'lat': -15.0, 'lon': -170.0}, {'lat': -20.0, 'lon': -160.0}, {'lat': -25.0, 'lon': -150.0}
                ],
                "plasma bubbles": [
                    # Южная Америка (плазменные пузыри)
                    {'lat': -5.0, 'lon': -50.0}, {'lat': 0.0, 'lon': -55.0}, {'lat': 5.0, 'lon': -60.0},
                    {'lat': 10.0, 'lon': -65.0}, {'lat': 15.0, 'lon': -70.0}, {'lat': 20.0, 'lon': -75.0},
                    # Африка (экваториальные пузыри)
                    {'lat': 20.0, 'lon': 5.0}, {'lat': 15.0, 'lon': 15.0}, {'lat': 10.0, 'lon': 25.0},
                    {'lat': 5.0, 'lon': 35.0}, {'lat': 0.0, 'lon': 40.0}, {'lat': -5.0, 'lon': 45.0},
                    # Индийский океан
                    {'lat': 15.0, 'lon': 70.0}, {'lat': 10.0, 'lon': 80.0}, {'lat': 5.0, 'lon': 90.0},
                    {'lat': 0.0, 'lon': 95.0}, {'lat': -5.0, 'lon': 100.0}, {'lat': -10.0, 'lon': 105.0},
                    # Тихий океан (низкие широты)
                    {'lat': 20.0, 'lon': 140.0}, {'lat': 15.0, 'lon': 150.0}, {'lat': 10.0, 'lon': 160.0},
                    {'lat': 5.0, 'lon': 170.0}, {'lat': 0.0, 'lon': 180.0}, {'lat': -5.0, 'lon': -170.0}
                ],
                "scintillation patches": [
                    # Северная полярная область
                    {'lat': 70.0, 'lon': -150.0}, {'lat': 75.0, 'lon': -120.0}, {'lat': 80.0, 'lon': -90.0},
                    {'lat': 85.0, 'lon': -60.0}, {'lat': 80.0, 'lon': -30.0}, {'lat': 75.0, 'lon': 0.0},
                    {'lat': 70.0, 'lon': 30.0}, {'lat': 75.0, 'lon': 60.0}, {'lat': 80.0, 'lon': 90.0},
                    {'lat': 85.0, 'lon': 120.0}, {'lat': 80.0, 'lon': 150.0}, {'lat': 75.0, 'lon': 180.0},
                    # Южная полярная область
                    {'lat': -70.0, 'lon': -150.0}, {'lat': -75.0, 'lon': -120.0}, {'lat': -80.0, 'lon': -90.0},
                    {'lat': -85.0, 'lon': -60.0}, {'lat': -80.0, 'lon': -30.0}, {'lat': -75.0, 'lon': 0.0},
                    {'lat': -70.0, 'lon': 30.0}, {'lat': -75.0, 'lon': 60.0}, {'lat': -80.0, 'lon': 90.0},
                    {'lat': -85.0, 'lon': 120.0}, {'lat': -80.0, 'lon': 150.0}, {'lat': -75.0, 'lon': 180.0}
                ],
                "tec gradients": [
                    # Северная Америка (средние широты)
                    {'lat': 35.0, 'lon': -120.0}, {'lat': 40.0, 'lon': -110.0}, {'lat': 45.0, 'lon': -100.0},
                    {'lat': 50.0, 'lon': -90.0}, {'lat': 55.0, 'lon': -80.0}, {'lat': 50.0, 'lon': -70.0},
                    # Европа (средние широты)
                    {'lat': 45.0, 'lon': -10.0}, {'lat': 50.0, 'lon': 0.0}, {'lat': 55.0, 'lon': 10.0},
                    {'lat': 60.0, 'lon': 20.0}, {'lat': 55.0, 'lon': 30.0}, {'lat': 50.0, 'lon': 40.0},
                    # Азия (средние широты)
                    {'lat': 40.0, 'lon': 80.0}, {'lat': 45.0, 'lon': 90.0}, {'lat': 50.0, 'lon': 100.0},
                    {'lat': 55.0, 'lon': 110.0}, {'lat': 50.0, 'lon': 120.0}, {'lat': 45.0, 'lon': 130.0},
                    # Южное полушарие (средние широты)
                    {'lat': -35.0, 'lon': -60.0}, {'lat': -40.0, 'lon': -50.0}, {'lat': -45.0, 'lon': -40.0},
                    {'lat': -40.0, 'lon': 140.0}, {'lat': -45.0, 'lon': 150.0}, {'lat': -50.0, 'lon': 160.0}
                ]
            }
            
            if selected_structure in structure_polygons:
                st.session_state['polygon_points'] = structure_polygons[selected_structure]
                st.session_state['polygon_completed'] = True
                st.session_state['polygon_mode'] = False
                save_session_data()  # Сохраняем данные
                st.success(f"📋 Загружен шаблон полигона для структуры: {selected_structure}")
            else:
                st.warning("⚠️ Шаблон для данного типа структуры не найден.")
    
    with col_mode_btn:
        if not st.session_state['polygon_mode']:
            if st.button("🖱️ Добавить точки", help="Включить режим добавления точек кликом по карте"):
                st.session_state['polygon_mode'] = True
                st.rerun()
        else:
            if st.button("❌ Выйти из режима", help="Выключить режим добавления точек"):
                st.session_state['polygon_mode'] = False
                st.rerun()
    
    with col_clear_btn:
        if st.button("🗑️ Очистить", help="Удалить все точки"):
            st.session_state['polygon_points'] = []
            st.session_state['polygon_completed'] = False
            st.session_state['polygon_mode'] = False
            save_session_data()  # Сохраняем данные
            st.rerun()
    
    with col_complete_btn:
        if st.button("✅ Готово", help="Завершить полигон", disabled=len(st.session_state['polygon_points']) < 3):
            st.session_state['polygon_completed'] = True
            st.session_state['polygon_mode'] = False
            save_session_data()  # Сохраняем данные
            st.rerun()

    # Отслеживание изменения типа структуры (БЕЗ автоматического создания полигона)
    if 'last_selected_structure' not in st.session_state:
        st.session_state['last_selected_structure'] = selected_structure
    
    # Просто обновляем последний выбранный тип без создания полигона
    if st.session_state['last_selected_structure'] != selected_structure:
        st.session_state['last_selected_structure'] = selected_structure

    # Статус полигона в одной строке
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        if st.session_state['polygon_mode']:
            st.info(f"🖱️ Режим добавления: {len(st.session_state['polygon_points'])} точек")
        elif st.session_state['polygon_completed']:
            st.success(f"✅ Полигон готов: {len(st.session_state['polygon_points'])} точек")
        else:
            st.warning(f"⚠️ Полигон: {len(st.session_state['polygon_points'])}/3 точек")
    
    with col_status2:
        if st.session_state['polygon_completed']:
            st.info(f"📍 Структура: {selected_structure}")
        else:
            st.info("📍 Выберите тип структуры")
    
    with col_status3:
        # Кнопка запроса данных
        if st.button("🔍 Запросить данные", type="primary", disabled=not st.session_state['polygon_completed']):
            if len(st.session_state['polygon_points']) < 3:
                st.error("❌ Необходимо создать полигон из минимум 3 точек")
            else:
                # Всегда используем станцию ARTU (ERKG не найдена в API)
                station_code = "ARTU"
                st.info(f"🔍 Поиск данных для даты: {selected_date} для станции {station_code}")
                
                with st.spinner("🌐 Загружаем данные с simurg.space..."):
                    polygon_coords = [(p['lat'], p['lon']) for p in st.session_state['polygon_points']]
                    
                    try:
                        # Передаем код станции ARTU
                        data = request_ionosphere_data(selected_date, selected_structure, polygon_coords, station_code)
                        
                        if data and 'points' in data and len(data['points']) > 0:
                            st.session_state['ionosphere_data'] = data
                            save_session_data()  # Сохраняем данные
                            metadata = data.get('metadata', {})
                            
                            st.success(f"✅ Загружено {len(data['points'])} точек данных")
                            st.rerun()  # Перезагружаем страницу для отображения данных
                        else:
                            st.error("❌ Данные не найдены или API вернул пустой ответ")
                    except Exception as e:
                        st.error(f"❌ Ошибка при запросе данных: {e}")
                        
                        # Загружаем тестовые данные если API недоступен
                        st.info("🔄 Загружаем тестовые данные...")
                        test_data = generate_test_ionosphere_data(selected_structure, st.session_state['polygon_points'])
                        st.session_state['ionosphere_data'] = test_data
                        save_session_data()  # Сохраняем данные
                        st.success(f"✅ Загружено {len(test_data['points'])} тестовых точек")
                        st.rerun()  # Перезагружаем страницу для отображения данных

    # HDF DATA ANALYSIS SECTION
    st.markdown("---")
    st.markdown("### 📊 Анализ HDF данных SIMuRG")
    
    # HDF data request and analysis
    col_hdf_date, col_hdf_download, col_hdf_region = st.columns([2, 1, 2])
    
    with col_hdf_date:
        # Выбор даты для HDF данных
        hdf_date = st.date_input(
            "📅 Дата для HDF данных",
            value=date(2025, 1, 5),  # Дата по умолчанию
            min_value=date(2020, 1, 1),
            max_value=date(2030, 12, 31),
            help="Выберите дату для загрузки HDF данных с сервера SIMuRG",
            key="hdf_date_input"
        )
    
    with col_hdf_download:
        st.write("")  # Пустая строка для выравнивания
        if st.button("📥 Загрузить HDF", key="download_hdf_btn"):
            # Формируем URL для загрузки HDF файла
            filename = hdf_date.strftime("%Y-%m-%d.h5")
            url = f"https://simurg.space/gen_file?data=obs&date={hdf_date.strftime('%Y-%m-%d')}"
            
            # Используем путь внутри директории приложения
            local_path = HDF_DIR / filename
            
            try:
                load_hdf_data(url, local_path, override=False)
                st.session_state['hdf_file_path'] = local_path
                st.session_state['hdf_date'] = hdf_date
                save_session_data()  # Сохраняем данные
                st.success(f"✅ HDF файл загружен в приложение: {filename}")
                st.info(f"📁 Файл сохранен в директории приложения: {local_path}")
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке HDF файла: {e}")
    
    with col_hdf_region:
        # Фильтр по региону для HDF данных
        hdf_lat_min = st.number_input("Мин. широта", value=30.0, min_value=-90.0, max_value=90.0, step=1.0, key="hdf_lat_min")
        hdf_lat_max = st.number_input("Макс. широта", value=50.0, min_value=-90.0, max_value=90.0, step=1.0, key="hdf_lat_max")
        hdf_lon_min = st.number_input("Мин. долгота", value=-120.0, min_value=-180.0, max_value=180.0, step=1.0, key="hdf_lon_min")
        hdf_lon_max = st.number_input("Макс. долгота", value=-90.0, min_value=-180.0, max_value=180.0, step=1.0, key="hdf_lon_max")

    # Отображение данных если HDF файл загружен
    if 'hdf_file_path' in st.session_state:
        hdf_path = st.session_state['hdf_file_path']
        # Проверяем и исправляем путь к HDF файлу
        hdf_path = validate_and_fix_hdf_path(hdf_path)
        
        # Проверяем, что файл существует и не является директорией
        if hdf_path is not None:
            st.markdown("### 📋 Site-Sat данные и геометрия")
            
            # Получаем данные из HDF файла
            
            with st.spinner("📖 Извлечение данных из HDF файла..."):
                # Получаем станции в указанном регионе
                hdf_sites = get_sites_from_hdf(
                    hdf_path, 
                    min_lat=hdf_lat_min, 
                    max_lat=hdf_lat_max,
                    min_lon=hdf_lon_min, 
                    max_lon=hdf_lon_max
                )
                
                if hdf_sites:
                    st.success(f"📡 Найдено {len(hdf_sites)} станций в регионе")
                    
                    # Показываем станции
                    col_sites, col_extract = st.columns([3, 1])
                    
                    with col_sites:
                        # Выбор станций для анализа
                        selected_hdf_sites = st.multiselect(
                            "Выберите станции для анализа:",
                            options=[f"{s.name} ({s.lat:.2f}°, {s.lon:.2f}°)" for s in hdf_sites],
                            default=[f"{s.name} ({s.lat:.2f}°, {s.lon:.2f}°)" for s in hdf_sites[:3]],  # Первые 3 по умолчанию
                            key="selected_hdf_sites"
                        )
                    
                    with col_extract:
                        st.write("")  # Пустая строка для выравнивания
                        if st.button("🔍 Извлечь данные", key="extract_hdf_data"):
                            # Извлекаем выбранные станции
                            selected_site_names = [s.split(' (')[0] for s in selected_hdf_sites]
                            filtered_sites = [s for s in hdf_sites if s.name in selected_site_names]
                            
                            if filtered_sites:
                                with st.spinner("🛰️ Извлечение site-sat данных..."):
                                    # Получаем данные спутников для выбранных станций
                                    site_sat_data = retrieve_visible_sats_data(hdf_path, filtered_sites)
                                    
                                    if site_sat_data:
                                        # Сохраняем данные в session_state
                                        st.session_state['site_sat_data'] = site_sat_data
                                        st.session_state['selected_sites'] = filtered_sites
                                        
                                        # Переупорядочиваем данные по спутникам
                                        sat_data = reorder_data_by_sat(site_sat_data)
                                        st.session_state['sat_data'] = sat_data
                                        
                                        # Сохраняем HDF данные на диск
                                        save_hdf_data()
                                        save_session_data()
                                        
                                        st.success(f"✅ Данные извлечены для {len(filtered_sites)} станций")
                                        
                                        # Отображаем табы с результатами
                                        tab_summary, tab_geometry, tab_data, tab_export = st.tabs(["Сводка", "Геометрия", "Данные", "Экспорт"])
                                        
                                        with tab_geometry:
                                            st.markdown("### 📐 Геометрические параметры")
                                            
                                            # Выбор станции и спутника для отображения
                                            col_geo_site, col_geo_sat = st.columns(2)
                                            
                                            with col_geo_site:
                                                selected_geo_site = st.selectbox(
                                                    "Выберите станцию:",
                                                    options=[site.name for site in filtered_sites],
                                                    key="geo_site_selector"
                                                )
                                            
                                            # Находим выбранную станцию
                                            selected_site_obj = next((site for site in filtered_sites if site.name == selected_geo_site), None)
                                            
                                            if selected_site_obj and selected_site_obj in site_sat_data:
                                                site_data = site_sat_data[selected_site_obj]
                                                
                                                with col_geo_sat:
                                                    available_sats = list(site_data.keys())
                                                    selected_geo_sat = st.selectbox(
                                                        "Выберите спутник:",
                                                        options=[sat.name for sat in available_sats],
                                                        key="geo_sat_selector"
                                                    )
                                                
                                                # Находим выбранный спутник
                                                selected_sat_obj = next((sat for sat in available_sats if sat.name == selected_geo_sat), None)
                                                
                                                if selected_sat_obj:
                                                    sat_data = site_data[selected_sat_obj]
                                                    
                                                    # Отображаем график угла места и азимута
                                                    if DataProducts.elevation in sat_data and DataProducts.azimuth in sat_data and DataProducts.timestamp in sat_data:
                                                        elevations = sat_data[DataProducts.elevation]
                                                        azimuths = sat_data[DataProducts.azimuth]
                                                        timestamps = sat_data[DataProducts.timestamp]
                                                        
                                                        # Преобразуем временные метки в datetime
                                                        times = [datetime.fromtimestamp(ts) for ts in timestamps]
                                                        
                                                        # Создаем график
                                                        fig = go.Figure()
                                                        
                                                        fig.add_trace(go.Scatter(
                                                            x=times,
                                                            y=elevations,
                                                            mode='lines+markers',
                                                            name='Угол места',
                                                            line=dict(color='blue'),
                                                            marker=dict(size=6)
                                                        ))
                                                        
                                                        fig.add_trace(go.Scatter(
                                                            x=times,
                                                            y=azimuths,
                                                            mode='lines+markers',
                                                            name='Азимут',
                                                            line=dict(color='red'),
                                                            marker=dict(size=6),
                                                            yaxis='y2'
                                                        ))
                                                        
                                                        fig.update_layout(
                                                            title=f"Геометрия для станции {selected_geo_site} и спутника {selected_geo_sat}",
                                                            xaxis_title="Время",
                                                            yaxis_title="Угол места (градусы)",
                                                            yaxis2=dict(
                                                                title="Азимут (градусы)",
                                                                overlaying='y',
                                                                side='right',
                                                                range=[0, 360]
                                                            ),
                                                            height=400,
                                                            margin=dict(l=0, r=0, t=30, b=0),
                                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                                        )
                                                        
                                                        st.plotly_chart(fig, use_container_width=True, key="geometry_plot_1")
                                                    else:
                                                        st.warning("⚠️ Недостаточно данных для построения графика геометрии")
                                            else:
                                                st.warning("⚠️ Нет данных для выбранной станции")
                                        
                                        with tab_data:
                                            st.markdown("### 📊 Данные измерений")
                                            
                                            # Отображение данных спутников как на второй фотографии
                                            if 'sat_data' in st.session_state and st.session_state['sat_data']:
                                                sat_data = st.session_state['sat_data']
                                                
                                                # Выбор спутников для отображения
                                                available_sats = list(sat_data.keys())
                                                col_sat_select, col_station_select = st.columns(2)
                                                
                                                with col_sat_select:
                                                    selected_sats = st.multiselect(
                                                        "Выберите спутники для отображения:",
                                                        options=[sat.name for sat in available_sats],
                                                        default=[available_sats[0].name] if available_sats else [],
                                                        key="data_sats_selector"
                                                    )
                                                
                                                with col_station_select:
                                                    # Получаем все доступные станции
                                                    all_stations = set()
                                                    for sat in available_sats:
                                                        all_stations.update(sat_data[sat].keys())
                                                    
                                                    selected_stations = st.multiselect(
                                                        "Выберите станции для отображения:",
                                                        options=[station.name for station in all_stations],
                                                        default=[list(all_stations)[0].name] if all_stations else [],
                                                        key="data_stations_selector"
                                                    )
                                                
                                                # Создаем график для отображения данных спутников
                                                if selected_sats and selected_stations:
                                                    fig = go.Figure()
                                                    
                                                    # Определяем цвета для станций
                                                    station_colors = {
                                                        'AREQ': 'blue',
                                                        'SCRZ': 'red',
                                                        'BRAZ': 'green'
                                                    }
                                                    
                                                    # Добавляем данные для каждого выбранного спутника и станции
                                                    for sat_name in selected_sats:
                                                        sat_obj = next((s for s in available_sats if s.name == sat_name), None)
                                                        if sat_obj and sat_obj in sat_data:
                                                            for station_name in selected_stations:
                                                                station_obj = next((s for s in all_stations if s.name == station_name), None)
                                                                if station_obj and station_obj in sat_data[sat_obj]:
                                                                    # Получаем данные для этой пары спутник-станция
                                                                    station_sat_data = sat_data[sat_obj][station_obj]
                                                                    
                                                                    # Проверяем наличие временных меток и TEC данных
                                                                    if DataProducts.time.value in station_sat_data and DataProducts.atec in station_sat_data:
                                                                        times = station_sat_data[DataProducts.time.value]
                                                                        tec_values = station_sat_data[DataProducts.atec]
                                                                        
                                                                        # Определяем цвет для станции
                                                                        color = station_colors.get(station_name, 'gray')
                                                                        
                                                                        # Добавляем линию на график
                                                                        fig.add_trace(go.Scatter(
                                                                            x=times,
                                                                            y=tec_values,
                                                                            mode='lines',
                                                                            name=f"{station_name} - {sat_name}",
                                                                            line=dict(color=color),
                                                                        ))
                                                    
                                                    # Настраиваем макет графика
                                                    fig.update_layout(
                                                        title="Временные ряды TEC для выбранных спутников и станций",
                                                        xaxis_title="Время",
                                                        yaxis_title="TEC",
                                                        height=400,
                                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                                    )
                                                    
                                                    # Отображаем график
                                                    st.plotly_chart(fig, use_container_width=True)
                                                    
                                                    # Добавляем кнопки оценки "Tinder-style"
                                                    st.markdown("### 📋 Оценка данных")
                                                    st.markdown("Оцените наличие ионосферного эффекта на данных:")
                                                    
                                                    col_effect, col_no_effect = st.columns(2)
                                                    
                                                    with col_effect:
                                                        if st.button("✅ Есть эффект", key="btn_effect", use_container_width=True):
                                                            st.session_state['last_evaluation'] = "effect"
                                                            st.success("✅ Отмечено наличие эффекта!")
                                                    
                                                    with col_no_effect:
                                                        if st.button("❌ Нет эффекта", key="btn_no_effect", use_container_width=True):
                                                            st.session_state['last_evaluation'] = "no_effect"
                                                            st.info("❌ Отмечено отсутствие эффекта.")
                                                    
                                                    # Отображаем историю оценок
                                                    if 'evaluations' not in st.session_state:
                                                        st.session_state['evaluations'] = []
                                                    
                                                    if 'last_evaluation' in st.session_state:
                                                        # Добавляем текущую оценку в историю
                                                        current_eval = {
                                                            'satellites': selected_sats,
                                                            'stations': selected_stations,
                                                            'evaluation': st.session_state['last_evaluation'],
                                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                        }
                                                        
                                                        # Проверяем, не дублируется ли оценка
                                                        is_duplicate = False
                                                        for eval in st.session_state['evaluations']:
                                                            if eval['satellites'] == current_eval['satellites'] and \
                                                               eval['stations'] == current_eval['stations'] and \
                                                               eval['evaluation'] == current_eval['evaluation']:
                                                                is_duplicate = True
                                                                break
                                                        
                                                        if not is_duplicate and 'last_evaluation' in st.session_state:
                                                            st.session_state['evaluations'].append(current_eval)
                                                            # Удаляем флаг последней оценки, чтобы не добавлять повторно
                                                            del st.session_state['last_evaluation']
                                                    
                                                    # Показываем историю оценок
                                                    if st.session_state['evaluations']:
                                                        st.markdown("### 📝 История оценок")
                                                        for i, eval in enumerate(st.session_state['evaluations']):
                                                            eval_type = "✅ Есть эффект" if eval['evaluation'] == "effect" else "❌ Нет эффекта"
                                                            st.markdown(f"**{i+1}. {eval_type}** - Спутники: {', '.join(eval['satellites'])} | Станции: {', '.join(eval['stations'])} | {eval['timestamp']}")
                                                else:
                                                    st.warning("⚠️ Выберите хотя бы один спутник и одну станцию для отображения данных")
                                            else:
                                                st.warning("⚠️ Нет данных для отображения. Извлеките данные из HDF файла.")
                                        
                                        with tab_export:
                                            st.markdown("### 💾 Экспорт данных")
                                            
                                            export_format = st.radio(
                                                "Выберите формат экспорта:",
                                                ["CSV", "JSON", "Excel"],
                                                horizontal=True
                                            )
                                            
                                            if st.button("📥 Скачать данные", use_container_width=True):
                                                st.info("Функция экспорта будет доступна в следующей версии")
                                    else:
                                        st.warning("⚠️ Не удалось извлечь данные")
                            else:
                                st.warning("⚠️ Не выбрано ни одной станции")
                else:
                    st.warning("⚠️ Не найдено станций в указанном регионе. Попробуйте изменить границы.")
        else:
            st.info("📥 Загрузите HDF файл для анализа site-sat данных")
            
        # Отображаем табы с результатами, если данные уже были извлечены
        if 'site_sat_data' in st.session_state and st.session_state['site_sat_data'] and 'selected_sites' in st.session_state and st.session_state['selected_sites']:
            filtered_sites = st.session_state['selected_sites']
            site_sat_data = st.session_state['site_sat_data']
            
            st.markdown("### 📊 Результаты анализа данных")
            tab_summary, tab_geometry, tab_data, tab_export = st.tabs(["Сводка", "Геометрия", "Данные", "Экспорт"])
            
            with tab_summary:
                st.markdown("### 📊 Сводка по данным")
                
                # Статистика
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric("Станции", len(filtered_sites))
                
                with col_stat2:
                    total_sats = sum(len(site_data) for site_data in site_sat_data.values())
                    st.metric("Спутники", total_sats)
                
                with col_stat3:
                    total_obs = sum(sum(1 for _ in sat_data.values()) for site_data in site_sat_data.values() for sat_data in site_data.values())
                    st.metric("Наблюдения", total_obs)
                
                # Таблицы данных
                st.markdown("#### 📡 Станции")
                site_df = pd.DataFrame([
                    {"Название": site.name, "Широта": site.lat, "Долгота": site.lon}
                    for site in filtered_sites
                ])
                st.dataframe(site_df, use_container_width=True)
                
                st.markdown("#### 🛰️ Спутники")
                # Собираем уникальные спутники
                all_sats = set()
                for site_data in site_sat_data.values():
                    all_sats.update(site_data.keys())
                
                sat_df = pd.DataFrame([
                    {"Название": sat.name, "Система": sat.system}
                    for sat in all_sats
                ])
                st.dataframe(sat_df, use_container_width=True)
            
            with tab_geometry:
                st.markdown("### 📐 Геометрические параметры")
                
                # Выбор станции и спутника для отображения
                col_geo_site, col_geo_sat = st.columns(2)
                
                with col_geo_site:
                    selected_geo_site = st.selectbox(
                        "Выберите станцию:",
                        options=[site.name for site in filtered_sites],
                        key="geo_site_selector_loaded"
                    )
                
                # Находим выбранную станцию
                selected_site_obj = next((site for site in filtered_sites if site.name == selected_geo_site), None)
                
                if selected_site_obj and selected_site_obj in site_sat_data:
                    site_data = site_sat_data[selected_site_obj]
                    
                    with col_geo_sat:
                        available_sats = list(site_data.keys())
                        selected_geo_sat = st.selectbox(
                            "Выберите спутник:",
                            options=[sat.name for sat in available_sats],
                            key="geo_sat_selector_loaded"
                        )
                    
                    # Находим выбранный спутник
                    selected_sat_obj = next((sat for sat in available_sats if sat.name == selected_geo_sat), None)
                    
                    if selected_sat_obj:
                        sat_data = site_data[selected_sat_obj]
                        
                        # Отображаем график угла места и азимута
                        if DataProducts.elevation in sat_data and DataProducts.azimuth in sat_data and DataProducts.timestamp in sat_data:
                            elevations = sat_data[DataProducts.elevation]
                            azimuths = sat_data[DataProducts.azimuth]
                            timestamps = sat_data[DataProducts.timestamp]
                            
                            # Преобразуем временные метки в datetime
                            times = [datetime.fromtimestamp(ts) for ts in timestamps]
                            
                            # Создаем график
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=times,
                                y=elevations,
                                mode='lines+markers',
                                name='Угол места',
                                line=dict(color='blue'),
                                marker=dict(size=6)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=times,
                                y=azimuths,
                                mode='lines+markers',
                                name='Азимут',
                                line=dict(color='red'),
                                marker=dict(size=6),
                                yaxis='y2'
                            ))
                            
                            fig.update_layout(
                                title=f"Геометрия для станции {selected_geo_site} и спутника {selected_geo_sat}",
                                xaxis_title="Время",
                                yaxis_title="Угол места (градусы)",
                                yaxis2=dict(
                                    title="Азимут (градусы)",
                                    overlaying='y',
                                    side='right',
                                    range=[0, 360]
                                ),
                                height=400,
                                margin=dict(l=0, r=0, t=30, b=0),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="geometry_plot_loaded")
                        else:
                            st.warning("⚠️ Недостаточно данных для построения графика геометрии")
                    else:
                        st.warning("⚠️ Нет данных для выбранной станции")
                
                with tab_data:
                    st.markdown("### 📊 Данные измерений")
                    st.info("Здесь будут отображаться данные измерений TEC, ROTI и других параметров")
                    
                    # Добавьте здесь код для отображения данных измерений
                
                with tab_export:
                    st.markdown("### 💾 Экспорт данных")
                    
                    export_format = st.radio(
                        "Выберите формат экспорта:",
                        ["CSV", "JSON", "Excel"],
                        horizontal=True,
                        key="export_format_loaded"
                    )
                    
                    if st.button("📥 Скачать данные", use_container_width=True, key="download_data_loaded"):
                        st.info("Функция экспорта будет доступна в следующей версии")

    # Разделитель перед настройками
    st.markdown("---")
    st.markdown("## ⚙️ Настройки и параметры")

    # Настройки анализа ионосферы
    st.subheader("⚙️ Настройки анализа ионосферы")
    
    # Настройка параметров отображения через набор горизонтальных колонок
    col_date, col_structure, col_names, col_hm, col_time, col_show, col_event, col_sat, col_data, col_thr, col_clear = st.columns([2,2,1,1,2,1,1,1,1,1,1])

    with col_date:
        # Расширяем выбор даты включая 2025 год
        current_year = datetime.now().year
        max_date = date(current_year, 12, 31)  # Включаем текущий год
        min_date = date(2020, 1, 1)  # С 2020 года данные более стабильны
        
        analysis_date = st.date_input(
            "📅 Дата для анализа",
            value=date(2025, 6, 15),  # Дата по умолчанию из 2025 года
            min_value=min_date,
            max_value=max_date,
            help="Выберите дату для анализа ионосферных данных"
        )
        
        # Проверка года и показ информации
        if analysis_date.year < 2023:
            st.warning("⚠️ Для дат до 2023 года данные могут быть ограничены")
        elif analysis_date.year >= 2025:
            # Убираем ненужную надпись о доступности данных
            pass
        
    with col_structure:
        structure_type = st.selectbox(
            "🌊 Тип ионосферной структуры",
            ["TEC", "Scintillation", "Gradient", "Anomaly"],
            help="Выберите тип ионосферной структуры для анализа"
        )

    # Отображение названий станций
    with col_names:
        show_names = st.checkbox("Site names", value=True)
    
    # Высота ионосферы для расчета SIP
    with col_hm:
        ion_height = st.number_input("hm:", min_value=100, max_value=1000, value=300, step=10)

    # Ввод времени наблюдения
    with col_time:
        time_str = st.text_input("Time:", value="00:00:00")
    
    # Кнопка "Показать" для построения карты
    with col_show:
        show_btn = st.button("Show")
    
    # Тип события (на будущее, может пригодиться)
    with col_event:
        event_type = st.selectbox("Event", ["Event"])
    
    # Выбор спутников
    all_sats = ["G03", "G07", "G12"]
    with col_sat:
        plot_sats = st.multiselect("Спутники", all_sats, default=all_sats)
    
    # Выбор типа данных (например, ROTI или TEC)
    with col_data:
        data_type = st.selectbox("Data", ["ROTI"])
    
    # Порог срабатывания
    with col_thr:
        threshold = st.number_input("Threshold", value=-0.5, step=0.1)
    
    # Кнопка очистки
    with col_clear:
        if st.button("🗑️ Очистить все", help="Очистить все сохраненные данные"):
            clear_all_data()
            st.rerun()

    # Загрузка файлов в горизонтальном ряду (перемещено сюда для правильного порядка)
    col_nav, col_data = st.columns(2)
    
    with col_nav:
        nav_file = st.file_uploader("Навигационный файл (RINEX NAV)", type=["rnx", "nav", "txt"])
    
    with col_data:
        data_file = st.file_uploader("Файл временных рядов (ROTI/TEC)", type=["txt", "csv"])

    # Обработка загруженного NAV файла
    if nav_file is not None:
        with st.spinner("📡 Обработка NAV файла..."):
            nav_stations = parse_nav_file(nav_file.read())
            
            if nav_stations:
                st.session_state['nav_file_stations'] = nav_stations
                st.success(f"✅ Загружено {len(nav_stations)} станций из NAV файла")
                
                # Показываем первые несколько станций
                if len(nav_stations) > 0:
                    st.info(f"📍 Примеры станций: {', '.join([s['name'] for s in nav_stations[:5]])}")
            else:
                st.warning("⚠️ Не удалось извлечь станции из NAV файла")
                st.session_state['nav_file_stations'] = None

    # Проверяем, есть ли загруженные станции из NAV файла
    if st.session_state['nav_file_stations'] is not None:
        # Используем станции из NAV файла, загруженного пользователем
        current_stations = st.session_state['nav_file_stations']
        
        col_nav_info, col_nav_clear = st.columns([3, 1])
        with col_nav_info:
            st.success(f"✅ Используются реальные данные из загруженного NAV файла ({len(current_stations)} станций)")
        
        with col_nav_clear:
            if st.button("🗑️ Очистить NAV", help="Вернуться к стандартным станциям"):
                st.session_state['nav_file_stations'] = None
                st.rerun()
                
    elif st.session_state.get('nav_date_loaded') and st.session_state.get('nav_date_stations'):
        # Используем станции из автоматически загруженного NAV файла для выбранной даты
        nav_date_stations = st.session_state['nav_date_stations']
        current_stations = [
            {
                'name': info['name'], 
                'lat': info['lat'], 
                'lon': info['lon'], 
                'color': 'orange'  # Оранжевый цвет для станций из даты
            }
            for info in nav_date_stations.values()
        ]
        
        # Информация о источнике данных
        nav_info = st.session_state.get('nav_date_info', {})
        col_nav_info, col_nav_clear = st.columns([3, 1])
        with col_nav_info:
            # Убираем лишнее сообщение о количестве станций
            pass
        
        with col_nav_clear:
            if st.button("🗑️ Очистить дату", help="Вернуться к стандартным станциям"):
                st.session_state['nav_date_loaded'] = False
                st.session_state['nav_date_stations'] = None
                st.session_state['last_selected_date'] = None
                st.rerun()
    else:
        # Используем стандартные станции
        current_stations = global_stations.get(selected_region, global_stations["🌍 Глобальная карта"])
        
        # Убираем информационные сообщения о стандартных станциях
        pass

    # Обновляем список имен станций
    site_names = [site['name'] for site in current_stations]

    # Выбор станций пользователем
    selected_sites = st.multiselect("Станции", site_names, default=site_names)
    sites = [s for s in current_stations if s["name"] in selected_sites]

    # Дополнительные настройки в горизонтальном ряду
    col_event_lat, col_event_lon, col_radius = st.columns(3)
    
    with col_event_lat:
        event_lat = st.number_input("Широта эпицентра", value=-7.0, min_value=-90.0, max_value=90.0, step=0.1)
    
    with col_event_lon:
        event_lon = st.number_input("Долгота эпицентра", value=-74.0, min_value=-180.0, max_value=180.0, step=0.1)
    
    with col_radius:
        radius_km = st.number_input("Радиус (км)", value=1000, min_value=100, max_value=5000, step=100)

    # Загрузка данных
    with st.expander("📂 Загрузка данных", expanded=True):
        st.markdown("### 📊 Загрузка HDF файла")
        
        # Загрузка файла
        uploaded_file = st.file_uploader("Выберите HDF файл", type=["h5", "hdf5"], key="hdf_uploader")
        
        # Если файл загружен, сохраняем его
        if uploaded_file is not None:
            # Сохраняем файл во временный каталог
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Сохраняем путь к файлу в состоянии сессии
            st.session_state['hdf_file'] = tmp_path
            st.success(f"✅ Файл загружен: {uploaded_file.name}")
            
            # Кнопка для извлечения данных
            if st.button("🔍 Извлечь данные из HDF", use_container_width=True):
                try:
                    with st.spinner("⏳ Извлечение данных из HDF файла..."):
                        # Получаем список станций из HDF файла
                        sites = get_sites_from_hdf(tmp_path)
                        
                        if sites:
                            # Извлекаем данные для всех видимых спутников
                            data = retrieve_visible_sats_data(tmp_path, sites)
                            
                            if data:
                                # Сохраняем данные в состоянии сессии
                                st.session_state['hdf_data'] = data
                                save_hdf_data()  # Сохраняем данные в файл
                                
                                # Отображаем информацию о загруженных данных
                                st.success(f"✅ Данные извлечены: {len(sites)} станций")
                                
                                # Информация о станциях
                                stations_info = []
                                for site in sites:
                                    # Получаем количество спутников для станции
                                    if site in data:
                                        satellites = len(data[site])
                                        stations_info.append(f"{site.name} ({satellites} спутников)")
                                    else:
                                        stations_info.append(f"{site.name} (нет данных)")
                                
                                st.info("📡 Станции: " + ", ".join(stations_info))
                                
                                # Перезагружаем страницу для обновления интерфейса
                                st.rerun()
                            else:
                                st.error("❌ Не удалось извлечь данные о спутниках")
                        else:
                            st.error("❌ Не удалось найти станции в HDF файле")
                except Exception as e:
                    st.error(f"❌ Ошибка при извлечении данных: {str(e)}")
        else:
            # Если файл не загружен, но есть сохраненные данные
            if 'hdf_data' in st.session_state and st.session_state['hdf_data']:
                st.success("✅ Используются ранее загруженные данные")
                
                # Информация о станциях
                data = st.session_state['hdf_data']
                stations_info = []
                for site_name in data.keys():
                    satellites = len(data[site_name])
                    stations_info.append(f"{site_name} ({satellites} спутников)")
                
                st.info("📡 Станции: " + ", ".join(stations_info))
                
                # Кнопка для очистки данных
                if st.button("🗑️ Очистить данные", use_container_width=True):
                    st.session_state.pop('hdf_data', None)
                    st.session_state.pop('hdf_file', None)
                    clear_all_data()
                    st.rerun()
            else:
                st.info("ℹ️ Загрузите HDF файл для анализа данных")

if mode == "Разметка (Tinder)":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Разметка траекторий SIP — экспертная оценка")
    
    # Настройка отображения
    col_settings1, col_settings2, col_settings3 = st.columns(3)
    
    with col_settings1:
        selected_date = st.date_input(
            "📅 Дата анализа", 
            value=date(2025, 6, 20),
            help="Дата для анализа ионосферных данных"
        )
    
    with col_settings2:
        ion_height = st.number_input("Высота ионосферы (hm):", min_value=100, max_value=1000, value=300, step=10, help="Высота ионосферы в км для расчета SIP")
    
    with col_settings3:
        projection_type = st.selectbox(
            "Проекция карты:",
            ["mercator", "orthographic", "natural earth"],
            index=0,
            help="Тип проекции для отображения карты"
        )
    
    # Загрузка HDF файла если он еще не загружен
    if 'hdf_file_path' not in st.session_state or not st.session_state['hdf_file_path']:
        st.info("📥 Загрузите HDF файл для анализа")
        
        col_upload1, col_upload2 = st.columns(2)
        
        with col_upload1:
            # Формируем URL для загрузки HDF файла
            filename = selected_date.strftime("simurg_data_%Y-%m-%d.h5")
            url = f"https://simurg.space/gen_file?data=obs&date={selected_date.strftime('%Y-%m-%d')}"
            
            # Используем путь внутри директории приложения
            local_path = HDF_DIR / filename
            
            if st.button("📥 Загрузить HDF файл", key="download_hdf_tinder"):
                try:
                    load_hdf_data(url, local_path, override=False)
                    st.session_state['hdf_file_path'] = local_path
                    st.session_state['hdf_date'] = selected_date
                    save_session_data()  # Сохраняем данные
                    st.success(f"✅ HDF файл загружен: {filename}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке HDF файла: {e}")
        
        with col_upload2:
            uploaded_file = st.file_uploader("Или загрузите свой HDF файл", type=["h5"])
            if uploaded_file is not None:
                try:
                    # Сохраняем загруженный файл
                    local_path = HDF_DIR / uploaded_file.name
                    with open(local_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state['hdf_file_path'] = local_path
                    save_session_data()  # Сохраняем данные
                    st.success(f"✅ HDF файл загружен: {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке загруженного файла: {e}")
    
    else:
        # Отображаем информацию о загруженном файле
        hdf_path = st.session_state['hdf_file_path']
        st.success(f"✅ HDF файл загружен: {hdf_path.name}")
        
        # Создаем две колонки для отображения станций и спутников
        col_stations, col_satellites = st.columns(2)
        
        with col_stations:
            # Получаем станции из HDF файла
            hdf_sites = get_sites_from_hdf(hdf_path)
            if hdf_sites:
                selected_site = st.selectbox(
                    "Выберите станцию:",
                    options=[f"{s.name} ({s.lat:.2f}°, {s.lon:.2f}°)" for s in hdf_sites],
                    key="tinder_site_selector"
                )
                
                # Извлекаем имя станции
                site_name = selected_site.split(' (')[0]
                selected_site_obj = next((site for site in hdf_sites if site.name == site_name), None)
            else:
                st.warning("⚠️ Не удалось найти станции в HDF файле")
                selected_site_obj = None
        
        # Если выбрана станция, извлекаем данные спутников
        if selected_site_obj:
            with col_satellites:
                # Извлекаем данные для выбранной станции
                with st.spinner("🛰️ Извлечение данных спутников..."):
                    site_data = retrieve_visible_sats_data(hdf_path, [selected_site_obj])
                    
                    if site_data and selected_site_obj in site_data:
                        # Получаем список спутников
                        available_sats = list(site_data[selected_site_obj].keys())
                        
                        if available_sats:
                            # Инициализируем индекс текущего спутника, если его еще нет
                            if 'tinder_current_sat_index' not in st.session_state:
                                st.session_state['tinder_current_sat_index'] = 0
                            
                            # Убеждаемся, что индекс находится в пределах списка
                            if st.session_state['tinder_current_sat_index'] >= len(available_sats):
                                st.session_state['tinder_current_sat_index'] = 0
                            
                            # Получаем текущий спутник по индексу
                            current_sat_index = st.session_state['tinder_current_sat_index']
                            current_sat_name = available_sats[current_sat_index].name
                            
                            # Отображаем информацию о текущем спутнике и прогрессе
                            col_progress, col_sat_info = st.columns([3, 1])
                            
                            with col_progress:
                                progress_text = f"Спутник {current_sat_index + 1} из {len(available_sats)}"
                                st.progress((current_sat_index + 1) / len(available_sats), text=progress_text)
                            
                            with col_sat_info:
                                st.info(f"🛰️ {current_sat_name}")
                            
                            # Находим выбранный спутник
                            selected_sat_obj = available_sats[current_sat_index]
                        else:
                            st.warning("⚠️ Нет доступных спутников для выбранной станции")
                            selected_sat_obj = None
                    else:
                        st.warning("⚠️ Не удалось извлечь данные для выбранной станции")
                        selected_sat_obj = None
            
            # Если выбраны станция и спутник, отображаем данные
            if selected_sat_obj and selected_site_obj in site_data and selected_sat_obj in site_data[selected_site_obj]:
                # Получаем данные для выбранной пары станция-спутник
                sat_data = site_data[selected_site_obj][selected_sat_obj]
                
                # Создаем уникальный ключ для текущей пары станция-спутник
                current_pair_key = f"{selected_site_obj.name}_{selected_sat_obj.name}"
                
                # Инициализируем словарь оценок, если его еще нет
                if 'satellite_evaluations' not in st.session_state:
                    st.session_state['satellite_evaluations'] = {}
                
                # Отображаем данные в двух колонках
                col_trajectory, col_data = st.columns([1, 1])
                
                with col_trajectory:
                    st.subheader("Траектория SIP")
                    
                    # Проверяем наличие данных о местоположении SIP
                    if DataProducts.elevation in sat_data and DataProducts.azimuth in sat_data:
                        # Создаем карту с траекторией SIP
                        fig_map = go.Figure()
                        
                        # Добавляем станцию на карту
                        fig_map.add_trace(go.Scattergeo(
                            lon=[selected_site_obj.lon], 
                            lat=[selected_site_obj.lat],
                            mode='markers+text',
                            marker=dict(size=10, color='blue'),
                            text=[selected_site_obj.name],
                            textposition="top center",
                            name=f"Станция: {selected_site_obj.name}"
                        ))
                        
                        # Рассчитываем координаты SIP
                        if DataProducts.elevation in sat_data and DataProducts.azimuth in sat_data:
                            elevations = sat_data[DataProducts.elevation]
                            azimuths = sat_data[DataProducts.azimuth]
                            
                            # Конвертируем в радианы для расчета
                            site_lat_rad = np.radians(selected_site_obj.lat)
                            site_lon_rad = np.radians(selected_site_obj.lon)
                            el_rad = np.radians(elevations)
                            az_rad = np.radians(azimuths)
                            
                            # Рассчитываем SIP координаты (упрощенная версия)
                            R = 6371.0  # Радиус Земли в км
                            h = ion_height / 1000.0  # Высота ионосферы в км
                            
                            # Расчет угла между зенитом и лучом к SIP
                            psi = np.pi/2 - el_rad
                            
                            # Расчет центрального угла между станцией и SIP
                            alpha = np.arcsin((R / (R + h)) * np.sin(psi))
                            
                            # Расчет угла от зенита до SIP
                            beta = psi - alpha
                            
                            # Расчет широты и долготы SIP
                            sip_lat = np.arcsin(np.sin(site_lat_rad) * np.cos(beta) + 
                                               np.cos(site_lat_rad) * np.sin(beta) * np.cos(az_rad))
                            
                            sip_lon = site_lon_rad + np.arctan2(np.sin(az_rad) * np.sin(beta) * np.cos(site_lat_rad),
                                                             np.cos(beta) - np.sin(site_lat_rad) * np.sin(sip_lat))
                            
                            # Конвертируем обратно в градусы
                            sip_lat_deg = np.degrees(sip_lat)
                            sip_lon_deg = np.degrees(sip_lon)
                            
                            # Добавляем траекторию SIP на карту
                            fig_map.add_trace(go.Scattergeo(
                                lon=sip_lon_deg, 
                                lat=sip_lat_deg,
                                mode='lines+markers',
                                marker=dict(size=4, color='red'),
                                line=dict(width=2, color='red'),
                                name=f"SIP: {selected_sat_obj.name}"
                            ))
                        
                        # Настройка карты
                        fig_map.update_geos(
                            projection_type=projection_type,
                            showcountries=True, 
                            showcoastlines=True, 
                            showland=True, 
                            landcolor="#f5f5f5",
                            resolution=50
                        )
                        
                        fig_map.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0),
                            title=f"Траектория SIP для {selected_site_obj.name}-{selected_sat_obj.name} (hm={ion_height} км)"
                        )
                        
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ Недостаточно данных для построения траектории SIP")
                
                with col_data:
                    st.subheader("Данные измерений")
                    
                    # Проверяем наличие временных меток и данных TEC
                    if DataProducts.time.value in sat_data and DataProducts.atec in sat_data:
                        times = sat_data[DataProducts.time.value]
                        tec_values = sat_data[DataProducts.atec]
                        
                        # Создаем график данных TEC
                        fig_data = go.Figure()
                        
                        # Определяем цвет для станции в соответствии с изображением
                        station_colors = {
                            'AREQ': 'blue',
                            'SCRZ': 'red',
                            'BRAZ': 'green'
                        }
                        
                        # Выбираем цвет для текущей станции
                        station_color = station_colors.get(selected_site_obj.name, 'blue')
                        
                        # Добавляем данные TEC с соответствующим цветом станции
                        fig_data.add_trace(go.Scatter(
                            x=times,
                            y=tec_values,
                            mode='lines',
                            name='TEC',
                            line=dict(color=station_color, width=2)
                        ))
                        
                        # Если есть данные ROTI, добавляем их на график
                        if DataProducts.roti in sat_data:
                            roti_values = sat_data[DataProducts.roti]
                            
                            fig_data.add_trace(go.Scatter(
                                x=times,
                                y=roti_values,
                                mode='lines',
                                name='ROTI',
                                line=dict(color='red' if station_color != 'red' else 'orange', width=2),
                                yaxis='y2'
                            ))
                        
                        # Настройка графика для соответствия изображению
                        fig_data.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0),
                            title=f"{selected_site_obj.name} - {selected_sat_obj.name}",
                            xaxis_title="Время",
                            yaxis_title="TEC (TECU)",
                            yaxis2=dict(
                                title="ROTI (TECU/min)",
                                overlaying='y',
                                side='right',
                                showgrid=False
                            ) if DataProducts.roti in sat_data else None,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            plot_bgcolor='rgb(20, 24, 35)',  # Темный фон как на изображении
                            paper_bgcolor='rgb(20, 24, 35)',  # Темный фон как на изображении
                            font=dict(color='white')  # Белый текст для контраста
                        )
                        
                        # Добавляем сетку как на изображении
                        fig_data.update_xaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(255, 255, 255, 0.2)'
                        )
                        
                        fig_data.update_yaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(255, 255, 255, 0.2)'
                        )
                        
                        st.plotly_chart(fig_data, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ Недостаточно данных для построения графика TEC")
                
                # Добавляем кнопки оценки в стиле Tinder
                st.markdown("### 📋 Оценка ионосферного эффекта")
                
                # Добавляем CSS для стилизации кнопок
                st.markdown("""
                <style>
                    .stButton > button {
                        font-size: 20px !important;
                        font-weight: bold !important;
                        padding: 15px !important;
                        border-radius: 10px !important;
                        margin-top: 10px !important;
                        margin-bottom: 10px !important;
                    }
                    .effect-button > button {
                        background-color: #4CAF50 !important;
                        color: white !important;
                    }
                    .no-effect-button > button {
                        background-color: #f44336 !important;
                        color: white !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                col_effect, col_no_effect = st.columns(2)
                
                # Получаем текущую оценку для этой пары (если есть)
                current_evaluation = st.session_state['satellite_evaluations'].get(current_pair_key, None)
                
                with col_effect:
                    # Используем markdown для создания стилизованной кнопки
                    st.markdown('<div class="effect-button">', unsafe_allow_html=True)
                    if st.button("✅ ЕСТЬ ЭФФЕКТ", key=f"effect_{current_pair_key}", use_container_width=True, help="Отметить наличие ионосферного эффекта"):
                        st.session_state['satellite_evaluations'][current_pair_key] = "effect"
                        st.success("✅ Отмечено наличие эффекта!")
                        
                        # Автоматическое переключение на следующий спутник
                        if st.session_state['tinder_current_sat_index'] < len(available_sats) - 1:
                            st.session_state['tinder_current_sat_index'] += 1
                        else:
                            # Если это последний спутник, переходим к первому
                            st.session_state['tinder_current_sat_index'] = 0
                            st.info("🔄 Все спутники оценены, начинаем заново")
                        
                        # Сохраняем данные
                        save_session_data()
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_no_effect:
                    # Используем markdown для создания стилизованной кнопки
                    st.markdown('<div class="no-effect-button">', unsafe_allow_html=True)
                    if st.button("❌ НЕТ ЭФФЕКТА", key=f"no_effect_{current_pair_key}", use_container_width=True, help="Отметить отсутствие ионосферного эффекта"):
                        st.session_state['satellite_evaluations'][current_pair_key] = "no_effect"
                        st.info("❌ Отмечено отсутствие эффекта.")
                        
                        # Автоматическое переключение на следующий спутник
                        if st.session_state['tinder_current_sat_index'] < len(available_sats) - 1:
                            st.session_state['tinder_current_sat_index'] += 1
                        else:
                            # Если это последний спутник, переходим к первому
                            st.session_state['tinder_current_sat_index'] = 0
                            st.info("🔄 Все спутники оценены, начинаем заново")
                        
                        # Сохраняем данные
                        save_session_data()
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Отображаем текущую оценку
                if current_evaluation:
                    if current_evaluation == "effect":
                        st.success(f"✅ Текущая оценка: ЕСТЬ ЭФФЕКТ")
                    else:
                        st.info(f"❌ Текущая оценка: НЕТ ЭФФЕКТА")
                
                # Отображаем историю оценок
                if st.session_state['satellite_evaluations']:
                    st.markdown("### 📝 История оценок")
                    
                    # Создаем DataFrame для отображения истории
                    evaluation_data = []
                    for pair_key, eval_type in st.session_state['satellite_evaluations'].items():
                        try:
                            site_name, sat_name = pair_key.split('_')
                            evaluation_data.append({
                                "Станция": site_name,
                                "Спутник": sat_name,
                                "Оценка": "✅ ЕСТЬ ЭФФЕКТ" if eval_type == "effect" else "❌ НЕТ ЭФФЕКТА",
                                "Тип": eval_type
                            })
                        except:
                            # Пропускаем некорректные ключи
                            pass
                    
                    if evaluation_data:
                        df = pd.DataFrame(evaluation_data)
                        
                        # Фильтры для истории
                        col_filter1, col_filter2, col_filter3 = st.columns(3)
                        
                        with col_filter1:
                            filter_station = st.multiselect(
                                "Фильтр по станциям:",
                                options=sorted(df["Станция"].unique()),
                                default=[]
                            )
                        
                        with col_filter2:
                            filter_sat = st.multiselect(
                                "Фильтр по спутникам:",
                                options=sorted(df["Спутник"].unique()),
                                default=[]
                            )
                        
                        with col_filter3:
                            filter_eval = st.multiselect(
                                "Фильтр по оценке:",
                                options=["✅ ЕСТЬ ЭФФЕКТ", "❌ НЕТ ЭФФЕКТА"],
                                default=[]
                            )
                        
                        # Применяем фильтры
                        if filter_station:
                            df = df[df["Станция"].isin(filter_station)]
                        
                        if filter_sat:
                            df = df[df["Спутник"].isin(filter_sat)]
                        
                        if filter_eval:
                            df = df[df["Оценка"].isin(filter_eval)]
                        
                        # Отображаем таблицу с историей
                        st.dataframe(
                            df[["Станция", "Спутник", "Оценка"]], 
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Статистика оценок
                        st.markdown("#### 📊 Статистика оценок")
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            total_evals = len(df)
                            st.metric("Всего оценок", total_evals)
                        
                        with col_stat2:
                            effect_count = len(df[df["Тип"] == "effect"])
                            effect_percent = (effect_count / total_evals * 100) if total_evals > 0 else 0
                            st.metric("Есть эффект", f"{effect_count} ({effect_percent:.1f}%)")
                        
                        with col_stat3:
                            no_effect_count = len(df[df["Тип"] == "no_effect"])
                            no_effect_percent = (no_effect_count / total_evals * 100) if total_evals > 0 else 0
                            st.metric("Нет эффекта", f"{no_effect_count} ({no_effect_percent:.1f}%)")
                        
                        # Кнопка для экспорта данных
                        if st.button("📥 Экспортировать оценки в CSV", use_container_width=True):
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать CSV файл",
                                data=csv,
                                file_name=f"sip_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.info("История оценок пуста")
            else:
                st.warning("⚠️ Выберите станцию и спутник для отображения данных")
        else:
            st.warning("⚠️ Выберите станцию для начала анализа")

    # Добавляем кнопки навигации
    st.markdown("### 🎯 Навигация по спутникам")
    
    col_prev, col_next, col_skip = st.columns(3)
    
    with col_prev:
        if st.button("⬅️ Предыдущий", key="prev_satellite", use_container_width=True, help="Перейти к предыдущему спутнику"):
            if st.session_state['tinder_current_sat_index'] > 0:
                st.session_state['tinder_current_sat_index'] -= 1
            else:
                st.session_state['tinder_current_sat_index'] = len(available_sats) - 1
            save_session_data()
            st.rerun()
    
    with col_next:
        if st.button("➡️ Следующий", key="next_satellite", use_container_width=True, help="Перейти к следующему спутнику"):
            if st.session_state['tinder_current_sat_index'] < len(available_sats) - 1:
                st.session_state['tinder_current_sat_index'] += 1
            else:
                st.session_state['tinder_current_sat_index'] = 0
            save_session_data()
            st.rerun()
    
    with col_skip:
        if st.button("⏭️ Пропустить", key="skip_satellite", use_container_width=True, help="Пропустить текущий спутник без оценки"):
            if st.session_state['tinder_current_sat_index'] < len(available_sats) - 1:
                st.session_state['tinder_current_sat_index'] += 1
            else:
                st.session_state['tinder_current_sat_index'] = 0
            save_session_data()
            st.rerun()
    
    # Добавляем кнопки оценки в стиле Tinder
    st.markdown("### 📋 Оценка ионосферного эффекта")
    
    # Добавляем CSS для стилизации кнопок
    st.markdown("""
    <style>
        .stButton > button {
            font-size: 20px !important;
            font-weight: bold !important;
            padding: 15px !important;
            border-radius: 10px !important;
            margin-top: 10px !important;
            margin-bottom: 10px !important;
        }
        .effect-button > button {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .no-effect-button > button {
            background-color: #f44336 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col_effect, col_no_effect = st.columns(2)
    
    # Получаем текущую оценку для этой пары (если есть)
    current_evaluation = st.session_state['satellite_evaluations'].get(current_pair_key, None)
    
    with col_effect:
        # Используем markdown для создания стилизованной кнопки
        st.markdown('<div class="effect-button">', unsafe_allow_html=True)
        if st.button("✅ ЕСТЬ ЭФФЕКТ", key=f"effect_{current_pair_key}", use_container_width=True, help="Отметить наличие ионосферного эффекта"):
            st.session_state['satellite_evaluations'][current_pair_key] = "effect"
            st.success("✅ Отмечено наличие эффекта!")
            
            # Автоматическое переключение на следующий спутник
            if st.session_state['tinder_current_sat_index'] < len(available_sats) - 1:
                st.session_state['tinder_current_sat_index'] += 1
            else:
                # Если это последний спутник, переходим к первому
                st.session_state['tinder_current_sat_index'] = 0
                st.info("🔄 Все спутники оценены, начинаем заново")
            
            # Сохраняем данные
            save_session_data()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_no_effect:
        # Используем markdown для создания стилизованной кнопки
        st.markdown('<div class="no-effect-button">', unsafe_allow_html=True)
        if st.button("❌ НЕТ ЭФФЕКТА", key=f"no_effect_{current_pair_key}", use_container_width=True, help="Отметить отсутствие ионосферного эффекта"):
            st.session_state['satellite_evaluations'][current_pair_key] = "no_effect"
            st.info("❌ Отмечено отсутствие эффекта.")
            
            # Автоматическое переключение на следующий спутник
            if st.session_state['tinder_current_sat_index'] < len(available_sats) - 1:
                st.session_state['tinder_current_sat_index'] += 1
            else:
                # Если это последний спутник, переходим к первому
                st.session_state['tinder_current_sat_index'] = 0
                st.info("🔄 Все спутники оценены, начинаем заново")
            
            # Сохраняем данные
            save_session_data()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Отображаем текущую оценку
    if current_evaluation:
        if current_evaluation == "effect":
            st.success(f"✅ Текущая оценка: ЕСТЬ ЭФФЕКТ")
        else:
            st.info(f"❌ Текущая оценка: НЕТ ЭФФЕКТА")
    
    # Отображаем историю оценок
    if st.session_state['satellite_evaluations']:
        st.markdown("### 📝 История оценок")
        
        # Создаем DataFrame для отображения истории
        evaluation_data = []
        for pair_key, eval_type in st.session_state['satellite_evaluations'].items():
            try:
                site_name, sat_name = pair_key.split('_')
                evaluation_data.append({
                    "Станция": site_name,
                    "Спутник": sat_name,
                    "Оценка": "✅ ЕСТЬ ЭФФЕКТ" if eval_type == "effect" else "❌ НЕТ ЭФФЕКТА",
                    "Тип": eval_type
                })
            except:
                # Пропускаем некорректные ключи
                pass
        
        if evaluation_data:
            df = pd.DataFrame(evaluation_data)
            
            # Фильтры для истории
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                filter_station = st.multiselect(
                    "Фильтр по станциям:",
                    options=sorted(df["Станция"].unique()),
                    default=[]
                )
            
            with col_filter2:
                filter_sat = st.multiselect(
                    "Фильтр по спутникам:",
                    options=sorted(df["Спутник"].unique()),
                    default=[]
                )
            
            with col_filter3:
                filter_eval = st.multiselect(
                    "Фильтр по оценке:",
                    options=["✅ ЕСТЬ ЭФФЕКТ", "❌ НЕТ ЭФФЕКТА"],
                    default=[]
                )
            
            # Применяем фильтры
            if filter_station:
                df = df[df["Станция"].isin(filter_station)]
            
            if filter_sat:
                df = df[df["Спутник"].isin(filter_sat)]
            
            if filter_eval:
                df = df[df["Оценка"].isin(filter_eval)]
            
            # Отображаем таблицу с историей
            st.dataframe(
                df[["Станция", "Спутник", "Оценка"]], 
                use_container_width=True,
                hide_index=True
            )
            
            # Статистика оценок
            st.markdown("#### 📊 Статистика оценок")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                total_evals = len(df)
                st.metric("Всего оценок", total_evals)
            
            with col_stat2:
                effect_count = len(df[df["Тип"] == "effect"])
                effect_percent = (effect_count / total_evals * 100) if total_evals > 0 else 0
                st.metric("Есть эффект", f"{effect_count} ({effect_percent:.1f}%)")
            
            with col_stat3:
                no_effect_count = len(df[df["Тип"] == "no_effect"])
                no_effect_percent = (no_effect_count / total_evals * 100) if total_evals > 0 else 0
                st.metric("Нет эффекта", f"{no_effect_count} ({no_effect_percent:.1f}%)")
            
            # Кнопка для экспорта данных
            if st.button("📥 Экспортировать оценки в CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать CSV файл",
                    data=csv,
                    file_name=f"sip_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# CSS стили для улучшения внешнего вида
st.markdown("""
<style>
/* Скрытие кнопки Deploy */
header [data-testid="stDeployButton"] {display: none !important;}

/* Стили для блока ионосферных структур */
.ionosphere-block {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}

/* Стили для кнопок полигона */
.polygon-buttons button {
    background-color: #ff6b6b !important;
    color: white !important;
    border: none !important;
    border-radius: 0.25rem !important;
    padding: 0.5rem 1rem !important;
    margin: 0.25rem !important;
}

.polygon-buttons button:hover {
    background-color: #ff5252 !important;
}

/* Стили для информационных блоков */
.info-box {
    background-color: #e3f2fd;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border-left: 3px solid #2196f3;
    margin: 0.5rem 0;
}

/* Стили для таблицы координат */
.coordinates-table {
    font-size: 0.9rem;
}

/* Стили для кнопок экспорта */
.export-button {
    background-color: #4caf50 !important;
    color: white !important;
}

.clear-button {
    background-color: #f44336 !important;
    color: white !important;
}

/* Улучшение отображения дат */
.date-input {
    font-weight: bold;
}

/* Стили для selectbox структур */
.structure-select {
    font-size: 1rem;
}

/* Анимация для загрузки */
.loading-spinner {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Улучшение отображения статистики */
.stats-container {
    display: flex;
    justify-content: space-between;
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.stat-item {
    text-align: center;
    flex: 1;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1f77b4;
}

.stat-label {
    font-size: 0.9rem;
    color: #666;
}

/* Стили для полигона на карте */
.polygon-info {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}

/* Улучшение кнопок тестирования полигона */
.test-polygon-buttons button {
    background-color: #17a2b8 !important;
    color: white !important;
    border: none !important;
    font-size: 0.8rem !important;
    padding: 0.25rem 0.5rem !important;
}

.test-polygon-buttons button:hover {
    background-color: #138496 !important;
}
</style>
""", unsafe_allow_html=True)

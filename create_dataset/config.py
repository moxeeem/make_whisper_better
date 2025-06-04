import os
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

NUM_VIDEOS = -1  # количество видео для анализа (-1 для обработки всех доступных видео)
MAX_SEARCH_RESULTS = 10_000  # максимальное количество результатов поиска при NUM_VIDEOS = -1
KEYWORD = 'авито'  # ключевое слово для поиска
MIN_CLIP_DURATION = 5  # минимальная длительность аудио сегмента в секундах
MAX_CLIP_DURATION = 25  # максимальная длительность аудио сегмента в секундах
MAX_VIDEO_DURATION = 30 * 60  # максимальная длительность исходного видео в секундах (30 минут)
VERBOSE = False
MIN_SEGMENT_GAP = 60  # минимальное расстояние между найденными сегментами (в секундах)

DATASET_DIR = SCRIPT_DIR / 'clips'
TEMP_DIR = SCRIPT_DIR / 'temp'
DATASET_CSV = DATASET_DIR / 'dataset.csv'
DATASET_JSON = DATASET_DIR / 'dataset.json'
PROCESSED_VIDEOS_FILE = DATASET_DIR / 'processed_videos.json'

DATASET_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True) 

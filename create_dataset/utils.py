import sys
import re
import json
import logging
import subprocess
import importlib
from pathlib import Path

from config import (
    DATASET_DIR,
    DATASET_JSON,
    DATASET_CSV,
    PROCESSED_VIDEOS_FILE
)

def setup_logging() -> logging.Logger:
    '''
    Настраивает логирование в файл и консоль

    Returns
    -------
    logger: logging.Logger
        Логгер
    '''
    log_file = DATASET_DIR / 'avito_parser.log'
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger('avito_parser')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def ensure_dependencies() -> None:
    '''
    Проверяет и устанавливает необходимые зависимости
    '''
    logger.info('Проверка зависимостей...')
    required_packages = {
        'yt_dlp': 'yt-dlp',
        'youtube_transcript_api': 'youtube-transcript-api',
        'pydub': 'pydub',
        'tqdm': 'tqdm' 
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f'✓ {package_name}')
        except ImportError:
            missing_packages.append(package_name)
            print(f'✗ {package_name} не установлен')
    
    if missing_packages:
        print(f'Установка пакетов: {", ".join(missing_packages)}')
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install'] + missing_packages
            )
            print('Установка завершена')
        except subprocess.CalledProcessError as e:
            print(f'Ошибка при установке пакетов: {e}')
            sys.exit(1)
    
    # ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        print('✓ FFmpeg')
    except Exception:
        print('✗ FFmpeg не найден!')
        print('Установите FFmpeg: https://ffmpeg.org/download.html')
        sys.exit(1)

def is_valid_clip_text(text: str) -> bool:
    '''
    Проверяет, содержит ли текст корректное упоминание слова "авито"

    Parameters
    ----------
    text: str
        Текст для проверки

    Returns
    -------
    bool
        True если содержит корректное упоминание, иначе False
    '''
    text = text.lower()
    
    words = re.findall(r'\b\w+\b', text)
    
    has_good = any(word in ['авито', 'avito'] for word in words)
    
    bad_words = [
        word for word in words if (
            (('авит' in word or 'avi' in word) and word not in ['авито', 'avito']) or
            word in [
                'авита', 'авитка', 'авитошоп', 'авиту', 'авите',
                'овито', 'савито', 'автио', 'аввито'
            ]
        )
    ]
    
    has_bad = len(bad_words) > 0
    
    return has_good and not has_bad

def load_processed_videos() -> set:
    '''
    Загружает список уже обработанных видео

    Returns
    -------
    set
        Множество ID обработанных видео
    '''
    if PROCESSED_VIDEOS_FILE.exists():
        try:
            with open(PROCESSED_VIDEOS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f'Загружен список обработанных видео: {len(data)} шт.')
            return set(data)
        except Exception as e:
            logger.error(f'Ошибка при загрузке списка обработанных видео: {e}')
    return set()

def save_processed_videos(processed_videos: set) -> None:
    '''
    Сохраняет список обработанных видео
    
    Parameters
    ----------
    processed_videos: set
        Множество ID обработанных видео
    '''
    with open(PROCESSED_VIDEOS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_videos), f, ensure_ascii=False, indent=2)
    logger.info(f'Сохранен список обработанных видео: {len(processed_videos)} шт.')

def load_dataset() -> list:
    '''
    Загружает существующий датасет

    Returns
    -------
    list
        Список записей датасета. Каждая запись - словарь
    '''
    if DATASET_JSON.exists():
        try:
            with open(DATASET_JSON, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f'Загружен датасет: {len(dataset)} записей')
            return dataset
        except Exception as e:
            logger.error(f'Ошибка при загрузке датасета: {e}')
    logger.info('Датасет не найден, создаем новый')
    return []

def save_dataset(dataset: list) -> None:
    '''
    Сохраняет датасет в JSON и CSV форматах

    Parameters
    ----------
        dataset: list
            Список записей датасета
            {
                'audio_file': str,
                'text': str,
                'video_id': str,
                'segment_index': int,
                'start_time': float,
                'end_time': float,
                'duration': float
            }
    '''
    with open(DATASET_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    with open(DATASET_CSV, 'w', encoding='utf-8') as f:
        if dataset:
            headers = list(dataset[0].keys())
            f.write(','.join(headers) + '\n')
            
            for item in dataset:
                row = []
                for key in headers:
                    value = item.get(key, '')
                    if isinstance(value, str) and (',' in value or '"' in value):
                        value = '"' + value.replace('"', '""') + '"'
                    row.append(str(value))
                f.write(','.join(row) + '\n')
    
    logger.info(f'Датасет сохранен: {len(dataset)} записей')
    print(f'Датасет обновлен, количество записей: {len(dataset)}') 

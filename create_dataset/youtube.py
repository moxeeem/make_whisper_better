import re
from pathlib import Path
from yt_dlp import YoutubeDL
from utils import logger, is_valid_clip_text
from config import (
    MIN_CLIP_DURATION, MAX_CLIP_DURATION, MAX_VIDEO_DURATION, MIN_SEGMENT_GAP, TEMP_DIR
)
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


def search_youtube(keyword: str, max_results: int = 50) -> list[dict]:
    '''
    Выполняет поиск видео на YouTube по ключевому слову
    
    Parameters
    ----------
    keyword: str
        Ключевое слово для поиска
    max_results: int
        Максимальное количество результатов
        
    Returns
    -------
    list[dict]
        Список словарей с информацией о видео
    '''
    try:
        logger.info(f"Поиск видео по запросу '{keyword}', лимит: {max_results}")
        print(f"Поиск видео по запросу '{keyword}'...")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'skip_download': True,
            'format': 'best',
            'ignoreerrors': True,
            'no_warnings': True,
        }
        
        search_url = f'ytsearch{max_results}:{keyword}'
        
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_url, download=False)
            
            if not result or 'entries' not in result:
                logger.warning('Не удалось получить результаты поиска')
                return []
            
            videos = []
            entries = result['entries']
            
            for entry in entries:
                if not entry:
                    continue
                
                video_id = entry.get('id')
                title = entry.get('title', '')
                duration = entry.get('duration', 0)
                
                if video_id and duration and duration <= MAX_VIDEO_DURATION:
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'duration': duration
                    })
            
            logger.info(f'Найдено {len(videos)} подходящих видео')
            return videos
    except Exception as e:
        logger.error(f'Ошибка при поиске видео: {e}')
        return []

def download_video_audio(video_id: str) -> Path | None:
    '''
    Скачивает аудио из видео
    
    Parameters
    ----------
    video_id: str
        Идентификатор видео на YouTube
        
    Returns
    -------
    Path | None
        Путь к скачанному аудиофайлу или None в случае ошибки
    '''
    try:
        output_path = TEMP_DIR / f'{video_id}.mp3'
        logger.info(f'Скачивание аудио для видео {video_id}')
        print(f'Скачивание аудио...')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{str(TEMP_DIR / video_id)}.%(ext)s',
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        
        if output_path.exists():
            logger.info(f'Аудио успешно скачано: {output_path}')
            return output_path
        
        for file in TEMP_DIR.glob(f'{video_id}.*'):
            if file.suffix.lower() in ['.mp3', '.m4a', '.wav']:
                logger.info(f'Найден аудиофайл с другим расширением: {file}')
                return file
        
        logger.error(f'Аудиофайл не найден после скачивания')
        return None
    except Exception as e:
        logger.error(f'Ошибка при скачивании аудио: {e}')
        return None

def get_transcript(video_id: str) -> list[dict] | None:
    '''
    Получает транскрипт видео
    
    Parameters
    ----------
    video_id: str
        Идентификатор видео на YouTube
        
    Returns
    -------
    list[dict] | None
        Список словарей с транскриптом или None в случае ошибки
    '''
    try:
        logger.info(f'Получение субтитров для видео {video_id}')
        print(f'Получение субтитров...')
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru'])
            logger.info(f'Получены субтитры: {len(transcript)} сегментов')
            return transcript
        except TranscriptsDisabled:
            logger.warning(f'Для видео {video_id} субтитры отключены')
        except Exception as e:
            logger.warning(f'Не удалось получить субтитры для видео {video_id}: {e}')
        return None

    except ImportError:
        logger.error('Библиотека youtube_transcript_api не установлена')
        return None

def find_all_keyword_segments(transcript: list[dict], min_gap: int | None = None) -> list[dict]:
    '''
    Находит все сегменты с ключевым словом в транскрипте одного видео
    
    Parameters
    ----------
    transcript: list[dict]
        Транскрипт видео
    min_gap: int | None
        Минимальное расстояние между сегментами в секундах

    Returns
    -------
    list[dict]
        Список найденных сегментов с ключевым словом
    '''
    if min_gap is None:
        min_gap = MIN_SEGMENT_GAP
    
    segments = []
    used_ranges = []
    
    for i, entry in enumerate(transcript):
        text = entry['text']
        if is_valid_clip_text(text):
            start_time = entry['start']
            end_time = start_time + entry.get('duration', 0)
            segments.append({
                'entry': entry, 
                'start': start_time, 
                'end': end_time,
                'index': i,
                'text': text
            })
    
    if not segments:
        logger.warning('Не найдено сегментов с ключевым словом')
        return []
    
    segments.sort(key=lambda x: x['start'])
    
    def is_text_unique(candidate_text: str, selected_texts: list[str], similarity_threshold: float = 0.5) -> bool:
        '''
        Проверяет, является ли текст уникальным среди выбранных текстов
        
        Parameters
        ----------
        candidate_text: str
            Текст для проверки
        selected_texts: list[str]
            Список выбранных текстов
        similarity_threshold: float
            Пороговое значение для сходства текстов
            
        Returns
        -------
        bool
            True, если текст уникален, False в противном случае
        '''
        candidate_words = set(candidate_text.lower().split())
        
        for text in selected_texts:
            existing_words = set(text.lower().split())
            
            if len(candidate_words) < 5 or len(existing_words) < 5:
                continue
                
            common_words = candidate_words.intersection(existing_words)
            similarity = len(common_words) / max(len(candidate_words), len(existing_words))
            
            if similarity > similarity_threshold:
                return False
        
        return True
    
    # выбираем сегменты с учетом временных промежутков и текстовой уникальности
    selected = []
    selected_texts = []
    
    # проходим по всем сегментам и выбираем наиболее разнообразные
    for segment in segments:
        # проверка временных промежутков
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (segment['end'] + min_gap <= used_start or segment['start'] - min_gap >= used_end):
                overlaps = True
                break
                
        # если нет временного пересечения и текст уникален
        if not overlaps and is_text_unique(segment['text'], selected_texts):
            selected.append({
                'entry': segment['entry'],
                'start': segment['start'],
                'end': segment['end'],
                'index': segment['index']
            })
            selected_texts.append(segment['text'])
            used_ranges.append((segment['start'], segment['end']))
    
    # если выбрано мало сегментов, ослабляем критерии отбора
    if len(selected) < 2 and len(segments) > 2:
        logger.warning(f'Найдено мало непересекающихся сегментов ({len(selected)}), ослабляем критерии')
        selected = []
        selected_texts = []
        used_ranges = []
        
        # уменьшаем минимальный промежуток в два раза для поиска большего количества фрагментов
        for segment in segments:
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (segment['end'] + min_gap/2 <= used_start or segment['start'] - min_gap/2 >= used_end):
                    overlaps = True
                    break
                    
            # если нет временного пересечения, добавляем сегмент
            if not overlaps:
                selected.append({
                    'entry': segment['entry'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'index': segment['index']
                })
                selected_texts.append(segment['text'])
                used_ranges.append((segment['start'], segment['end']))
    
    logger.info(f'Найдено {len(selected)} непересекающихся сегментов с ключевым словом')
    return selected


def cut_transcript(transcript: list[dict], word_time: float, word_index: int | None = None) -> tuple[str, float, float]:
    '''
    Вырезает фрагмент транскрипта, содержащий слово на указанной временной метке
    
    Parameters
    ----------
    transcript: list[dict]
        Транскрипт видео
    word_time: float
        Временная метка слова в секундах
    word_index: int | None
        Индекс субтитра с ключевым словом
        
    Returns
    -------
    tuple[str, float, float]
        Кортеж с фрагментом текста, временем начала и окончания
    '''
    # находим субтитр, в котором находится слово
    word_entry = None
    if word_index is None:
        # если индекс не передан, ищем по времени
        for i, entry in enumerate(transcript):
            start_time = entry['start']
            end_time = start_time + entry.get('duration', 0)
            
            if start_time <= word_time <= end_time:
                word_entry = entry
                word_index = i
                break
    else:
        # если индекс передан, используем его
        if 0 <= word_index < len(transcript):
            word_entry = transcript[word_index]
    
    if word_entry is None:
        logger.warning(f'Не найден субтитр для временной метки {word_time}')
        return '', 0, 0
    
    word_text = word_entry['text'].lower()
    avito_pos = max(word_text.find('авито'), word_text.find('avito'))
    
    if avito_pos >= 0:
        text_pos_percent = avito_pos / len(word_text)
        duration = word_entry.get('duration', 0)
        word_time = word_entry['start'] + duration * text_pos_percent
    
    sentence_start_index = word_index
    sentence_end_index = word_index
    
    # ищем начало предложения (до ближайшего знака препинания или начала транскрипта)
    for i in range(word_index - 1, -1, -1):
        prev_text = transcript[i]['text']
        if re.search(r'[.!?]', prev_text):
            sentence_start_index = i + 1
            break
        sentence_start_index = i
    
    # ищем конец предложения (до ближайшего знака препинания или конца транскрипта)
    for i in range(word_index + 1, len(transcript)):
        curr_text = transcript[i]['text']
        if re.search(r'[.!?]', curr_text):
            sentence_end_index = i
            break
        sentence_end_index = i
    
    # расширяем для захвата контекста
    start_index = max(0, sentence_start_index - 1)
    end_index = min(len(transcript) - 1, sentence_end_index + 1)
    
    # проверяем длину и корректируем при необходимости
    start_time = transcript[start_index]['start']
    end_time = transcript[end_index]['start'] + transcript[end_index].get('duration', 0)
    duration = end_time - start_time
    
    # корректируем для соблюдения ограничений по длительности
    while duration > MAX_CLIP_DURATION and end_index > word_index:
        end_index -= 1
        end_time = transcript[end_index]['start'] + transcript[end_index].get('duration', 0)
        duration = end_time - start_time
    
    while duration > MAX_CLIP_DURATION and start_index < word_index:
        start_index += 1
        start_time = transcript[start_index]['start']
        duration = end_time - start_time
    
    # если слишком короткий - расширяем
    while duration < MIN_CLIP_DURATION and (start_index > 0 or end_index < len(transcript) - 1):
        if end_index < len(transcript) - 1:
            end_index += 1
            end_time = transcript[end_index]['start'] + transcript[end_index].get('duration', 0)
        elif start_index > 0:
            start_index -= 1
            start_time = transcript[start_index]['start']
        
        duration = end_time - start_time
        
        if duration > MAX_CLIP_DURATION:
            if end_index > word_index + 1:
                end_index -= 1
                end_time = transcript[end_index]['start'] + transcript[end_index].get('duration', 0)
            break
    
    # собираем текст
    snippet = [entry['text'] for entry in transcript[start_index:end_index+1]]
    full_text = ' '.join(snippet).strip()
    
    # финальная проверка текста
    if not is_valid_clip_text(full_text):
        logger.warning('В вырезанном фрагменте нет корректного упоминания ключевого слова')
        return '', 0, 0
    
    logger.info(f'Вырезан фрагмент текста: {start_time:.1f}с - {end_time:.1f}с, длительность: {duration:.1f}с')
    return full_text, start_time, end_time

import time

from config import (
    NUM_VIDEOS, MAX_SEARCH_RESULTS, KEYWORD, 
    DATASET_DIR, TEMP_DIR, MIN_CLIP_DURATION, MAX_CLIP_DURATION
)
from utils import (
    logger, ensure_dependencies, load_processed_videos, 
    save_processed_videos, load_dataset, save_dataset
)
from youtube import (
    search_youtube, get_transcript, find_all_keyword_segments,
    cut_transcript, download_video_audio
)
from audio import get_audio_duration, has_audio_content, cut_audio

def clean_temp_dir():
    '''
    Очищает временную директорию
    '''
    try:
        for file in TEMP_DIR.glob('*'):
            try:
                file.unlink()
            except Exception as e:
                logger.error(f'Не удалось удалить временный файл {file}: {e}')
        logger.info(f'Временная директория очищена: {TEMP_DIR}')
    except Exception as e:
        logger.error(f'Ошибка при очистке временной директории: {e}')


def create_progress_bar(progress: float, width: int = 30) -> str:
    '''
    Создает графический прогресс-бар
    
    Parameters
    ----------
    progress : float
        Прогресс в виде дроби (0-1)
    width : int
        Ширина прогресс-бара
    
    Returns
    -------
    str
        Строка с прогресс-баром
    '''
    filled_width = int(width * progress)
    empty_width = width - filled_width
    percent = int(progress * 100)
    
    bar = '#' * filled_width + '-' * empty_width
    return f'[{bar}] {percent}%'


def process_video(video: dict, video_num: int, total_videos: int, dataset: list) -> tuple[bool, int]:
    '''
    Обрабатывает одно видео - последовательно находит и обрабатывает все фрагменты
    
    Parameters
    ----------
    video : dict
        Информация о видео
    video_num : int
        Номер текущего видео
    total_videos : int
        Общее количество видео
    dataset : list
        Текущий датасет
    
    Returns
    -------
    tuple[bool, int]
        (успех, количество добавленных сегментов)
    '''
    video_id = video['id']
    title = video.get('title', 'Без названия')
    title_short = title[:50] + '...' if len(title) > 50 else title
    
    video_progress = video_num / total_videos
    video_bar = create_progress_bar(video_progress)
    
    print('\n' + '=' * 70)
    print(f'Видео {video_num}/{total_videos}: {video_bar}')
    print(f"'{title_short}'")
    print('=' * 70)
    
    logger.info(f'Начало обработки видео {video_id}: {title}')
    
    transcript = get_transcript(video_id)
    if not transcript:
        print('Нет субтитров')
        return False, 0
    
    segments = find_all_keyword_segments(transcript)
    if not segments:
        print('Нет упоминаний ключевого слова')
        return False, 0
    
    print(f'Найдено {len(segments)} фрагментов')
    
    audio_file = download_video_audio(video_id)
    if not audio_file:
        print('Ошибка скачивания')
        return False, 0
    
    segments_added = 0
    total_fragments = len(segments)
    
    for i, segment in enumerate(segments, 1):
        segment_entry = segment['entry']
        segment_start = segment['start']
        segment_index = segment.get('index')
        
        segment_progress = i / total_fragments
        segment_bar = create_progress_bar(segment_progress)
        
        print(f'\nФрагмент {i}/{total_fragments}: {segment_bar}')
        print(f'Успешно: {segments_added}')
        
        snippet_text, start_time, end_time = cut_transcript(transcript, segment_start, segment_index)
        if not snippet_text or start_time >= end_time:
            print('  × Не удалось вырезать текст')
            continue
        
        audio_output = DATASET_DIR / f'audio_{video_id}_{i}.mp3'
        
        if not cut_audio(audio_file, audio_output, start_time, end_time):
            print('  × Не удалось вырезать аудио')
            if audio_output.exists():
                audio_output.unlink()
            continue
        
        duration = get_audio_duration(audio_output)
        if not (MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION + 5):
            print(f'  × Некорректная длительность: {duration:.1f}с')
            if audio_output.exists():
                audio_output.unlink()
            continue
        
        if not has_audio_content(audio_output):
            print('  × Аудио не содержит звука')
            if audio_output.exists():
                audio_output.unlink()
            continue
        
        dataset_item = {
            'audio_file': str(audio_output),
            'text': snippet_text,
            'video_id': video_id,
            'segment_index': i,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }
        
        dataset.append(dataset_item)
        segments_added += 1
        
        save_dataset(dataset)
        
        print(f'  ✓ Успешно: [{duration:.1f}с] {snippet_text[:50]}...')
    
    print(f'\nИтог видео: добавлено {segments_added} фрагментов из {total_fragments}')
    
    if audio_file and audio_file.exists():
        try:
            audio_file.unlink()
        except Exception as e:
            logger.error(f'Не удалось удалить временный файл {audio_file}: {e}')
    
    return segments_added > 0, segments_added

def main():
    '''
    Основная функция скрипта
    '''
    
    print('\n' + '=' * 70)
    print("Сбор аудио-фрагментов с ключевым словом с YouTube")
    print('=' * 70)
    
    ensure_dependencies()
    
    processed_videos = load_processed_videos()
    dataset = load_dataset()
    
    print('\nНастройки:')
    print(f"- Ключевое слово: '{KEYWORD}'")
    print(f"- Режим: {'все видео' if NUM_VIDEOS <= 0 else str(NUM_VIDEOS) + ' видео'}")
    print(f"- Обработано видео: {len(processed_videos)}")
    print(f"- Размер датасета: {len(dataset)} фрагментов")
    
    try:
        input('\nНажмите Enter для начала или Ctrl+C для отмены...')
    except KeyboardInterrupt:
        print('\nОтмена операции')
        return
    
    clean_temp_dir()
    
    print('\nПоиск видео...')
    videos = search_youtube(KEYWORD, MAX_SEARCH_RESULTS)
    if not videos:
        print('Не найдено подходящих видео')
        return
    
    unprocessed_videos = [v for v in videos if v['id'] not in processed_videos]
    
    if NUM_VIDEOS > 0:
        unprocessed_videos = unprocessed_videos[:NUM_VIDEOS]
    
    if not unprocessed_videos:
        print('Все доступные видео уже обработаны')
        return
    
    print(f'\nНайдено {len(unprocessed_videos)} новых видео')
    print(f'Текущий размер датасета: {len(dataset)} фрагментов')
    
    videos_processed = 0
    segments_added = 0
    start_time = time.time()
    
    try:
        for i, video in enumerate(unprocessed_videos, 1):
            video_id = video['id']
            
            success, video_segments = process_video(video, i, len(unprocessed_videos), dataset)
            
            processed_videos.add(video_id)
            save_processed_videos(processed_videos)
            
            if success:
                videos_processed += 1
                segments_added += video_segments
            
            elapsed = time.time() - start_time
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            
            print(f'Время работы: {elapsed_str}')
    
    except KeyboardInterrupt:
        print('\nПрерывание пользователем')
        logger.warning('Прерывание пользователем')
    
    print('\n' + '=' * 70)
    print('ИТОГИ СБОРА ДАТАСЕТА:')
    print(f'- Обработано видео: {videos_processed}')
    print(f'- Добавлено фрагментов: {segments_added}')
    print(f'- Всего в датасете: {len(dataset)} фрагментов')
    print(f'- Сохранено в: {DATASET_DIR}')
    print('=' * 70)
    
    clean_temp_dir()

if __name__ == '__main__':
    main() 

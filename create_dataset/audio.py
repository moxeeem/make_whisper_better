from tqdm import tqdm
from utils import logger
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import mediainfo
from config import MIN_CLIP_DURATION, MAX_CLIP_DURATION


def get_audio_duration(filepath: Path) -> float:
    '''
    Возвращает длительность аудиофайла в секундах
    
    Parameters
    ----------
    filepath: Path
        Путь к аудиофайлу
        
    Returns
    -------
    float
        Длительность в секундах
    '''
    try:
        info = mediainfo(str(filepath))
        duration = float(info.get('duration', 0))
        logger.debug(f'Длительность аудио {filepath}: {duration:.2f}с')
        return duration

    except Exception as e:
        logger.error(f'Ошибка при получении длительности аудио {filepath}: {e}')
        return 0

def has_audio_content(audio_path: Path, silence_thresh: float = -50.0) -> bool:
    '''
    Проверяет, есть ли в аудиофайле содержимое громче порога
    
    Parameters
    ----------
    audio_path: Path
        Путь к аудиофайлу
    silence_thresh: float
        Порог тишины в dB
        
    Returns
    -------
    bool
        True если аудио содержит звук громче порога, иначе False
    '''
    try:
        sound = AudioSegment.from_file(str(audio_path))
        if sound.dBFS > silence_thresh:
            logger.debug(f'Аудио {audio_path} содержит звук (dBFS={sound.dBFS:.2f})')
            return True
        else:
            logger.warning(f'Аудио {audio_path} не содержит звука (dBFS={sound.dBFS:.2f})')
            return False
    except Exception as e:
        logger.error(f'Ошибка при проверке содержимого аудио {audio_path}: {e}')
        return False

def cut_audio(input_file: Path, output_file: Path, start_time: float, end_time: float) -> bool:
    '''
    Вырезает фрагмент аудио из файла
    
    Parameters
    ----------
    input_file: Path
        Путь к исходному аудиофайлу
    output_file: Path
        Путь для сохранения вырезанного фрагмента
    start_time: float
        Время начала фрагмента в секундах
    end_time: float
        Время окончания фрагмента в секундах
        
    Returns
    -------
    bool
        True если операция выполнена успешно, иначе False
    '''
    try:        
        logger.info(f'Вырезаем аудио из {input_file} (c {start_time:.1f}с до {end_time:.1f}с)')
        
        try:
            pbar = tqdm(total=4, desc='    Обработка', leave=False)
        except ImportError:
            pbar = None
            
        sound = AudioSegment.from_file(str(input_file))
        if pbar: pbar.update(1)
        
        start_ms = max(int(start_time * 1000), 0)
        end_ms = min(int(end_time * 1000), len(sound))
        clip_duration_ms = end_ms - start_ms
        
        if clip_duration_ms < MIN_CLIP_DURATION * 1000:
            logger.warning(f'Отрывок слишком короткий ({clip_duration_ms/1000:.1f}с), расширяем')
            end_ms = min(start_ms + (MIN_CLIP_DURATION + 2) * 1000, len(sound))
            clip_duration_ms = end_ms - start_ms
            
        if clip_duration_ms > MAX_CLIP_DURATION * 1000:
            logger.warning(f'Отрывок слишком длинный ({clip_duration_ms/1000:.1f}с), обрезаем')
            end_ms = start_ms + MAX_CLIP_DURATION * 1000
            end_ms = min(end_ms, len(sound))
            clip_duration_ms = end_ms - start_ms
        
        if pbar: pbar.update(1)
        
        snippet = sound[start_ms:end_ms]
        if pbar: pbar.update(1)
        
        if snippet.dBFS < -50:
            logger.warning(f'Вырезанный фрагмент не содержит звука (dBFS={snippet.dBFS:.2f})')
            if pbar: pbar.close()
            return False
        
        snippet.export(str(output_file), format='mp3')
        if pbar: pbar.update(1)
        
        real_duration = get_audio_duration(output_file)
        logger.info(f'Создан аудиофайл: {output_file}, длительность: {real_duration:.1f}с')
        
        if pbar: pbar.close()
        return True
    except Exception as e:
        logger.error(f'Ошибка при вырезании аудио: {e}')
        return False 

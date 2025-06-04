import os
import gc
import json
import torch
import torchaudio
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'openai/whisper-small'
CHECKPOINT_DIR = "./lora_checkpoint"
AUDIO_FILE = './hello_avito.wav' # замените на свой аудиофайл


def load_best_model(checkpoint_dir: str = CHECKPOINT_DIR) -> tuple[object, WhisperProcessor]:
    '''
    Загружает лучшую LoRA модель из чекпоинта
    
    Parameters
    ----------
    checkpoint_dir : str
        Путь к директории с лучшим чекпоинтом
        
    Returns
    -------
    tuple[object, WhisperProcessor]
        Загруженная модель и процессор
    '''
    print(f'Загрузка модели из: {checkpoint_dir}')
    
    try:
        processor = WhisperProcessor.from_pretrained(checkpoint_dir)
        print('Процессор загружен из чекпоинта')
    except Exception as e:
        print(f'Не удалось загрузить процессор из чекпоинта: {e}')
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        print('Использован базовый процессор')
    
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    print('Базовая модель загружена')
    
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(DEVICE)
    model.eval()
    print(f'LoRA модель загружена на устройство: {DEVICE}')
    
    return model, processor


def load_and_preprocess_audio(audio_path: str, 
                             target_sr: int = 16000) -> torch.Tensor:
    '''
    Загружает и предобрабатывает аудиофайл
    
    Parameters
    ----------
    audio_path : str
        Путь к аудиофайлу
    target_sr : int, default=16000
        Целевая частота дискретизации
        
    Returns
    -------
    torch.Tensor
        Предобработанный аудиосигнал
    '''
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Аудиофайл не найден: {audio_path}')
    
    print(f'Загрузка аудиофайла: {audio_path}')
    waveform, sr = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print('Конвертировано в моно')
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        print(f'Ресемплировано с {sr} Hz до {target_sr} Hz')
    
    print(f'Форма аудиосигнала: {waveform.shape}')
    print(f'Длительность: {waveform.shape[1] / target_sr:.2f} сек')
    
    return waveform


def transcribe_audio(model: object, 
                    processor: WhisperProcessor, 
                    waveform: torch.Tensor) -> str:
    '''
    Выполняет транскрипцию аудио
    
    Parameters
    ----------
    model : object
        LoRA модель для транскрипции
    processor : WhisperProcessor
        Процессор для обработки аудио
    waveform : torch.Tensor
        Аудиосигнал для транскрипции
        
    Returns
    -------
    str
        Транскрибированный текст
    '''
    print('Начинаем транскрипцию...')
    input_features = processor.feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(DEVICE)
    
    forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language='ru',
                                                                    task='transcribe')
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_features, 
            max_length=448,
            forced_decoder_ids=forced_decoder_ids
        )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('Транскрипция завершена')
    return transcription.strip()


if __name__ == '__main__':
    try:
        model, processor = load_best_model(CHECKPOINT_DIR)
        waveform = load_and_preprocess_audio(AUDIO_FILE)
        transcription = transcribe_audio(model, processor, waveform)
        
        print('\n')
        print(f'Файл: {AUDIO_FILE}')
        print(f'Модель: {CHECKPOINT_DIR}')
        print(f'Текст: "{transcription}"')
        
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f'Ошибка во время инференса: {e}')

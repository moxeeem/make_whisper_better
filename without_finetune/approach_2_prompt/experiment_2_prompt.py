import os
import sys
import torch
import warnings
import argparse
import torchaudio
import numpy as np
import pandas as pd
import jiwer
from pathlib import Path
from typing import List, Tuple, Optional
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm


AUDIO_FILE = Path('/workspace/AAA_project/make_whisper_better/without_finetune/samples/hello_avito.wav')

MODEL_REPO = 'openai/whisper-small'
SAFE_MODEL_DIR = Path(__file__).resolve().parent.parent / 'base_checkpoint'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16000

DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../clips/dataset_clean.csv')
AUDIO_DIR = os.path.join(os.path.dirname(__file__), '../../clips')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../approach_results/prompt')


def download_checkpoint() -> None:
    '''
    Загружает модель, если она не существует
    '''
    if (SAFE_MODEL_DIR / 'config.json').exists():
        print(f'[OK] Модель существует: {SAFE_MODEL_DIR}')
        return

    print(f'[DL] Скачивание модели {MODEL_REPO} в {SAFE_MODEL_DIR} ...')
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(SAFE_MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print('[DL] Модель успешно скачана!')


def sanitize_generation_config(model: WhisperForConditionalGeneration) -> None:
    '''
    Очищает конфигурацию генерации модели от лишних параметров,
    чтобы избежать конфликтов при использовании prompt_ids

    Parameters
    ----------
    model : WhisperForConditionalGeneration
        Модель Whisper, для которой нужно очистить конфигурацию генерации
    '''
    cfg = model.config
    gen_cfg = model.generation_config

    if cfg.decoder_start_token_id is None:
        cfg.decoder_start_token_id = 50257  # <|startoftranscript|>
        gen_cfg.decoder_start_token_id = 50257

    if gen_cfg.forced_decoder_ids:
        clean: List[Tuple[int, int]] = []
        for pos, tok in gen_cfg.forced_decoder_ids:
            clean.append((0 if pos is None else pos, tok))
        gen_cfg.forced_decoder_ids = clean


def load_model() -> (WhisperForConditionalGeneration, WhisperProcessor):
    '''
    Загружает модель Whisper и процессор из локального чекпоинта или скачивает его
    если он не существует

    Returns
    -------
    tuple[WhisperForConditionalGeneration, WhisperProcessor]
        Кортеж из модели и процессора Whisper
    '''
    download_checkpoint()

    print(f'[INFO] Загружаем модель из {SAFE_MODEL_DIR} ...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = WhisperForConditionalGeneration.from_pretrained(
            SAFE_MODEL_DIR, device_map="auto"
        ).to(DEVICE)
        processor = WhisperProcessor.from_pretrained(SAFE_MODEL_DIR)

    sanitize_generation_config(model)

    model.save_pretrained(SAFE_MODEL_DIR)
    return model.eval(), processor


def load_audio(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    '''
    Загружает аудиофайл, конвертирует в моно и ресэмплирует до target_sr

    Parameters
    ----------
    path : Path
        Путь к аудиофайлу (WAV/FLAC/MP3)
    target_sr : int
        Целевая частота дискретизации, по умолчанию 16000 Гц

    Returns
    -------
    torch.Tensor
        Аудио в формате Tensor, моно и с частотой дискретизации target_sr
    '''
    if not path.exists():
        sys.exit(f'[ERROR] Файл не найден: {path}')

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:  # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)
        print(f'[AUDIO] Аудио преобразовано из стерео в моно: {wav.shape}')

    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        print(f'[AUDIO] Аудио ресэмплировано с {sr} Гц до {target_sr} Гц')

    return wav


def transcribe(model: WhisperForConditionalGeneration, 
               processor: WhisperProcessor, 
               waveform: torch.Tensor, 
               prompt_ru: str) -> str:
    '''
    Транскрибирует аудиофайл с использованием начального промпта

    Parameters
    ----------
    model : WhisperForConditionalGeneration
        Модель Whisper для транскрипции
    processor : WhisperProcessor
        Процессор Whisper для обработки аудио и текста
    waveform : torch.Tensor
        Аудио в формате Tensor, моно и с частотой дискретизации 16 кГц
    prompt_ru : str
        Начальный промпт

    Returns
    -------
    str
        Транскрибированный текст
    '''
    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    inputs = processor(
        waveform.squeeze().numpy(), 
        sampling_rate=TARGET_SR, 
        return_tensors="pt",
        padding=True
    )
    input_features = inputs.input_features.to(DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)
    else:
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=DEVICE)

    prompt_ids = processor.get_prompt_ids(prompt_ru)
    if isinstance(prompt_ids, np.ndarray):
        prompt_ids = torch.from_numpy(prompt_ids).long()
    elif isinstance(prompt_ids, list):
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
    elif torch.is_tensor(prompt_ids): 
        prompt_ids = prompt_ids.long()

    prompt_ids = prompt_ids.to(DEVICE)   

    original_forced_decoder_ids = model.generation_config.forced_decoder_ids
    model.generation_config.forced_decoder_ids = None

    generate_kwargs = {
        "prompt_ids": prompt_ids,
        "max_new_tokens": 256,
        "attention_mask": attention_mask,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "language": "ru",
        "task": "transcribe"
    }
    
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    with torch.no_grad(), torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generated = model.generate(input_features, **generate_kwargs)
    
    model.generation_config.forced_decoder_ids = original_forced_decoder_ids

    text: str = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return text


def compute_ker(reference: str,
                hypothesis: str,
                keyword: str = 'авито') -> Optional[float]:
    '''
    Вычисляет Keyword Error Rate (KER) для заданного ключевого слова
    в распознанном тексте

    Parameters
    ----------
    reference : str
        Истинный текст
    hypothesis : str
        Распознанный текст
    keyword : str, optional
        Ключевое слово, по умолчанию 'авито'

    Returns
    -------
    float
        KER для заданного ключевого слова
    '''
    if not isinstance(reference, str):
        reference = str(reference)
    if not isinstance(hypothesis, str):
        hypothesis = str(hypothesis)
    
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    keyword = keyword.lower()

    ref_keywords = [word for word in reference.split() if word == keyword]
    hyp_keywords = [word for word in hypothesis.split() if word == keyword]

    ref_keywords = " ".join(ref_keywords)
    hyp_keywords = " ".join(hyp_keywords)

    wer_transforms = jiwer.transforms.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords()
    ])

    if len(ref_keywords) == 0:
        return None
    ker = jiwer.wer(
        ref_keywords, 
        hyp_keywords, 
        reference_transform=wer_transforms, 
        hypothesis_transform=wer_transforms
    )
    return ker


def evaluate_dataset(model: WhisperForConditionalGeneration, 
                    processor: WhisperProcessor) -> None:
    '''
    Применяет модель к датасету и сохраняет результаты

    Parameters
    ----------
    model : WhisperForConditionalGeneration
        Модель Whisper для транскрипции
    processor : WhisperProcessor
        Процессор Whisper для обработки аудио и текста
    '''
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Загружен датасет с {len(df)} записями")
    
    predictions = []
    references = []
    
    initial_prompt = (
        'компания авито, авито это сайт для объявлений, звонил продавцу с авито, '
        'купили на авито, продают на авито, сайт авито, товар на авито, '
        'купил с авито, работа на авито, заказал с авито, одежда с авито'
    )
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Транскрибирование"):
        audio_file = row['audio_file']
        reference_text = row['text']
        
        audio_path = Path(AUDIO_DIR) / audio_file
        
        try:
            wav = load_audio(audio_path)
            predicted_text = transcribe(model, processor, wav, initial_prompt)
            
            predictions.append(predicted_text)
            references.append(reference_text)
            
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке {audio_file}: {e}")
            predictions.append("")
            references.append(reference_text)
    
    df['predicted_text'] = predictions
    
    predictions_path = os.path.join(RESULTS_DIR, 'all_predictions.csv')
    df.to_csv(predictions_path, index=False)
    print(f"[INFO] Предсказания сохранены в {predictions_path}")
    
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) if ref and pred]
    valid_references = [pair[0] for pair in valid_pairs]
    valid_predictions = [pair[1] for pair in valid_pairs]
    
    if valid_pairs:
        wer = jiwer.wer(valid_references, valid_predictions)
        cer = jiwer.cer(valid_references, valid_predictions)
        
        ker_values = []
        for ref, pred in valid_pairs:
            ker = compute_ker(ref, pred)
            if ker is not None:
                ker_values.append(ker)
        
        avg_ker = np.mean(ker_values) if ker_values else 0.0
        
        metrics_path = os.path.join(RESULTS_DIR, 'all_metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"WER: {wer}\n")
            f.write(f"CER: {cer}\n")
            f.write(f"KER: {avg_ker}\n")
        
        print(f"[INFO] Метрики сохранены в {metrics_path}")
        print(f"WER: {wer}")
        print(f"CER: {cer}")
        print(f"KER: {avg_ker}")
    else:
        print("[ERROR] Нет валидных пар для вычисления метрик")


if __name__ == "__main__":
    model, processor = load_model()
    evaluate_dataset(model, processor)

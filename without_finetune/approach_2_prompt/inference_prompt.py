import os
import sys
import torch
import warnings
import argparse
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor


AUDIO_FILE = Path('/workspace/AAA_project/make_whisper_better/without_finetune/samples/hello_avito.wav')

MODEL_REPO = 'openai/whisper-small'
SAFE_MODEL_DIR = Path(__file__).resolve().parent.parent / 'base_checkpoint'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16000


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


if __name__ == "__main__":
    wav = load_audio(AUDIO_FILE)
    initial_prompt = (
        'компания авито, авито это сайт для объявлений, звонил продавцу с авито, '
        'купили на авито, продают на авито, сайт авито, товар на авито, '
        'купил с авито, работа на авито, заказал с авито, одежда с авито'
    )

    model, processor = load_model()
    text = transcribe(model, processor, wav, initial_prompt)
    print(text)

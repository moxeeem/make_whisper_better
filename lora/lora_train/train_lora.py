import os
import gc
import torch
import jiwer
import logging
import warnings
import torchaudio
import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
from dataclasses import dataclass
from datasets import Dataset, Audio
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig 
from sklearn.model_selection import train_test_split
from transformers import (WhisperProcessor, WhisperForConditionalGeneration, 
                         Seq2SeqTrainingArguments, Seq2SeqTrainer)
from transformers import (TrainerCallback, TrainingArguments, TrainerState, TrainerControl)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import random
import glob
from scipy import signal
import soundfile as sf
import librosa
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, 
    Shift, Normalize, AddBackgroundNoise, ApplyImpulseResponse
)

warnings.filterwarnings(
    'ignore', 
    message='Due to a bug fix in https://github.com/huggingface/transformers/pull/28687'
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RESULTS_DIR = '/workspace/AAA_project/make_whisper_better/lora/lora_metrics'
CHECKPOINT_DIR = '/workspace/AAA_project/make_whisper_better/lora/approach_5_lora/lora_checkpoint'
DATASET_PATH = '/workspace/AAA_project/make_whisper_better/clips/dataset_clean.csv'
AUDIO_CLIPS = '/workspace/AAA_project/make_whisper_better/clips'
RIRS_ROOT = '/workspace/AAA_project/make_whisper_better/lora/lora_train/RIRS_NOISES'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_NAME = 'openai/whisper-small'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 42


def find_noise_rir_dirs(root_dir: str) -> tuple[list[str], list[str]]:
    '''
    Автоматически ищет папки с шумами и RIR (импульсными характеристиками)
    
    Parameters
    ----------
    root_dir : str
        Корневая директория для поиска
        
    Returns
    -------
    tuple[list[str], list[str]]
        Два списка папок: шумовые и с RIR
    '''
    noise_dirs = set()
    rir_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.lower().endswith(('.wav', '.flac')) for f in filenames):
            dir_low = dirpath.lower()
            if any(x in dir_low for x in ['noise', 'ambient', 'background']):
                noise_dirs.add(dirpath)
            if any(x in dir_low for x in ['rir', 'room', 'reverb', 'impulse']):
                rir_dirs.add(dirpath)
    return sorted(list(noise_dirs)), sorted(list(rir_dirs))

NOISES_DIRS, RIRS_DIRS = find_noise_rir_dirs(RIRS_ROOT)
print('Найдены папки с шумами:', NOISES_DIRS)
print('Найдены папки с RIR:', RIRS_DIRS)


def apply_professional_augmentation(waveform: torch.Tensor, 
                                    sample_rate: int = 16000) -> torch.Tensor:
    '''
    Применяет аугментации к аудиосигналу
    
    Parameters
    ----------
    waveform : torch.Tensor
        Входной аудиосигнал
    sample_rate : int, default=16000
        Частота дискретизации
        
    Returns
    -------
    torch.Tensor
        Аугментированный аудиосигнал
    '''
    device = waveform.device
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    audio_np = waveform.cpu().numpy()
    if audio_np.ndim == 2 and audio_np.shape[0] > 1:
        audio_np = audio_np.mean(axis=0)
    elif audio_np.ndim == 2 and audio_np.shape[0] == 1:
        audio_np = audio_np[0]
    augmentations = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.25),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
        Shift(min_shift=-0.2, max_shift=0.2, shift_unit='fraction', 
              rollover=True, p=0.15),
        AddBackgroundNoise(
            sounds_path=NOISES_DIRS if NOISES_DIRS else None,
            min_snr_db=15, max_snr_db=30, p=0.4 if NOISES_DIRS else 0.0
        ),
        ApplyImpulseResponse(
            ir_path=RIRS_DIRS if RIRS_DIRS else None,
            p=0.35 if RIRS_DIRS else 0.0
        ),
        Normalize(p=1.0)
    ])
    try:
        augmented = augmentations(samples=audio_np, sample_rate=sample_rate)
        if isinstance(augmented, np.ndarray) and augmented.ndim == 1:
            augmented = np.expand_dims(augmented, axis=0)
        augmented = augmented.astype(np.float32)
        return torch.tensor(augmented, dtype=torch.float32)
    except Exception as e:
        print(f'[Augmentation Error]: {e}')
        return waveform


def create_augmented_dataset(df: pd.DataFrame,
                             multiplier: int = 2, 
                             augmented_dir: str = None) -> pd.DataFrame:
    '''
    Создаёт аугментированный датасет
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм
    multiplier : int, default=2
        Множитель увеличения размера датасета
    augmented_dir : str, optional
        Директория для сохранения аугментированных файлов
        
    Returns
    -------
    pd.DataFrame
        Расширенный датафрейм с аугментированными данными
    '''
    if augmented_dir is None:
        augmented_dir = os.path.join(AUDIO_CLIPS, 'augmented')
    os.makedirs(augmented_dir, exist_ok=True)

    augmented_rows = []
    print(f'Создание аугментированного датасета с множителем {multiplier}')

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Аугментация аудио'):
        augmented_rows.append(row)
        for aug_idx in range(multiplier - 1):
            try:
                waveform, sr = torchaudio.load(row['audio_path'])
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                augmented_waveform = apply_professional_augmentation(
                    waveform, 16000)
                original_name = os.path.splitext(
                    os.path.basename(row['audio_path']))[0]
                augmented_filename = f'{original_name}_aug_{aug_idx + 1}.wav'
                augmented_path = os.path.join(augmented_dir, augmented_filename)
                torchaudio.save(augmented_path, augmented_waveform, 16000)
                new_row = row.copy()
                new_row['audio_path'] = augmented_path
                augmented_rows.append(new_row)
            except Exception as e:
                print(f'[Augment Error] {row["audio_path"]}: {e}')
                continue

    result_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
    print(f'Создан расширенный датасет: {len(result_df)} записей (было {len(df)})')
    return result_df


def compute_ker(reference: str,
                hypothesis: str,
                keyword: str = 'авито') -> float | None:
    '''
    Вычисляет Keyword Error Rate (KER) для определенного ключевого слова
    
    Parameters
    ----------
    reference : str
        Эталонный текст
    hypothesis : str  
        Предсказанный текст
    keyword : str, default='авито'
        Ключевое слово для оценки KER
        
    Returns
    -------
    float or None
        KER для ключевого слова или None, если слово отсутствует в эталоне
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
    ref_keywords = ' '.join(ref_keywords)
    hyp_keywords = ' '.join(hyp_keywords)
    wer_transforms = jiwer.Compose([
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


def prepare_dataset_init(dataset_path: str) -> pd.DataFrame:
    '''
    Загружает и подготавливает исходный датасет из CSV файла
    
    Parameters
    ----------
    dataset_path : str
        Путь к CSV файлу с датасетом
        
    Returns
    -------
    pd.DataFrame
        Подготовленный датафрейм с аудиофайлами и текстами
    '''
    logging.info(f'Загрузка датасета из {dataset_path}')
    df = pd.read_csv(dataset_path)
    
    required_columns = ['audio_file', 'text']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'В датасете отсутствует столбец {col}')
            
    df = df.rename(columns={'audio_file': 'audio_path'})
    
    def fix_path(path):
        if isinstance(path, str):
            if '\\' in path:
                filename = path.split('\\')[-1]
            else:
                filename = os.path.basename(path)
            return os.path.join(AUDIO_CLIPS, filename)
        return path
    
    df['audio_path'] = df['audio_path'].apply(fix_path)
    logging.info(f'Исправлены пути к аудиофайлам: {df["audio_path"].iloc[0]}')

    df['text'] = df['text'].apply(lambda x: x.lower())
    
    return df[['audio_path', 'text']].reset_index(drop=True)


def check_and_resample(audio_path: str,
                       target_sr: int = 16000,
                       apply_augmentation: bool = False) -> torch.Tensor | None:
    '''
    Загружает и ресемплирует аудиофайл при необходимости
    
    Parameters
    ----------
    audio_path : str
        Путь к аудиофайлу
    target_sr : int, default=16000
        Целевая частота дискретизации для ресемплинга
    apply_augmentation : bool, default=False
        Применять ли аугментации к аудио
        
    Returns
    -------
    torch.Tensor or None
        Аудиосигнал в виде тензора или None, если файл не удалось загрузить
    '''
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr)(waveform)
        
        if apply_augmentation:
            waveform = apply_professional_augmentation(waveform, target_sr)
            
        return waveform

    except Exception as e:
        return None


def prepare_dataset(df: pd.DataFrame,
                    audio_col: str = 'audio_path',
                    text_col: str = 'text',
                    target_sr: int = 16000) -> pd.DataFrame:
    '''
    Валидирует и фильтрует датасет, проверяя аудиофайлы
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм с аудиофайлами и текстами
    audio_col : str, default='audio_path'
        Имя столбца с аудиофайлами
    text_col : str, default='text' 
        Имя столбца с текстами
    target_sr : int, default=16000
        Целевая частота дискретизации для ресемплинга
        
    Returns
    -------
    pd.DataFrame
        Отфильтрованный датафрейм с валидными аудиофайлами и текстами
    '''
    new_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), 
                        desc='Проверка и ресемплинг аудио'):
        waveform = check_and_resample(row[audio_col], target_sr)
        if waveform is not None:
            new_rows.append(row)

    return pd.DataFrame(new_rows)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    '''
    Коллатор данных для обучения speech-to-text seq2seq с padding
    
    Parameters
    ----------
    processor : WhisperProcessor
        Whisper processor для обработки аудио и текста
    '''
    processor: object
    
    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        '''
        Обрабатывает батч признаков для обучения
        
        Parameters
        ----------
        features : list[dict[str, object]]
            Список признаков с аудио и текстами
            
        Returns
        -------
        dict[str, object]
            Обработанный батч с аудио и текстами
        '''
        input_features = [{'input_features': feature['input_features']} 
                         for feature in features]
        batch = self.processor.feature_extractor.pad(input_features,
                                                     return_tensors='pt')
        label_features = [{'input_ids': feature['labels']} 
                         for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features,
                                                    return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch['labels'] = labels
        return batch


def preprocess_with_processor(processor: WhisperProcessor):
    '''
    Создает функцию предобработки с закрытым processor
    
    Parameters
    ----------
    processor : WhisperProcessor
        Whisper processor для обработки аудио и текста
        
    Returns
    -------
    function
        Функция предобработки для применения к датасету
    '''
    def preprocess_func(batch: dict) -> dict:
        audio = batch['audio_path']
        inputs = processor.feature_extractor(audio['array'],
                                             sampling_rate=audio['sampling_rate'])
        batch['input_features'] = inputs.input_features[0]
        batch['labels'] = processor.tokenizer(batch['text']).input_ids
        return batch
    return preprocess_func


def compute_metrics(pred: object, processor: WhisperProcessor) -> dict[str, float]:
    '''
    Вычисляет метрики оценки для предсказаний модели
    
    Parameters
    ----------
    pred : object
        Предсказания модели, содержащие идентификаторы предсказаний и меток
    processor : WhisperProcessor
        Whisper processor для обработки аудио и текста
        
    Returns
    -------
    dict[str, float]
        Вычисленные метрики WER, CER и KER
    '''
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.tokenizer.batch_decode(pred_ids,
                                                skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids,
                                                 skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    cer_score = cer(label_str, pred_str)
    ker_scores = []
    for ref, hyp in zip(label_str, pred_str):
        ker_val = compute_ker(ref, hyp)
        if ker_val is not None:
            ker_scores.append(ker_val)

    ker_score = sum(ker_scores) / len(ker_scores) if ker_scores else None

    return {'wer': wer_score, 'cer': cer_score, 'ker': ker_score}


def load_model_from_checkpoint(checkpoint_dir: str, 
                               model_name: str = MODEL_NAME) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    '''
    Загружает LoRA адаптированную модель из чекпоинта
    
    Parameters
    ----------
    checkpoint_dir : str
        Путь к директории с чекпоинтом
    model_name : str, default=MODEL_NAME
        Имя модели для загрузки - openai/whisper-small
        
    Returns
    -------
    tuple[WhisperForConditionalGeneration, WhisperProcessor]
        Загруженная модель и процессор
    '''
    try:
        processor = WhisperProcessor.from_pretrained(checkpoint_dir)
    except:
        processor = WhisperProcessor.from_pretrained(model_name)
    
    base_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(DEVICE)
    
    return model, processor


def apply_model_to_pandas_and_save(model: object,
                                   processor: object,
                                   df: pd.DataFrame,
                                   csv_path: str,
                                   batch_size: int = 16) -> pd.DataFrame:
    '''
    Применяет модель к датафрейму и сохраняет результаты
    
    Parameters
    ----------
    model : object
        Модель для предсказания
    processor : object
        Процессор для обработки аудио и текста
    df : pd.DataFrame
        Датафрейм с аудиофайлами и текстами
    csv_path : str
        Путь для сохранения результатов предсказания
    batch_size : int, default=16
        Размер батча для предсказания
        
    Returns
    -------
    pd.DataFrame
        Датафрейм с предсказанными текстами
    '''
    ds = Dataset.from_pandas(df).cast_column('audio_path', Audio(sampling_rate=16000))
    preprocess_func = preprocess_with_processor(processor)
    ds = ds.map(preprocess_func, remove_columns=ds.column_names)

    preds = []
    model.eval()
    for i in tqdm(range(0, len(ds), batch_size), desc='Batch Inference'):
            feats = [torch.tensor(ds[j]['input_features']) 
                    for j in range(i, min(i+batch_size, len(ds)))]
            feats = torch.stack(feats).to(DEVICE)

            attention_mask = torch.ones(feats.shape[:2], device=DEVICE)

            with torch.no_grad():
                pred_ids = model.generate(feats,
                                          attention_mask=attention_mask,
                                          max_length=225,
                                          do_sample=False
                                          )
            preds.extend(processor.tokenizer.batch_decode(pred_ids, 
                                                         skip_special_tokens=True))

    result_df = df.copy()
    result_df['predicted_text'] = preds
    result_df.to_csv(csv_path, index=False)
    return result_df


def compute_and_save_metrics(df: pd.DataFrame,
                           txt_path: str,
                           reference_col: str = 'text',
                           prediction_col: str = 'predicted_text') -> dict[str, float]:
    '''
    Вычисляет и сохраняет метрики оценки
    
    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с эталонными и предсказанными текстами
    txt_path : str
        Путь для сохранения метрик
    reference_col : str, default='text'
        Эталонный текстовый столбец
    prediction_col : str, default='predicted_text'
        Предсказанный текстовый столбец
        
    Returns
    -------
    dict[str, float]
        Вычисленные метрики WER, CER и KER
    '''
    refs = df[reference_col].tolist()
    preds = df[prediction_col].tolist()
    wer_score = wer(refs, preds)
    cer_score = cer(refs, preds)
    ker_vals = [compute_ker(r, p) for r, p in zip(refs, preds) 
               if compute_ker(r, p) is not None]
    ker_score = sum(ker_vals) / len(ker_vals) if ker_vals else None
    metrics = {
        'WER': wer_score,
        'CER': cer_score,
        'KER': ker_score
    }
    with open(txt_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
            
    return metrics


class KERCallback(TrainerCallback):
    '''
    Callback для отслеживания метрики KER во время обучения
    '''
    
    def __init__(self, eval_dataset: Dataset, processor: WhisperProcessor) -> None:
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.epoch = []
        self.ker_history = []
        self.eval_loss_history = []
        self.eval_wer_history = []
        self.eval_cer_history = []
        self.train_loss_history = []
        
        self.best_ker = float('inf')
        self.best_ker_epoch = 0
        self.best_wer = float('inf')
        self.best_wer_epoch = 0
        self.best_cer = float('inf')
        self.best_cer_epoch = 0
        self.best_loss = float('inf')
        self.best_loss_epoch = 0
        
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(15, 6))

    def on_epoch_end(self, 
                     args: TrainingArguments, 
                     state: TrainerState, 
                     control: TrainerControl, 
                     **kwargs) -> None:
        trainer = kwargs.get('model')
        model = trainer if hasattr(trainer, 'generate') else trainer.model
        model.eval()
        pred_ids = []
        label_ids = []

        for batch in self.eval_dataset:
            input_features = torch.tensor(batch['input_features']).unsqueeze(0).to(model.device)
            with torch.no_grad():
                attention_mask = torch.ones(input_features.shape[:-1]).to(model.device)
                pred = model.generate(input_features, attention_mask=attention_mask)
            pred_ids.append(pred[0])
            label_ids.append(torch.tensor(batch['labels']))

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        ker_scores = []
        for ref, hyp in zip(label_str, pred_str):
            ker_val = compute_ker(ref, hyp, keyword='авито')
            if ker_val is not None:
                ker_scores.append(ker_val)
    
        ker_score = sum(ker_scores) / len(ker_scores) if ker_scores else None
        ker_score_rounded = round(ker_score, 4) if ker_score is not None else None
        print(f'KER по слову \'авито\' на эпохе {state.epoch}: {ker_score_rounded}')

        self.epoch.append(state.epoch)
        self.ker_history.append(ker_score_rounded)

        if ker_score_rounded is not None and ker_score_rounded < self.best_ker:
            self.best_ker = ker_score_rounded
            self.best_ker_epoch = state.epoch

        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            last_eval = next((x for x in reversed(state.log_history) if 'eval_loss' in x), None)
            if last_eval:
                eval_loss = last_eval.get('eval_loss')
                eval_wer = last_eval.get('eval_wer')
                eval_cer = last_eval.get('eval_cer')
                
                self.eval_loss_history.append(eval_loss)
                self.eval_wer_history.append(eval_wer)
                self.eval_cer_history.append(eval_cer)
                
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.best_loss_epoch = state.epoch
                if eval_wer < self.best_wer:
                    self.best_wer = eval_wer
                    self.best_wer_epoch = state.epoch
                if eval_cer < self.best_cer:
                    self.best_cer = eval_cer
                    self.best_cer_epoch = state.epoch
                    
            last_train = next((x for x in reversed(state.log_history) if 'loss' in x and 'eval_loss' not in x), None)
            if last_train:
                self.train_loss_history.append(last_train.get('loss'))

        self.ax[0].cla()
        self.ax[0].set_title(f'KER по эпохам\nЛучший KER: {self.best_ker:.4f} (эпоха {self.best_ker_epoch})')
        self.ax[0].plot(self.epoch, self.ker_history, 'b-', 
                       label=f'KER (авито), лучший: {self.best_ker:.4f}')
        self.ax[0].axhline(y=self.best_ker, color='b', linestyle='--', alpha=0.7)
        self.ax[0].set_xlabel('Эпоха')
        self.ax[0].set_ylabel('KER')
        self.ax[0].legend()
        self.ax[0].grid(True, alpha=0.3)
        
        self.ax[1].cla()
        self.ax[1].set_title('Метрики по эпохам')
        
        if self.eval_loss_history:
            self.ax[1].plot(self.epoch[:len(self.eval_loss_history)], self.eval_loss_history, 
                           'r-', label=f'Eval Loss (лучший: {self.best_loss:.4f}, эп. {self.best_loss_epoch})')
        if self.train_loss_history:
            self.ax[1].plot(self.epoch[:len(self.train_loss_history)], self.train_loss_history, 
                           'g-', label='Train Loss')
        if self.eval_wer_history:
            self.ax[1].plot(self.epoch[:len(self.eval_wer_history)], self.eval_wer_history, 
                           'orange', label=f'Eval WER (лучший: {self.best_wer:.4f}, эп. {self.best_wer_epoch})')
        if self.eval_cer_history:
            self.ax[1].plot(self.epoch[:len(self.eval_cer_history)], self.eval_cer_history, 
                           'purple', label=f'Eval CER (лучший: {self.best_cer:.4f}, эп. {self.best_cer_epoch})')
        
        if self.eval_loss_history:
            self.ax[1].axhline(y=self.best_loss, color='r', linestyle='--', alpha=0.5)
        if self.eval_wer_history:
            self.ax[1].axhline(y=self.best_wer, color='orange', linestyle='--', alpha=0.5)
        if self.eval_cer_history:
            self.ax[1].axhline(y=self.best_cer, color='purple', linestyle='--', alpha=0.5)
            
        self.ax[1].set_xlabel('Эпоха')
        self.ax[1].legend(fontsize=8)
        self.ax[1].grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        self.fig.savefig(os.path.join(RESULTS_DIR, 'metrics_plot.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    original_df = prepare_dataset_init(DATASET_PATH)
    original_df = prepare_dataset(original_df, audio_col='audio_path', text_col='text')
    df = create_augmented_dataset(original_df, multiplier=2)
    
    print(f'Размер итогового датасета: {len(df)} записей')

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds = train_ds.cast_column('audio_path', Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column('audio_path', Audio(sampling_rate=16000))
    test_ds = test_ds.cast_column('audio_path', Audio(sampling_rate=16000))

    torch.cuda.empty_cache()
    gc.collect()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    config = LoraConfig(r=32, lora_alpha=64, 
                        target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj'],
                        lora_dropout=0.1, bias='none')
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    preprocess_func = preprocess_with_processor(processor)
    train_ds = train_ds.map(preprocess_func, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_func, remove_columns=val_ds.column_names)
    test_ds = test_ds.map(preprocess_func, remove_columns=test_ds.column_names)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    ker_callback = KERCallback(val_ds, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,  
        gradient_accumulation_steps=4, 
        
        learning_rate=5e-5,       
        lr_scheduler_type='cosine',  
        lr_scheduler_kwargs={'num_cycles': 0.5}, 
        
        weight_decay=0.01, 
        warmup_steps=500,    
        
        logging_steps=50,
        save_steps=50,               
        eval_steps=50,   
        save_total_limit=3,
        
        num_train_epochs=10, 
        fp16=True if DEVICE == 'cuda' else False,
        dataloader_drop_last=False,
        report_to=['tensorboard'],
        
        # Генерация
        predict_with_generate=True,
        generation_max_length=225,
        
        load_best_model_at_end=True,
        metric_for_best_model='eval_ker',
        greater_is_better=False,
        
        label_names=['labels'],
        eval_strategy='steps'
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        callbacks=[ker_callback],
    )

    print('*** Before training ***')
    metrics = trainer.evaluate()
    print(metrics)

    trainer.train()

    print('*** After training ***')
    metrics = trainer.evaluate()
    print(metrics)

    print('\n*** Примеры предсказаний после обучения ***')
    test_samples = test_ds.shuffle(seed=RANDOM_STATE).select(range(5))
    
    sample_inputs = []
    for i in range(5):
        sample_inputs.append(torch.tensor(test_samples[i]['input_features']).unsqueeze(0).to(DEVICE))
    
    model.eval()
    sample_predictions = []
    for input_features in sample_inputs:
        with torch.no_grad():
            pred_ids = model.generate(input_features)
        pred_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        sample_predictions.append(pred_text)
    
    for i in range(5):
        print(f'*** Пример {i+1} ***')
        print(f'Оригинальный текст: {processor.tokenizer.decode(test_samples[i]["labels"], skip_special_tokens=True)}')
        print(f'Предсказанный текст: {sample_predictions[i]}')

    final_model_dir = os.path.join(CHECKPOINT_DIR, 'final_best_model')
    os.makedirs(final_model_dir, exist_ok=True)

    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)

    print(f'Лучшая модель сохранена в: {final_model_dir}')

    del model, processor
    torch.cuda.empty_cache()
    model, processor = load_model_from_checkpoint(final_model_dir)

    result_df = apply_model_to_pandas_and_save(model, processor, test_df, 
                                              os.path.join(RESULTS_DIR, 'test_predictions.csv'))
    compute_and_save_metrics(result_df, os.path.join(RESULTS_DIR, 'test_metrics.txt'))

    result_df_all = apply_model_to_pandas_and_save(model, processor, original_df, 
                                                  os.path.join(RESULTS_DIR, 'all_predictions.csv'))
    compute_and_save_metrics(result_df_all, os.path.join(RESULTS_DIR, 'all_metrics.txt'))

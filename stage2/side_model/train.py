import sys
import copy
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from typing import Optional
import numpy as np
from utils.labels import ID_TO_CONDITION, ID_TO_LEVEL

import source.datasets as datasets
from source.datasets import build_transforms, DatasetPhase

from side_classifier_attention import RSNA2024AttentionNet

# from source.metrics import RSNA2024Metrics
# from source.submit import Submit
# from src.utils import load_settings


from omegaconf import OmegaConf



# ==================== FILE SETTING ====================
def load_settings(json_path: str = 'SETTINGS.json') -> dict:
    import json
    from pathlib import Path
    with open(json_path, 'r') as f:
        data = json.load(f)
        return {
            'raw_data_dir': Path(data['RAW_DATA_DIR']),
            'train_data_clean_dir': Path(data['TRAIN_DATA_CLEAN_DIR']),
            'model_checkpoint_dir': Path(data['MODEL_CHECKPOINT_DIR']),
            'pretrained_checkpoint_dir': Path(data['PRETRAINED_CHECKPOINT_DIR']),
            'submission_dir': Path(data['SUBMISSION_DIR'])
        }

# ==================== SET SEED ====================
def fix_seed(seed: int = 2025) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# ==================== OUTPUT DF FILE ====================
def submit(
    preds: np.ndarray,
    study_ids: np.ndarray,
    segment_ids: np.ndarray,
    condition_ids: np.ndarray,
    available_flags: Optional[np.ndarray] = None,
    labels: list[str] = ['row_id', 'normal_mild', 'moderate', 'severe'],
    base_submit_csv: Optional[str] = None,
    drop_row_ids: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Create submission DataFrame from predictions and metadata.
    
    Args:
        preds: Model predictions array
        study_ids: Study IDs array
        segment_ids: Segment IDs array
        condition_ids: Condition IDs array
        available_flags: Optional availability flags
        labels: Column labels for output DataFrame
        base_submit_csv: Optional path to base submission CSV
        drop_row_ids: Optional list of row IDs to drop
    
    Returns:
        Submission DataFrame
    """
    base_submit_df = None
    if base_submit_csv is not None:
        base_submit_df = pd.read_csv(base_submit_csv)
        base_submit_df['study_id'] = base_submit_df['row_id'].apply(lambda x: x.split('_')[0])
        base_submit_df['cond'] = base_submit_df['row_id'].apply(lambda x: x.split('_', 1)[1].rsplit('_', 2)[0])

    # Generate row names and available mask
    row_names = []
    if available_flags is None:
        available_flags = np.ones(len(study_ids), dtype=bool)
    
    available_mask = []
    for study_id, segment_id_list, condition_id_list, available_flag in zip(study_ids, segment_ids, condition_ids, available_flags):
        for condition_id in condition_id_list:
            for segment_id in segment_id_list:
                condition = ID_TO_CONDITION[condition_id]
                segment = ID_TO_LEVEL[segment_id]
                if '/' in segment:
                    segment = segment.replace('/', '_').lower()
                row_names.append(str(study_id) + '_' + condition + '_' + segment)
                available_mask.append(available_flag)

    row_names = np.asarray(row_names)
    preds = preds.reshape(-1, len(labels[1:]))
    available_mask = np.asarray(available_mask)
    row_names = row_names[available_mask]
    preds = preds[available_mask]

    # Create submission df
    assert len(preds) == len(row_names)
    submit_df = pd.DataFrame()
    submit_df[labels[0]] = row_names
    submit_df[labels[1:]] = preds.reshape(-1, len(labels[1:]))

    if drop_row_ids is not None:
        submit_df = submit_df[~submit_df['row_id'].str.contains('|'.join(drop_row_ids))]

    if base_submit_df is not None:
        submit_df_condition_set = set([row_id.split('_', 1)[1].rsplit('_', 2)[0] for row_id in submit_df['row_id']])
        add_conditions = list(set(list(ID_TO_CONDITION.values())) - submit_df_condition_set)
        if len(add_conditions) > 0:
            study_id_set = set(study_ids.astype(str).tolist())
            add_df = base_submit_df[base_submit_df['study_id'].isin(study_id_set)].copy()
            add_df = add_df[add_df['cond'].isin(add_conditions)][['row_id', 'normal_mild', 'moderate', 'severe']]
            submit_df = pd.concat([submit_df, add_df], axis=0).reset_index(drop=True)
    return submit_df



# ==================== CONFIG ====================
def get_config(config_path: str, dot_list: list) -> dict:
    config_omega_from_yaml = OmegaConf.load(config_path)
    config_omega_from_args = OmegaConf.from_dotlist(dot_list)
    config_omega = OmegaConf.merge(config_omega_from_yaml, config_omega_from_args)
    config = OmegaConf.to_container(config_omega, resolve=True)
    return config



# ==================== METRIC ====================
def get_condition(full_location: str) -> str:
    for injury_condition in ['spinal', 'foraminal', 'subarticular']:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f'condition not found in {full_location}')

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, any_severe_scalar: float) -> float:
    import pandas.api.types
    import sklearn.metrics
    
    target_levels = ['normal_mild', 'moderate', 'severe']
    
    if not pandas.api.types.is_numeric_dtype(submission[target_levels].values):
        raise Exception('All submission values must be numeric')
    if not np.isfinite(submission[target_levels].values).all():
        raise Exception('All submission values must be finite')
    if solution[target_levels].min().min() < 0:
        raise Exception('All labels must be at least zero')
    if submission[target_levels].min().min() < 0:
        raise Exception('All predictions must be at least zero')

    solution['study_id'] = solution['row_id'].apply(lambda x: x.split('_')[0])
    solution['location'] = solution['row_id'].apply(lambda x: '_'.join(x.split('_')[1:]))
    solution['condition'] = solution['row_id'].apply(get_condition)
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)
    submission['study_id'] = solution['study_id']
    submission['location'] = solution['location']
    submission['condition'] = solution['condition']

    target_conditions = submission['condition'].unique().tolist()
    condition_losses = []
    condition_weights = []
    for condition in target_conditions:
        condition_indices = solution.loc[solution['condition'] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, 'sample_weight'].values
        )
        condition_losses.append(condition_loss)
        condition_weights.append(1)

    if 'spinal' in target_conditions:
        any_severe_spinal_labels = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['severe'].max())
        any_severe_spinal_weights = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['sample_weight'].max())
        any_severe_spinal_predictions = pd.Series(submission.loc[submission['condition'] == 'spinal'].groupby('study_id')['severe'].max())
        any_severe_spinal_loss = sklearn.metrics.log_loss(
            y_true=any_severe_spinal_labels,
            y_pred=any_severe_spinal_predictions,
            sample_weight=any_severe_spinal_weights
        )
        condition_losses.append(any_severe_spinal_loss)
        condition_weights.append(any_severe_scalar)

    return np.average(condition_losses, weights=condition_weights)

def metric(
    submission: pd.DataFrame,
    train_df: pd.DataFrame,
    row_id_column_name: str = "row_id",
    any_severe_scalar: float = 1.0,
    sample_weights: dict[str, int] = {"normal_mild": 1, "moderate": 2, "severe": 4}
) -> float:
    target_cols = list(sample_weights.keys())
    pred = submission.copy()
    pred[target_cols] = pred[target_cols].div(pred[target_cols].sum(axis=1), axis=0)
    train_df = train_df.copy()
    indexed_train_df = train_df.set_index("study_id", verify_integrity=True)
    row_ids = pred[row_id_column_name]
    study_ids = row_ids.apply(lambda x: x.split('_')[0])
    locations = row_ids.apply(lambda x: '_'.join(x.split('_')[1:]))
    
    solution_data = np.zeros_like(pred[target_cols].values)
    sample_weight_list = []
    nan_row_ids = set()
    for idx, (row, study_id, location) in enumerate(zip(row_ids, study_ids, locations)):
        severity = str(indexed_train_df.at[int(study_id), location]).replace("/", "_").lower()
        if severity in sample_weights:
            solution_data[idx, target_cols.index(severity)] = 1.0
            sample_weight_list.append(sample_weights[severity])
        else:
            solution_data[idx] = np.nan
            nan_row_ids.add(row)
            sample_weight_list.append(np.nan)

    solution = pd.DataFrame({
        row_id_column_name: pred[row_id_column_name],
        "sample_weight": sample_weight_list
    })
    solution[target_cols] = solution_data
    pred.loc[pred[row_id_column_name].isin(nan_row_ids), target_cols] = np.nan
    
    return score(solution.dropna().copy(), pred.dropna().copy(), row_id_column_name, any_severe_scalar)




# ==================== TRAINING ====================
def train_one_epoch(model, train_dataloader, optimizer, scheduler, device, autocast, scaler, use_tqdm):
    model.train()
    total_loss = 0.0
    dataloader = tqdm(train_dataloader, leave=True) if use_tqdm else train_dataloader
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_ids, segment_ids, condition_ids, available_flags = batch
        if torch.all(~available_flags):
            if scheduler is not None:
                scheduler.step()
            continue

        sagittal_t1_image = sagittal_t1_image.to(device)
        sagittal_t2_image = sagittal_t2_image.to(device)
        axial_t2_image = axial_t2_image.to(device)
        label = label.to(device)

        with autocast:
            outputs = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image, label)
            loss = outputs['losses']['loss']
            total_loss += loss.item()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    train_loss = total_loss / len(train_dataloader)
    return dict(train_loss=train_loss)


def validation_one_epoch(model, valid_dataloader, device, autocast, metrics, submit, use_tqdm):
    model.eval()
    total_loss = 0
    logits, labels = [], []
    study_ids_list, segment_ids_list, condition_ids_list, available_flags_list = [], [], [], []

    dataloader = tqdm(valid_dataloader, leave=True) if use_tqdm else valid_dataloader
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_ids, segment_ids, condition_ids, available_flags = batch
            if torch.all(~available_flags):
                continue

            sagittal_t1_image = sagittal_t1_image.to(device)
            sagittal_t2_image = sagittal_t2_image.to(device)
            axial_t2_image = axial_t2_image.to(device)
            label = label.to(device)

            with autocast:
                outputs = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, force_loss_execute=True)
                total_loss += outputs['losses']['loss'].item()
                logits.append(outputs['logits'].detach().cpu())
                labels.append(label.detach().cpu())
                study_ids_list.append(study_ids)
                segment_ids_list.append(segment_ids)
                condition_ids_list.append(condition_ids)
                available_flags_list.append(available_flags)

    val_loss = total_loss / len(valid_dataloader)
    logits = torch.cat(logits, dim=0)
    preds = torch.softmax(logits.float(), dim=-1)

    preds = preds.numpy()
    targets = torch.cat(labels).numpy()
    available_flags = torch.cat(available_flags_list).numpy()
    study_ids = torch.cat(study_ids_list).numpy()
    segment_ids = torch.cat(segment_ids_list).numpy()
    condition_ids = torch.cat(condition_ids_list).numpy()

    submit_df = submit(preds, study_ids, segment_ids, condition_ids, available_flags)
    val_score = metrics(submit_df)

    return dict(val_loss=val_loss, val_score=val_score, submit_df=submit_df)


# ==================== TRAIN METHOD ====================
def run_training(config_path, experiment, dst_root, options=None):
    settings = load_settings()
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    config = get_config(config_path, options or [])
    config['dataset']['image_root'] = str(settings.train_data_clean_dir / config['dataset']['image_root'])
    config['dataset']['label_csv_path'] = str(settings.train_data_clean_dir / config['dataset']['label_csv_path'])
    config['dataset']['image_csv_path'] = str(settings.train_data_clean_dir / config['dataset']['image_csv_path'])

    fix_seed(config['seed'])
    device = torch.device(config['device'])

    autocast = torch.cuda.amp.autocast(enabled=config['use_amp'], dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'], init_scale=4096)

    label_csv_path = config['dataset'].pop('label_csv_path')
    label_ori_df = pd.read_csv(label_csv_path)
    label_df = label_ori_df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    label_df = label_df.replace(label2id)

    image_csv_path = config['dataset'].pop('image_csv_path')
    image_df = pd.read_csv(image_csv_path)

    # metrics = RSNA2024Metrics(label_ori_df)
    metrics = metric(label_ori_df)
    submit = submit(**config['submit'])

    base_config = copy.deepcopy(config)
    for fold in config['folds']:
        print(f'===== Fold {fold} =====')
        config = copy.deepcopy(base_config)

        train_df = label_df[label_df['fold'] != fold].reset_index(drop=True)
        valid_df = label_df[label_df['fold'] == fold].reset_index(drop=True)

        train_transforms, valid_transforms = {}, {}
        for plane in ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']:
            if plane in config['transform']:
                train_transforms[plane] = build_transforms(DatasetPhase.TRAIN, **config['transform'][plane])
                valid_transforms[plane] = build_transforms(DatasetPhase.VALIDATION, **config['transform'][plane])

        dataset_name = config['dataset'].pop('name')
        dataset_class = getattr(datasets, dataset_name)
        train_dataset = dataset_class(**config['dataset'], train_df=train_df, train_image_df=image_df, phase=DatasetPhase.TRAIN, transforms=train_transforms)
        valid_dataset = dataset_class(**config['dataset'], train_df=valid_df, train_image_df=image_df, phase=DatasetPhase.VALIDATION, transforms=valid_transforms)

        train_loader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], num_workers=config['dataloader']['num_workers'], shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['dataloader']['batch_size'], num_workers=config['dataloader']['num_workers'], shuffle=False, pin_memory=True, drop_last=False)

        model = RSNA2024AttentionNet(**config['model']).to(device)

        optimizer_class = getattr(torch.optim, config['optimizer'].pop('name'))
        optimizer = optimizer_class(model.parameters(), **config['optimizer'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **config['scheduler'], epochs=config['epochs'], steps_per_epoch=len(train_loader))

        best_val_score = float('inf')
        for epoch in range(1, config['epochs'] + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, autocast, scaler, config['use_tqdm'])
            valid_metrics = validation_one_epoch(model, valid_loader, device, autocast, metrics, submit, config['use_tqdm'])
            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, val_loss={valid_metrics['val_loss']:.4f}, val_score={valid_metrics['val_score']:.4f}")

            if valid_metrics['val_score'] < best_val_score:
                best_val_score = valid_metrics['val_score']
                torch.save(model.state_dict(), dst_root / f"exp{experiment}_fold{fold}_best.pth")

    print("Train complete")

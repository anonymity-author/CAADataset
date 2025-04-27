from itertools import count
import os
import sys
import random
import logging
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms

# local functions
from dataset.dataset import CAADataset, Padding, Rescale, RandomCrop, ToTensor
from models.convtcn import ConvTCN_Visual, ConvTCN_Audio, ConvTCN_Text
from models.evaluator import Evaluator
from models.fusion import Bottleneck
from models.sam import SAM
from models.tcn import TCN

def init_seed(manual_seed):
    """
    Set random seed for torch and numpy.
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filepath, log_title):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 54 + log_title + '-' * 54)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_sampler_atm_binary(atm_binary_gt):
    # sampler for atm_binary_gt
    class_sample_count = np.unique(atm_binary_gt, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[atm_binary_gt]
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_sampler_atm_score(atm_score_gt):
    class_sample_ID, class_sample_count = np.unique(atm_score_gt, return_counts=True)
    weight = 1. / class_sample_count
    samples_weight = np.zeros(atm_score_gt.shape)
    for i, sample_id in enumerate(class_sample_ID):
        indices = np.where(atm_score_gt == sample_id)[0]
        value = weight[i]
        samples_weight[indices] = value
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_dataloaders(data_config):

    dataloaders = {}
    for mode in ['train', 'test']:
        if mode == 'test':
            # for test dataset, we don't need shuffle, sampler and augmentation
            dataset = CAADataset(data_config[f'{mode}_ROOT_DIR'.upper()], mode,
                                        use_mel_spectrogram=data_config['USE_MEL_SPECTROGRAM'],
                                        visual_with_gaze=data_config['VISUAL_WITH_GAZE'],
                                        transform=transforms.Compose([ToTensor(mode)]))
            dataloaders[mode] = DataLoader(dataset,
                                           batch_size=data_config['BATCH_SIZE'],
                                           num_workers=data_config['NUM_WORKERS'])

        else:
            dataset = CAADataset(data_config[f'{mode}_ROOT_DIR'.upper()], mode,
                                        use_mel_spectrogram=data_config['USE_MEL_SPECTROGRAM'],
                                        visual_with_gaze=data_config['VISUAL_WITH_GAZE'],
                                        transform=transforms.Compose([ToTensor(mode)]))  # Rescale(data_config['RESCALE_SIZE']), Padding(data_config['PADDING']) + Augmentation TODO !!!
            sampler = get_sampler_atm_score(dataset.atm_score_gt)
            dataloaders[mode] = DataLoader(dataset,
                                           batch_size=data_config['BATCH_SIZE'],
                                           num_workers=data_config['NUM_WORKERS'], 
                                           sampler=sampler)

    return dataloaders


def find_last_ckpts(path, key, date=None):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Arguments:
        path: str, path to the checkpoint
        key: str, model type
        date: str, a specific date in string format 'YYYY-MM-DD'
    Returns:
        The path of the last checkpoint file
    """
    ckpts = list(sorted(os.listdir(path)))

    if date is not None:
        # match the date format
        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(date, date_format)
            # print("This is the correct date string format.")
            matched = True
        except ValueError:
            # print("This is the incorrect date string format. It should be YYYY-MM-DD")
            matched = False
        assert matched, "The given date is the incorrect date string format. It should be YYYY-MM-DD"

        key = '{}_{}'.format(key, date)
    else:
        key = str(key)

    # filter the files
    ckpts = list(filter(lambda f: f.startswith(key), ckpts))
    # get whole file path
    last_ckpt = os.path.join(path, ckpts[-1])

    return last_ckpt


def get_models(model_config, args, model_type=None, ckpt_path=None):
    """
    Get the different deep model net as encoder backbone and the evaluator with parameters moved to GPU.
    """
    visual_net = ConvTCN_Visual(input_dim=model_config['VISUAL_NET']['INPUT_DIM'],
                                 output_dim=model_config['VISUAL_NET']['OUTPUT_DIM'], 
                                 conv_hidden=model_config['VISUAL_NET']['CONV_HIDDEN'], 
                                 tcn_hidden=model_config['VISUAL_NET']['TCN_HIDDEN'],
                                 activation=model_config['VISUAL_NET']['ACTIVATION'],
                                 norm = model_config['VISUAL_NET']['NORM'], 
                                 dropout=model_config['VISUAL_NET']['DROPOUT'])

    audio_net = ConvTCN_Audio(input_dim=model_config['AUDIO_NET']['INPUT_DIM'],
                               output_dim=model_config['AUDIO_NET']['OUTPUT_DIM'], 
                               conv_hidden=model_config['AUDIO_NET']['CONV_HIDDEN'], 
                               tcn_hidden=model_config['AUDIO_NET']['TCN_HIDDEN'],
                               activation=model_config['AUDIO_NET']['ACTIVATION'],
                               norm = model_config['AUDIO_NET']['NORM'], 
                               dropout=model_config['AUDIO_NET']['DROPOUT'])

    text_net = ConvTCN_Text(input_dim=model_config['TEXT_NET']['INPUT_DIM'],
                             output_dim=model_config['TEXT_NET']['OUTPUT_DIM'], 
                             conv_hidden=model_config['TEXT_NET']['CONV_HIDDEN'], 
                             tcn_hidden=model_config['TEXT_NET']['TCN_HIDDEN'],
                             activation=model_config['TEXT_NET']['ACTIVATION'],
                             norm = model_config['TEXT_NET']['NORM'], 
                             dropout=model_config['TEXT_NET']['DROPOUT'])


    evaluator = Evaluator(feature_dim=model_config['EVALUATOR']['INPUT_FEATURE_DIM'],
                          output_dim=model_config['EVALUATOR']['CLASSES_RESOLUTION'],
                          dropout=model_config['EVALUATOR']['DROPOUT'],
                          attention_config=model_config['EVALUATOR']['ATTENTION'])

    if len(args.gpu.split(',')) > 1:
        visual_net = nn.DataParallel(visual_net)
        audio_net = nn.DataParallel(audio_net)
        text_net = nn.DataParallel(text_net)
        evaluator = nn.DataParallel(evaluator)

    # move to GPU
    visual_net = visual_net.to(args.device)
    audio_net = audio_net.to(args.device)
    text_net = text_net.to(args.device)
    evaluator = evaluator.to(args.device)

    # find the model weights
    if model_config['WEIGHTS']['TYPE'].lower() == 'last':
        assert ckpt_path is not None, \
            "'ckpt_path' must be given for the function 'get_models' "
        weights_path = find_last_ckpts(path=ckpt_path,
                                       key=model_type,
                                       date=model_config['WEIGHTS']['DATE'])

    elif model_config['WEIGHTS']['TYPE'].lower() == 'absolute_path':
        assert model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'] is not None, \
            "'CUSTOM_ABSOLUTE_PATH' (absolute path to wights file) in config file under 'WEIGHTS' must be given"
        assert os.path.isabs(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH']), \
            "The given 'CUSTOM_ABSOLUTE_PATH' is not an absolute path to wights file, please give an absolute"

        weights_path = str(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'])

    elif model_config['WEIGHTS']['TYPE'].lower() != 'new':
        assert model_config['WEIGHTS']['NAME'] is not None, \
            "'NAME' (name of the wights file) in config file under 'WEIGHTS' must be given"
        weights_path = os.path.join(model_config['WEIGHTS']['PATH'], model_config['WEIGHTS']['NAME'])
    
    else:
        weights_path = None


    # load model weights
    if weights_path is not None:
        model_config['WEIGHTS']['INCLUDED'] = [x.lower() for x in model_config['WEIGHTS']['INCLUDED']]

        checkpoint = torch.load(weights_path)

        if 'visual_net' in model_config['WEIGHTS']['INCLUDED']:
            print("Loading Deep Visual Net weights from {}".format(weights_path))
            visual_net.load_state_dict(checkpoint['visual_net'])

        if 'audio_net' in model_config['WEIGHTS']['INCLUDED']:
            print("Loading Deep Audio Net weights from {}".format(weights_path))
            audio_net.load_state_dict(checkpoint['audio_net'])
        
        if 'text_net' in model_config['WEIGHTS']['INCLUDED']:
            print("Loading Deep Text Net weights from {}".format(weights_path))
            text_net.load_state_dict(checkpoint['text_net'])


        if 'evaluator' in model_config['WEIGHTS']['INCLUDED']:
            print("Loading MUSDL weights from {}".format(weights_path))
            evaluator.load_state_dict(checkpoint['evaluator'])

    return visual_net, audio_net, text_net, evaluator




def get_crossentropy_weights(gt, evaluator_config):
    weights = np.zeros(evaluator_config['N_CLASSES'])
    labels, counts = np.unique(gt, return_counts=True)
    for i in range(len(labels)):
        weights[int(labels[i])] = 1. / counts[i]
    return weights


def get_criterion(criterion_config, args):

    if criterion_config['USE_SOFT_LABEL']:
        criterion = nn.KLDivLoss()
    else:
        if criterion_config['USE_WEIGHTS']:

            weights = torch.tensor(criterion_config['WEIGHTS']).type(torch.FloatTensor).to(args.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        
        else:
            
            criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer_scheduler(model_parameters, optimizer_config, scheduler_config):

    if optimizer_config['USE_SAM']:
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model_parameters, base_optimizer, rho=2, adaptive=True, betas=(0.9, 0.999),
                        lr=float(optimizer_config['LR']), weight_decay=float(optimizer_config['WEIGHT_DECAY']))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.base_optimizer,
                                                    step_size=scheduler_config['STEP_SIZE'],
                                                    gamma=scheduler_config['GAMMA'])
    else:
        optimizer = torch.optim.Adam(model_parameters, betas=(0.9, 0.999), 
                                     lr=optimizer_config['LR'],
                                     weight_decay=optimizer_config['WEIGHT_DECAY'])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=scheduler_config['STEP_SIZE'],
                                                    gamma=scheduler_config['GAMMA'])
    
    return optimizer, scheduler


def get_gt(data):
    gt = data['atm_score_gt']

    return gt


def compute_score(probs, evaluator_config, args):

    factor = evaluator_config['N_CLASSES'] / evaluator_config['CLASSES_RESOLUTION']
    score_pred = (probs.argmax(dim=-1) * factor).to(int).to(float)
    return score_pred.to(args.device)
    


def convert_soft_gt(gt, evaluator_config):
    factor = (evaluator_config['N_CLASSES'] - 1) / (evaluator_config['CLASSES_RESOLUTION'] - 1)
    tmp = stats.norm.pdf(np.arange(evaluator_config['CLASSES_RESOLUTION']), loc=gt / factor,
                             scale=evaluator_config['STD']).astype(np.float32)

    return torch.from_numpy(tmp / tmp.sum(axis=-1, keepdims=True))


def get_soft_gt(gt, evaluator_config):

    soft_gt = torch.tensor([[]])

    # iterate through each batch 
    for i in range(len(gt)):

        current_gt = gt[i]
        converted_current_gt = convert_soft_gt(current_gt, evaluator_config)
        if i == 0:
            soft_gt = converted_current_gt.unsqueeze(dim=0)
        else:
            soft_gt = torch.cat([soft_gt, converted_current_gt.unsqueeze(dim=0)], dim=0)

    return soft_gt


def compute_loss(criterion, probs, gt, evaluator_config, args, use_soft_label=False):
    if use_soft_label:
        soft_gt = get_soft_gt(gt, evaluator_config)
        loss = criterion(torch.log(probs), soft_gt.to(args.device))
            
    return loss




def get_regression_scores(gt, pred):
    gt = np.array(gt).astype(float)
    pred = np.array(pred).astype(float)
    mae = metrics.mean_absolute_error(gt, pred)
    mse = metrics.mean_squared_error(gt, pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(gt, pred)
    return mae, mse, rmse, r2



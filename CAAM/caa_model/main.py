import os
import sys
import math
import time
import shutil
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime
from scipy import stats
from autolab_core import YamlConfig

# local functions
from utils import *
from models.bypass_bn import enable_running_stats, disable_running_stats


def main(dataloaders, visual_net, audio_net, text_net, evaluator, base_logger, writer, config, args, model_type, ckpt_path):

    if not config['CRITERION']['USE_SOFT_LABEL']:
        assert config['EVALUATOR']['CLASSES_RESOLUTION'] == config['EVALUATOR']['N_CLASSES'], \
            "Argument --config['EVALUATOR']['CLASSES_RESOLUTION'] should be the same as --config['EVALUATOR']['N_CLASSES'] when soft label is not used!"

    model_parameters = [*visual_net.parameters()] + [*audio_net.parameters()] + [*text_net.parameters()]  + [*evaluator.parameters()]
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    for epoch in range(config['EPOCHS']):

        for mode in ['train', 'test']:
            mode_start_time = time.time()

            atm_score_gt = []
            atm_score_pred = []


            if mode == 'train':
                visual_net.train()
                audio_net.train()
                text_net.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                visual_net.eval()
                audio_net.eval()
                text_net.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            total_loss = 0
            log_interval_loss = 0
            log_interval = 10
            batch_number = 0
            n_batches = len(dataloaders[mode])
            batches_start_time = time.time()
            
            for data in tqdm(dataloaders[mode]):
                batch_size = data['ID'].size(0)

                # store ground truth
                atm_score_gt.extend(data['atm_score_gt'].numpy().astype(float))  # 1D list

                # TODO: extract features with multi-model ...
                # combine all models into a function
                def model_processing(input):
                    # get facial visual feature with Deep Visual Net'
                    # input shape for visual_net must be (B, C, F, T) = (batch_size, channels, features, time series)
                    B, T, F, C = input['visual'].shape
                    visual_input = input['visual'].permute(0, 3, 2, 1).contiguous()
                    visual_features = visual_net(visual_input.to(args.device))  # output dim: [B, visual net output dim]

                    # get audio feature with Deep Audio Net'
                    # input shape for audio_net must be (B, F, T) = (batch_size, features, time series)
                    B, F, T = input['audio'].shape
                    audio_input = input['audio'].view(B, F, T)
                    audio_features = audio_net(audio_input.to(args.device))  # output dim: [B, audio net output dim]

                    # get Text features with Deep Text Net'
                    # input shape for text_net must be (B, F, T) = (batch_size, features, time series))
                    B, T, F = input['text'].shape
                    text_input = input['text'].permute(0, 2, 1).contiguous()
                    text_features = text_net(text_input.to(args.device))  # output dim: [B, text net output dim]

                    # ---------------------- Start evaluating with sub-attentional feature fusion ----------------------
                    # combine all features into shape: B, C=1, num_modal, audio net output dim
                    all_features = torch.stack([visual_features,audio_features,text_features], dim=1).unsqueeze(dim=1)
                    probs = evaluator(all_features)
                    """ 
                    Arguments:
                        'features' should have size (batch_size, channels(=1), num_modal, feature_dim of each branch)
                    """
                    return probs
                
                if mode == 'train':
                    
                    # choose the right GT for criterion based on prediciton type
                    gt = get_gt(data)

                    # get dynamic weights for cross entropy loss if needed
                    if config['CRITERION']['USE_WEIGHTS']:
                        config['CRITERION']['WEIGHTS'] = get_crossentropy_weights(gt, config['EVALUATOR'])
                    criterion = get_criterion(config['CRITERION'], args)

                    if config['OPTIMIZER']['USE_SAM']:
                        models = [visual_net, audio_net, evaluator]
                        # first forward-backward pass
                        for model in models:
                            enable_running_stats(model)
                        probs = model_processing(input=data)
                        loss = compute_loss(criterion, probs, gt, config['EVALUATOR'], args, 
                                            use_soft_label=config['CRITERION']['USE_SOFT_LABEL'])
                        loss.backward()                      # backpropagation - use this loss for any training statistics
                        optimizer.first_step(zero_grad=True)

                        # second forward-backward pass
                        for model in models:
                            disable_running_stats(model)
                        compute_loss(criterion, model_processing(input=data), gt, config['EVALUATOR'], 
                                    args, use_soft_label=config['CRITERION']['USE_SOFT_LABEL']).backward()
                        optimizer.second_step(zero_grad=True)
                    
                    else:
                        # only one time forward-backward pass 
                        probs = model_processing(input=data)
                        loss = compute_loss(criterion, probs, gt, config['EVALUATOR'], args, 
                                            use_soft_label=config['CRITERION']['USE_SOFT_LABEL'])
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(models_parameters, max_norm=2.0, norm_type=2)
                        optimizer.step()

                else:
                    # for test set, only do prediction
                    probs = model_processing(input=data)

                # predict the final score
                pred_score = compute_score(probs, config['EVALUATOR'], args)
                atm_score_pred.extend([pred_score[i].item() for i in range(batch_size)])  # 1D list

                if mode == 'train':
                    # information per batch
                    total_loss += loss.item()
                    log_interval_loss += loss.item()
                    if batch_number % log_interval == 0 and batch_number > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - batches_start_time) * 1000 / log_interval
                        current_loss = log_interval_loss / log_interval
                        print(f'| epoch {epoch:3d} | {mode} | {batch_number:3d}/{n_batches:3d} batches | '
                            f'LR {lr:7.6f} | ms/batch {ms_per_batch:5.2f} | loss {current_loss:8.5f} |')

                        # tensorboard
                        writer.add_scalar('Loss_per_{}_batches/{}'.format(log_interval, mode),
                                        current_loss, epoch*n_batches+batch_number)

                        log_interval_loss = 0
                        batches_start_time = time.time()
                else:
                    # for test set we don't need to calculate the loss so just leave it 'nan'
                    total_loss = np.nan

                batch_number += 1

            # information per mode
            print('ATM Score prediction: {}'.format(atm_score_pred[:20]))
            print('ATM Score ground truth: {}'.format(atm_score_gt[:20]))

            print('-' * 110)

            # regression related
            mae, mse, rmse, r2 = get_regression_scores(atm_score_gt, atm_score_pred)
            msg = ('  - Regression:\n'
                   '      MAE: {0:7.4f}\n'
                   '      MSE: {1:7.4f}\n'
                   '      RMSE: {2:7.4f}\n'
                   '      R2: {3:7.4f}\n').format(mae, mse, rmse, r2)
            log_and_print(base_logger, msg)


            log_and_print(base_logger, msg)


            writer.add_scalars('Regression/Scores', {'MAE': mae,
                                                     'MSE': mse,
                                                     'RMSE': rmse,
                                                     'R2': r2}, epoch)



        scheduler.step()
                        
    if args.save:

        best_model_weights_path = find_last_ckpts(ckpt_path, model_type, date=None)
        shutil.copy(best_model_weights_path, config['WEIGHTS']['PATH'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        help="path to yaml file",
                        required=False,
                        default='config/config.yaml')
    parser.add_argument('--device',
                        type=str,
                        help="set up torch device: 'cpu' or 'cuda' (GPU)",
                        required=False,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # remember to set the gpu device number
    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        required=False,
                        default='0')
    parser.add_argument('--save',
                        type=bool,
                        help='if set true, save the best model',
                        required=False,
                        default=False)
    args = parser.parse_args()

    # set up GPU
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load config file into dict() format
    config = YamlConfig(args.config_file)

    # create the output folder (name of experiment) for storing model result such as logger information
    if not os.path.exists(config['OUTPUT_DIR']):
        os.mkdir(config['OUTPUT_DIR'])
    # create the root folder for storing checkpoints during training
    if not os.path.exists(config['CKPTS_DIR']):
        os.mkdir(config['CKPTS_DIR'])
    # create the subfolder for storing checkpoints based on the model type
    if not os.path.exists(os.path.join(config['CKPTS_DIR'], config['TYPE'])):
        os.mkdir(os.path.join(config['CKPTS_DIR'], config['TYPE']))
    # create the folder for storing the best model after all epochs
    if not os.path.exists(config['MODEL']['WEIGHTS']['PATH']):
        os.mkdir(config['MODEL']['WEIGHTS']['PATH'])

    # print configuration
    print('=' * 40)
    print(config.file_contents)
    config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
    print('=' * 40)

    # initialize random seed for torch and numpy
    init_seed(config['MANUAL_SEED'])

    # get logger os.path.join(config['OUTPUT_DIR'], f'{config['TYPE']}_{config['LOG_TITLE']}.log')
    file_name = os.path.join(config['OUTPUT_DIR'], '{}.log'.format(config['TYPE']))
    base_logger = get_logger(file_name, config['LOG_TITLE'])
    # get summary writer for TensorBoard
    writer = SummaryWriter(os.path.join(config['OUTPUT_DIR'], 'runs'))
    # get dataloaders
    dataloaders = get_dataloaders(config['DATA'])
    # get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    visual_net, audio_net, text_net, evaluator = get_models(config['MODEL'], args, model_type, ckpt_path)

    main(dataloaders, visual_net, audio_net, text_net, evaluator, base_logger, writer, config['MODEL'], args, model_type, ckpt_path)

    writer.close()

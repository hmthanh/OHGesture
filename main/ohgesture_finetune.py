import sys

[sys.path.append(i) for i in ['.', '..', '../model', '../train']]

import pdb
import os
import logging
import torch
import yaml
from torch.utils.data import DataLoader
from pprint import pprint
from easydict import EasyDict

from configs.parse_args import parse_args
from data_loader.deepgesture_dataset import DeepGestureDataset, custom_collate
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from deepgesture_training_loop import DeepGestureTrainLoop
from model.deepgesture import DeepGesture
from model.mdm import MDM


logging.getLogger().setLevel(logging.INFO)


def create_model_and_diffusion(args):
    model = DeepGesture(modeltype='', njoints=1141, nfeats=1,
                        translation=True, pose_rep='rot6d', glob=True, glob_rot=True,
                        cond_mode='cross_local_attention3_style1', clip_version='ViT-B/32', action_emb='tensor',
                        audio_feat=args.audio_feat,
                        text_feat=args.text_feat,
                        arch='trans_enc', latent_dim=256,
                        n_seed=8, cond_mask_prob=0.1)
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def load_mdm_model(args):
    mdm_model = MDM(modeltype='', njoints=1141, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                    glob_rot=True, cond_mode='cross_local_attention3_style1', clip_version='ViT-B/32',
                    action_emb='tensor',
                    audio_feat=args.audio_feat,
                    arch='trans_enc', latent_dim=256, n_seed=8)
    return mdm_model


def get_submodule(model, module_name):
    module = model
    for attr in module_name.split("."):
        module = getattr(module, attr, None)
        if module is None:
            return None
    return module


def load_finetune_model_weights(args, ohgesture_model):
    # , weight_mdm

    mdm_model = load_mdm_model(args=args)
    print(f"Loading checkpoints from [{args.model_finetune_path}]...")
    state_dict = torch.load(args.model_finetune_path, map_location='cpu', weights_only=True)
    weight_mdm = load_model_wo_clip(mdm_model, state_dict)
    all_parameters = weight_mdm.parameters()
    for param_name, param in weight_mdm.named_parameters():
        print(f"Parameter name: {param_name}")
        # print(f"Parameter values: {param}")
        print(f"Shape: {param.shape}\n")

    total_params = sum(p.numel() for p in all_parameters)

    print(f'Total parameters: {total_params}')
    # shared_layers = [
    #     ("cross_local_attention", "cross_local_attention"),
    #     ("embed_timestep.sequence_pos_encoder", "embed_timestep.sequence_pos_encoder"),
    #     ("embed_timestep.time_embedding.0", "embed_timestep.time_embedding.0"),
    #     ("embed_timestep.time_embedding.2", "embed_timestep.time_embedding.2"),
    #     ("input_process.poseEmbedding", "input_process.poseEmbedding"),
    #     ("input_process2", "input_process2"),
    #     ("output_process.poseFinal", "output_process.poseFinal"),
    #     ("rel_pos", "rel_pos"),
    #     ("seqTransEncoder", "seqTransEncoder"),
    #     ("sequence_pos_encoder", "sequence_pos_encoder"),
    #     ("WavEncoder.audio_feature_map", "speech_linear_encoder.audio_feature_map"),
    #     ("embed_style", "style_linear_encoder"),
    #     # ("embed_text", "text_linear_encoder.text_feature_map")
    # ]
    #
    # for layer1_name, layer2_name in shared_layers:
    #     layer1 = get_submodule(weight_mdm, layer1_name)
    #     layer2 = get_submodule(ohgesture_model, layer2_name)
    #
    #     if layer1 and layer2:
    #         # Copy weights and biases if applicable
    #         if hasattr(layer1, "weight") and hasattr(layer2, "weight"):
    #             layer2.weight.data = layer1.weight.data.clone()
    #         if hasattr(layer1, "bias") and hasattr(layer2, "bias"):
    #             layer2.bias.data = layer1.bias.data.clone()
    #         print(f"Copied weights from {layer1_name} to {layer2_name}")

    # # Initialize layers unique to model2
    # unique_layers_model2 = [
    #     "seed_gesture_linear",
    #     "text_linear_encoder"  # Already shared but with an additional feature
    # ]
    #
    # for layer_name in unique_layers_model2:
    #     layer = get_submodule(ohgesture_model, layer_name)
    #     if layer:
    #         if hasattr(layer, "weight"):
    #             torch.nn.init.xavier_uniform_(layer.weight)
    #         if hasattr(layer, "bias"):
    #             torch.nn.init.zeros_(layer.bias)
    #         print(f"Initialized weights for {layer_name}")
    #
    #
    # # ohgesture_model.
    ohgesture_model.speech_linear_encoder.audio_feature_map.weight.data = weight_mdm.WavEncoder.audio_feature_map.weight.data.clone()
    ohgesture_model.speech_linear_encoder.audio_feature_map.bias.data = weight_mdm.WavEncoder.audio_feature_map.bias.data.clone()

    ohgesture_model.sequence_pos_encoder = weight_mdm.sequence_pos_encoder
    # ohgesture_model.sequence_pos_encoder.bias.data = weight_mdm.sequence_pos_encoder.bias.data.clone()

    return ohgesture_model


def main(args, device):
    # ~~~~~~~~~~~~~~~ Train ~~~~~~~~~~~~~~~
    train_dataset = DeepGestureDataset(args.train_h5,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=custom_collate)

    # ~~~~~~~~~~~~~~~ Valid ~~~~~~~~~~~~~~~
    # val_dataset = DeepGestureDataset(args.valid_h5,
    #                              n_poses=args.n_poses,
    #                              subdivision_stride=args.subdivision_stride,
    #                              pose_resampling_fps=args.motion_resampling_framerate)
    # test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
    #                          shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=False)

    # logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))
    logging.info('len of train loader: {}'.format(len(train_loader)))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    ohgesture_model, diffusion = create_model_and_diffusion(args)

    model = load_finetune_model_weights(args, ohgesture_model)

    # model.to(device)
    # train_loop = DeepGestureTrainLoop(args, model, diffusion, device, data=train_loader)
    # train_loop.run_loop()


if __name__ == '__main__':
    """
    cd main/
    python ohgesture.py --config=./configs/OHGesture.yml --gpu mps
    """
    args = parse_args()
    device = torch.device(args.gpu)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    main(config, device)

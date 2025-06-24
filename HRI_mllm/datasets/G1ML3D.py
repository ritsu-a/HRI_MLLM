import numpy as np
import torch
import os 
from os.path import join as pjoin
# from .humanml.utils.word_vectorizer import WordVectorizer
from HRI_mllm.utils.motion_utils.g1ml3d import (process_file, recover_from_ric)
from .BaseDataModule import BASEDataModule
from .T2M_dataset import Text2MotionDataset
# from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T

def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def humanml3d_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) == 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    return adapted_batch



class G1ML3DDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'g1ml3d'
        self.name = "g1ml3d"
        self.njoints = 41
        
        # Path to the dataset
        data_root = cfg.DATASET.HUMANML3D.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs')
        
        # Mean and std of the dataset
        dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH)
        self.hparams.mean = np.load(pjoin(dis_data_root, "Mean.npy"))
        self.hparams.std = np.load(pjoin(dis_data_root, "Std.npy"))
        
        # Mean and std for fair evaluation
        dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "Comp_v6_KLD01", "meta")
        self.hparams.mean_eval = self.hparams.mean 
        self.hparams.std_eval = self.hparams.std
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.HUMANML3D.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.HUMANML3D.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.HUMANML3D.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.HUMANML3D.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        # self.hparams.w_vectorizer = WordVectorizer(
        #     cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval

        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                self.hparams.win_size = 64
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.HUMANML3D.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        
        

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

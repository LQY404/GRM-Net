# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from numpy.core.fromnumeric import nonzero
import torch
from torch import nn
import os 
import torch.nn.functional as F
import time
import cv2
from transformers import AutoTokenizer, CLIPTokenizerFast, CLIPTokenizer

from detectron2.layers import cat
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
# from .rpn_ref import build_proposal_generator_ref
from .rpn_gcnref import build_proposal_generator_gcnref
# from .roi_heads_ref import build_roi_heads
from .roi_heads_gcnref import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .langencoder.build_lang import init_ref
from .langencoder.phraseAttn import AsyCA, MATTN

from .langencoder.rnn import RNNEncoder, BertEncoder, BertTextCNN, CLIPEncoder
from .utils.comm import generate_coord
from .resnet_sketch import build_resnet_sketch_fpn_backbone
from .utils.sketch_visualizer import visualize_sem_inst_mask_withGT
from .data_mapper_sketch import SKETCH_CLASS_NAME

from .data_mapper_sketch import ICATE_SKETCH
# from .datamapper_phrasecut import ICATE_PHRASE



__all__ = ["MRCNNRef", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GCNRef(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        textencoder: None, 
        input_format: Optional[str] = None,
        vis_period: int = 0,
        cfg = None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


        # print("MRCCN需要训练的参数")
        # for name, m in self.named_parameters():
        #     if m.requires_grad:
        #         print(name)
        #         # m.requires_grad = False  # 


        self.textencoder = init_ref(cfg, is_training=self.training)

        self.use_bert = cfg.MODEL.USE_BERT
        self.use_roberta = cfg.MODEL.USE_ROBERTA
        self.use_clip = cfg.MODEL.USE_CLIP
        
        if self.use_bert:
            assert not self.use_roberta
            assert not self.use_clip
            self.textencoder = None
            # self.bert = CustomerBert()
            self.bert = BertEncoder(cfg)
            # self.bert = BertTextCNN()
            
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
        elif self.use_roberta:
            raise
            assert not self.use_bert
            self.textencoder = None
            self.bert = CustomerRoberta()
        
        elif self.use_clip:
            # raise
            self.textencoder = None
            
            assert not self.use_bert
            assert not self.use_roberta
            
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.bert = CLIPEncoder() # 方便
            
        else:
            self.bert = None

        self.sketch = False
        if cfg.MODEL.NUM_CLASSES == 46:
            self.sketch = True
            print("ignore the bg")

        self.num_class = cfg.MODEL.NUM_CLASSES
        # print("使用MRCNNRef")

        # assert 1 == 0
        self.rnn_dim = cfg.REF_RNN_DIM
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional if not self.use_bert else 1024
        # self.hn_dim = 768  # use bert

        self.hs_dim = self.rnn_dim * bidirectional
        self.embedding_dim = cfg.WORD_VEC_DIM 
        self.roi_dim = 256

        m_dim = self.roi_dim * 2 + 8
        

        self.c = 1  # 1: RPN和ROI融合；-1：FPN融合；-2：backbone融合

        if self.c == -1:
            # del self.textencoder

            self.mback = nn.Sequential(
                nn.Conv2d(m_dim, self.roi_dim, 3, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(self.roi_dim),
            )

            self.lfc = nn.Sequential(
                nn.Linear(self.hn_dim, self.roi_dim),
                # nn.LayerNorm(self.roi_dim),
                # nn.Dropout(0.1 if self.training else 0.0),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.roi_dim, self.roi_dim),
                # nn.LayerNorm(self.roi_dim),
                # nn.Dropout(0.5 if self.training else 0.0),
                nn.ReLU(),
            )

            self.attn = AsyCA(self.roi_dim, 2)
            # self.attn = MATTN(cfg)


        # for name, m in self.named_parameters():
        #     if m.requires_grad:
        #         print(name)
        self.time = 0

        # self.save_path = "/home/lingpeng/project/Adet_new/effe_RPN_pred_in_"  + ("test/" if not self.training else "train/") + cfg.RPN_SAVE + "wodilation_worpnpos"
        self.save_path = "/home/lingpeng/project/Adet_2022/effe_RPN_pred_in_"  + ("test/" if not self.training else "train/") + cfg.RPN_SAVE + "_gcnref_v1"
        # self.save_path = "/home/lingpeng/project/Adet_new/effe_RPN_pred_in_"  + ("test/" if not self.training else "train/") + cfg.RPN_SAVE + "res5dilation_512"
        
        
        os.makedirs(self.save_path, exist_ok=True)
                
                
    @classmethod
    def from_config(cls, cfg):
        print("#"*20)
        print("GCNRef")
        
        print("#"*20)
        # assert 1 == 0
        if cfg.MODEL.BACKBONE.NAME ==  "build_resnet_fpn_backbone":
            backbone = build_backbone(cfg)
        else:
            print("SKETCH fpn")
            # assert 1 == 0
            backbone = build_resnet_sketch_fpn_backbone(cfg)

        # print("使用MRCNNRef")

        # assert 1 == 0
        
        return {
            "backbone": backbone,
            # "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "proposal_generator": build_proposal_generator_gcnref(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # "textencoder": init_ref(cfg),
            "textencoder": None,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def save_train_pred(self, batched_inputs, proposals, stage='RPN'):
        # storage = get_event_storage()
        max_vis_prop = 20

        t = 0
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            image_id = input['image_id']
            cref_id = input['cref_id']


            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            gt_boxes = input["instances"].gt_boxes.tensor.cpu().numpy()
            gt_labels = input['instances'].gt_classes.cpu().numpy()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)

            pred_boxes = prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            pred_labels = prop.objectness_logits[0:box_size].cpu().sigmoid().numpy()

            fileter = pred_labels >= 0.9
            pred_labels = pred_labels[fileter]
            pred_boxes = pred_boxes[fileter]
            
            sent = input['raw_sentence']

            # assert self.sketch
            save_file = os.path.join(self.save_path, "RPN_PRED_"+str(self.time)+"iter_"+str(image_id)+"_"+str(cref_id)+'.png')
            visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, scores=pred_labels, save_path=save_file, sent=sent)
            # self.time += 1
            if t >= 6:
                break
            t += 1
    
    def save_test_pred(self, batched_inputs, proposals, stage='RPN'):
        # storage = get_event_storage()
        print("save RPN results to " + self.save_path)
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]

            image_id = input['image_id']
            cref_id = input['cref_id']

            file_root_per_image = os.path.join(self.save_path, str(image_id))
            os.makedirs(file_root_per_image, exist_ok=True)

            rpn_pred_inst_file = os.path.join(file_root_per_image, str(image_id) + '_' + str(cref_id) + "_rpn_pred_image_inst.png")
            rpn_pred_inst_list_file = os.path.join(file_root_per_image, str(image_id) + '_' + str(cref_id) + "_rpn_pred_image_inst.txt")

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            out_height, out_width = img.shape[:2]
            img = cv2.resize(img.astype(np.float32), (768, 768)).astype(img.dtype)

            gt_boxes = input["instances"].gt_boxes.tensor.cpu().numpy()
            gt_labels = input['instances'].gt_classes.cpu().numpy()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)

            pred_boxes = prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            pred_labels = prop.objectness_logits[0:box_size].cpu().sigmoid().numpy()
            scale_factor_x, scale_factor_y = 768*1.0 / out_width, 768*1.0 / out_height

            tpred_boxes = []

            for i in range(pred_boxes.shape[0]):
                x0, y0, x1, y1 = pred_boxes[i]
                x0 *= scale_factor_x
                x1 *= scale_factor_x
                y0 *= scale_factor_y
                y1 *= scale_factor_y

                tpred_boxes.append([x0, y0, x1, y1])

            pred_boxes = np.array(tpred_boxes)
            # 类id映射回初始
            if self.num_class == 46:
                gt_labels = np.array([ICATE_SKETCH[x] for x in gt_labels])

            elif self.num_class == 1272:
                gt_labels = np.array([ICATE_PHRASE[x] for x in gt_labels])

            else:
                raise


            # fileter = pred_labels >= 0.001
            # pred_labels = pred_labels[fileter]
            # pred_boxes = pred_boxes[fileter]

            # assert self.sketch
            # save_file = os.path.join(self.save_path, "RPN_PRED_"+str(self.time)+"iter_"+str(image_id)+"_"+str(cref_id)+'.png')
            visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, class_names=list(SKETCH_CLASS_NAME.values()), scores=pred_labels, save_path=rpn_pred_inst_file, list_file_name=rpn_pred_inst_list_file)
            # visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, scores=None, save_path=rpn_pred_inst_file)
            # self.time += 1


    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # start = time.time()
        self.time += 1
        # print("#" * 20)
        # print(len(batched_inputs))
        # print("#" * 20)
        # print(batched_inputs)
        # print("#" * 20)

        # assert 1 == 0

        if not self.training:
            return self.inference(batched_inputs)

        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."

        images = self.preprocess_image(batched_inputs)
        # print(images.tensor)
        # assert 1 == 0
        # if "sent_encode" in batched_inputs[0]:
        #     assert self.textencoder is not None 

        if self.textencoder is not None:
            sent_encodes = cat([x['sent_encode'].unsqueeze(0).to(self.device) for x in batched_inputs])
        else:
            assert self.use_roberta or self.use_bert or self.use_clip
            captions = [e['raw_sentence'] for e in batched_inputs]
            # if len(batched_inputs) == 1:
        #     # print(decode(encode_sent, self.idx2word))
        #     sent_encodes = sent_encodes.unsqueeze(0)

        # print(images)
        # print(sent_encodes.shape, images.tensor.shape)
        
        # assert 1 == 0

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # for f in features:
        #     print(features[f].shape)

        # assert 1 == 0

        

        if self.use_bert or self.use_roberta:
            # assert self.use_roberta
            
            # mask_attention = cat([x["mask_attention"].unsqueeze(0).to(self.device) for x in batched_inputs])
            # word_embeddings, hn = self.bert(sent_encodes, attention_mask=mask_attention)
            tokenized = self.tokenizer.batch_encode_plus(captions,
                    max_length=256,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True).to(self.device)
            
            tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
            if self.freeze_bert:
                with torch.no_grad():
                    language_dict_features = self.bert(tokenizer_input)
            else:
                language_dict_features = self.bert(tokenizer_input)
                 
            sent_dict = {
                "hs": None,
                "hn": language_dict_features["aggregate"],
                "embedding": language_dict_features["embedded"],
                'words': None
            }
            
        elif self.use_clip:
            tokenized = self.tokenizer.batch_encode_plus(captions,
                    max_length=77,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True).to(self.device)
            
            tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
            
            if self.freeze_bert:
                with torch.no_grad():
                    language_dict_features = self.bert(tokenizer_input)
            else:
                language_dict_features = self.bert(tokenizer_input)
                 
            sent_dict = {
                "hs": None,
                "hn": language_dict_features["aggregate"],
                "embedding": language_dict_features["embedded"],
                'words': None
            }
            
            # print(sent_dict['hn'].shape)
            
            # assert 1 == 0
            
        elif self.textencoder is not None:
            seq_len = (sent_encodes != 0).sum(1).max().item()
            sent_encodes = sent_encodes[:, :seq_len] 
            # hs, hn, embedding = self.textencoder(sent_encodes)
            lang_dict = self.textencoder(sent_encodes)
            
            sent_dict = {
                "hs": lang_dict["output"],
                "hn": lang_dict['final_output'],
                "embedding": lang_dict['embedded'],
                'words': sent_encodes
            }
            

        features = self.backbone(images.tensor)
        # backbone_time = time.time()
        # print("backbone处理时间：", backbone_time - start)


        if self.sketch:
            imgs = [x["image"].to(self.device).unsqueeze(0) for x in batched_inputs]
            imgs = cat(imgs, dim=0)
            imgs = imgs[:, 0, :, :]
            # print(imgs.shape) # [N, H, W]
            # print(imgs)
            # assert 1 == 0
            image_bin = torch.ones_like(imgs)
            image_bin[imgs == 255] = 0
        else:
            image_bin = None


        if self.proposal_generator is not None:
            if self.c == 1:
                if self.use_bert or self.use_roberta or self.use_clip:
                    # print("use bert")
                    proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, sent_dict=None, sent_encodes=None, bert_hn=sent_dict['hn'])
                else:
                    proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, sent_dict=None if self.textencoder is None else sent_dict, sent_encodes=sent_encodes if self.textencoder is None else None)
            else:
                raise 

        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        # RPN_time = time.time()
        # print("RPN处理时间：", RPN_time - backbone_time)

        if self.vis_period > 0 and self.time > 0 and self.time % self.vis_period == 0:
        # if self.vis_period > 0:
            # storage = get_event_storage()
            # if storage.iter > 0 and storage.iter % self.vis_period == 0:
            #     self.visualize_training(batched_inputs, proposals, stage='RPN')

            self.save_train_pred(batched_inputs, proposals, stage='RPN')

        del batched_inputs
        
        if self.c == 1:
            if self.use_bert or self.use_roberta or self.use_clip:
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, sent_dict=None, sent_encodes=None, image_bin=image_bin, bert_hn=sent_dict['hn'])
        
            else:
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, sent_dict=None if self.textencoder is None else sent_dict, sent_encodes=sent_encodes if self.textencoder is None else None, image_bin=image_bin)
        else:
            raise 
        # ROI_time = time.time()
        # print("ROI处理时间：", ROI_time - RPN_time)
        # roi不再可视化
        #if self.vis_period > 0:
        #    storage = get_event_storage()
        #    if storage.iter > 0 and storage.iter % self.vis_period == 0:
        #        self.visualize_training(batched_inputs, proposals, stage='ROI')

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # final_time = time.time()
        # print("一次forward需要时间：", final_time-start)
        # print(losses)
        return losses
    
    @torch.no_grad()
    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        # if "sent_encode" in batched_inputs[0]:
        #     assert self.textencoder is not None 

        if self.textencoder is not None:
            sent_encodes = cat([x['sent_encode'].unsqueeze(0).to(self.device) for x in batched_inputs])
        else:
            assert self.use_roberta or self.use_bert or self.use_clip
            captions = [e['raw_sentence'] for e in batched_inputs]# if len(batched_inputs) == 1:
        #     # print(decode(encode_sent, self.idx2word))
        #     sent_encodes = sent_encodes.unsqueeze(0)

        # print(images)
        # print(sent_encodes.shape, images.tensor.shape)
        
        # assert 1 == 0


        if self.use_bert or self.use_roberta:
            # assert self.use_roberta
            
            # mask_attention = cat([x["mask_attention"].unsqueeze(0).to(self.device) for x in batched_inputs])
            # word_embeddings, hn = self.bert(sent_encodes, attention_mask=mask_attention)
            tokenized = self.tokenizer.batch_encode_plus(captions,
                    max_length=256,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True).to(self.device)
            
            tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
            with torch.no_grad():
                language_dict_features = self.bert(tokenizer_input)
                
            sent_dict = {
                "hs": None,
                "hn": language_dict_features["aggregate"],
                "embedding": language_dict_features["embedded"],
                'words': None
            }

        elif self.use_clip:
            tokenized = self.tokenizer.batch_encode_plus(captions,
                    max_length=77,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True).to(self.device)
            
            tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
            
            with torch.no_grad():
                language_dict_features = self.bert(tokenizer_input)
            
            sent_dict = {
                "hs": None,
                "hn": language_dict_features["aggregate"],
                "embedding": language_dict_features["embedded"],
                'words': None
            }
            
            
        elif self.textencoder is not None:
            seq_len = (sent_encodes != 0).sum(1).max().item()
            sent_encodes = sent_encodes[:, :seq_len] 
            # hs, hn, embedding = self.textencoder(sent_encodes)
            lang_dict = self.textencoder(sent_encodes)
            sent_dict = {
                "hs": lang_dict["output"],
                "hn": lang_dict['final_output'],
                "embedding": lang_dict['embedded'],
                'words': sent_encodes
            }

        features = self.backbone(images.tensor)


        if detected_instances is None:
            if self.proposal_generator is not None:
                if self.c == 1:
                    if self.use_bert or self.use_roberta or self.use_clip:
                        proposals, _ = self.proposal_generator(images, features, None, sent_dict=None, sent_encodes=None, bert_hn=sent_dict['hn'])
                
                    else:
                        proposals, _ = self.proposal_generator(images, features, None, sent_dict=None if self.textencoder is None else sent_dict, sent_encodes=sent_encodes if self.textencoder is None else None)
                
                else:
                    raise

            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            # self.save_test_pred(batched_inputs, proposals)

            if self.c == 1:
                if self.use_bert or self.use_roberta or self.use_clip:
                    results, _ = self.roi_heads(images, features, proposals, None, sent_dict=None, sent_encodes=None, bert_hn=sent_dict['hn'])
            
                else:
                    results, _ = self.roi_heads(images, features, proposals, None, sent_dict=None if self.textencoder is None else sent_dict, sent_encodes=sent_encodes if self.textencoder is None else None)
            
            else:
                raise 
            
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # print(results)
    
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GCNRef._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        # 先输出看看，应该是要添加NMS和matching_Score筛选的
        
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


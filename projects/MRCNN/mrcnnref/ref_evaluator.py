import contextlib
import copy
from detectron2.structures.masks import BitMasks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode

import glob
import shutil

import zipfile
import matplotlib.pyplot as plt

from .data_mapper_sketch import SKETCH_CLASS_NAME


class RefcocoEvaluator(DatasetEvaluator):
    def __init__(self, cfg=None):
        self._cpu_device = torch.device("cpu")
        self.cfg = cfg

        self.mean_bbox_iou = 0.0
        self.mean_mask_iou = 0.0
        self.data_len = 0
        self.iou_count = {}

        self.map = 0.0

        self.APs = []

        self.type = ['iep_ref', 'refcoco'][1]


        self.APs_list = []
        self.precision_list = []
        self.recall_list = []

        self.cum_I = 0.0
        self.cum_U = 0.0
        
        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        self.seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
        self.seg_total = 0.

        # if self.cfg is not None:
        #     self.entity = SketchDatasetMapper(self.cfg, False)

        self.APs_list_category = dict()
        self.precision_list_category = dict()
        self.recall_list_category = dict()
        
        self.cum_I_category = dict()
        self.cum_U_category = dict()
        self.seg_correct_category = dict()
        self.seg_total_category = dict()
        
        self.data_len_category = dict()
        
        
        for k, v in SKETCH_CLASS_NAME.items():
            self.APs_list_category[k] = []
            self.precision_list_category[k] = []
            self.recall_list_category[k] = []
            
            self.cum_I_category[k] = 0.0
            self.cum_U_category[k] = 0.0
            
            self.seg_correct_category[k] = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
            self.seg_total_category[k] = 0.
            
            self.data_len_category[k] = 0

            self.save_res = "../evaluation_res"
            os.makedirs(self.save_res, exist_ok=True)

    def process(self, inputs, outputs):
        # score_threshold = 0.0
        # iou_threshold = 0.5
        # iou_thresholds = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)
        # return 
        self.process3(inputs, outputs)
        # self.process2(inputs, outputs)
        # return 

        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        iou_thresholds = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)


        output_str = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        for k, v in SKETCH_CLASS_NAME.items():
            category = k
            output_str += "\n ###### for categroy: " + str(SKETCH_CLASS_NAME[category]) + " ###### \n"
            if self.cum_U_category[category] > 0:
                output_str += "累积mean IoU：" + str(self.cum_I_category[category]/self.cum_U_category[category]) + "\n"
            else:
                output_str += "累积mean IoU：" + str(0) + "\n"
                
            for e_eval_iou_index, e_eval_iou in enumerate(eval_seg_iou_list):
                if self.seg_total_category[category] > 0:
                    output_str += "Precision[@"+str(e_eval_iou) + "]: " + str(self.seg_correct_category[category][e_eval_iou_index] / self.seg_total_category[category]) + "\n"
                else:
                    output_str += "Precision[@"+str(e_eval_iou) + "]: " + str(0) + "\n"
            
            output_str += "累积的mAP: " + str(np.mean(self.APs_list_category[category])) + "\n"
            output_str += "累积的mAP_list[@.5:0.95]: " + str(np.mean(self.APs_list_category[category], axis=0)) + '\n'
        
        output_str += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n"
        
        with open(os.path.join(self.save_res, "test.txt"), 'w') as f:
            f.write(output_str)
        

    def process3(self, inputs, outputs): # 不仅统计总体的量化，还分类别进行统计
        score_threshold = 0.0
        iou_threshold = 0.5
        iou_thresholds = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)

        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        
        for input, output in zip(inputs, outputs):
            # gt_instances = input["instances"].to(self._cpu_device)
            sent = input['raw_sentence']
            # category = set(gt_instances.gt_classes.numpy())
            # print(category)
            # assert len(category) == 1
            # category = list(category)[-1]
            
            # if choice == 'refcoco':
            #     # labels.append(str(round(s, 3)) + ": " + str(ICATE[c]))
            #     category = ICATE[category]
            # elif choice == 'iepref':
            #     # labels.append(str(round(s, 3)) + ": " + str((c+1) % 49))
            #     category = (category + 1) % 49
            # elif choice == 'sketch':
            #     # labels.append(str(round(s, 3)) + ": " + str((c+1) % 47))
            #     category = (category + 1) % 47
            
            pred_instances = output["instances"].to(self._cpu_device)
            
            # 整体
            self.seg_total += 1
            self.data_len += 1
            
            # 单个类别
            # self.seg_total_category[category] += 1
            # self.data_len_category[category] += 1
            
            image = input['oimage'].permute(1, 2, 0).cpu().numpy().copy()  #一定要这个copy()函数，存在数据冲突
            results, semantic_mask, inst_image_file, list_file_name = instances_to_refcoco_json(pred_instances, input["image_id"], input['cref_id'], image.copy(), sent_encode=None, cfg=self.cfg, sent=sent)

            if len(pred_instances) == 0 or not pred_instances.has("pred_masks") or len(results) == 0:
                print("没有预测到任何实例")
                # self.data_len += 1
                self.APs.append(0)
                

                print("累积mean AP: ", self.map / self.data_len)

                print("mean AP: ", np.mean(self.APs))

                AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
                self.APs_list.append(AP_list)
                print("累积的mAP: ", np.mean(self.APs_list))
                print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list, axis=0))
                
                # for each category
                # self.APs_list_category[category].append(AP_list)
                
                self.precision_list.append(AP_list)
                # print("累积的precision：", np.mean(self.precision_list))
                # print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list, axis=0))
                
                # for each category
                # self.precision_list_category[category].append(AP_list)

                self.recall_list.append(AP_list)
                # print("累积的recall：", np.mean(self.recall_list))
                # print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list, axis=0))
                
                # for each category
                # self.recall_list_category[category].append(AP_list)
                
                if choice == 'sketch':
                    sent = input['raw_sentence']
                
                    index_infos = "IoU: " + str(0.00) + '\n' + \
                                "mAP: " + str(0.00) + '\n' + \
                                "mRecall: " + str(0.00) + '\n' + \
                                'mPR: ' + str(0.00)
                    visualize_sem_inst_mask(image, None, torch.tensor([]), None, None, save_path=inst_image_file, sent=sent, ap=index_infos)

                
                continue
            
            pred_boxes = []
            pred_masks = []
            pred_masks_s = np.zeros((image.shape[0], image.shape[1], len(results)))
            b_pred_mask = np.zeros((image.shape[0], image.shape[1]))
            pred_classes = []
            scores = []

            oheight, owidth = 768, 768
            
            for i, r in enumerate(results):
                single_pred_mask = r['bit mask']
                
                pred_box = r['bbox']
                x0, y0, x1, y1 = pred_box
                # x0 *= scale_factor_x
                # x1 *= scale_factor_x
                # y0 *= scale_factor_y
                # y1 *= scale_factor_y
                pred_boxes.append([x0, y0, x1, y1])
                pred_masks.append(single_pred_mask)
                pred_masks_s[:, :, i] = single_pred_mask

                b_pred_mask = np.where((b_pred_mask == 0) & (single_pred_mask != 0), b_pred_mask+single_pred_mask, b_pred_mask)

                pred_classes.append(r['category_id'])
                scores.append(r['score'])
            
            
            pred_boxes = np.array(pred_boxes)
            pred_masks = np.array(pred_masks)
            pred_classes = np.array(pred_classes)

            # sent_token = self.entity.from_encode2sentence(input['sent_encode'].cpu().numpy())
            # sent = sent_token[0]
            # for t in sent_token[1:]:
            #     sent += ' ' + t   
            sent = input['raw_sentence']
            
            # 只有在自己输入文本时用这个
            visualize_sem_inst_mask(image, semantic_mask, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
                                        class_names=list(SKETCH_CLASS_NAME.values()), scores=scores, save_path=inst_image_file, sent=sent)
            # visualize_sem_inst_mask(image, semantic_mask, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
            #                             class_names=None, scores=None, save_path=inst_image_file, sent=sent)
            
            #     # visual_compare(image, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
                #                 class_names=list(SKETCH_CLASS_NAME.values()), scores=scores, sent=sent, save_path=inst_image_file, list_file_name=list_file_name)
                # assert 1== 0

            assert len(scores) > 0

            if "instances" not in input.keys():  # 此时就是自己输入文本状态，不需要往下计算量化
                print("inference")
                continue
            
            gt_boxes = input['instances'].gt_boxes.tensor.numpy()
            gt_classes = input['instances'].gt_classes.numpy()
            gt_masks = input['instances'].gt_masks.tensor.numpy()
            
            gt_masks_s = np.zeros((gt_masks.shape[1], gt_masks.shape[2], gt_masks.shape[0]))
            b_gt_mask = np.zeros((gt_masks.shape[1], gt_masks.shape[2]))

            for i in range(gt_masks.shape[0]):
                single_gt_mask = gt_masks[i]

                gt_masks_s[:, :, i] = single_gt_mask

                b_gt_mask = np.where((b_gt_mask == 0) & (single_gt_mask != 0), b_gt_mask+single_gt_mask, b_gt_mask)
                
            print("mean IoU this time: ")
            I, U = compute_mask_IU(b_pred_mask, b_gt_mask)
            self.cum_I += I
            self.cum_U += U
            
            # for each category
            # self.cum_I_category[category] += I
            # self.cum_U_category[category] += U
            
            
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                
                self.seg_correct[n_eval_iou] += ((I / U) >= eval_seg_iou)
                
                # for each category
                # self.seg_correct_category[n_eval_iou] += ((I / U) >= eval_seg_iou)

            print("本次mean Iou：", I/U)
            print("累积mean IoU：", self.cum_I/self.cum_U)

            print("segmentation evluation (cumulate): ")
            for e_eval_iou_index, e_eval_iou in enumerate(eval_seg_iou_list):
                print("Precision[@"+str(e_eval_iou) + "]: " + str(self.seg_correct[e_eval_iou_index] / self.seg_total))
        

            AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            Pre_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            Recall_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            for j, iou_thr in enumerate(iou_thresholds):
                mAP, precisions, recalls, overlaps = compute_ap(gt_masks_s, scores, pred_masks_s, iou_thr)
  
                AP_list[j] = mAP
                Pre_list[j] = np.mean(precisions)
                Recall_list[j] = np.mean(recalls)
                
            print("此次的AP list为：", AP_list)
            print("此次的mean AP为：")
            print("[@.5:0.95]: " + str(np.mean(AP_list)))
            self.APs_list.append(AP_list)
            print("累积的mAP: ", np.mean(self.APs_list))
            print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list, axis=0))
            
            # for each category
            # self.APs_list_category[category].append(AP_list)

            # print("此次的precision：", Pre_list)
            # print("此次的mean precision：", np.mean(Pre_list))
            self.precision_list.append(Pre_list)
            # print("累积的precision：", np.mean(self.precision_list))
            # print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list, axis=0))
            
            # for each category
            # self.precision_list_category[category].append(Pre_list)

            # print("此次的recall：", Recall_list)
            # print("此次的mean recall:", np.mean(Recall_list))
            self.recall_list.append(Recall_list)
            # print("累积的recall：", np.mean(self.recall_list))
            # print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list, axis=0))
            
            # for each category
            # self.recall_list_category[category].append(Recall_list)
            
            # for each category
            # print("###### for categroy: " + str(SKETCH_CLASS_NAME[category]) + " ######")
            
            # if self.cum_U_category[category] > 0:
            #     print("累积mean IoU：", self.cum_I_category[category]/self.cum_U_category[category])
            # else:
            #     print("累积mean IoU：", 0)
                
            # for e_eval_iou_index, e_eval_iou in enumerate(eval_seg_iou_list):
            #     if self.seg_total_category[category] > 0:
            #         print("Precision[@"+str(e_eval_iou) + "]: " + str(self.seg_correct_category[category][e_eval_iou_index] / self.seg_total_category[category]))
            #     else:
            #         print("Precision[@"+str(e_eval_iou) + "]: " + str(0))
                
            # print("累积的mAP: ", np.mean(self.APs_list_category[category]))
            # print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list_category[category], axis=0))
            
            # print("累积的precision：", np.mean(self.precision_list_category[category]))
            # print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list_category[category], axis=0))
            
            # print("累积的recall：", np.mean(self.recall_list_category[category]))
            # print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list_category[category], axis=0))
            
            if choice == 'sketch':
                # continue
                # assert input['sent_encode'] is not None and self.cfg is not None
                # inst_image_file = os.path.join(save_root, str(img_id) + '_' + str(cref_id) + "_inst_image" + '.png')
                #entity = SketchDatasetMapper(self.cfg, False)
                # sent_token = self.entity.from_encode2sentence(input['sent_encode'].cpu().numpy())
                # sent = sent_token[0]
                # for t in sent_token[1:]:
                    # sent += ' ' + t
                    
                sent = input['raw_sentence']
                print("decode sent in evaluator:")
                print(sent)
                print("#"*20)
                # plt.imshow(image)
                # plt.show()
                index_infos = "IoU: " + str(I/U) + '\n' + \
                              "mAP: " + str(np.mean(AP_list)) + '\n' + \
                              "mRecall: " + str(np.mean(Recall_list)) + '\n' + \
                              'mPR: ' + str(np.mean(Pre_list))
                              
                visualize_sem_inst_mask(image, semantic_mask, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
                                            class_names=list(SKETCH_CLASS_NAME.values()), scores=scores, save_path=inst_image_file, sent=sent, ap=index_infos)
                # visual_compare(image, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
                #                 class_names=list(SKETCH_CLASS_NAME.values()), scores=scores, sent=sent, save_path=inst_image_file, list_file_name=list_file_name)
                # assert 1== 0


    def process2(self, inputs, outputs):
        score_threshold = 0.0
        iou_threshold = 0.5
        iou_thresholds = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)

        eval_seg_iou_list = [.5, .6, .7, .8, .9]

        for input, output in zip(inputs, outputs):
            # N = len(inputs)
            # 
            instances = output["instances"].to(self._cpu_device)
            self.seg_total += 1
            self.data_len += 1

            # oimage = input['image'].permute(1, 2, 0).cpu().numpy().copy()  #一定要这个copy()函数，存在数据冲突
            image = input['oimage'].permute(1, 2, 0).cpu().numpy().copy()  #一定要这个copy()函数，存在数据冲突
            results, semantic_mask, inst_image_file, _ = instances_to_refcoco_json(instances, input["image_id"], input['cref_id'], image.copy(), sent_encode=input['sent_encode'], cfg=None)

            if len(instances) == 0 or not instances.has("pred_masks") or len(results) == 0:
                print("没有预测到任何实例")
                # self.data_len += 1
                self.APs.append(0)

                print("累积mean AP: ", self.map / self.data_len)

                print("mean AP: ", np.mean(self.APs))

                AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
                self.APs_list.append(AP_list)
                print("累积的mAP: ", np.mean(self.APs_list))
                print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list, axis=0))

                self.precision_list.append(AP_list)
                print("累积的precision：", np.mean(self.precision_list))
                print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list, axis=0))

                self.recall_list.append(AP_list)
                print("累积的recall：", np.mean(self.recall_list))
                print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list, axis=0))


                continue


            gt_boxes = input['instances'].gt_boxes.tensor.numpy()
            gt_classes = input['instances'].gt_classes.numpy()
            gt_masks = input['instances'].gt_masks.tensor.numpy()
            
            gt_masks_s = np.zeros((gt_masks.shape[1], gt_masks.shape[2], gt_masks.shape[0]))
            b_gt_mask = np.zeros((gt_masks.shape[1], gt_masks.shape[2]))

            for i in range(gt_masks.shape[0]):
                single_gt_mask = gt_masks[i]

                gt_masks_s[:, :, i] = single_gt_mask

                b_gt_mask = np.where((b_gt_mask == 0) & (single_gt_mask != 0), b_gt_mask+single_gt_mask, b_gt_mask)
 
            pred_boxes = []
            pred_masks = []
            pred_masks_s = np.zeros((gt_masks.shape[1], gt_masks.shape[2], len(results)))
            b_pred_mask = np.zeros((gt_masks.shape[1], gt_masks.shape[2]))
            pred_classes = []
            scores = []

            oheight, owidth = 768, 768

            # semantic_mask = cv2.resize(semantic_mask.astype(np.float32), (owidth, oheight))
            # scale_factor_x, scale_factor_y = owidth*1.0 / 512, oheight*1.0 / 512
            for i, r in enumerate(results):
                single_pred_mask = r['bit mask']
                # scale_factor_x, scale_factor_y = owidth*1.0 / 512, oheight*1.0 / 512

                # single_pred_mask = cv2.resize(single_pred_mask.astype(np.float32), (owidth, oheight))

                pred_box = r['bbox']
                x0, y0, x1, y1 = pred_box

                # x0 *= scale_factor_x
                # x1 *= scale_factor_x
                # y0 *= scale_factor_y
                # y1 *= scale_factor_y

                pred_boxes.append([x0, y0, x1, y1])
                pred_masks.append(single_pred_mask)
                pred_masks_s[:, :, i] = single_pred_mask

                b_pred_mask = np.where((b_pred_mask == 0) & (single_pred_mask != 0), b_pred_mask+single_pred_mask, b_pred_mask)

                pred_classes.append(r['category_id'])
                scores.append(r['score'])
            
            assert len(scores) > 0

            print("mean IoU this time: ")
            I, U = compute_mask_IU(b_pred_mask, b_gt_mask)
            self.cum_I += I
            self.cum_U += U
            
            
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                
                self.seg_correct[n_eval_iou] += ((I / U) >= eval_seg_iou)

            print("本次mean Iou：", I/U)
            print("累积mean IoU：", self.cum_I/self.cum_U)

            print("segmentation evluation (cumulate): ")
            for e_eval_iou_index, e_eval_iou in enumerate(eval_seg_iou_list):
                print("Precision[@"+str(e_eval_iou) + "]: " + str(self.seg_correct[e_eval_iou_index] / self.seg_total))




            pred_boxes = np.array(pred_boxes)
            pred_masks = np.array(pred_masks)
            pred_classes = np.array(pred_classes)
            
            AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            Pre_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            Recall_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            for j, iou_thr in enumerate(iou_thresholds):
                mAP, precisions, recalls, overlaps = compute_ap(gt_masks_s, scores, pred_masks_s, iou_thr)
  
                AP_list[j] = mAP
                Pre_list[j] = np.mean(precisions)
                Recall_list[j] = np.mean(recalls)
            
            # self.APs_list.append(AP_list)
            # self.precision_list.append(Pre_list)
            # self.recall_list.append(Recall_list)


            print("此次的AP list为：", AP_list)
            print("此次的mean AP为：")
            print("[@.5:0.95]: " + str(np.mean(AP_list)))
            self.APs_list.append(AP_list)
            print("累积的mAP: ", np.mean(self.APs_list))
            print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list, axis=0))

            print("此次的precision：", Pre_list)
            print("此次的mean precision：", np.mean(Pre_list))
            self.precision_list.append(Pre_list)
            print("累积的precision：", np.mean(self.precision_list))
            print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list, axis=0))

            print("此次的recall：", Recall_list)
            print("此次的mean recall:", np.mean(Recall_list))
            self.recall_list.append(Recall_list)
            print("累积的recall：", np.mean(self.recall_list))
            print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list, axis=0))


            
            if choice == 'sketch':
                assert input['sent_encode'] is not None and self.cfg is not None
                # inst_image_file = os.path.join(save_root, str(img_id) + '_' + str(cref_id) + "_inst_image" + '.png')
                #entity = SketchDatasetMapper(self.cfg, False)
                sent_token = self.entity.from_encode2sentence(input['sent_encode'].cpu().numpy())
                sent = sent_token[0]
                for t in sent_token[1:]:
                    sent += ' ' + t
                print("decode sent in evaluator:")
                print(sent)
                print("#"*20)
                # plt.imshow(image)
                # plt.show()
                index_infos = "IoU: " + str(I/U) + '\n' + \
                              "mAP: " + str(np.mean(AP_list)) + '\n' + \
                              "mRecall: " + str(np.mean(Recall_list))
                            #   'mPR: ' + str(np.mean(Pre_list)) + '\n' + \
                              

                              
                visualize_sem_inst_mask(image, semantic_mask, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
                                        class_names=list(SKETCH_CLASS_NAME.values()), scores=scores, save_path=inst_image_file, sent=sent, ap=index_infos)


    def process1(self, inputs, outputs):
        score_threshold = 0.0
        iou_threshold = 0.5
        iou_thresholds = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)


        
        # 准备工作
        for input, output in zip(inputs, outputs):
            instances = output["instances"].to(self._cpu_device)
            # print(instances.fields)
            print("图像：", input["image_id"], "数据：", input['cref_id'])
            if len(instances) == 0 or not instances.has("pred_masks"):
                print("没有预测到任何实例")
                self.data_len += 1
                self.APs.append(0)

                print("累积mean AP: ", self.map / self.data_len)

                print("mean AP: ", np.mean(self.APs))

                AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
                self.APs_list.append(AP_list)
                print("累积的mAP: ", np.mean(self.APs_list))
                print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list, axis=0))

                self.precision_list.append(AP_list)
                print("累积的precision：", np.mean(self.precision_list))
                print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list, axis=0))

                self.recall_list.append(AP_list)
                print("累积的recall：", np.mean(self.recall_list))
                print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list, axis=0))


                continue
            
            image = input['image'].permute(1, 2, 0).cpu().numpy().copy()  #一定要这个copy()函数，存在数据冲突

            if self.type == 'iep_ref':
                results, semantic_mask, inst_image_file = instances_to_iep_json(instances, input["image_id"], input['cref_id'], image.copy())
            
            else:

                results, semantic_mask, inst_image_file = instances_to_refcoco_json(instances, input["image_id"], input['cref_id'], image.copy(), sent_encode=None, cfg=self.cfg)
            
            gt_boxes = input['instances'].gt_boxes.tensor.numpy()
            gt_classes = input['instances'].gt_classes.numpy()
            gt_masks = input['instances'].gt_masks.tensor.numpy()


            pred_boxes = []
            pred_masks = []
            pred_classes = []
            scores = []


            for r in results:
                pred_boxes.append(r['bbox'])
                pred_masks.append(r['bit mask'])
                pred_classes.append(r['category_id'])
                scores.append(r['score'])

            pred_boxes = np.array(pred_boxes)
            pred_masks = np.array(pred_masks)
            pred_classes = np.array(pred_classes)
            # instances = output["instances"].to(self._cpu_device)
            # scores = instances.scores.tolist()
            print("预测到实例" + str(len(scores)) + "个")

            # assert 1 == 0
            # keep = 0
            # for score in scores:
            #     if score > 0:
            #         keep += 1
            # if keep == 0:
            #     print("没有预测到任何实例")

            #     return []
            # keep = 10000
            # scores = scores[: keep]
            # pred_boxes = instances.pred_boxes.tensor.cpu().numpy()[: keep]
            # pred_masks = instances.pred_masks.cpu().numpy()[: keep]
            # pred_classes = instances.pred_classes.cpu().numpy()[: keep]

            scores = np.array(scores)
            indices = np.argsort(scores)[: : -1]
            # print(indices)
            # n_bboxes = []
            # for pred_boxe in pred_boxes:
            #     # print(type(pred_boxe))
            #     if type(pred_boxe) == torch.Tensor:
            #         pred_boxe = pred_boxe.numpy()
            #     n_bboxes.append(pred_boxe)

            # pred按照得分逆序排序
            # pred_boxes = np.array(n_bboxes)
            pred_boxes = pred_boxes[indices]

            pred_masks = pred_masks[indices]
            pred_classes = pred_classes[indices]

            # 计算AP
            # if type(iou_threshold) == int:
            # if choice == 'sketch':
            #     image_bin = np.zeros(image.shape[:-1])
            #     image_bin[image[:, :, 0] == 0] = 1
            # else:
            image_bin = None

            ap, precisions, recalls, overlaps = self.compute_ap(gt_boxes, gt_classes, gt_masks, 
                                                                    pred_boxes, pred_classes, scores, pred_masks, 
                                                                    iou_threshold=iou_threshold, score_threshold=score_threshold, image_bin=image_bin)
            self.APs.append(ap)
            self.map += ap
            self.data_len += 1

            print("[@" + str(iou_threshold) + "]: " + str(np.mean(self.APs)))


            # 计算mean AP
            AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            pre_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
            recall_list = np.zeros([len(iou_thresholds)], dtype=np.float32)

            for j in range(len(iou_thresholds)):
                iouThr = iou_thresholds[j]
                AP_single_iouThr, precisions, recalls, overlaps = self.compute_ap(gt_boxes, gt_classes, gt_masks, 
                                                                    pred_boxes, pred_classes, scores, pred_masks, 
                                                                    iou_threshold=iouThr, score_threshold=score_threshold, image_bin=image_bin)
                
                AP_list[j] = AP_single_iouThr
                pre_list[j] = np.mean(precisions)
                # pre_list[j] = precisions[-1]
                recall_list[j] = np.mean(recalls)
                # recall_list[j] = recalls[-1]

            print()
            print("此次的AP list为：", AP_list)
            print("此次的mean AP为：")
            print("[@.5:0.95]: " + str(np.mean(AP_list)))
            self.APs_list.append(AP_list)
            print("累积的mAP: ", np.mean(self.APs_list))
            print("累积的mAP_list[@.5:0.95]: ", np.mean(self.APs_list, axis=0))

            print("此次的precision：", pre_list)
            print("此次的mean precision：", np.mean(pre_list))
            self.precision_list.append(pre_list)
            print("累积的precision：", np.mean(self.precision_list))
            print("累积的precision_list[@.5:0.95]: ", np.mean(self.precision_list, axis=0))

            print("此次的recall：", recall_list)
            print("此次的mean recall:", np.mean(recall_list))
            self.recall_list.append(recall_list)
            print("累积的recall：", np.mean(self.recall_list))
            print("累积的recall_list[@.5:0.95]: ", np.mean(self.recall_list, axis=0))


            if choice == 'sketch':
                assert input['sent_encode'] is not None and self.cfg is not None
                # inst_image_file = os.path.join(save_root, str(img_id) + '_' + str(cref_id) + "_inst_image" + '.png')
                #entity = SketchDatasetMapper(self.cfg, False)
                sent_token = self.entity.from_encode2sentence(input['sent_encode'].cpu().numpy())
                sent = sent_token[0]
                for t in sent_token[1:]:
                    sent += ' ' + t
                print("decode sent in evaluator:")
                print(sent)
                print("#"*20)
                # plt.imshow(image)
                # plt.show()
                index_infos = "mAP: " + str(np.mean(AP_list)) + '\n' + 'mPR: ' + str(np.mean(pre_list)) + '\n' + "mRecall: " + str(np.mean(recall_list))
                visualize_sem_inst_mask(image, semantic_mask, pred_boxes, pred_masks, np.array([c for c in pred_classes]), 
                                        scores=scores, save_path=inst_image_file, sent=sent, ap=index_infos)

                


    def box_iou_xyxy(self, box1, box2):
        x1min, y1min, x1max, y1max = box1
        x2min, y2min, x2max, y2max = box2
        
        area1 = (y1max - y1min + 1) * (x1max - x1min + 1)
        area2 = (y2max - y2min + 1) * (x2max - x2min + 1)
        
        xmin = np.maximum(x1min, x2min)
        ymin = np.maximum(y1min, y2min)
        
        xmax = np.minimum(x1max, x2max)
        ymax = np.minimum(y1max, y2max)
        
        inter_h = np.maximum(ymax - ymin + 1, 0)
        inter_w = np.maximum(xmax - xmin + 1, 0)
        intersection = inter_h * inter_w
        
        union = area1 + area2 - intersection
    
        iou = intersection / union
        
        return iou
    
    def mask_iou(self, mask1, mask2):
        # print(mask1.dtype, mask2.dtype)

        # assert 1 == 0
        # mask1 = mask1.astype(np.uint8)
        # mask2 = mask2.astype(np.uint8)

        # intersection = np.sum(np.logical_and(mask1, mask2))
        # union = np.sum(np.logical_or(mask1, mask2))
        mask1 = np.reshape(mask1.astype(np.float32), (-1))
        mask2 = np.reshape(mask2.astype(np.float32), (-1))

        area1 = np.sum(mask1)
        area2 = np.sum(mask2)

        intersection = np.dot(mask1.T, mask2)
        union = area1 + area2 - intersection

        return intersection / union

    def compute_ap(self, gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5, score_threshold=0.0, image_bin=None):
        

        # Compute IoU overlaps [pred_masks, gt_masks]
        overlaps = np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)  # 以mask为基础计算，在sketch中会存在一些问题

        for pi in range(pred_boxes.shape[0]):

            pred_bbox = pred_boxes[pi]
            pred_class = pred_class_ids[pi]
            pred_mask = pred_masks[pi]
            # print(pred_mask)

            for gi in range(gt_boxes.shape[0]):
                gt_bbox = gt_boxes[gi]
                gt_class = gt_class_ids[gi]
                gt_mask = gt_masks[gi]
                # print(gt_mask)

                # assert 1 == 0

                # intersection = np.sum(np.logical_and(pred_mask, gt_mask))
                # union = np.sum(np.logical_or(pred_mask, gt_mask))

                # overlaps[pi, gi] = intersection*1.0 / union
                overlaps[pi, gi] = self.mask_iou(pred_mask, gt_mask)
                # print(pred_bbox)
                # print(gt_bbox)
                # overlaps[pi, gi] = self.box_iou_xyxy(pred_bbox, gt_bbox)

                # assert 1 == 0
        
        # print(overlaps)
        # Loop through predictions and find matching ground truth boxes
        match_count = 0
        # pred_match = -1 * np.ones([pred_boxes.shape[0]])
        pred_match = np.zeros([pred_boxes.shape[0]])
        # gt_match = -1 * np.ones([gt_boxes.shape[0]])
        gt_match = np.zeros([gt_boxes.shape[0]])
        for i in range(pred_boxes.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            # low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            # if low_score_idx.size > 0:
            #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            # 找到与pred_i最匹配的Gt
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                # if gt_match[j] > 0:
                if gt_match[j] == 1:
                    continue
                            
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_threshold:
                    break

                # Do we have a match?
                # if pred_class_ids[i] == gt_class_ids[j]:
                #     match_count += 1
                #     gt_match[j] = i
                #     pred_match[i] = j
                #     break
                # else:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                
                break

        # 利用上面产生的gt_match, pred_match, overlaps计算
        # Compute precision and recall at each prediction box step
        # precisions = np.cumsum(pred_match > -1).astype(np.float32) / (np.arange(len(pred_match)) + 1)
        precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
        # recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
        recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0.], precisions, [0.]])
        recalls = np.concatenate([[0.], recalls, [1.]])
        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])
        
        # Compute mean AP over recall range
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                    precisions[indices])
        
        assert mAP <= 1.0
        return mAP, precisions, recalls, overlaps


import imgviz
from PIL import Image
import cv2
from .dataset_mapper import ICATE

from .data_mapper_sketch import ICATE_SKETCH


choice = ['refcoco', 'iepref', 'sketch'][2]

class RefcocoVisualizer(Visualizer):

    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None)
        labels = []
        for s, c in zip(scores.tolist(), classes):
            if choice == 'refcoco':
                labels.append(str(round(s, 3)) + ": " + str((c+1) % 91))
            elif choice == 'iepref':
                labels.append(str(round(s, 3)) + ": " + str((c+1) % 49))
            elif choice == 'sketch':
                labels.append(str(round(s, 3)) + ": " + str((c+1) % 47))
            else:
                raise 

        # labels = [(str(s) + ": " + str(CATEGORY[c])) for s, c in zip(scores, classes)]
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)

            if choice == 'sketch':
                image_bin = np.zeros(self.img.shape[:-1])
                image_bin[self.img[:, :, 0] == 0] = 1

                nmasks = []
                for m in masks:
                    m = m * image_bin
                    nmasks.append(m)
                
                masks = np.array(nmasks)

            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
        
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[ICATE[c]]]) for c in classes
            ]
            alpha = 1.0
        else:
            colors = None
            alpha = 1.0

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.1

        alpha = 0.5
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


def instances_to_iep_json(instances, img_id, cref_id, image):
    simple = False
    easy = True
    begin = True
    use_attn = False
    if use_attn:
        easy = False

    use_iter = True
    # if simple:
    #     # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_50_1x_iep_ref_simple" + ('_easyconcat' if easy else "") + ("_atend" if not begin else "") + ("_useattn" if use_attn else "")
    #     save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_101_1x_iep_ref"  + ("_simple" if simple else "") + ("_atend" if not begin else "_atbegin") + ("_useattn" if use_attn else "")
    # else:
        # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_101_1x_iep_ref" + ('_easyconcat' if easy else "") + ("_atend" if not begin else "_atbegin") + ("_useattn" if use_attn else "")
    # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_50_1x_iep_ref_v2"  + ("_simple" if simple else "") + ("_atend" if not begin else "") + "_newmethod"
    save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/mrcnn_R_101_1x_iep_ref_rubber"  \
                                                        +  ("_simple" if simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_useattn" if use_attn else "")  \
                                                        + ("_use_1iter_withheatmap" if use_iter else "")
    
    # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/CondInst_MS_R_50_1x_iep_ref_atend_selfattn_6time"
    print("pred save to " + save_root)

    num_instances = len(instances)
    if num_instances == 0:
        print("没有预测到任何实例")
        return []

    print("输出预测得到的instance")
    # print(instances)
    # assert 1 == 0
    os.makedirs(save_root, exist_ok=True)

    image_file = os.path.join(save_root, str(img_id) + '_' + str(cref_id) + "_image" + '.png')
    sem_mask_file = os.path.join(save_root, str(img_id) + '_' + str(cref_id) + "_semantic_mask.png")
    with_bbox_image_file = os.path.join(save_root, str(img_id) + '_' + str(cref_id) + "_image_with_bbox_" + '.png')
    
    
    # Image.fromarray(image.astype(np.uint8)).save(image_file)
    # image = input['image'].permute(1, 2, 0).cpu().numpy().copy()  #一定要这个copy()函数，存在数据冲突
    scores = instances.scores.tolist()
    keep = 0
    for score in scores:
        if score > 0:
            keep += 1
    # if keep == 0:
    #     print("没有预测到任何实例")

    #     return []
    # keep = 10000

    pred_boxes = instances.pred_boxes[: keep]
    pred_masks = instances.pred_masks.cpu().numpy()[: keep]
    pred_classes = instances.pred_classes.cpu().numpy()[: keep]
    

    out_height, out_width = pred_masks[-1].shape

    semantic_mask = np.zeros(image.shape[: 2])

    for index, mask in enumerate(pred_masks):
        mask = mask.astype(np.long)
        mask = cv2.resize(mask.astype(np.float32), (768, 768)).astype(np.long)
        # semantic_masks.append(mask * pred_classes[index])
        semantic_mask = np.where((semantic_mask == 0) & (mask != 0), semantic_mask + mask * pred_classes[index], semantic_mask)

    saveMask(semantic_mask, sem_mask_file)

    scale_factor_x, scale_factor_y = 768*1.0 / out_width, 768*1.0 / out_height
    for index, box in enumerate(pred_boxes):
        
            box = np.array(box)
            # print(box)
            # color = [int(1.5*pred_classes[index]), int(0.5*pred_classes[index]), int(4.5*pred_classes[index])]
            color = [220, 20, 60]

            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

            x0 *= scale_factor_x
            x1 *= scale_factor_x
            y0 *= scale_factor_y
            y1 *= scale_factor_y

            image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
            text_score = "score: " + str(round(scores[index], 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            image = cv2.putText(image, text_score, (int(x0+10), int(y0+10)), font, 0.5, color, 2)

            # text_category = str(CATEGORY_NAME[pred_classes[index]])
            text_category = str(pred_classes[index])
            image = cv2.putText(image, text_category, (int(x1)-40, int(y1)-10), font, 0.5, color, 2)

    Image.fromarray(image.astype(np.uint8)).save(with_bbox_image_file)

    results = []

    for score, pred_box, pred_mask, pred_class in zip(scores, pred_boxes, pred_masks, pred_classes):
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (768, 768)).astype(np.long)

        box = np.array(pred_box)

        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

        x0 *= scale_factor_x
        x1 *= scale_factor_x
        y0 *= scale_factor_y
        y1 *= scale_factor_y

        result = {
            "image_id": img_id,
            "category_id": pred_class,
            "score": score,
            "sem seg": pred_mask * pred_class,
            "bit mask": pred_mask,
            "bbox": [x0, y0, x1, y1]
        }

        results.append(result)
    
    
    
    return results



from .utils.sketch_visualizer import visualize_sem_inst_mask, visualize_sem_seg
from .data_mapper_sketch import SketchDatasetMapper

def instances_to_refcoco_json(instances, img_id, cref_id, image, sent_encode=None, cfg=None, sent=""):
    simple = False
    easy = True
    begin = True
    use_attn = True
    use_iter = False
    # if simple:
    #     # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_50_1x_iep_ref_simple" + ('_easyconcat' if easy else "") + ("_atend" if not begin else "") + ("_useattn" if use_attn else "")
    #     save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_101_1x_iep_ref"  + ("_simple" if simple else "") + ("_atend" if not begin else "_atbegin") + ("_useattn" if use_attn else "")
    # else:
        # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_101_1x_iep_ref" + ('_easyconcat' if easy else "") + ("_atend" if not begin else "_atbegin") + ("_useattn" if use_attn else "")
    # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_50_1x_iep_ref_v2"  + ("_simple" if simple else "") + ("_atend" if not begin else "") + "_newmethod"
    save_root = "/home/lingpeng/project/Adet_pgshort/effe_inference_dir_new/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "") + '_' + choice \
                                                        + '_onenewgraph'
    
    # save_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/CondInst_MS_R_50_1x_iep_ref_atend_selfattn_6time"
    os.makedirs(save_root, exist_ok=True)

    file_root_per_image = os.path.join(save_root, str(img_id))
    os.makedirs(file_root_per_image, exist_ok=True)

    print("pred save to " + file_root_per_image)

    # print(instances)
    # assert 1 == 0
    
    if choice == 'sketch':
        visual_image_file = os.path.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_visual_image" + '.png')
        sem_mask_file = os.path.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_semantic_mask.png")
        with_bbox_image_file = os.path.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_image_with_bbox_" + '.png')
        inst_image_file = os.path.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_inst_image" + '.png')

        list_file_name = os.path.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_inst_score_list" + '.txt')
    else:
        visual_image_file = os.path.join(file_root_per_image, str(cref_id) + '_' + str(img_id) + "_visual_image" + '.png')
        sem_mask_file = os.path.join(file_root_per_image, str(cref_id) + '_' + str(img_id) + "_semantic_mask.png")
        with_bbox_image_file = os.path.join(file_root_per_image, str(cref_id) + '_' + str(img_id) + "_image_with_bbox_" + '.png')
        inst_image_file = os.path.join(file_root_per_image, str(cref_id) + '_' + str(img_id) + "_inst_image" + '.png')

        list_file_name = os.path.join(file_root_per_image, str(cref_id) + '_' + str(img_id) + "_inst_score_list" + '.txt')
    
    # Image.fromarray(image.astype(np.uint8)).save(image_file)
    # image = input['image'].permute(1, 2, 0).cpu().numpy().copy()  #一定要这个copy()函数，存在数据冲突
    
    num_instances = len(instances)
    if num_instances == 0:
        print("没有预测到任何实例")
        # assert 1 == 0
        return [], None, inst_image_file, list_file_name

    print("输出预测得到的instance")
    
    scores = instances.scores.tolist()
    keep = 0
    # print(scores)
    # assert 1 == 0
    for score in scores:
        if score > 0.5:
            keep += 1
    # if keep == 0:
    #     print("没有预测到任何实例")

    #     return []
    # keep = 10000
    # print(keep)
    
    keep = 1  # 取最大那个（for single）
    
    scores = scores[: keep]
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()[: keep]
    pred_masks = instances.pred_masks.cpu().numpy()[: keep]
    pred_classes = instances.pred_classes.cpu().numpy()[: keep]

    if len(scores) == 0:
        print("筛选后无实例输出")
        return [], None, inst_image_file, list_file_name

    ninstances = Instances(instances.image_size)
    ninstances.pred_boxes = Boxes(torch.tensor(pred_boxes))
    ninstances.pred_masks = torch.tensor(pred_masks)
    ninstances.pred_classes = torch.tensor(pred_classes)
    ninstances.scores = torch.tensor(scores)

    if choice != 'sketch':
        v = RefcocoVisualizer(image)
        v.draw_instance_predictions(ninstances).save(visual_image_file)

    out_height, out_width = pred_masks[-1].shape

    if choice == 'sketch':
        # pass
        image_bin = np.ones(image.shape[:-1])
        # print(image.shape)
        # print(image[:, :, 0])

        # assert 1 == 0
        image_bin[image[:, :, 0] == 255] = 0
    
        npred_masks = []
        for m in pred_masks:
            m = cv2.resize(m.astype(np.float32), (768, 768)).astype(m.dtype)
            m = m * image_bin
            npred_masks.append(m)
        
        pred_masks = np.array(npred_masks)

    semantic_mask = np.zeros(image.shape[: 2])
    
    binary_mask = np.zeros(image.shape[: 2])
    
    
    for index, mask in enumerate(pred_masks):
        mask = mask.astype(np.long)

        binary_mask = np.where((binary_mask == 0) & (mask != 0), binary_mask + mask, binary_mask)
        if choice == 'refcoco':
            semantic_mask = np.where((semantic_mask == 0) & (mask != 0), semantic_mask + mask * ((pred_classes[index]+1) % 91), semantic_mask)
        elif choice == 'iepref':
            semantic_mask = np.where((semantic_mask == 0) & (mask != 0), semantic_mask + mask * ((pred_classes[index]+1) % 49), semantic_mask)
        elif choice == 'sketch':
            semantic_mask = np.where((semantic_mask == 0) & (mask != 0), semantic_mask + mask * ((pred_classes[index]+1) % 47), semantic_mask)
        else:
            raise 
    
    # if choice != 'sketch':
    #     saveMask(semantic_mask, sem_mask_file)
    # saveMask(semantic_mask, sem_mask_file)
    # saveMask(binary_mask, sem_mask_file)
    visualize_sem_seg(image.copy(), binary_mask, sent, save_path=sem_mask_file)

    if choice == 'sketch' and 1 == 0:
        pass
        assert sent_encode is not None and cfg is not None
        entity = SketchDatasetMapper(cfg, False)
        sent_token = entity.from_encode2sentence(sent_encode.cpu().numpy())
        sent = sent_token[0]
        for t in sent_token[1:]:
            sent += ' ' + t
        print("decode sent in evaluator:")
        print(sent)
        print("#"*20)
        # visualize_sem_inst_mask(image, semantic_mask, pred_boxes, pred_masks, np.array([(c+1)%47 for c in pred_classes]), 
        #                         scores=scores, save_path=inst_image_file, sent=sent)


    # image_copy = image.copy()
    if choice == 'sketch':
        scale_factor_x, scale_factor_y = 768*1.0 / out_width, 768*1.0 / out_height  # for sketch
    else:
        scale_factor_x, scale_factor_y = 512*1.0 / out_width, 512*1.0 / out_height   # for refcoco
    
    npred_boxes = []
    image_bbox = image.copy()
    for index, box in enumerate(pred_boxes):
        
            box = np.array(box)
            # print(box)
            # color = [int(1.5*pred_classes[index]), int(0.5*pred_classes[index]), int(4.5*pred_classes[index])]
            color = [220, 20, 60]

            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

            x0 *= scale_factor_x
            x1 *= scale_factor_x
            y0 *= scale_factor_y
            y1 *= scale_factor_y

            npred_boxes.append([x0, y0, x1, y1])

            image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
            text_score = "score: " + str(round(scores[index], 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            image_bbox = cv2.putText(image_bbox, text_score, (int(x0+10), int(y0+10)), font, 0.5, color, 2)

            if choice == 'refcoco':
                text_category = str((pred_classes[index]+1) % 91)
            elif choice == 'iepref':
                text_category = str((pred_classes[index]+1) % 49)  # for iep ref
            elif choice == 'sketch':
                text_category = str((pred_classes[index]+1) % 47)  # for iep ref

            image_bbox = cv2.putText(image_bbox, text_category, (int(x1)-40, int(y1)-10), font, 0.5, color, 2)

    pred_boxes = np.array(npred_boxes)

    if choice != 'sketch':
        Image.fromarray(image_bbox.astype(np.uint8)).save(with_bbox_image_file)

    results = []


    for score, pred_box, pred_mask, pred_class in zip(scores, pred_boxes, pred_masks, pred_classes):

        if choice == 'refcoco':
            sem_seg = pred_mask * (pred_class+1) % 91

        elif choice == 'iepref':
            sem_seg = pred_mask * (pred_class+1) % 49

        elif choice == 'sketch':
            sem_seg = pred_mask * (pred_class+1) % 47,
            # sem_seg *= image_bin

        else:
            raise

        result = {
            "image_id": img_id,
            # "category_id": (pred_class+1) % 47, 
            "category_id": (pred_class+1) % 91, 
            "score": score,
            "sem seg": sem_seg,
            "bit mask": pred_mask,
            "bbox": pred_box
        }

        results.append(result)
    
    
    
    return results, semantic_mask, inst_image_file, list_file_name


def saveMask(mask, save_dir):
    # print("保存预测的语义掩膜")
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())

    lbl_pil.save(save_dir)


import numpy as np

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_bbox_iou(bboxes, target):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    if isinstance(target, list):
        target = np.array(target)
    target = target.reshape((-1, 4))

    A_bboxes = (bboxes[..., 2]-bboxes[..., 0]+1) * (bboxes[..., 3]-bboxes[..., 1]+1)
    A_target = (target[..., 2]-target[..., 0]+1) * (target[..., 3]-target[..., 1]+1)
    assert(np.all(A_bboxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(bboxes[..., 0], target[..., 0])
    I_y1 = np.maximum(bboxes[..., 1], target[..., 1])
    I_x2 = np.minimum(bboxes[..., 2], target[..., 2])
    I_y2 = np.minimum(bboxes[..., 3], target[..., 3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_bboxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def compute_ap(gt_masks, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    gt_masks: (768, 768, nGT)
    pred_scores: (nRoIs), the mask occupied percentage
    pred_masks: (768, 768, nRoIs)

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    indices = np.argsort(pred_scores)[::-1]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_masks.shape[2]])
    gt_match = np.zeros([gt_masks.shape[2]])
    for i in range(pred_masks.shape[2]):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            else:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps
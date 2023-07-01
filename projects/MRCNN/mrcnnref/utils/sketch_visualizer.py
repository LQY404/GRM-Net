import os
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import random


def format_decimal(num):
    a, b = str(num).split('.')

    return float(a + '.' + b[0: 3])

def visualize_sem_seg(im, predicts, sent, save_path=''):
    # im_seg = im.copy()
    masked_image = im.astype(np.uint32).copy()
    
    def generate_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        # hsv = [(i / N, 1, brightness) for i in range(N)]
        hsv = [(0. , 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    def apply_mask(image, mask, color, alpha=1.):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
    
    # im_seg[:, :, 0] += predicts.astype('uint8') * 200
    # im_seg[:, :, 1] += predicts.astype('uint8') * 0
    # im_seg[:, :, 2] += predicts.astype('uint8') * 0
    color = generate_colors(1)[0]
    masked_image = apply_mask(masked_image, predicts, color)
    
    plt.imshow(masked_image.astype('uint8'))
    if sent is not None:
        plt.title(sent)

    if save_path != '':
        plt.savefig(save_path)
        # im_seg_png = Image.fromarray(im_seg, 'RGB')
        # im_seg_png.save(save_path)
    else:
        plt.show()
    
    plt.close()


def visualize_inst_seg(im, predict_inst_seg, sent):
    """

    :param im:
    :param predict_inst_seg: [H, W, N]
    :param sent:
    :return:
    """
    predicts = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)
    if predict_inst_seg.shape[0] != 0:
        for inst_idx in range(predict_inst_seg.shape[2]):
            predicts = np.logical_or(predicts, predict_inst_seg[:, :, inst_idx])

    im_seg = im.copy()
    im_seg[:, :, 0] += predicts.astype('uint8') * 250
    plt.imshow(im_seg.astype('uint8'))
    if sent is not None:
        plt.title(sent)

    plt.show()

    plt.close()


# for RPN output
def visualize_sem_inst_mask_withGT(im, boxes_gt, boxes_pred, class_ids, class_names=None, sent=None, scores=None, save_path='', gt=False, list_file_name=None):
    
    
    def generate_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def draw_dash_line(x1, y1, x2, y2, dash_gap):
        assert x1 - x2 == 0 or y1 - y2 == 0
        len = abs(x1 - x2) + abs(y1 - y2)
        segm = int(len // dash_gap + 1)
        # print(segm)
        for seg_idx in range(segm):
            if x1 - x2 == 0:
                draw.line((x1, y1 + seg_idx * dash_gap, x2, min((y1 + seg_idx * dash_gap + 20), y2)),
                          fill=color_str, width=3)
            else:
                draw.line((x1 + seg_idx * dash_gap, y1, min((x1 + seg_idx * dash_gap + 20), x2), y2),
                          fill=color_str, width=3)

    N = boxes_gt.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes_gt.shape[0] == class_ids.shape[0]

    # Generate random colors
    colors = generate_colors(N)

    masked_image = im.astype(np.uint32).copy()

    masked_image_out_gt = Image.fromarray(np.array(masked_image, dtype=np.uint8))
    draw = ImageDraw.Draw(masked_image_out_gt)
    font_path = '/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/TakaoPGothic.ttf'

    if not os.path.exists(font_path):
        font_path = '../data/TakaoPGothic.ttf'

    font = ImageFont.truetype(font_path, 32)
    
    rows = 1
    cols = 2
    plt.figure(figsize=(8 * cols, 8 * rows))

    # for gt
    font = ImageFont.truetype(font_path, 32)

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes_gt[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        # y1_, x1_, y2_, x2_ = boxes[i]
        x1_, y1_, x2_, y2_ = boxes_gt[i]

        # Label
        class_id = class_ids[i]
        score = None
        label = class_names[class_id] if class_names is not None else str(class_id)
        caption = label

        draw.text((x1_ + 2, y1_ + 1), caption, font=font, fill='#000000')
        color_str = '#'

        def get_color_str(color_val):
            nor_color = int(color_val * 255)
            if nor_color < 16:
                return '0' + str(hex(nor_color))[2:]
            else:
                return str(hex(nor_color))[2:]

        color_str += get_color_str(color[0])
        color_str += get_color_str(color[1])
        color_str += get_color_str(color[2])

        dash_gap_ = 30

        draw_dash_line(x1_, y1_, x1_, y2_, dash_gap_)
        draw_dash_line(x2_, y1_, x2_, y2_, dash_gap_)
        draw_dash_line(x1_, y1_, x2_, y1_, dash_gap_)
        draw_dash_line(x1_, y2_, x2_, y2_, dash_gap_)

    plt.subplot(rows, cols, 1)
    plt.title(str('Gt') + ' instance: ' + sent, fontsize=14) if sent is not None else None
    plt.axis('on')
    plt.imshow(np.array(masked_image_out_gt, dtype=np.uint8))

    # for pred
    N = boxes_pred.shape[0]
    # print(N)
    if not N:
        print("\n*** No instances to display *** \n")
    
    colors = generate_colors(N)
    masked_image = im.astype(np.uint32).copy()

    masked_image_out_pred = Image.fromarray(np.array(masked_image, dtype=np.uint8))
    draw = ImageDraw.Draw(masked_image_out_pred)

    rpn_ins_ids = list(range(len(scores)))
    li = str(rpn_ins_ids) + '\n' + str(scores)

    final = 0
    for i in range(N):
        color = colors[i]

        # rpn_ins_id = rpn_ins_ids[i]

        # if rpn_ins_id not in [0, 1, 2]: # for 4
        #     continue

        # if rpn_ins_id not in [0, 1]:  # for 24, reltop0 & reltop10
        #     continue

        # if rpn_ins_id not in [0, 1]:  # for 30, reltop0 & reltop10
        #     continue

        # if rpn_ins_id not in [0, 2]: # for 42, reltop0
        #     continue
        # if rpn_ins_id not in [0, 1]: # for 42, reltop10
        #     continue

        # if rpn_ins_id not in [0, 2]: # for 63, reltop0
        #     continue
        # if rpn_ins_id not in [0, 5]: # for 63, reltop10:
        #     continue

        # if rpn_ins_id != 3 and rpn_ins_id != 4:  # specific for 136 reltop10
        #     continue
        # if rpn_ins_id not in [0, 6]: # specific for 136 reltop0
        #     continue

        # if rpn_ins_id not in [1, 3, 5]:  # for 202, reltop0
        #     continue
        # if rpn_ins_id not in [1, 3, 4]: # for 202, reltop10
        #     continue

        # if rpn_ins_id != 3 and rpn_ins_id != 4 and rpn_ins_id != 15: # for 238
        #     continue
        # if rpn_ins_id not in [3, 4]: # for 238, reltop0
        #     continue

        # if rpn_ins_id not in [0, 2, 3, 9]: # for 457
        #     continue
        # if rpn_ins_id not in [0, 1, 2]: # for 457, reltop0
        #     continue


        # if rpn_ins_id not in [0, 3]: # for 475
        #     continue
        # if rpn_ins_id not in [0, 1, 2, 3]: # for 262
        #     continue
        # if rpn_ins_id not in [0, 1, 2]:  # for 262, reltop0
        #     continue

        # if rpn_ins_id not in [2, 6]:  # for 408
        #     continue

        # if rpn_ins_id not in [0, 1, 2]:  # for 778, reltop0
        #     continue

        # if rpn_ins_id not in list(range(10)):  # general
        #     continue
        
        # Bounding box
        if not np.any(boxes_pred[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        # y1_, x1_, y2_, x2_ = boxes[i]
        x1_, y1_, x2_, y2_ = boxes_pred[i]

        # Label
        # class_id = class_ids[i]
        score = scores[i] if scores is not None else None

        # if score is not None and score <= 1e-8:
        #     continue
        label = class_names[class_id] if class_names is not None else str(class_id)
        caption = "{:.3f}".format(score) if scores is not None else None
        # caption = str(rpn_ins_id)

        if scores is not None and final == 0:
            draw.text((x1_ + 2, (y1_ - 30) if score is not None else (y1_ + 1)), caption, font=font, fill='#000000')

        color_str = '#'

        def get_color_str(color_val):
            nor_color = int(color_val * 255)
            if nor_color < 16:
                return '0' + str(hex(nor_color))[2:]
            else:
                return str(hex(nor_color))[2:]

        color_str += get_color_str(color[0])
        color_str += get_color_str(color[1])
        color_str += get_color_str(color[2])

        dash_gap_ = 30

        draw_dash_line(x1_, y1_, x1_, y2_, dash_gap_)
        draw_dash_line(x2_, y1_, x2_, y2_, dash_gap_)
        draw_dash_line(x1_, y1_, x2_, y1_, dash_gap_)
        draw_dash_line(x1_, y2_, x2_, y2_, dash_gap_)

    

    plt.subplot(rows, cols, 2)
    plt.title(str('Pred') + ' instance: ' + sent, fontsize=14) if sent is not None else None
    plt.axis('on')
    plt.imshow(np.array(masked_image_out_pred, dtype=np.uint8))
    # plt.show()

    if list_file_name:
        save_ids_scores_to_file(list_file_name, li)

    if save_path != '':
        plt.savefig(save_path)
        # im_seg_png = Image.fromarray(im_seg, 'RGB')
        # im_seg_png.save(save_path)
    else:
        plt.show()

    plt.close()

# for final output
def visual_compare(im, boxes_set, inst_masks_set, class_ids_set, class_names=None, sent=None, scores=None, save_path='', gt=False, list_file_name=None):
    # 不可视化confidence，自己做
    assert not gt # 仅用于测试结果可视化
    assert class_names is not None
    assert scores is not None or gt

    ## instance mask
    def apply_mask(image, mask, color, alpha=1.):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def generate_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def draw_dash_line(x1, y1, x2, y2, dash_gap):
        assert x1 - x2 == 0 or y1 - y2 == 0
        len = abs(x1 - x2) + abs(y1 - y2)
        if dash_gap > 0:
            segm = int(len // dash_gap + 1)
            # print(segm)
            for seg_idx in range(segm):
                if x1 - x2 == 0:
                    draw.line((x1, y1 + seg_idx * dash_gap, x2, min((y1 + seg_idx * dash_gap + 20), y2)),
                            fill=color_str, width=3)
                else:
                    draw.line((x1 + seg_idx * dash_gap, y1, min((x1 + seg_idx * dash_gap + 20), x2), y2),
                            fill=color_str, width=3)

        else:
            draw.rectangle((x1, y1, x2, y2), outline=color_str, width=2)
            # return

        

    N = boxes_set.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes_set.shape[0] == inst_masks_set.shape[0] == class_ids_set.shape[0]


    ids = list(range(len(scores))) # for each predicted ins
    # print("#"*20)
    # print(ids)
    # print(scores)
    # print("#"*20)

    li = str(ids) + '\n' + str(scores)

    use_ids = []
    # Generate random colors
    colors = generate_colors(N)

    masked_image = im.astype(np.uint32).copy()
    for i in range(N):
        score = scores[i] if scores else None
        if score and score <= 0.5: # 与阈值对应
            continue
        # if ids[i] not in [0, 1]:  # for 30, reltop0
        #     continue
        # if ids[i] not in [0, 1, 5]: # for 778, rletop0
        #     continue

        # if ids[i] == 3: # only for 262, reltop0
        #     continue

        # if ids[i] not in [0, 2]: # for 238, reltop0
        #     continue


        use_ids.append(ids[i])
        color = colors[i]
        # mask = inst_masks[:, :, i]
        mask = inst_masks_set[i, :, :]
        masked_image = apply_mask(masked_image, mask, color)

    masked_image_out = Image.fromarray(np.array(masked_image, dtype=np.uint8))
    draw = ImageDraw.Draw(masked_image_out)
    font_path = '/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/TakaoPGothic.ttf'

    if not os.path.exists(font_path):
        font_path = '../data/TakaoPGothic.ttf'
    
    font = ImageFont.truetype(font_path, 32) # 字体调大


    final = 0
    for i, id in enumerate(ids):
        score = scores[i] if scores else None
        if score and score < 1e-8:  # 这个阈值可以设置很低
            continue

        # if id not in [0, 1]: # for 24, reltop0 & reltop10
        #     continue

        # if id not in [0, 1]: # for 30, reltop0 & reltop10
        #     continue

        # if id not in [0, 1]: # for 42, reltop0
        #     continue
        # if id not in [0, 3]: # for 42, reltop10
        #     continue

        # if id not in [0]: # for 63, reltop0
        #     continue
        # if id not in [0, 3]: # for 63, reltop10
        #     continue

        # if id != 0 and id != 1: # specific for 136 reltop10
        #     continue 
        # if id not in [0, 1]:  # for 136, reltop0
        #     continue

        # if id not in [0, 1, 2]: # for 202, reltop0
        #     continue
        # if id not in [0, 2, 3]: # for 202, reltop10
        #     continue

        # if id != 0 and id != 1 and id != 2: # specific for 238
        #     continue 
        # if id not in [0, 2]: # for 238, reltop0
        #     continue

        # if id not in [0, 1, 2, 6]: # for 457
        #     continue
        # if id not in [0, 1, 2]: # for 457, reltop0
        #     continue

        # if id not in [0, 5]: # for 475
        #     continue
        # if id not in [0, 1, 2, 3]: # for 262 
        #     continue
        # if id not in [0, 1, 2] and id not in use_ids: # for 262, reltop0
        #     continue
        # if id not in [0, 3]:  # for 408
        #     continue

        # if id not in [0, 1, 2]: # for 778, reltop10
        #     continue
        # if id not in [0, 1, 5]: # for 778, reltop0
        #     continue

        if id not in list(range(15)):  # for general
            continue

        color = colors[i]
        box = boxes_set[i]
        # Bounding box
        if not np.any(box):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        # mask = inst_masks_set[i]
        class_id = class_ids_set[i]
        class_name = class_names[class_id]

        x1_, y1_, x2_, y2_ = box

        if final == 0:
            caption = "{}\n{}".format(id, class_name)
            draw.text((x1_ + 4, (y1_ - 34)), caption, font=font, fill='#000000')
        else:
            caption = "{}".format(class_name)
            draw.text((x1_ + 4, (y1_ - 1)), caption, font=font, fill='#000000')

        color_str = '#'

        def get_color_str(color_val):
            nor_color = int(color_val * 255)
            if nor_color < 16:
                return '0' + str(hex(nor_color))[2:]
            else:
                return str(hex(nor_color))[2:]

        color_str += get_color_str(color[0])
        color_str += get_color_str(color[1])
        color_str += get_color_str(color[2])

        if id in use_ids and final == 1:  # 随着不同的inference不同，注意时刻调整
            dash_gap_ = 0  # 如果用实线设为0
        
        else:
            dash_gap_ = 30

        draw_dash_line(x1_, y1_, x1_, y2_, dash_gap_)
        draw_dash_line(x2_, y1_, x2_, y2_, dash_gap_)
        draw_dash_line(x1_, y1_, x2_, y1_, dash_gap_)
        draw_dash_line(x1_, y2_, x2_, y2_, dash_gap_)


    rows = 1
    cols = 1
    plt.figure(figsize=(8 * cols, 8 * rows))
    plt.subplot(rows, cols, 1)
    plt.title('pred instance: ' + sent, fontsize=14) if sent is not None else None
    plt.axis('on')
    plt.imshow(np.array(masked_image_out, dtype=np.uint8))

    # plt.xlabel(str(li), fontdict={'family': 'Times New Romen', 'size': 12})
    if list_file_name:
        save_ids_scores_to_file(list_file_name, li)


    if save_path != '':
        plt.savefig(save_path)
        # im_seg_png = Image.fromarray(im_seg, 'RGB')
        # im_seg_png.save(save_path)
    else:
        plt.show()

    plt.close()

    



def visualize_sem_inst_mask(im, sem_mask, boxes, inst_masks, class_ids, class_names=None, sent=None, scores=None, save_path='', gt=False, ap=None):
    rows = 1
    cols = 1
    plt.figure(figsize=(8 * cols, 8 * rows))

    ## semantic mask

    # im_mask = im.copy()
    # im_mask[:, :, 0] += sem_mask.astype('uint8') * 250

    # plt.subplot(rows, cols, 1)
    # plt.title(str('Pred' if not gt else 'Gt') + ' semantic: ' + sent, fontsize=14) if sent is not None else None
    # plt.axis('on')
    # plt.imshow(im_mask.astype('uint8'))

    ## instance mask
    def apply_mask(image, mask, color, alpha=1.):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def generate_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def draw_dash_line(x1, y1, x2, y2, dash_gap):
        assert x1 - x2 == 0 or y1 - y2 == 0
        len = abs(x1 - x2) + abs(y1 - y2)
        if dash_gap > 0:
            segm = int(len // dash_gap + 1)
            # print(segm)
            for seg_idx in range(segm):
                if x1 - x2 == 0:
                    draw.line((x1, y1 + seg_idx * dash_gap, x2, min((y1 + seg_idx * dash_gap + 20), y2)),
                            fill=color_str, width=3)
                else:
                    draw.line((x1 + seg_idx * dash_gap, y1, min((x1 + seg_idx * dash_gap + 20), x2), y2),
                            fill=color_str, width=3)

        else:
            draw.rectangle((x1, y1, x2, y2), outline=color_str, width=2)
            # return

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == inst_masks.shape[0] == class_ids.shape[0]

    # Generate random colors
    colors = generate_colors(2*N)

    masked_image = im.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        # mask = inst_masks[:, :, i]
        mask = inst_masks[i, :, :]
        masked_image = apply_mask(masked_image, mask, color)

    # ax.imshow(masked_image.astype(np.uint8))
    masked_image_out = Image.fromarray(np.array(masked_image, dtype=np.uint8))
    draw = ImageDraw.Draw(masked_image_out)
    font_path = '/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/TakaoPGothic.ttf'

    if not os.path.exists(font_path):
        font_path = '../data/TakaoPGothic.ttf'

    font = ImageFont.truetype(font_path, 32)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        # y1_, x1_, y2_, x2_ = boxes[i]
        x1_, y1_, x2_, y2_ = boxes[i]

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id] if class_names is not None else str(class_id)
        
        if score:
            score = format_decimal(score)
        caption = "{:.3f}\n{}".format(score, label) if score else label
        draw.text((x1_ + 4, (y1_ - 34) if score is not None else (y1_ + 1)), caption, font=font, fill='#000000')

        # draw.text((x1_ + 2, y1_ + 2), label, font=font, fill='#000000')
        # draw.text((x1_, y1_ - 2), score, font=font, fill='#000000')
        
        color_str = '#'

        def get_color_str(color_val):
            nor_color = int(color_val * 255)
            if nor_color < 16:
                return '0' + str(hex(nor_color))[2:]
            else:
                return str(hex(nor_color))[2:]

        color_str += get_color_str(color[0])
        color_str += get_color_str(color[1])
        color_str += get_color_str(color[2])

        dash_gap_ = 30

        dash_gap_ = 0

        draw_dash_line(x1_, y1_, x1_, y2_, dash_gap_)
        draw_dash_line(x2_, y1_, x2_, y2_, dash_gap_)
        draw_dash_line(x1_, y1_, x2_, y1_, dash_gap_)
        draw_dash_line(x1_, y2_, x2_, y2_, dash_gap_)

    plt.subplot(rows, cols, 1)
    plt.title(str('Pred' if not gt else 'Gt') + ' instance: ' + sent, fontsize=14) if sent is not None else None
    plt.axis('on')
    plt.imshow(np.array(masked_image_out, dtype=np.uint8))

    if ap is not None:
        plt.xlabel(str(ap), fontdict={'family': 'Times New Romen', 'size': 12})
        

    # plt.show()
    if save_path != '':
        plt.savefig(save_path)
        # im_seg_png = Image.fromarray(im_seg, 'RGB')
        # im_seg_png.save(save_path)
    else:
        plt.show()

    plt.close()


def save_ids_scores_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()
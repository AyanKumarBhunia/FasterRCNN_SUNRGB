from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib
from coco import COCOMeta
print(device_lib.list_local_devices())
from skimage import img_as_ubyte
import imageio
import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass
from common import CustomResize, clip_boxes
assert six.PY3, "FasterRCNN requires Python 3!"
from tensorflow.python.framework import ops
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
#from tensorpack.tfutils.common import get_tf_version_tuple
import tensorpack.utils.viz as tpviz
from tensorpack.tfutils.tower import *
from coco import COCODetection
from basemodel import (image_preprocess, resnet_c4_backbone, resnet_conv5,resnet_fpn_backbone)

import model_frcnn

from model_frcnn import (sample_fast_rcnn_targets, fastrcnn_outputs, fastrcnn_losses, fastrcnn_predictions)
from model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from model_box import (clip_boxes, decode_bbox_target, encode_bbox_target, crop_and_resize, roi_align, RPNAnchors)

from data import(get_train_dataflow, get_eval_dataflow, get_all_anchors, get_all_anchors_fpn)
from viz import(draw_annotation, draw_proposal_recall, draw_predictions, draw_final_outputs)
from eval import (eval_coco, detect_one_image, print_evaluation_scores, DetectionResult)
from config import finalize_configs, config as cfg
from tensorpack.utils import viz

def load_weights(saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        iter_val = int(ckpt_name.split('-')[-1])
        print("Model Loaded")
        return iter_val, True
    else:
        return False, False

def preprocess(image):
    image = tf.expand_dims(image, 0)
    image = image_preprocess(image, bgr=True)
    return tf.transpose(image, [0, 3, 1, 2])


def fastrcnn_training(image,
                      rcnn_labels, fg_rcnn_boxes, gt_boxes_per_fg,
                      rcnn_label_logits, fg_rcnn_box_logits):
    """
    Args:
        image (NCHW):
        rcnn_labels (n): labels for each sampled targets
        fg_rcnn_boxes (fg x 4): proposal boxes for each sampled foreground targets
        gt_boxes_per_fg (fg x 4): matching gt boxes for each sampled foreground targets
        rcnn_label_logits (n): label logits for each sampled targets
        fg_rcnn_box_logits (fg x #class x 4): box logits for each sampled foreground targets
    """

    with tf.name_scope('fg_sample_patch_viz'):
        fg_sampled_patches = crop_and_resize(image, fg_rcnn_boxes,
                                             tf.zeros([tf.shape(fg_rcnn_boxes)[0]], dtype=tf.int32), 300)
        fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
        fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
        tf.summary.image('viz', fg_sampled_patches, max_outputs=30)

    encoded_boxes = encode_bbox_target(gt_boxes_per_fg, fg_rcnn_boxes) * tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS,
                                                                                     dtype=tf.float32)
    fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(rcnn_labels, rcnn_label_logits, encoded_boxes,
                                                             fg_rcnn_box_logits)
    return fastrcnn_label_loss, fastrcnn_box_loss

def fastrcnn_inference(image_shape2d, rcnn_boxes, rcnn_label_logits, rcnn_box_logits):
    """
    Args:
        image_shape2d: h, w
        rcnn_boxes (nx4): the proposal boxes
        rcnn_label_logits (n):
        rcnn_box_logits (nx #class x 4):

    Returns:
        boxes (mx4):
        labels (m): each >= 1
    """
    rcnn_box_logits = rcnn_box_logits[:, 1:, :]
    rcnn_box_logits.set_shape([None, cfg.DATA.NUM_CATEGORY, None])
    label_probs = tf.nn.softmax(rcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
    anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1), [1, cfg.DATA.NUM_CATEGORY, 1])   # #proposal x #Cat x 4
    decoded_boxes = decode_bbox_target(rcnn_box_logits /tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32), anchors)
    decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

    # indices: Nx2. Each index into (#proposal, #category)
    pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
    final_probs = tf.identity(final_probs, 'final_probs')
    final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
    final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
    return final_boxes, final_labels, final_probs

def initialize_FasterRCNN(filename):
    obj = np.load(filename)
    all_vars = tf.global_variables()
    Var_List = []
    for i in all_vars:
        Var_List.append(i.name)
    count = 0
    for i in obj.iterkeys():
        if i in Var_List:
            var = all_vars[Var_List.index(i)]
            sess.run(var.assign(obj[var.name]))
            print(i  + '==>>Loaded')
            count =  count + 1
        else:
            print(i  + '==>> NOT_Loaded')
    print('Total Parameter Loaded:' + str(count))


def save(saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(sess, dir, step)

if __name__ == '__main__':

    ops.reset_default_graph()
    global sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()
    graph = tf.get_default_graph()

    parser = argparse.ArgumentParser()
    parser.add_argument('--load',
                        default='/media/ayan/Drive/All_Object/tensorpack-master/FasterRCNN/ImageNet-R50-AlignPadding.npz',
                        help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/fasterRCNN')
    parser.add_argument('--visualize', action='store_true', default=True, help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation on COCO. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    # if get_tf_version_tuple() < (1, 6):
    # https://github.com/tensorflow/tensorflow/issues/14657
    #    logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN training if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    assert args.load
#############################################################################################
#############################################################################################
    Trainining_is = False
    finalize_configs(is_training=Trainining_is)  ### IMP
    cfg.TRAIN.BASE_LR = 0.001
    cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS
#############################################################################################
#############################################################################################
    is_training = Trainining_is
    image_P = tf.placeholder(tf.float32, (800, 1067, 3), 'image')
    anchor_labels = tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels')
    anchor_boxes = tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes')
    gt_boxes = tf.placeholder(tf.float32, (None, 4), 'gt_boxes')
    gt_labels = tf.placeholder(tf.int64, (None,), 'gt_labels')

    image = preprocess(image_P)
    Load_Weights = []

    with TowerContext('', is_training= Trainining_is):


        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)

        anchors = RPNAnchors(get_all_anchors(), anchor_labels, anchor_boxes)
        anchors = anchors.narrow_to(featuremap)

        image_shape2d = tf.shape(image)[2:]  # h,w
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(tf.reshape(pred_boxes_decoded, [-1, 4]),
                                                                 tf.reshape(rpn_label_logits, [-1]), image_shape2d,
                                                                 cfg.RPN.TRAIN_PRE_NMS_TOPK if is_training else cfg.RPN.TEST_PRE_NMS_TOPK,
                                                                 cfg.RPN.TRAIN_POST_NMS_TOPK if is_training else cfg.RPN.TEST_POST_NMS_TOPK)
        if is_training:
            # sample proposal boxes in training
            rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(proposal_boxes, gt_boxes, gt_labels)
        else:
            # The boxes to be used to crop RoIs.
            # Use all proposal boxes in inference
            rcnn_boxes = proposal_boxes

        boxes_on_featuremap = rcnn_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)


        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])  # nxcx7x7
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)

    if is_training:
        # rpn loss
        rpn_label_loss, rpn_box_loss = rpn_losses(anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits,
                                                  rpn_box_logits)

        # fastrcnn loss
        matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

        fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])  # fg inds w.r.t all samples
        fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
        fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

        fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_training(image, rcnn_labels, fg_sampled_boxes,
                                                                        matched_gt_boxes, fastrcnn_label_logits,
                                                                        fg_fastrcnn_box_logits)
        with TowerContext('', is_training = Trainining_is):
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')

        total_cost = tf.add_n([rpn_label_loss, rpn_box_loss, fastrcnn_label_loss, fastrcnn_box_loss, wd_cost], 'total_cost')
        final_boxes, final_labels, final_probs = fastrcnn_inference(image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)

        tf.summary.scalar('Total_Loss', total_cost)
        tf.summary.scalar('rpn_label_loss', rpn_label_loss)
        tf.summary.scalar('rpn_box_loss', rpn_box_loss)
        tf.summary.scalar('fastrcnn_label_loss', fastrcnn_label_loss)
        tf.summary.scalar('fastrcnn_box_loss', fastrcnn_box_loss)
        lr = tf.Variable(cfg.TRAIN.BASE_LR, trainable=False)
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(total_cost)
        saver = tf.train.Saver(max_to_keep=5)
        summary = tf.summary.merge_all()
        trainwriter = tf.summary.FileWriter("./logs/train", sess.graph)

    else:
        final_boxes, final_labels, final_probs = fastrcnn_inference(image_shape2d, rcnn_boxes, fastrcnn_label_logits,
                                                                    fastrcnn_box_logits)


    init_op = tf.global_variables_initializer()
    sess.run(init_op)

##############################################################
##############################################################
##############################################################
    cfg.DATA.CLASS_NAMES = ('BG',  # always index 0
               'bathtub', 'bed', 'bookshelf', 'box',
                'chair', 'counter', 'desk', 'door',
                'dresser', 'garbage_bin', 'lamp', 'monitor',
                'night_stand', 'pillow', 'sink',
                'sofa', 'table', 'toilet', 'tv')
''' 
##############################################################
##############################################################
##############################################################
# # # # # # # # # # # # Training # # # # # # # # # # # # # # #
##############################################################
##############################################################
##############################################################
    initialize_FasterRCNN(args.load)
    max_iters = 100000
    LR_Update = np.array([70000, 320000, 360000])
    LR_List = np.array([0.001, 0.0001, 0.00001, 0.000001])
    prev_LRsum = 0
    save_path = './Results'
    src_Train = '/media/ayan/Drive/IMI-Research/Datasets/Datasets_OP_Train/'
##############################################################
##############################################################
##############################################################
    df = get_train_dataflow(src_Train)
    df.reset_state()
    Total_Images = df.size()
    data_generator = df.get_data()
    iter = -1
##############################################################
##############################################################
##############################################################
    #for iter, dp in enumerate(data_generator):
    while iter < max_iters:
        iter = iter + 1
        print(iter)
        try:
            batch_image, batch_anchor_labels, batch_anchor_boxes, batch_gt_boxes, batch_gt_labels = next(data_generator)
        except StopIteration:
            data_generator = df.get_data()
            continue

        if sum(LR_Update < iter) > prev_LRsum:
            prev_LRsum = sum(iter > LR_Update)
            sess.run(tf.assign(lr, LR_List[prev_LRsum]))

        feed_dict = {image_P: batch_image, anchor_labels: batch_anchor_labels,
                     anchor_boxes: batch_anchor_boxes, gt_boxes: batch_gt_boxes, gt_labels: batch_gt_labels}

        if iter % 10 == 0:
            _, total_cost_, summary_ = sess.run([train_op, total_cost, summary], feed_dict)
            trainwriter.add_summary(summary_, iter)
            print('#Iteration:' + str(iter), 'Loss Value:' + str(total_cost_))
        else:
            _, total_cost_ = sess.run([train_op, total_cost], feed_dict)

        if total_cost_ > 100:
            pass

        if np.isnan(total_cost_):
            print('Ayan_NAN')

        if iter % 1000 == 0:
            final_boxes_, final_labels_, final_probs_, fastrcnn_box_logits_ = sess.run([final_boxes, final_labels, final_probs, fastrcnn_box_logits], feed_dict)
            orig_shape = batch_image.shape[:2]
            final_boxes_ = clip_boxes(final_boxes_, orig_shape)
            final_boxes_ = sess.run(final_boxes_)
            final_boxes_ = final_boxes_.astype('int32')
            if np.any(final_boxes_):
                tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in
                        zip(final_labels_, final_probs_)]
                final_viz = viz.draw_boxes(batch_image, final_boxes_, tags)
                gt_viz = draw_annotation(batch_image, batch_gt_boxes, batch_gt_labels)
                img_out = np.hstack((final_viz, gt_viz))
                imageio.imwrite(os.path.join(save_path, str(iter) + ".jpg"), img_out)

        if iter % 500 == 0:
            save(saver, './model', iter)
            print('Model Saved!!')

'''
##############################################################
##############################################################
##############################################################
# # # # # # # # # # # # Testing Type 2 # # # # # # # # # # # #
##############################################################
##############################################################
##############################################################

src_Test = '/media/ayan/Drive/IMI-Research/Datasets/Datasets_OP_Test/'
save_path = './generated_outputLast/'
#initialize_FasterRCNN(args.load)
saver = tf.train.Saver()
itr, _ = load_weights(saver, './model/')
output_file = 'out.json'
all_results = []
df = get_train_dataflow(src_Test)
df.reset_state()
iter = 0
data_generator = df.get_data()
max_iters = df.size()
save_folder = '/media/ayan/Drive/All_Object/tensorpack-master/Faster_RCNN_Test/Object-Detection-Metrics-master_2/'

while iter < max_iters:
    iter = iter + 1
    print(iter)
    try:
        batch_image, batch_anchor_labels, batch_anchor_boxes, batch_gt_boxes, batch_gt_labels = next(data_generator)
    except StopIteration:
        break
    orig_shape = batch_image.shape[:2]
    feed_dict = {image_P: batch_image}
    final_boxes_, final_labels_, final_probs_ = sess.run([final_boxes, final_labels, final_probs], feed_dict)
    final_boxes_ = clip_boxes(final_boxes_, orig_shape)
    final_boxes_ = sess.run(final_boxes_)
    final_boxes_ = final_boxes_.astype('int32')

    if np.any(final_boxes_):
        tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in zip(final_labels_, final_probs_)]
        final_viz = viz.draw_boxes(batch_image, final_boxes_, tags)
        gt_viz = draw_annotation(batch_image, batch_gt_boxes, batch_gt_labels)
        img_out = np.hstack((final_viz, gt_viz))
        imageio.imwrite(os.path.join(save_path, str(iter) + ".jpg"), img_out)




    Detection = []
    for ik in range(final_boxes_.shape[0]):
        Detection.append([cfg.DATA.CLASS_NAMES[final_labels_[ik]], float(final_probs_[ik]),
                          final_boxes_[ik,0], final_boxes_[ik,1], final_boxes_[ik,2] - final_boxes_[ik,0],
                          final_boxes_[ik,3] - final_boxes_[ik,1]])
    Detection = np.array(Detection)

    #np.savetxt(os.path.join(save_folder, 'detections/',str(iter) + '.txt'), Detection, fmt='%s %1.2f %1.0f %1.0f %1.0f %1.0f')
    np.savetxt(os.path.join(save_folder, 'detections/',str(iter) + '.txt'), Detection, delimiter=" ", fmt="%s")

    Groundtruth = []
    for ik in range(batch_gt_labels.shape[0]):
        Groundtruth.append([cfg.DATA.CLASS_NAMES[batch_gt_labels[ik]],
                            batch_gt_boxes[ik,0], batch_gt_boxes[ik,1], batch_gt_boxes[ik,2] - batch_gt_boxes[ik,0],
                            batch_gt_boxes[ik,3] - batch_gt_boxes[ik,1] ])
    Groundtruth = np.array(Groundtruth)
    #np.savetxt(os.path.join(save_folder, 'groundtruths/', iter + '.txt'), Groundtruth, fmt='%s %1.0f %1.0f %1.0f %1.0f')
    np.savetxt(os.path.join(save_folder, 'groundtruths/', str(iter) + '.txt'), Groundtruth, delimiter=" ", fmt="%s")
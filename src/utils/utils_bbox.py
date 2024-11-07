import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
 
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
    
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])

    print('##################')
    print(boxes)
    print(boxes.shape)
    
    print(image_shape)
    print('##################')
    
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)   #先验框的数量

    grid_shape = K.shape(feats)[1:3]
 
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    #---------------------------------------------------------------#
    #   将先验框进行拓展，生成的shape为(20, 20, num_anchors, 2)
    #---------------------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size, 20, 20, 3, 85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, (num_classes + 5)])
    print('K.sigmoid(feats[..., 2:4])',K.sigmoid(feats[..., 2:4]).shape)
    print('anchors_tensor',anchors_tensor.shape)

    box_xy          = (K.sigmoid(feats[..., :2]) * 2 - 0.5 + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh          = (K.sigmoid(feats[..., 2:4]) * 2) ** 2 * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #------------------------------------------#
    #   获得预测框的置信度
    #------------------------------------------#
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
   
    if calc_loss == True: 
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def DecodeBox(outputs,
            anchors,
            num_classes,
            input_shape1,
            input_shape2,
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
    image_shape = K.reshape(outputs[-1], [-1])
    print("image_shape:",image_shape)
    print("outputs[-1]:",outputs[-1])
    box_xy = []
    box_wh = []
    box_confidence = []
    box_class_probs = []
    for i in range(len(anchor_mask)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape1, input_shape2)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    box_xy = K.concatenate(box_xy, axis=0)
    box_wh = K.concatenate(box_wh, axis=0)
    box_confidence = K.concatenate(box_confidence, axis=0)
    box_class_probs = K.concatenate(box_class_probs, axis=0)

    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape1, image_shape, letterbox_image)

    box_scores = box_confidence * box_class_probs

    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out = []
    scores_out = []
    classes_out = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out = K.concatenate(boxes_out, axis=0)
    scores_out = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


class DecodeBoxNP():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBoxNP, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            batch_size      = np.shape(input)[0]
            input_height    = np.shape(input)[2]
            input_width     = np.shape(input)[3]

            #-----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
        
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            prediction = np.transpose(np.reshape(input, (batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width)), (0, 1, 3, 4, 2))

            x = self.sigmoid(prediction[..., 0])  
            y = self.sigmoid(prediction[..., 1])
         
            w = self.sigmoid(prediction[..., 2]) 
            h = self.sigmoid(prediction[..., 3]) 

            conf        = self.sigmoid(prediction[..., 4])
 
            pred_cls    = self.sigmoid(prediction[..., 5:])
            grid_x = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.linspace(0, input_width - 1, input_width), 0), input_height, axis=0), 0), batch_size * len(self.anchors_mask[i]), axis=0)
            grid_x = np.reshape(grid_x, np.shape(x))
            grid_y = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.linspace(0, input_height - 1, input_height), 0), input_width, axis=0).T, 0), batch_size * len(self.anchors_mask[i]), axis=0)
            grid_y = np.reshape(grid_y, np.shape(y))
    
         
            anchor_w = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(scaled_anchors)[:, 0], 0), batch_size, axis=0), -1), input_height * input_width, axis=-1)
            anchor_h = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(scaled_anchors)[:, 1], 0), batch_size, axis=0), -1), input_height * input_width, axis=-1)
            anchor_w = np.reshape(anchor_w, np.shape(w))
            anchor_h = np.reshape(anchor_h, np.shape(h))
  
            pred_boxes          = np.zeros(np.shape(prediction[..., :4]))
            pred_boxes[..., 0]  = x * 2. - 0.5 + grid_x
            pred_boxes[..., 1]  = y * 2. - 0.5 + grid_y
            pred_boxes[..., 2]  = (w * 2) ** 2 * anchor_w
            pred_boxes[..., 3]  = (h * 2) ** 2 * anchor_h

 
            _scale = np.array([input_width, input_height, input_width, input_height])
            output = np.concatenate([np.reshape(pred_boxes, (batch_size, -1, 4)) / _scale,
                                np.reshape(conf, (batch_size, -1, 1)), np.reshape(pred_cls, (batch_size, -1, self.num_classes))], -1)
            outputs.append(output)
        return outputs
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
            计算IOU
        """
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
                    
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        iou = inter_area / np.maximum(b1_area + b2_area - inter_area, 1e-6)

        return iou

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        box_corner          = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
          
            class_conf = np.max(image_pred[:, 5:5 + num_classes], 1, keepdims=True)
            class_pred = np.expand_dims(np.argmax(image_pred[:, 5:5 + num_classes], 1), -1)
 
            conf_mask = np.squeeze((image_pred[:, 4] * class_conf[:, 0] >= conf_thres))

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not np.shape(image_pred)[0]:
                continue
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)

            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:
       
                detections_class = detections[detections[:, -1] == c]

                conf_sort_index     = np.argsort(detections_class[:, 4] * detections_class[:, 5])[::-1]
                detections_class    = detections_class[conf_sort_index]
               
                max_detections = []
                while np.shape(detections_class)[0]:
  
                    max_detections.append(detections_class[0:1])
                    if len(detections_class) == 1:
                        break
                    ious                = self.bbox_iou(max_detections[-1], detections_class[1:])
                    detections_class    = detections_class[1:][ious < nms_thres]
                # 堆叠
                max_detections = np.concatenate(max_detections, 0)
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i]
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def get_anchors_and_decode(feats, anchors, num_classes):
        # feats     [batch_size, 20, 20, 3 * (5 + num_classes)]
        # anchors   [3, 2]
        # num_classes 
        # 3
        num_anchors = len(anchors)       
        grid_shape = np.shape(feats)[1:3]

        grid_x  = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
        grid_y  = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
        grid    = np.concatenate([grid_x, grid_y], -1)
        #---------------------------------------------------------------#
        #   将先验框进行拓展，生成的shape为(20, 20, num_anchors, 2)
        #   [1, 1, 3, 2]
        #   [20, 20, 3, 2]
        #---------------------------------------------------------------#
        anchors_tensor = np.reshape(anchors, [1, 1, num_anchors, 2])
        anchors_tensor = np.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1]) 

        feats           = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])


        box_xy          = (sigmoid(feats[..., :2]) * 2 - 0.5 + grid)   #先验框的回归参数
        box_wh          = (sigmoid(feats[..., 2:4]) * 2) ** 2 * anchors_tensor

        box_confidence  = sigmoid(feats[..., 4:5]) 
        box_class_probs = sigmoid(feats[..., 5:])  

        box_wh          = box_wh / 32
        anchors_tensor  = anchors_tensor / 32
        fig = plt.figure()
        ax  = fig.add_subplot(121)
        plt.ylim(-2, 22)
        plt.xlim(-2, 22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5, 5, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchors_tensor/2 
        anchor_top  = grid_y - anchors_tensor/2 
        print(np.shape(anchors_tensor))
        print(np.shape(box_xy))
        rect1 = plt.Rectangle([anchor_left[5,5,0,0],anchor_top[5,5,0,1]],anchors_tensor[0,0,0,0],anchors_tensor[0,0,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[5,5,1,0],anchor_top[5,5,1,1]],anchors_tensor[0,0,1,0],anchors_tensor[0,0,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[5,5,2,0],anchor_top[5,5,2,1]],anchors_tensor[0,0,2,0],anchors_tensor[0,0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.ylim(-2, 22)
        plt.xlim(-2, 22)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5, 5, c='black')
        plt.scatter(box_xy[0, 5, 5, :, 0],box_xy[0, 5, 5, :, 1],c='r')
        plt.gca().invert_yaxis()

        pre_left    = box_xy[...,0] - box_wh[...,0] / 2 
        pre_top     = box_xy[...,1] - box_wh[...,1] / 2 

        rect1 = plt.Rectangle([pre_left[0,5,5,0],pre_top[0,5,5,0]],box_wh[0,5,5,0,0],box_wh[0,5,5,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0,5,5,1],pre_top[0,5,5,1]],box_wh[0,5,5,1,0],box_wh[0,5,5,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0,5,5,2],pre_top[0,5,5,2]],box_wh[0,5,5,2,0],box_wh[0,5,5,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #
    feat = np.random.normal(-0.5,0.5, [4, 20, 20, 75])
    anchors = [[116, 90], [156, 198], [373, 326]]
    get_anchors_and_decode(feat, anchors, 20)

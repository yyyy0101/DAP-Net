import colorsys
import os
import time
import cv2
import gc

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont, Image
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo import yolo_body
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP


class YOLO(object):
    _defaults = {
       
        "model_path"        : '',
        "classes_path"      : 'model_data/voc_classes.txt',
     
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
  
        "input_shape1"       : [640, 640],
        "input_shape2"       : [640, 640],
        #---------------------------------------------------------------------#
        #   YoloV5:s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        "att"               : 5,
      
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,

        "max_boxes"         : 100,
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
    
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.model = yolo_body([None, None, 3], [None, None, 3], self.anchors_mask, self.num_classes, self.phi, self.att)
        self.model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))
    
        self.input_image_shape1 = Input([2,],batch_size=1)
        self.input_image_shape2 = Input([2,],batch_size=1)
        inputs  = [*self.model.output, self.input_image_shape1, self.input_image_shape2]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'anchors'           : self.anchors, 
                'num_classes'       : self.num_classes, 
                # 'input_shape'       : self.input_shape, 
                'input_shape1'       :self.input_shape1,
                'input_shape2'       :self.input_shape2,
                'anchor_mask'       : self.anchors_mask,
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.model.input, self.input_image_shape1,self.input_image_shape2], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape1, input_image_shape2):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape1, input_image_shape2], training=False)
        return out_boxes, out_scores, out_classes
    #---------------------------------------------------#
    #  detect
    #---------------------------------------------------#
    def detect_image(self, image1, image2, crop = False, count = False):
        image1       = cvtColor(image1)
        image2       = cvtColor(image2)
      
        image_data1  = resize_image(image1, (self.input_shape1[1], self.input_shape1[0]), self.letterbox_image)
        image_data2  = resize_image(image2, (self.input_shape1[1], self.input_shape1[0]), self.letterbox_image)
        
 
        image_data1  = np.expand_dims(preprocess_input(np.array(image_data1, dtype='float32')), 0)
        image_data2  = np.expand_dims(preprocess_input(np.array(image_data2, dtype='float32')), 0)

        input_image_shape = np.expand_dims(np.array([image1.size[1], image1.size[0]], dtype='float32'), 0)

        image_data = [image_data1, image_data2]
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape, input_image_shape) 

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image1.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image1.size[0] + image1.size[1]) // np.mean(self.input_shape1), 1))
        if count:
            print("top_label:", out_classes)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(out_classes == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
   
        if crop:
            for i, c in list(enumerate(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image1.size[1], np.floor(bottom).astype('int32'))
                right   = min(image1.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image1.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
   
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image1.size[1], np.floor(bottom).astype('int32'))
            right   = min(image1.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image1)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image1

    def get_FPS(self, image1, image2, test_interval):
        image1       = cvtColor(image1)
        image2       = cvtColor(image2)
        image_data1  = resize_image(image1, (self.input_shape1[1], self.input_shape1[0]), self.letterbox_image)
        image_data2  = resize_image(image2, (self.input_shape2[1], self.input_shape2[0]), self.letterbox_image)
    
        image_data1  = np.expand_dims(preprocess_input(np.array(image_data1, dtype='float32')), 0)
        image_data2  = np.expand_dims(preprocess_input(np.array(image_data2, dtype='float32')), 0)
        
        input_image_shape1 = np.expand_dims(np.array([image1.size[1], image1.size[0]], dtype='float32'), 0)
        input_image_shape2 = np.expand_dims(np.array([image2.size[1], image2.size[0]], dtype='float32'), 0)
        image_data = [image_data1, image_data2]
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape1, input_image_shape2) 

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape1, input_image_shape2) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
   
        image       = cvtColor(image)
      
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        output  = self.model.predict(image_data)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in output:
            b, h, w, c = np.shape(sub_output)
            sub_output = np.reshape(sub_output, [b, h, w, 3, -1])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()
        
    def convert_to_onnx(self, simplify, model_path):
        import onnx
        import tf2onnx
        spec = (tf.TensorSpec((None, *self.input_shape, 3), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13, output_path=model_path)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
 
    def get_map_txt(self, image_id1, image_id2, image1, image2, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id1+".txt"),"w") 
        image1       = cvtColor(image1)
        image2       = cvtColor(image2)
        
        image_data1  = resize_image(image1, (self.input_shape1[1], self.input_shape1[0]), self.letterbox_image)
        image_data2  = resize_image(image2, (self.input_shape1[1], self.input_shape1[0]), self.letterbox_image)

        image_data1  = np.expand_dims(preprocess_input(np.array(image_data1, dtype='float32')), 0)
        image_data2  = np.expand_dims(preprocess_input(np.array(image_data2, dtype='float32')), 0)

        input_image_shape = np.expand_dims(np.array([image1.size[1], image1.size[0]], dtype='float32'), 0)
        image_data = [image_data1,image_data2]
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape, input_image_shape) 

        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
        
class YOLO_ONNX(object):
    _defaults = {

        "onnx_path"         : 'model_data/models.onnx',
        "classes_path"      : 'model_data/coco_classes.txt',

        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],

        "input_shape"       : [640, 640],

        "confidence"        : 0.5,
 
        "nms_iou"           : 0.3,
   
        "letterbox_image"   : True
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        import onnxruntime
        self.onnx_session   = onnxruntime.InferenceSession(self.onnx_path)

        self.input_name     = self.get_input_name()

        self.output_name    = self.get_output_name()

        self.class_names, self.num_classes  = self.get_classes(self.classes_path)
        self.anchors, self.num_anchors      = self.get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBoxNP(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        show_config(**self._defaults)
 
    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
    
    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors, len(anchors)

    def get_input_name(self):
        # 获得所有的输入node
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_feed(self,image_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor
        return input_feed

    def resize_image(self, image, size, letterbox_image, mode='PIL'):
        if mode == 'PIL':
            iw, ih  = image.size
            w, h    = size

            if letterbox_image:
                scale   = min(w/iw, h/ih)
                nw      = int(iw*scale)
                nh      = int(ih*scale)

                image   = image.resize((nw,nh), Image.BICUBIC)
                new_image = Image.new('RGB', size, (128,128,128))
                new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            else:
                new_image = image.resize((w, h), Image.BICUBIC)
        else:
            image = np.array(image)
            if letterbox_image:      
                shape       = np.shape(image)[:2]
                if isinstance(size, int):
                    size    = (size, size)
                r = min(size[0] / shape[0], size[1] / shape[1])

                new_unpad   = int(round(shape[1] * r)), int(round(shape[0] * r))
                dw, dh      = size[1] - new_unpad[0], size[0] - new_unpad[1]

                dw          /= 2  
                dh          /= 2

                if shape[::-1] != new_unpad:  # resize
                    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
                new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
            else:
                new_image = cv2.resize(image, (w, h))

        return new_image
 
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
 
        image_data  = self.resize_image(image, self.input_shape, True)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
 
        input_feed  = self.get_input_feed(image_data)
        outputs     = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)

        feature_map_shape   = [[int(j / (2 ** (i + 3))) for j in self.input_shape] for i in range(len(self.anchors_mask))][::-1]
        for i in range(len(self.anchors_mask)):
            outputs[i] = np.transpose(np.reshape(outputs[i], (1, feature_map_shape[i][0], feature_map_shape[i][1], len(self.anchors_mask[i]) * (5 + self.num_classes))), (0, 3, 1, 2))
        
        outputs = self.bbox_util.decode_box(outputs)

        results = self.bbox_util.non_max_suppression(np.concatenate(outputs, 1), self.num_classes, self.input_shape, 
                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                
        if results[0] is None: 
            return image

        top_label   = np.array(results[0][:, 6], dtype = 'int32')
        top_conf    = results[0][:, 4] * results[0][:, 5]
        top_boxes   = results[0][:, :4]

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

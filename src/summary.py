
from nets.yolo import yolo_body
from utils.utils import net_flops
from contextlib import redirect_stdout

with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        

        if __name__ == "__main__":
            input_shape     = [640, 640, 3]
            anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_classes     = 80
            phi             = 's'            

            model = yolo_body(input_shape, input_shape, anchors_mask, num_classes, phi, att = 5)

            model.summary()
        
            net_flops(model, table=False)

import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
from model import TextRemoval
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./examples/images/test.jpg', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='./examples/masks/00001.png', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='./examples/results/output.png', type=str,
                    help='Where to save output image.')
parser.add_argument('--checkpoint_dir', default='./model_logs/text_output_0308', type=str,
                    help='The directory of tensorflow checkpoint.')

def data_batch(list1, list2,list3):
    test_dataset = tf.data.Dataset.from_tensor_slices((list1, list2,list3))
    input_size=256
    def image_fn(gt_path,img_path, mask_path):
        x = tf.read_file(gt_path)
        x_decode = tf.image.decode_jpeg(x, channels=3)
        gt = tf.image.resize_images(x_decode, [input_size, input_size])
        gt = tf.cast(gt, tf.float32)

        x = tf.read_file(img_path)
        x_decode = tf.image.decode_jpeg(x, channels=3)
        img = tf.image.resize_images(x_decode, [input_size, input_size])
        img = tf.cast(img, tf.float32)

        x = tf.read_file(mask_path)
        x_decode = tf.image.decode_jpeg(x, channels=1)
        mask = tf.image.resize_images(x_decode, [input_size, input_size])
        mask = tf.cast(mask, tf.float32)
        return gt,img, mask

    test_dataset = test_dataset. \
        repeat(1). \
        map(image_fn). \
        apply(tf.contrib.data.batch_and_drop_remainder(1)). \
        prefetch(1)

    test_gt,test_image, test_mask = test_dataset.make_one_shot_iterator().get_next()
    return test_gt,test_image, test_mask


if __name__ == "__main__":
    # ng.get_gpus(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parser.parse_args()

    config_path = os.path.join('config.yml')
    config = Config(config_path)
    model = TextRemoval(config)

    # dataset_name = 2000
    # path_img = '/data/cqwang/64_backup/cqwang/dataset/IJCIA_dataset/test/test_{}_256/text/'.format(str(dataset_name))
    # path_mask = '/data/cqwang/64_backup/cqwang/dataset/IJCIA_dataset/test/test_{}_256/mask/'.format(str(dataset_name))
    path_img = args.image
    path_mask = args.mask

    list_img = list(glob.glob(path_img + '/*.jpg')) + list(glob.glob(path_img + '/*.png'))
    list_img.sort()
    list_mask = list(glob.glob(path_mask + '/*.jpg')) + list(glob.glob(path_mask + '/*.png'))
    list_mask.sort()

    gt,images, masks = data_batch(list_img,list_img, list_mask)
    
    images = (images / 255 - 0.5) / 0.5
    
    masks = masks / 255

    images_masked = (images * (1 - masks)) + masks
    # input of the model
    inputs = tf.concat([images_masked, masks], axis=3)

    # process outputs
    stroke_mask1, output1, stroke_mask2, output2 = model.generator(
            images, masks, reuse=False, training=False,name='textremoval_generator',
            padding='SAME')
    output = output2

    outputs_merged = (output * masks) + (images * (1 - masks))
    images = (images + 1) / 2 * 255
    
    images_masked = (images_masked + 1) / 2 * 255
    outputs = (output + 1) / 2 * 255
    masks=masks*255
    outputs_merged = (outputs_merged + 1) / 2 * 255

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        sess.run(assign_ops)
        
        # res_path="./result_{}/".format(str(dataset_name))
        res_path = args.output
        if os.path.exists(res_path):
            print("res_path已经存在")
        else:
            os.makedirs(res_path)


        for num in range(0, len(list_img)):
            outputs_merge, mas, out,img = sess.run([outputs_merged, masks, outputs,images])
            outputs_merge = outputs_merge[0][:, :, ::-1].astype(np.uint8)
            
            picname = list_img[num].split('/')[-1]
            cv2.imwrite(res_path+picname, outputs_merge)
            print(res_path+picname)
            

import os
import glob
import numpy as np
import tensorflow as tf

class Dataset(object):
    def __init__(self, config):
        super(Dataset, self).__init__()
        self.train_concat_list = self.load_flist(config.TRAIN_CONCAT_FLIST)
        self.val_concat_list = self.load_flist(config.VAL_CONCAT_FLIST)

        self.len_train = len(self.train_concat_list)
        self.len_val = len(self.val_concat_list)

        self.input_size = config.INPUT_SIZE
        self.epoch = config.EPOCH
        self.batch_size = config.BATCH_SIZE
        self.val_batch_size = config.VAL_BATCH_SIZE

        self.data_batch()

    def load_flist(self, flist):

        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def data_batch(self):
        train_concat = tf.data.Dataset.from_tensor_slices(self.train_concat_list)
        val_concat = tf.data.Dataset.from_tensor_slices(self.val_concat_list)

        def image_fn(img_path):
            x = tf.read_file(img_path)
            x_decode = tf.image.decode_jpeg(x, channels=3)
            img = tf.image.resize_images(x_decode, [256,1280])
            return img

        train_concat = train_concat.map(image_fn, num_parallel_calls=self.batch_size)
        train_dataset = tf.data.Dataset.zip((train_concat))
        train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(1000, 10*self.epoch)).batch(self.batch_size, drop_remainder=True)

        val_concat = val_concat.map(image_fn, num_parallel_calls=self.val_batch_size)
        val_dataset = tf.data.Dataset.zip((val_concat))
        val_dataset = val_dataset.apply(tf.data.experimental.shuffle_and_repeat(1000, 10*self.epoch)).batch(self.val_batch_size, drop_remainder=True)

        self.batch_concat= train_dataset.make_one_shot_iterator().get_next()
        self.val_concat = val_dataset.make_one_shot_iterator().get_next()

        # get the epoch of dataset train_image
        self.dataset = train_dataset



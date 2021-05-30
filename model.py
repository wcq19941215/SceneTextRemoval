from network import *
from metrics import *
from loss import *
from tensorflow.contrib.framework.python.ops import arg_scope
class TextRemoval(object):
    def __init__(self, config):
        self.config = config
        self.res_num = config.RES_NUM
        self.base_channel = config.BASE_CHANNEL
        self.sample_num = config.SAMPLE_NUM
        self.model_name = 'textremoval'
        self.rate=config.RATE

        self.gen_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )
        self.dis_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )

    def build_whole_model(self, batch_data,is_training=True):
        batch_predicted, batch_complete, batch_gt, gen_loss, dis_loss=self.textremoval_model(batch_data, training=is_training)
        outputs_merged = (batch_complete + 1) / 2 * 255
        gt = (batch_gt + 1) / 2 * 255
        _, psnr = mean_psnr(gt, outputs_merged)
        _, ssim = mean_ssim(gt, outputs_merged)
        tf.summary.scalar('train/psnr', psnr)
        tf.summary.scalar('train/ssim', ssim)
        tf.summary.scalar('train_loss/gen_loss', gen_loss)
        tf.summary.scalar('train_loss/dis_loss', dis_loss)
        return gen_loss,dis_loss,psnr,ssim
        
    # def build_validation_model(self, batch_data):
    def build_validation_model(self, batch_data, reuse=True, is_training=False):
        batch_batch = batch_data
        batch_width = int(batch_batch.get_shape().as_list()[2]/5)
        batch_gt = batch_batch[:, :, :batch_width,:] / 127.5 - 1.
        batch_img = batch_batch[:, :, batch_width:batch_width * 2,:] / 127.5 - 1.
        batch_mask = tf.cast(batch_batch[:, :, batch_width*2:batch_width * 3,0:1] > 127.5, tf.float32)
        batch_text = batch_mask
        # process outputs
        stroke_mask1, output1, stroke_mask2, output2 = self.generator(
            batch_img, batch_mask, reuse=reuse, training=is_training,name=self.model_name + '_generator',
            padding='SAME')
        batch_predicted = output2
        
        batch_complete = batch_predicted * batch_mask + batch_gt * (1.-batch_mask)

        _, psnr = mean_psnr((batch_gt+1.)*127.5, (batch_complete+1.)*127.5)
        _, ssim = mean_ssim((batch_gt+1.)*127.5, (batch_complete+1.)*127.5)
        return psnr,ssim

    # def build_optim(self, gen_loss, dis_loss):
    def build_optim(self, gen_loss, dis_loss):
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + '_generator')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_gradient = self.gen_optimizer.compute_gradients(gen_loss, var_list=g_vars)
        d_gradient = self.dis_optimizer.compute_gradients(dis_loss, var_list=d_vars)
        return self.gen_optimizer.apply_gradients(g_gradient), self.dis_optimizer.apply_gradients(d_gradient)


    #def inpaint_model(self,batch_data,is_training=False,reuse=False):
    def textremoval_model(self, batch_data, training=True,reuse=False):
        batch_batch = batch_data
        batch_width = int(batch_batch.get_shape().as_list()[2]/5)
        batch_gt = batch_batch[:, :, :batch_width,:] / 127.5 - 1.
        batch_img = batch_batch[:, :, batch_width:batch_width*2,:] / 127.5 - 1. # raw image with text
        batch_mask = tf.cast(batch_batch[:, :, batch_width*2:batch_width*3,0:1] > 127.5, tf.float32) # text region mask
        batch_text = tf.cast(batch_batch[:, :, batch_width*3:batch_width*4,0:1] > 127.5, tf.float32) # text stroke mask
        # process outputs x1, x2, s1, s2

        stroke_mask1, output1, stroke_mask2, output2 = self.generator(
            batch_img, batch_mask, reuse=reuse, training=training,name=self.model_name + '_generator',
            padding='SAME')
        batch_predicted = output2
        
        batch_complete = batch_predicted * batch_mask + batch_img * (1. - batch_mask)
        if training:
            losses = {}
            l1_alpha = 1.2
            losses['output1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_gt - output1))#
            losses['output1_loss'] += 10. * tf.reduce_mean(tf.abs(batch_gt - output1) * batch_mask)
            losses['output2_loss'] = tf.reduce_mean(tf.abs(batch_gt - output2))#
            losses['output2_loss'] += 10. * tf.reduce_mean(tf.abs(batch_gt - output2) * batch_mask)
            
            losses['stroke_mask1_loss'] = tf.reduce_mean(tf.abs(batch_text - stroke_mask1))# * (1.-bbox_mask))
            losses['stroke_mask1_loss'] += 10. * tf.reduce_mean(tf.abs(batch_text - stroke_mask1) * batch_mask)
            losses['stroke_mask2_loss'] = tf.reduce_mean(tf.abs(batch_text - stroke_mask2))# * (1.-bbox_mask))
            losses['stroke_mask2_loss'] += 10. * tf.reduce_mean(tf.abs(batch_text - stroke_mask2) * batch_mask)
    
            # seperate gan
            batch_pos_feature = self.sngan_discriminator(batch_gt, training=training, reuse=reuse)
            batch_neg_feature = self.sngan_discriminator(batch_complete, training=training, reuse=tf.AUTO_REUSE)
            
            # wgan loss
            loss_discriminator, loss_generator = hinge_gan_loss(batch_pos_feature, batch_neg_feature)
            
            losses['g_loss'] = 0.001 * loss_generator
            losses['d_loss'] = loss_discriminator
    
            losses['g_loss'] = 0.001  * losses['g_loss']
            losses['g_loss'] += 1. * losses['output1_loss']
            losses['g_loss'] += 5. * losses['output2_loss']

            losses['g_loss'] += losses['stroke_mask1_loss']
            losses['g_loss'] += losses['stroke_mask2_loss']
            viz_img = [batch_gt, batch_img, tf.tile(stroke_mask1,[1,1,1,3]), tf.tile(stroke_mask2,[1,1,1,3]), output1, output2]

            images_summary(
                tf.concat(viz_img, axis=2),
                'batchgt_batchimg_strokemask1_strokemask2_output1_output2', 10)
        
            return batch_predicted,batch_complete,batch_gt,losses['g_loss'],losses['d_loss']
        else:
            return batch_predicted,batch_complete,batch_gt,batch_img,batch_mask,batch_text

    def generator(self, image, mask, reuse=False,
                          training=True, padding='SAME', name='generator'):
        """Inpaint network.

        Args:
            image: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """

        image2 = image
        ones_image = tf.ones_like(image)[:, :, :, 0:1]
        image = tf.concat([image, ones_image, ones_image * mask], axis=3)
        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage 1 stroke mask
            t1_conv1 = gen_conv(image, cnum//2, 3, 1, name='t1conv1')
            t1_conv2 = gen_conv(t1_conv1, cnum//2, 3, 1, name='t1conv2')
            t1_conv3 = gen_conv(t1_conv2, cnum, 3, 2, name='t1conv3_128')
            t1_conv4 = gen_conv(t1_conv3, cnum, 3, 1, name='t1conv4')
            t1_conv5 = gen_conv(t1_conv4, cnum, 3, 1, name='t1conv5')
            t1_conv6 = gen_conv(t1_conv5, 2*cnum, 3, 2, name='t1conv6_64')
            t1_conv7 = gen_conv(t1_conv6, 2*cnum, 3, 1, name='t1conv7')
            t1_conv8 = gen_conv(t1_conv7, 2*cnum, 3, 1, name='t1conv8')
            t1_conv9 = gen_conv(t1_conv8, 4*cnum, 3, 2, name='t1conv9_32')
            t1_conv10 = gen_conv(t1_conv9, 4*cnum, 3, 1, name='t1conv10')
            t1_conv11 = gen_deconv(t1_conv10, 2*cnum, name='t1conv11_64')
            t1_conv11 = tf.concat([t1_conv8, t1_conv11], axis=3)
            t1_conv12 = gen_conv(t1_conv11, 2*cnum, 3, 1, name='t1conv12')
            t1_conv13 = gen_conv(t1_conv12, 2*cnum, 3, 1, name='t1conv13')
            t1_conv14 = gen_conv(t1_conv13, 2*cnum, 3, 1, name='t1conv14')
            t1_conv15 = gen_deconv(t1_conv14, cnum, name='t1conv15_128')
            t1_conv15 = tf.concat([t1_conv5, t1_conv15], axis=3)
            t1_conv16 = gen_conv(t1_conv15, cnum, 3, 1, name='t1conv16')
            t1_conv17 = gen_conv(t1_conv16, cnum, 3, 1, name='t1conv17')
            t1_conv18 = gen_conv(t1_conv17, cnum, 3, 1, name='t1conv18')
            t1_conv19 = gen_deconv(t1_conv18, cnum//2, name='t1conv19_256')
            t1_conv19 = tf.concat([t1_conv2, t1_conv19], axis=3)
            t1_conv20 = gen_conv(t1_conv19, cnum//2, 3, 1, name='t1conv20')
            
            stroke_mask1 = gen_conv(t1_conv20, 1, 3, 1, name='stroke_mask1')
            
            # stage 1 output
            xnow = tf.concat([image2, ones_image, ones_image * mask, stroke_mask1 * mask], axis=3)
            s1_conv1 = gen_conv(xnow, cnum, 5, 1, name='conv1')
            s1_conv2 = gen_conv(s1_conv1, 2*cnum, 3, 2, name='conv2_downsample')
            s1_conv3 = gen_conv(s1_conv2, 2*cnum, 3, 1, name='conv3')
            s1_conv4 = gen_conv(s1_conv3, 4*cnum, 3, 2, name='conv4_downsample')
            s1_conv5 = gen_conv(s1_conv4, 4*cnum, 3, 1, name='conv5')
            s1_conv6 = gen_conv(s1_conv5, 4*cnum, 3, 1, name='conv6')

            s1_conv7 = res_block(s1_conv6, name='s1res_block1')
            s1_conv8 = res_block(s1_conv7, name='s1res_block2')
            s1_conv9 = res_block(s1_conv8, name='s1res_block3')
            s1_conv10 = res_block(s1_conv9, name='s1res_block4')

            s1_conv11 = gen_conv(s1_conv10, 4*cnum, 3, 1, name='conv11')
            s1_conv11 = tf.concat([s1_conv6, s1_conv11], axis=3)
            s1_conv12 = gen_conv(s1_conv11, 4*cnum, 3, 1, name='conv12')
            s1_conv12 = tf.concat([s1_conv5, s1_conv12], axis=3)
            s1_conv13 = gen_deconv(s1_conv12, 2*cnum, name='conv13_upsample')
            s1_conv13 = tf.concat([s1_conv3, s1_conv13], axis=3)
            s1_conv14 = gen_conv(s1_conv13, 2*cnum, 3, 1, name='conv14')
            s1_conv14 = tf.concat([s1_conv2, s1_conv14], axis=3)
            s1_conv15 = gen_deconv(s1_conv14, cnum, name='conv15_upsample')
            s1_conv15 = tf.concat([s1_conv1, s1_conv15], axis=3)
            s1_conv16 = gen_conv(s1_conv15, cnum//2, 3, 1, name='conv16')
            s1_conv17 = gen_conv(s1_conv16, 3, 3, 1, activation=None, name='conv17')
            s1_conv = tf.clip_by_value(s1_conv17, -1., 1., name='stage1')
            output1 = s1_conv

            # stage 2 stroke mask
            sin = tf.concat([output1, ones_image, ones_image * mask, stroke_mask1 * mask], axis=3)
            t2_conv1 = gen_conv(sin, cnum//2, 3, 1, name='t2conv1')
            t2_conv2 = gen_conv(t2_conv1, cnum//2, 3, 1, name='t2conv2')
            t2_conv3 = gen_conv(t2_conv2, cnum, 3, 2, name='t2conv3_128')
            t2_conv4 = gen_conv(t2_conv3, cnum, 3, 1, name='t2conv4')
            t2_conv5 = gen_conv(t2_conv4, cnum, 3, 1, name='t2conv5')
            t2_conv6 = gen_conv(t2_conv5, 2*cnum, 3, 2, name='t2conv6_64')
            t2_conv7 = gen_conv(t2_conv6, 2*cnum, 3, 1, name='t2conv7')
            t2_conv8 = gen_conv(t2_conv7, 2*cnum, 3, 1, name='t2conv8')
            t2_conv9 = gen_conv(t2_conv8, 4*cnum, 3, 2, name='t2conv9_32')
            t2_conv10 = gen_conv(t2_conv9, 4*cnum, 3, 1, name='t2conv10')
            t2_conv11 = gen_deconv(t2_conv10, 2*cnum, name='t2conv11_64')
            t2_conv11 = tf.concat([t2_conv8, t2_conv11], axis=3)
            t2_conv12 = gen_conv(t2_conv11, 2*cnum, 3, 1, name='t2conv12')
            t2_conv13 = gen_conv(t2_conv12, 2*cnum, 3, 1, name='t2conv13')
            t2_conv14 = gen_conv(t2_conv13, 2*cnum, 3, 1, name='t2conv14')
            t2_conv15 = gen_deconv(t2_conv14, cnum, name='t2conv15_128')
            t2_conv15 = tf.concat([t2_conv5, t2_conv15], axis=3)
            t2_conv16 = gen_conv(t2_conv15, cnum, 3, 1, name='t2conv16')
            t2_conv17 = gen_conv(t2_conv16, cnum, 3, 1, name='t2conv17')
            t2_conv18 = gen_conv(t2_conv17, cnum, 3, 1, name='t2conv18')
            t2_conv19 = gen_deconv(t2_conv18, cnum//2, name='t2conv19_256')
            t2_conv19 = tf.concat([t2_conv2, t2_conv19], axis=3)
            t2_conv20 = gen_conv(t2_conv19, cnum//2, 3, 1, name='t2conv20')

            stroke_mask2 = gen_conv(t2_conv20, 1, 3, 1, name='stroke_mask2')

            # stage 2 output
            xnow = tf.concat([output1, ones_image, ones_image * mask, stroke_mask2 * mask], axis=3)
            s2c_conv1 = gen_conv(xnow, cnum, 5, 1, name='s2conv1')
            s2c_conv2 = gen_conv(s2c_conv1, cnum, 3, 2, name='s2conv2_downsample')
            s2c_conv3 = gen_conv(s2c_conv2, 2*cnum, 3, 1, name='s2conv3')
            s2c_conv4 = gen_conv(s2c_conv3, 2*cnum, 3, 2, name='s2conv4_downsample')
            s2c_conv5 = gen_conv(s2c_conv4, 4*cnum, 3, 1, name='s2conv5')
            s2c_conv6 = gen_conv(s2c_conv5, 4*cnum, 3, 1, name='s2conv6')

            s2c_conv7 = res_block(s2c_conv6, name='s2res_block1')
            s2c_conv8 = res_block(s2c_conv7, name='s2res_block2')
            s2c_conv9 = res_block(s2c_conv8, name='s2res_block3')
            s2c_conv10 = res_block(s2c_conv9, name='s2res_block4')

            s2_conv11 = gen_conv(s2c_conv10, 4*cnum, 3, 1, name='s2conv11')
            s2_conv11 = tf.concat([s2c_conv6, s2_conv11], axis=3)
            s2_conv12 = gen_conv(s2_conv11, 4*cnum, 3, 1, name='s2conv12')
            s2_conv12 = tf.concat([s2c_conv5, s2_conv12], axis=3)
            s2_conv13 = gen_deconv(s2_conv12, 2*cnum, name='s2conv13_upsample')
            s2_conv13 = tf.concat([s2c_conv3, s2_conv13], axis=3)
            s2_conv14 = gen_conv(s2_conv13, 2*cnum, 3, 1, name='s2conv14')
            s2_conv14 = tf.concat([s2c_conv2, s2_conv14], axis=3)
            s2_conv15 = gen_deconv(s2_conv14, cnum, name='s2conv15_upsample')
            s2_conv15 = tf.concat([s2c_conv1, s2_conv15], axis=3)
            s2_conv16 = gen_conv(s2_conv15, cnum//2, 3, 1, name='s2conv16')
            s2_conv17 = gen_conv(s2_conv16, 3, 3, 1, activation=None, name='s2conv17')
            output2 = tf.clip_by_value(s2_conv17, -1., 1., name='output')
        return stroke_mask1, output1, stroke_mask2, output2

    def sngan_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 64
            x = sndis_conv(x, cnum, 5, 1, name='conv1', training=training)
            x = sndis_conv(x, cnum*2, 5, 2, name='conv2', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv3', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv4', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv5', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv6', training=training)
            return x

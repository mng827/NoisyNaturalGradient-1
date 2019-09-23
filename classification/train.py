from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from core.base_train import BaseTrain
from tqdm import tqdm

import numpy as np
from misc import image_utils, distribution_stats, utils


class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            self.test_epoch()

            if (cur_epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.model.save(self.sess)

    def train_epoch(self):
        total_loss_list = []
        loss_list = []
        acc_list = []
        dice_list = []

        for itr, (x, y, _, _, _) in enumerate(tqdm(self.train_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles
            }
            feed_dict.update({self.model.is_training: True})
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            feed_dict.update({self.model.is_training: False})  # Note: that's important
            total_loss, loss, acc, dice = self.sess.run([self.model.total_loss, self.model.loss, self.model.acc, self.model.dice], feed_dict=feed_dict)

            total_loss_list.append(total_loss)
            loss_list.append(loss)
            acc_list.append(acc)
            dice_list.append(dice)

            cur_iter = self.model.global_step_tensor.eval(self.sess)

            if cur_iter % self.config.get('TCov', 10) == 0 and \
                    self.model.cov_update_op is not None:
                self.sess.run([self.model.cov_update_op], feed_dict=feed_dict)

                if self.config.optimizer == "diag":
                    self.sess.run([self.model.var_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.get('TInv', 200) == 0 and \
                    self.model.inv_update_op is not None:
                self.sess.run([self.model.inv_update_op, self.model.var_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.get('TEigen', 200) == 0 and \
                    self.model.eigen_basis_update_op is not None:
                self.sess.run([self.model.eigen_basis_update_op, self.model.var_update_op], feed_dict=feed_dict)

                if self.config.kfac_init_after_basis:
                    self.sess.run(self.model.re_init_kfac_scale_op)

            if cur_iter % self.config.get('TScale', 10) == 0 and \
                    self.model.scale_update_op is not None:
                self.sess.run([self.model.scale_update_op, self.model.var_scale_update_op], feed_dict=feed_dict)

        avg_total_loss = np.mean(total_loss_list)
        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        avg_dice = np.mean(dice_list, axis=0)

        self.logger.info("train | total_loss: %5.6f | loss: %5.6f | accuracy: %5.6f | dice: %s"
                         %(float(avg_total_loss), float(avg_loss), float(avg_acc), avg_dice))

        # Summarize
        summaries_dict = dict()
        summaries_dict['train_total_loss'] = avg_total_loss
        summaries_dict['train_loss'] = avg_loss
        summaries_dict['train_acc'] = avg_acc
        for i in range(self.model.num_classes):
            summaries_dict['train_dice_{}'.format(i)] = avg_dice[i]


        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def test_epoch(self):
        total_loss_list = []
        loss_list = []
        acc_list = []
        dice_list = []

        for itr, (x, y, _, _, _) in enumerate(tqdm(self.test_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: 1
            }

            for _ in range(self.config.test_particles):
                total_loss, loss, acc, dice = self.sess.run([self.model.total_loss, self.model.loss, self.model.acc, self.model.dice], feed_dict=feed_dict)

                total_loss_list.append(total_loss)
                loss_list.append(loss)
                acc_list.append(acc)
                dice_list.append(dice)

        avg_total_loss = np.mean(total_loss_list)
        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        avg_dice = np.mean(dice_list, axis=0)

        self.logger.info("test | total_loss: %5.6f | loss: %5.6f | accuracy: %5.6f | dice: %s\n"
                         %(float(avg_total_loss), float(avg_loss), float(avg_acc), avg_dice))

        # Summarize
        summaries_dict = dict()
        summaries_dict['test_total_loss'] = avg_total_loss
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc
        for i in range(self.model.num_classes):
            summaries_dict['test_dice_{}'.format(i)] = avg_dice[i]

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def test_epoch_with_misc_metrics(self):
        loss_list = []
        acc_list = []
        dice_list = []

        stats_logger = utils.StatsLogger()

        for itr, (x, y, info, whole_x, whole_y) in enumerate(tqdm(self.test_loader)):
            n_particles = 1
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: n_particles
            }

            logits_list = []

            for _ in range(int(self.config.test_particles / n_particles)):
                logits, loss, acc, dice = self.sess.run([self.model.logits, self.model.loss, self.model.acc, self.model.dice], feed_dict=feed_dict)

                print("Loss: {}".format(loss))
                print("Dice: {}".format(dice))

                loss_list.append(loss)
                acc_list.append(acc)
                dice_list.append(dice)

                logits = np.reshape(logits, [n_particles, -1] + self.model.input_dim + [self.model.num_classes]) # [n_particles, batch_size, 256, 256, 1, num_classes]
                logits_list.extend(np.split(logits, n_particles, axis=0)) # [1, batch_size, 256, 256, 1, num_classes]

            prob_samples = [utils.softmax(sample, axis=-1) for sample in logits_list] # [1, batch_size, 256, 256, 1, num_classes]

            x, y = prob_samples[0].shape[2:4]
            x_centre, y_centre = int(x / 2), int(y / 2)

            whole_prob_samples = []

            for sample in prob_samples:
                sample = np.transpose(sample[0,:,:,:,0,:], axes=(1,2,0,3))
                whole_sample = np.zeros(np.concatenate([info['ImageSize'][0:2], sample.shape[2:]]))

                whole_sample[:,:,:,0] = image_utils.crop_image(sample[:,:,:,0], x_centre, y_centre, size=info['ImageSize'][0:2], constant_values=1)
                for c in range(1, self.model.num_classes):
                    whole_sample[:,:,:,c] = image_utils.crop_image(sample[:,:,:,c], x_centre, y_centre, size=info['ImageSize'][0:2], constant_values=0)

                whole_prob_samples.append(whole_sample)

            dice_mean_to_samples_per_slice = distribution_stats.get_dice_mean_to_samples(whole_prob_samples,
                                                                                         self.model.num_classes)
            assd_mean_to_samples_per_slice = distribution_stats.get_assd_mean_to_samples(whole_prob_samples,
                                                                                         self.model.num_classes,
                                                                                         info['PixelResolution'][0:2])
            whole_pred = np.argmax(np.mean(whole_prob_samples, axis=0), axis=-1)
            whole_y = whole_y[0]

            dice_per_class_per_slice = image_utils.np_categorical_dice(whole_pred, whole_y, self.model.num_classes,
                                                                       axis=(0, 1), smooth_epsilon=1e-5)

            assd_per_class_per_slice, hd_per_class_per_slice = image_utils.np_categorical_assd_hd_per_slice(whole_pred,
                                                                                                            whole_y,
                                                                                                            self.model.num_classes,
                                                                                                            info['PixelResolution'][0:2],
                                                                                                            fill_nan=True)

            stats_logger.append(info, dice_mean_to_samples_per_slice, assd_mean_to_samples_per_slice, dice_per_class_per_slice, assd_per_class_per_slice)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        avg_dice = np.mean(dice_list, axis=0)

        self.logger.info("test | loss: %5.6f | accuracy: %5.6f | dice: %s\n"%(float(avg_loss), float(avg_acc), avg_dice))

        # Summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc
        for i in range(self.model.num_classes):
            summaries_dict['test_dice_{}'.format(i)] = avg_dice[i]

        stats_logger.save(os.path.join(self.config.log_dir, 'results'))

    def load_checkpoint(self, checkpoint):
        self.model.saver.restore(self.sess, checkpoint)
        print("Loaded {}".format(checkpoint))

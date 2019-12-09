### Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
### Licensed under the Nvidia Source Code License.
import numpy as np
import os
import ntpath
import time
import glob
import scipy.misc
from io import BytesIO
from util import util
from util import html
from util.distributed import master_only_print as print

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_visdom = opt.use_visdom
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize #* opt.aspect_ratio
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_visdom:
            import visdom
            self.vis = visdom.Visdom()
            self.visdom_id = opt.visdom_id

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            if hasattr(opt, 'model_idx') and opt.model_idx != -1:
                self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_%03d.txt' % opt.model_idx)
            else:
                self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_visdom_results(self, visuals, epoch, step):
        ncols = self.ncols
        if ncols > 0:
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = """<style>
                    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)
            title = self.name
            label_html = ''
            label_html_row = ''
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                label_html_row += '<td>%s</td>' % label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % ncols != 0:
                images.append(white_image)
                label_html_row += '<td></td>'
                idx += 1
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row
            # pane col = image row
            self.vis.images(images, nrow=ncols, win=self.visdom_id + 1,
                            padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=self.visdom_id + 2,
                          opts=dict(title=title + ' labels'))


    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.use_visdom:
            self.display_visdom_results(visuals, epoch, step)
            
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if image_numpy is None: continue
                ext = 'png' if 'label' in label else 'jpg'
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%03d_iter%07d_%s_%d.%s' % (epoch, step, label, i, ext))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%03d_iter%07d_%s.%s' % (epoch, step, label, ext))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]                    
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=300)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if image_numpy is None: continue
                    ext = 'png' if 'label' in label else 'jpg'
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            if n == epoch:
                                img_path = 'epoch%03d_iter%07d_%s_%d.%s' % (n, step, label, i, ext)
                            else:
                                img_paths = sorted(glob.glob(os.path.join(self.img_dir, 'epoch%03d_iter*_%s_%d.%s' % (n, label, i, ext))))
                                img_path = os.path.basename(img_paths[-1]) if len(img_paths) else 'img.jpg'
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        if n == epoch:
                            img_path = 'epoch%03d_iter%07d_%s.%s' % (n, step, label, ext)
                        else:
                            img_paths = sorted(glob.glob(os.path.join(self.img_dir, 'epoch%03d_iter*_%s.%s' % (n, label, ext))))
                            img_path = os.path.basename(img_paths[-1]) if len(img_paths) else 'img.jpg'
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 6:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            if image_numpy is None: continue
            ext = 'png' if 'label' in label else 'jpg'
            image_name = os.path.join(label, '%s.%s' % (name, ext))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    @staticmethod
    def vis_print(opt, message):
        print(message)
        if opt.isTrain and not opt.debug:
            log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)
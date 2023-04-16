from options.test_options import TestOptions
from models.models import create_model

import imageio
from tqdm import tqdm
from skimage.transform import resize
import torch
import numpy as np
from skimage import img_as_ubyte

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def show(img):
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


reader = imageio.get_reader('demo_video_trim.mp4')
fps = reader.get_meta_data()['fps']
frame_list = []
frame_raw_list = []
for im in tqdm(reader, desc="reading video", total=reader.count_frames()):
    frame_list.append(resize(im, (256, 256), order=3))
    frame_raw_list.append(im/256)
reader.close()
frame_shape = frame_list[0].shape[:2]
frame_raw_shape = frame_raw_list[0].shape[:2]

opt = TestOptions()
opt.initialize()

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False

opt.dataset_mode = "aligned"
opt.dataroot = "./"
# opt.dataroot = "./datasets/soccer_seg_detection"
opt.phase = "test"
opt.resize_or_crop = 'resize_and_crop'
opt.model = "two_pix2pix"
opt.gpu_ids = [0, 1]
opt.isTrain = False
opt.checkpoints_dir = "checkpoints"
opt.name = "soccer_seg_detection_pix2pix"
opt.input_nc = 3
opt.output_nc = 1
opt.ngf = 64
opt.which_model_netG = "unet_256"
opt.norm = "batch"
opt.no_dropout = True
opt.init_type = 'normal'
opt.which_direction = "AtoB"
opt.how_many = 186
opt.loadSize = 256
opt.which_epoch = "latest"

model = create_model(opt)
model.initialize(opt)

result_masked = []
result_line = []
i = 0
frame = frame_list[0]
for i, frame in tqdm(enumerate(frame_list), total=len(frame_list)):
    frame = np.transpose(frame*2-1, (2, 0, 1))
    frame = torch.FloatTensor(frame).cuda()
    frame = frame.unsqueeze(0)

    seg_result = model.seg_netG(frame)
    masked = (((frame + 1) / 2) * ((seg_result + 1) / 2)) * 2.0 - 1

    seg_tmp = np.transpose(seg_result.squeeze(0).cpu().numpy(force=True), (1, 2, 0))
    seg_tmp = resize(seg_tmp, frame_raw_shape, order=3)
    masked_tmp = (((frame_raw_list[i] + 1) / 2) * ((seg_tmp + 1) / 2)) * 2.0 - 1

    result_masked.append(masked_tmp)

    det_result = model.detec_netG(masked)
    det_result = (det_result + 1) / 2

    res_frame = np.tile(np.transpose(det_result.squeeze(0).cpu().numpy(force=True), (1, 2, 0)), (1, 1, 3))
    result_line.append(resize(res_frame, frame_raw_shape, order=3))

imageio.mimsave("masked_demo.mp4", [img_as_ubyte(p) for p in tqdm(result_masked, desc="generating masked video", total=len(result_masked))], fps=fps)
imageio.mimsave("line_demo.mp4", [img_as_ubyte(p) for p in tqdm(result_line, desc="generating line video", total=len(result_line))], fps=fps)

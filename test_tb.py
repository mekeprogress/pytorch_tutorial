import cv2 as cv

from torch.utils.tensorboard import SummaryWriter

image_path = 'dataset/train/ants_image/5650366_e22b7e1065.jpg'
img = cv.imread(image_path)

writer = SummaryWriter("logs")
writer.add_image('test', img, 2,dataformats='HWC')

writer.close()

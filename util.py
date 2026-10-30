import logging
from colorlog import ColoredFormatter
import os


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)


def check_path(path):
    os.makedirs(path, exist_ok=True)

def fill_image(all_imgs, idx, full_img, p):
	h = all_imgs[idx].shape[0]
	w = all_imgs[idx].shape[1]
	x = p[0]
	y = p[1]
	if h%2==0 and w%2!=0:
		full_img[x- h//2:x+ h//2 , y- w//2:y+ w//2 +1 ] = all_imgs[idx]
	elif h%2!=0 and w%2==0:
		full_img[x- h//2:x+ h//2 +1 , y- w//2:y+ w//2  ] = all_imgs[idx]
	elif h%2!=0 and w%2!=0:
		full_img[x- h//2:x+ h//2 +1 , y- w//2:y+ w//2 +1 ] = all_imgs[idx]
	else:
		full_img[x- h//2:x+ h//2 , y- w//2:y+ w//2  ] = all_imgs[idx]

	return full_img

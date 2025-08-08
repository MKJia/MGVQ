import numpy as np
from PIL import Image

def resize_center_crop(img_path_data, test_res_w, test_res_h):
        # resize to eval_res
        scale = test_res_w / test_res_h
        im_pil = Image.open(img_path_data).convert('RGB')
        w, h = im_pil.size
        scale_im = w / h
        if scale_im < scale:
            h_1 = round(h / w * test_res_w) 
            im = np.array(im_pil.resize((test_res_w, h_1), resample=Image.BICUBIC))
        else:
            w_1 = round(w / h * test_res_h) 
            im = np.array(im_pil.resize((w_1, test_res_h), resample=Image.BICUBIC)) 
        # center crop
        ih, iw, _ = im.shape
        if iw == test_res_w:
            x = int(ih/2-test_res_h/2)
            y = 0
            assert y + test_res_w == iw
            im = im[x:x+test_res_h, y:y+test_res_w, :]
        else: # ih == test_res_h
            x = 0
            y = int(iw/2-test_res_w/2)
            assert x + test_res_h == ih
            im = im[x:x+test_res_h, y:y+test_res_w, :]
        return im
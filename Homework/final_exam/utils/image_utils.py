import os
import traceback

import numpy as np
import cv2

def load_dataset2(img_dir, categories, new_size = None, standardize = True, fit_new_shape = False):
    if not os.path.isdir(img_dir): # check if the init_dir exist or not
        return [], []
    
    image_file_info = []
    for root, dirs, files in os.walk(img_dir):
        image_file_info.extend([ (root, f) for f in files if f.endswith('.jpg') ])
    

        try:
            # img01 = cv2.imread(os.path.join(image_dir, image_file_name)) # type of img01 = 'numpy.ndarray'

            # alternative way for reading the image if the file path contains the Chinese characters. !!!
            img01 = cv2.imdecode(np.fromfile(os.path.join(image_dir, image_file_name), dtype = np.uint8), -1)
            if img01 is None:
                print(f"img01 is None !! { os.path.join(image_dir, image_file_name) }")
                continue
           
            if img01.ndim != 3:
                print(f"img01.ndim != 3 !! { img01.shape }, { os.path.join(image_dir, image_file_name) }")
                continue

            # True, 只過濾與 new_size 相同的, 正方形, height > width, height < width 的三種
            if fit_new_shape:  
                if shape == 0:    # 要正方形, 不是的離開
                    if img01.shape[0] != img01.shape[1]: 
                        continue
                elif shape == 1:  # 要 height > width 的, 不是的離開
                    if img01.shape[0] <= img01.shape[1]: 
                        continue
                else:             # 要 height < width 的, 不是的離開
                    if img01.shape[0] >= img01.shape[1]: 
                        continue
            
            #print(f"{ img01.shape }, { os.path.join(image_dir, image_file_name) }")

            img02 = resize_img(img = img01, new_height = new_size[0], new_width = new_size[1])
            img02 = img02.astype('float32')
            img03 = std_img(img02) if standardize else img02
            if img03 is None:
                continue
            else:
                images.append(img03)
                label = categories[folder_name.upper()]
                labels.append(label)
        except:
            ex = traceback.format_exc()
            print(ex) # log the error

    return images, labels

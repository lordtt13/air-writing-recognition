# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:00:55 2020

@author: Tanmay Thakur
"""
import numpy as np
import pickle

from collections import deque


letter_count = {0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna', 6: 'character_06_cha',
                    7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna',
                    11: '11_taamatar',
                    12: '12_thaa', 13: '13_daa', 14: '14_dhaa', 15: '15_adna', 16: '16_tabala', 17: '17_tha',
                    18: '18_da',
                    19: '19_dha', 20: '20_na', 21: '21_pa', 22: '22_pha',
                    23: '23_ba',
                    24: '24_bha', 25: '25_ma', 26: '26_yaw', 27: '27_ra', 28: '28_la', 29: '29_waw', 30: '30_motosaw',
                    31: '31_petchiryakha',32: '32_patalosaw', 33: '33_ha',
                    34: '34_chhya', 35: '35_tra', 36: '36_gya', 37: 'CHECK'}


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

pickle_out = open("dict.pickle","wb")
pickle.dump(letter_count, pickle_out)
pickle_out.close()
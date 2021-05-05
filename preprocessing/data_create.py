import os
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')

def select_images(mask_list, img_list):
    filtered_img_list = []
    filtered_mask_list = []
    print(len(img_list))
    cv2.namedWindow('kokoti', cv2.WINDOW_AUTOSIZE)
    for img_index, image in enumerate(img_list):
        if image.shape[0] > 0 and image.shape[1] > 0:
            while True:
                cv2.imshow('kokoti', image)
                key = cv2.waitKey(0) & 0xFF
                print(key)
                if key == 102:  # f key - aprove img
                    filtered_img_list.append(image)
                    filtered_mask_list.append(mask_list[img_index])
                    break
                elif key == 97:    #a key for interupt
                    return filtered_img_list, filtered_mask_list
                else:
                    break
    return filtered_img_list, filtered_mask_list

def create_klestik_img(final_size,size_img,mask_list,img_list):
    full_img_mask = np.zeros(shape=(final_size * size_img, final_size * size_img))
    k = 0
    for i in range(0, final_size):
        for j in range(0, final_size):
            try:
                full_img_mask[i * size_img:i * size_img + size_img, j * size_img:j * size_img + size_img] = mask_list[k]
            except:
                print("klestik na kraji")
            k = k + 1

    matplotlib.image.imsave('tst_tenzor_mask.jpg', full_img_mask)


    full_img_img = np.zeros(shape=(final_size * size_img, final_size * size_img, 3),dtype=np.uint8)
    k = 0
    for i in range(0, final_size):
        for j in range(0, final_size):
            try:
                full_img_img[i * size_img:i * size_img + size_img, j * size_img:j * size_img + size_img] = img_list[k]
            except:
                print("klestik na kraji")
            k = k + 1

    matplotlib.image.imsave('tst_tenzor_klestici.jpg', full_img_img)
    return 0

# def mask_create_add(mask_list):
#     retyped = []
#     for i in mask_list:
#         retyped.append(i.astype("float32"))
#     template=sum(retyped)/len(retyped)
#     return template/template.max()
#
#
# def mask_create_multiply(mask_list):
#     retyped = []
#     for i in mask_list:
#         retyped.append(i.astype("float32"))
#     template=retyped[0]
#     for index in range(1,len(retyped)):
#         template=template*retyped[index]
#     return template

def create_rgb_template(image_list, mask_list):
    if len(image_list) == len(mask_list):
        print('OK')
    pure_klestik_list=[]
    for i in range(0, len(image_list)-1):
        pure_klestik_list.append(cv2.bitwise_and(image_list[i], image_list[i], mask=mask_list[i]))

    pure_klestik_clist=[]
    for val in pure_klestik_list:
        if type(val).__module__ == np.__name__:
            pure_klestik_clist.append(val)

    img_add = pure_klestik_clist[0]
    img_test = np.array(img_add, np.uint32)
    klestik_counter = 1
    for j in range(1,len(pure_klestik_clist)-1):
        if pure_klestik_clist[j].shape[0] == 60 and pure_klestik_clist[j].shape[1] == 60:
            img_test += pure_klestik_clist[j]
            klestik_counter += 1
    img_div = img_test / klestik_counter
    im_show = np.array(img_div, np.uint8)

    return im_show

if __name__ == '__main__':

    # relevant_path_mask = "../MASK-sieberm/SegmentationClass"
    # relevant_path_img = "../MASK-sieberm/JPEGImages"
    relevant_path_mask = "../MASK-sieberm2/SegmentationClass"
    relevant_path_img = "../MASK-sieberm2/JPEGImages"
    extension_jpg = ['jpg']
    extension_png = ['png']
    img_names = [fn for fn in os.listdir(relevant_path_img)
                  if any(fn.endswith(ext) for ext in extension_jpg)]
    mask_names = [fn for fn in os.listdir(relevant_path_mask)
                  if any(fn.endswith(ext) for ext in extension_png)]

    print(img_names)
    print(mask_names)

    neighborhood = 30
    size_img = 2*neighborhood

    mask_list = []
    img_list = []

    help_list = []
    cutoff_mask = []
    cutoff_jpg = []

    for index in range(0, len(mask_names)):
        im = cv2.imread(relevant_path_mask+'/'+mask_names[index])         # mask
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #grey mask

        im2_unrot = cv2.imread(relevant_path_img + '/' + img_names[index])  # img
        im2= cv2.rotate(im2_unrot, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY)  # klestici v masce
        countours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(countours))
        help_list=[]
        img_s = True
        for c in countours:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # vyrez obrazu masky
            mat = imgray[cY - neighborhood:cY + neighborhood, cX - neighborhood:cX + neighborhood]
            mask_list.append(mat)
            mat2 = im2[cY - neighborhood:cY + neighborhood, cX - neighborhood:cX + neighborhood]
            if img_s:
                cv2.imshow('a',mat2)
                cv2.waitKey(0)
                print(img_names[index])
                img_s = False
            img_list.append(mat2)
            help_list.append([cY,cX])

        # stred_cX = int(np.mean([elem[0] for elem in help_list]))
        # stred_cY = int(np.mean([elem[1] for elem in help_list]))
        #
        # cutoff_jpg.append(im2[stred_cX-200:stred_cX+200][:])
        # cutoff_mask.append(im[stred_cX-200:stred_cX+200][:])



    filter_img_list, filter_mask_list = select_images(mask_list, img_list)
    np.save('filter_img.npy', filter_img_list)
    np.save('filter_mask.npy', filter_mask_list)

    # create_klestik_img(10, size_img, filter_mask_list, filter_img_list)


    # concat_mask = np.concatenate(cutoff_mask)
    # concat_img = np.concatenate(cutoff_jpg)

    # plt.figure()
    # plt.imshow(concat_mask)
    # plt.figure()
    # plt.imshow(concat_img)
    # plt.show()

    # cv2.imwrite('../images/concat_jpg.png',concat_img)
    # cv2.imwrite('../images/concat_mask.png',concat_mask)

    # rgb_template=create_rgb_template(img_list,mask_list)
    # cv2.imshow('klestik', rgb_template)
    # cv2.waitKey(0)
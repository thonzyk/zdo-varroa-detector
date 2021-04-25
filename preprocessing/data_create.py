import os
import cv2
import numpy as np


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

    relevant_path_mask = "../MASK-sieberm/SegmentationClass"
    relevant_path_img = "../MASK-sieberm/JPEGImages"
    extension_jpg = ['jpg']
    extension_png = ['png']
    img_names = [fn for fn in os.listdir(relevant_path_img)
                  if any(fn.endswith(ext) for ext in extension_jpg)]
    mask_names = [fn for fn in os.listdir(relevant_path_mask)
                  if any(fn.endswith(ext) for ext in extension_png)]

    print(img_names)
    print(mask_names)

    neighborhood=30
    size_img=2*neighborhood

    mask_list=[]
    img_list=[]

    for index in range(0, len(mask_names)):
        im = cv2.imread(relevant_path_mask+'/'+mask_names[index])         # mask
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #grey mask

        im2_unrot = cv2.imread(relevant_path_img + '/' + img_names[index])  # img
        im2= cv2.rotate(im2_unrot, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY)  # klestici v masce
        _, countours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(countours))

        for c in countours:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # vyrez obrazu masky
            mat = imgray[cY - neighborhood:cY + neighborhood, cX - neighborhood:cX + neighborhood]
            mask_list.append(mat)
            mat2=im2[cY - neighborhood:cY + neighborhood, cX - neighborhood:cX + neighborhood]
            # cv2.imshow('a',mat2)
            # cv2.waitKey(0)
            img_list.append(mat2)

    rgb_template=create_rgb_template(img_list,mask_list)
    cv2.imshow('klestik', rgb_template)
    cv2.waitKey(0)
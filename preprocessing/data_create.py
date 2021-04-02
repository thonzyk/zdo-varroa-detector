import os
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def mask_create_add(mask_list):
    retyped=[]
    for i in mask_list:
        retyped.append(i.astype("float32"))
    template=sum(retyped)/len(retyped)
    return template/template.max()


def mask_create_multiply(mask_list):
    retyped = []
    for i in mask_list:
        retyped.append(i.astype("float32"))
    template=retyped[0]
    for index in range(1,len(retyped)):
        template=template*retyped[index]
    return template

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

    for index in range(0,len(mask_names)):
        im = cv2.imread(relevant_path_mask+'/'+mask_names[index])         #mask
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #grey mask

        im2_rot = cv2.imread(relevant_path_img + '/' + img_names[index])  # img
        im2= cv2.rotate(im2_rot,cv2.ROTATE_90_COUNTERCLOCKWISE)

        ret, thresh = cv2.threshold(imgray,20,255,cv2.THRESH_BINARY)
        _, countours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(countours))

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

    #np.save('maska_tenzor',mask_list)


    final_size=12

    full_img_mask=np.zeros(shape=(final_size*size_img,final_size*size_img))
    k=0
    for i in range(0,final_size):
        for j in range(0, final_size):
            try:
                full_img_mask[i*size_img:i*size_img+size_img,j*size_img:j*size_img+size_img]=mask_list[k]
            except:
                print("klestik na kraji")
            k = k + 1

    matplotlib.image.imsave('tenzor_mask.jpg', full_img_mask)

    full_img_img=np.zeros(shape=(final_size*size_img,final_size*size_img,3))
    k=0
    for i in range(0,final_size):
        for j in range(0, final_size):
            try:
                full_img_img[i*size_img:i*size_img+size_img,j*size_img:j*size_img+size_img]=img_list[k]
            except:
                print("klestik na kraji")
            k=k+1

    cv2.imwrite('tenzor_klestici.jpg', full_img_img)

    #tmp1=mask_create_add(mask_list)
    #tmp2=mask_create_multiply(mask_list)

    # np.save('mask_add',tmp1)
    # np.save('mask_multiply',tmp2)

print()
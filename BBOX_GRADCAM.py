#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
try:
    import google.colab
    from google.colab.patches import cv2_imshow
except:
    from cv2 import imshow as cv2_imshow
import matplotlib.pyplot as plt
import os
import numpy as np

class BBoxerwGradCAM():
    
    def __init__(self,learner,heatmap,image_path,resize_scale_list,bbox_scale_list):
        self.learner = learner
        self.heatmap = heatmap
        self.image_path = image_path
        self.resize_list = resize_scale_list
        self.scale_list = bbox_scale_list
        
        self.og_img, self.smooth_heatmap = self.heatmap_smoothing()
        
        self.bbox = self.form_bbox()
        
        self.get_bbox()
        
    def heatmap_smoothing(self):
        og_img = cv2.imread(self.image_path)
        heatmap = cv2.resize(self.heatmap, (self.resize_list[0],self.resize_list[1]))
        og_img = cv2.resize(og_img, (self.resize_list[0],self.resize_list[1]))
        #plt.matshow(heatmap)
        #plt.show()
        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        return og_img, heatmapshow
    
    def show_smoothheatmap(self):
        cv2_imshow(self.smooth_heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def show_bboxonimage(self):
        cv2.rectangle(self.og_img,(self.bbox[0],self.bbox[1]),(self.bbox[0]+self.bbox[2],self.bbox[1]+self.bbox[3]),
                      (0,255,0),2)
        cv2_imshow(self.og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def form_bbox(self):
        img = cv2.cvtColor(self.smooth_heatmap, cv2.COLOR_BGR2GRAY);
        ret,thresh = cv2.threshold(img,127,255,0)
        im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)

        for item in range(len(contours)):
            cnt = contours[item]
            if len(cnt)>20:
                #print(len(cnt))
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)

                '''
                x, y is the top left corner, and w, h are the width and height respectively

                '''
                x = int(x*self.scale_list[0])
                y = int(y*self.scale_list[1])
                w = int(w*self.scale_list[2])
                h = int(h*self.scale_list[3])

                return [x,y,w,h]
            
    def get_bbox(self):
        return self.bbox


# TSAFN
Texture and Structure Awareness Network

This network contains three sub-models, which are TPN, SPN and TSAFN. Corresponding paper is

[Lu, K., You, S., & Barnes, N. (2018, September). Deep Texture and Structure Aware Filtering Network for Image Smoothing. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 217-233).](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kaiyue_Lu_Deep_Texture_and_ECCV_2018_paper.pdf)

Be carefult that, to fully functionize the model, user must create their own dataset based on the description from the paper. The dataset is quite rigirous due to generalization problems. Thus, when making the dataset, please always think about if the data will make the model overfitting to the object rather than generalize on the strucure and texture.


The so-called SPN is same as HED which comes from paper:

[Xie, S., & Tu, Z. (2015). Holistically-nested edge detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 1395-1403).](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)


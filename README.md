# Awesome MXNet [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/jtoy/awesome)

<p align="center">
    <a href="https://mxnet.incubator.apache.org/"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png"></a><br>
	<img src="https://img.shields.io/badge/stars-750+-brightgreen.svg?style=flat"/>
	<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
</p>

A curated list of MXNet examples, tutorials, papers, conferences and blogs.

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.

## Table of Contents
- [1. Tutorials and Resources](#Tutorials)
- [2. Vision](#Vision)
- [3. NLP](#NLP)
- [4. Speech](#Speech)
- [5. Time series forecasting](#Time_series_forecasting)
- [6. Spatiotemporal](#Spatiotemporal )
- [7. CTR](#CTR)
- [8. DRL](#DRL)
- [9. Neuro Evolution](#Neuro_evolution)
- [10. One Class Learning](#One_class_learning)
- [11. Probabilistic Programming](#PPL)
- [12. Transfer Learning](#TL)
- [13. Tools](#Tools)

________________


## <a name="Tutorials and Resources"></a>1. Tutorials and Resources
- [Documents](https://mxnet.incubator.apache.org/) [[site]](https://github.com/apache/incubator-mxnet-site)
- Tutorial Documentation [[English]](https://mxnet.incubator.apache.org/tutorials/) [[Chinese]](https://github.com/wangx404/symbol_coding_tutorials_of_MXNet)
- [New version of Documentation](https://github.com/mli/new-docs)
- Gluon Tutorial Documentation [[English]](http://en.diveintodeeplearning.org//) [[Chinese]](https://zh.diveintodeeplearning.org/) [[Japanese]](https://github.com/harusametime/mxnet-the-straight-dope-ja)
- [Gluon Api](https://github.com/gluon-api/gluon-api)
- [CheatSheet](https://github.com/chinakook/Awesome-MXNet/blob/master/apache-mxnet-cheat.pdf)
- [Using MXNet](https://github.com/JONGGON/Mxnet_Tutorial)
- [TVM Documentation](http://docs.tvmlang.org/)
- [NNVM Documentation](http://nnvm.tvmlang.org/)
- [Linalg examples](https://github.com/ARCambridge/MXNet_linalg_examples)
- [NNVM Vison Demo](https://github.com/masahi/nnvm-vision-demo)
- [im2rec_tutorial](https://github.com/leocvml/mxnet-im2rec_tutorial)
- [MXNet Blog (Chinese)](https://zh.mxnet.io/blog/)
- MXNet Discuss Forum [[English]](https://discuss.mxnet.io/) [[Chinese]](https://discuss.gluon.ai/)
- [r/mxnet subreddit](https://www.reddit.com/r/mxnet/)
- Apache MXNet youtube channel [[English]](https://www.youtube.com/channel/UCQua2ZAkbr_Shsgfk1LCy6A) [[Chinese]](https://www.youtube.com/channel/UCjeLwTKPMlDt2segkZzw2ZQ)
- [GluonCV](http://gluon-cv.mxnet.io/)
- [GluonNLP](http://gluon-nlp.mxnet.io/)

## <a name="Vision"></a>2. Vision
### 2.1 Image Classification
 - ResNet [[sym]](https://github.com/tornadomeet/ResNet) [[gluon]](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/resnet.py) [[v1b]](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py)
 - DenseNet [[sym]](https://github.com/bruinxiong/densenet.mxnet) [[gluon]](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/densenet.py)
 - SENet [[sym]](https://github.com/bruinxiong/SENet.mxnet) [[gluon]](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/se_resnet.py) [[caffe]](https://github.com/IIMarch/SENet-mxnet)
 - [MobileNet V3](https://github.com/AmigoCDT/MXNet-MobileNetV3)
 - Xception [[sym]](https://github.com/bruinxiong/xception.mxnet) [[gluon]](https://github.com/osmr/imgclsmob/blob/master/gluon/gluoncv2/models/xception.py) [[Keras]](https://github.com/u1234x1234/mxnet-xception)
 - [DPN](https://github.com/cypw/DPNs)
 - [CapsNet](https://github.com/Soonhwan-Kwon/capsnet.mxnet)
 - [NASNet-A(Gluon:star:)](https://github.com/qingzhouzhen/incubator-mxnet/blob/nasnet/python/mxnet/gluon/model_zoo/vision/nasnet.py)
 - [CRU-Net](https://github.com/cypw/CRU-Net)
 - [ShuffleNet v1/v2](https://github.com/Tveek/mxnet-shufflenet)
 - [**IGCV3**](https://github.com/homles11/IGCV3)
 - [SqeezeNet](https://github.com/miaow1988/SqueezeNet_v1.2)
 - [FractalNet](https://github.com/newuser-16824/mxnet-fractalnet)
 - [BMXNet](https://github.com/hpi-xnor/BMXNet)
 - [**BMXNet v2**](https://github.com/hpi-xnor/BMXNet-v2)
 - [fusenet](https://github.com/zlmzju/fusenet)
 - [Self-Norm Nets](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/SelfNormNets)
 - [Factorized-Bilinear-Network](https://github.com/lyttonhao/Factorized-Bilinear-Network)
 - [AOGNet](https://github.com/xilaili/AOGNet)
 - [NonLocal+SENet](https://github.com/WillSuen/NonLocalandSEnet)
 - [mixup](https://github.com/unsky/mixup)
 - [sparse-structure-selection](https://github.com/TuSimple/sparse-structure-selection)
 - [neuron-selectivity-transfer](https://github.com/TuSimple/neuron-selectivity-transfer)
 - [L-GM-Loss](https://github.com/LeeJuly30/L-GM-Loss-For-Gluon)
 - [**CoordConv**](https://github.com/ZwX1616/mxnet-CoordConv)
 - [IBN-Net](https://github.com/bruinxiong/IBN-Net.mxnet)
 - [Mnasnet](https://github.com/chinakook/Mnasnet.MXNet) [[pretrained model]](https://github.com/zeusees/Mnasnet-Pretrained-Model)
 - [**CompetitiveSENet**](https://github.com/scut-aitcm/CompetitiveSENet)
 - [Residual-Attention-Network](https://github.com/haoxintong/Residual-Attention-Network-Gluon)
 - [SNAIL](https://github.com/seujung/SNAIL-gluon)
 - [DropBlock](https://github.com/chenzx921020/DropBlock-mxnet)
 - [DropBlock(c++ implementaion)](https://github.com/yuyijie1995/dropblock_mxnet_bottom_implemention)
 - [Modified-CBAMnet](https://github.com/bruinxiong/Modified-CBAMnet.mxnet)
 - [**OctConv**](https://github.com/facebookresearch/OctConv)
 - [tasn](https://github.com/researchmm/tasn)
 - 3rdparty Resnet/Resnext/Inception/Xception/Air/DPN/SENet [pretrained models](https://github.com/soeaver/mxnet-model)
 - Collection of [pretrained models (Gluon:star:)](https://github.com/osmr/imgclsmob)

 ### 2.2 Object Detection
 - [PVANet](https://github.com/apache/incubator-mxnet/pull/7786)
 - SSD [[Origin]](https://github.com/zhreshold/mxnet-ssd) [[Focal Loss]](https://github.com/eldercrow/focal_loss_mxnet_ssd) [[FPN]](https://github.com/zunzhumu/ssd) [[DSSD/TDM]](https://github.com/MTCloudVision/mxnet-dssd) [[RetinaNet]](https://github.com/jkznst/RetinaNet-mxnet) [[RefineDet]](https://github.com/MTCloudVision/RefineDet-Mxnet)
 - [DSOD](https://github.com/leocvml/DSOD-gluon-mxnet)
 - YOLO [[sym v1/v2]](https://github.com/zhreshold/mxnet-yolo) [[darknet]](https://github.com/bowenc0221/MXNet-YOLO) [[gluon]](https://github.com/MashiMaroLjc/YOLO) [[v3]](https://github.com/Fermes/yolov3-mxnet)
 - Faster RCNN [[Origin]](https://github.com/precedenceguo/mx-rcnn) [[gluon]](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo/faster_rcnn) [[ya_mxdet]](https://github.com/linmx0130/ya_mxdet) [[Focal Loss]](https://github.com/unsky/focal-loss) [[Light-Head]](https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784)
 - [**Deformable-ConvNets**](https://github.com/msracver/Deformable-ConvNets) with Faster RCNN/R-FCN/FPN/SoftNMS and Deeplab
 - [**Deformable-ConvNets v2**](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op)
 - [**Relation-Networks**](https://github.com/msracver/Relation-Networks-for-Object-Detection) with FPN
 - [FPN-gluon-cv](https://github.com/Angzz/fpn-gluon-cv)
 - [FCIS](https://github.com/msracver/FCIS)
 - [Mask R-CNN](https://github.com/TuSimple/mx-maskrcnn)
 - [SqueezeDet](https://github.com/alvinwan/squeezeDetMX)
 - [IOULoss](https://github.com/wcj-Ford/IOULoss)
 - [FocalLoss(CUDA)](https://github.com/yuantangliang/softmaxfocalloss)
 - [dspnet](https://github.com/liangfu/dspnet)
 - [Faster_RCNN_for_DOTA](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA)
 - [RoITransformer](https://github.com/dingjiansw101/RoITransformer_DOTA)
 - [cascade-rcnn-gluon(Gluon:star:)](https://github.com/lizhenbang56/cascade-rcnn-gluon)
 - [**SNIPER**](https://github.com/mahyarnajibi/SNIPER) with R-FCN-3K and SSH Face Detector
 - [Car-Detector-and-Tracker](https://github.com/YvesHarrison/Car-Detector-and-Tracker-Using-MXNet-and-KCF)
 - [detnet](https://github.com/BigDeviltjj/mxnet-detnet)
 - [CornerNet](https://github.com/BigDeviltjj/mxnet-cornernet)
 - [GroupNormalization](https://github.com/JaggerYoung/mxnet-GroupNormalization)
 - [faster-rcnn-rotate](https://github.com/shihan19911126/mxnet-faster-rcnn-rotate)
 - [Detection and Recognition in Remote Sensing Image](https://github.com/whywhs/Detection_and_Recognition_in_Remote_Sensing_Image)
 - [**simpledet**](https://github.com/TuSimple/simpledet) with FP16 and distributed training
 - [**FCOS**](https://github.com/Angzz/fcos-gluon-cv)

 ### 2.3 Image Segmentation
 - [FCN](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/fcn.py)
 - Deeplab [[v2]](https://github.com/buptweixin/mxnet-deeplab)  [[v3+Vortex Pooling]](https://github.com/MTCloudVision/deeplabv3-mxnet_gluon) [[v3plus]](https://github.com/duducheng/deeplabv3p_gluon) [[v3plus+densenet]](https://github.com/leocvml/deeplabv3plus-gluon-mxnet)
 - U-Net [[gluon]](https://github.com/chinakook/U-Net) [[kaggle dstl]](https://github.com/u1234x1234/kaggle-dstl-satellite-imagery-feature-detection)
 - [SegNet](https://github.com/solin319/incubator-mxnet/tree/solin-patch-segnet)
 - [PSPNet](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/pspnet.py) with [SyncBN](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/syncbn.py)
 - [DUC](https://github.com/TuSimple/TuSimple-DUC)
 - [ResNet-38](https://github.com/itijyou/ademxapp)
 - [SEC](https://github.com/ascust/SEC-MXNet)
 - [**DRN**](https://github.com/zhuangyqin/DRN)
 - [panoptic-fpn](https://github.com/Angzz/panoptic-fpn-gluon)
 - [AdaptIS](https://github.com/saic-vul/adaptis)

 ### 2.4 Video Recognition and Object Detection
 - [Deep Feature Flow](https://github.com/msracver/Deep-Feature-Flow)
 - [Flow-Guided Feature Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation)
 - [st-resnet](https://github.com/jay1204/st-resnet)

 ### 2.5 Face Detection and Recognition
 - [MXNet Face](https://github.com/tornadomeet/mxnet-face)
 - MTCNN [[w/ train]](https://github.com/Seanlinx/mtcnn) [[caffe]](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
 - Tiny Face [[w/ train]](https://github.com/IIMarch/tiny-face-mxnet) [[matconvnet]](https://github.com/chinakook/hr101_mxnet)
 - [S3FD](https://github.com/zunzhumu/S3FD)
 - [S3FD-gluoncv](https://github.com/yangfly/sfd.gluoncv)
 - [SSH](https://github.com/deepinsight/mxnet-SSH)
 - [FaceDetection-ConvNet-3D](https://github.com/tfwu/FaceDetection-ConvNet-3D)
 - [DeepID v1](https://github.com/AihahaFox/deepid-mxnet)
 - Range Loss [[CustomOp]](https://github.com/ShownX/mxnet-rangeloss) [[gluon]](https://github.com/LeeJuly30/RangeLoss-For-Gluno)
 - [Convolutional Sketch Inversion](https://github.com/VinniaKemala/sketch-inversion)
 - [Face68Pts](https://github.com/LaoDar/mxnet_cnn_face68pts)
 - [DCGAN face generation(Gluon:star:)](https://github.com/dbsheta/dcgan_face_generation)
 - [**InsightFace**](https://github.com/deepinsight/insightface)
 - [Modified-CRUNet+Residual-Attention-Network](https://github.com/bruinxiong/Modified-CRUNet-and-Residual-Attention-Network.mxnet)
 - [LightCNN](https://github.com/ly-atdawn/LightCNN-mxnet)
 - [E2FAR](https://github.com/ShownX/mxnet-E2FAR)
 - [FacialLandmark](https://github.com/BobLiu20/FacialLandmark_MXNet)
 - [batch_hard_triplet_loss](https://github.com/IcewineChen/mxnet-batch_hard_triplet_loss)
 - [facial-emotion-recognition](https://github.com/TalkAI/facial-emotion-recognition-gluon)
 - [RSA(prediction only)](https://github.com/ElegantGod/RSA-for-object-detection-mxnet-version)
 - [gender_age_estimation_mxnet](https://github.com/wayen820/gender_age_estimation_mxnet)
 - [Ringloss](https://github.com/haoxintong/Ringloss-Gluon)
 - [gluon-face](https://github.com/THUFutureLab/gluon-face)
 - [age-gender-estimation](https://github.com/deepinx/age-gender-estimation)
 - [iqiyi-vid-challenge 1st code](https://github.com/deepinx/iqiyi-vid-challenge)
 - [sdu-face-alignment](https://github.com/deepinx/sdu-face-alignment)
 - [PyramidBox](https://github.com/JJXiangJiaoJun/gluon_PyramidBox)
 - [A-Light-and-Fast-Face-Detector-for-Edge-Devices](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)


 ### 2.6 ReID
 - [rl-multishot-reid](https://github.com/TuSimple/rl-multishot-reid)
 - [DarkRank](https://github.com/TuSimple/DarkRank)
 - [reid_baseline_gluon](https://github.com/L1aoXingyu/reid_baseline_gluon)
 - [beyond-part-models](https://github.com/Tyhye/beyond-part-models-gluon)
 - [**gluon-reid**](https://github.com/xiaolai-sqlai/gluon-reid)

 ### 2.7 Human Analyzation and Activity Recognition
 - [Head Pose](https://github.com/LaoDar/cnn_head_pose_estimator)
 - [Convolutional Pose Machines](https://github.com/li-haoran/mxnet-convolutional_pose_machines_Testing)
 - [Realtime Multi-Person Pose Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation)
 - [Realtime Multi-Person Pose Estimation (Gluon :star:)](https://github.com/ThomasDelteil/MultiPoseEstimation_MXNet)
 - [OpenPose](https://github.com/kohillyang/mx-openpose)
 - [Dynamic pose estimation](https://github.com/gengshan-y/dyn_pose)
 - [LSTM for HAR](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition)
 - [C3D](https://github.com/JaggerYoung/C3D-mxnet)
 - [P3D](https://github.com/IIMarch/pseudo-3d-residual-networks-mxnet)
 - [DeepHumanPrediction](https://github.com/JONGGON/DeepHumanPrediction)
 - [Reinspect](https://github.com/NoListen/mxnet-reinspect)
 - [COCO Human keypoint](https://github.com/wangsr126/mxnet-pose)
 - [R2Plus1D](https://github.com/starsdeep/R2Plus1D-MXNet)
 - [CSRNet](https://github.com/wkcn/CSRNet-mx)

 ### 2.8 Image Enhancement
 - [**learning-to-see-in-the-dark**](https://github.com/anzhao0503/learning-to-see-in-the-dark.mxnet)
 - SRCNN [[1]](https://github.com/Codersadis/SRCNN-MXNET) [[2]](https://github.com/galad-loth/SuperResolutionCNN)
 - [**Super-Resolution-Zoo**](https://github.com/WolframRhodium/Super-Resolution-Zoo) MXNet pretrained models for super resolution, denoising and deblocking

 ### 2.9 OCR
 - [SSD Text Detection](https://github.com/oyxhust/ssd-text_detection)
 - [EAST](https://github.com/wangpan8154/east-text-detection-with-mxnet/tree/1a63083d69954e7c1c7ac277cf6b8ed5af4ec770)
 - [**CTPN.mxnet**](https://github.com/chinakook/CTPN.mxnet)
 - CRNN [[Chinese]](https://github.com/diaomin/crnn-mxnet-chinese-text-recognition) [ [[insightocr]](https://github.com/deepinsight/insightocr) [[A full version]](https://github.com/WenmuZhou/crnn.gluon)
 - [Handwritten OCR CRNN (Gluon :star:)](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet)
 - [PSENet](https://github.com/saicoco/Gluon-PSENet)


 ### 2.10 Point cloud & 3D
 - [mx-pointnet](https://github.com/Zehaos/mx-pointnet) [[gluon version]](https://github.com/hnVfly/pointnet.mxnet)
 - [PointCNN.MX](https://github.com/chinakook/PointCNN.MX)
 - [RC3D](https://github.com/likelyzhao/MxNet-RC3D/blob/master/RC3D/symbols/RC3D.py)
 - [DeepIM](https://github.com/liyi14/mx-DeepIM)

 ### 2.11 Images Generation
 - [pix2pix](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/Pix2Pix)
 - [Image colorization](https://github.com/skirdey/mxnet-pix2pix)
 - [Neural-Style-MMD](https://github.com/lyttonhao/Neural-Style-MMD)
 - [MSG-Net(Gluon:star:)](https://github.com/zhanghang1989/MXNet-Gluon-Style-Transfer)
 - [fast-style-transfer](https://github.com/SineYuan/mxnet-fast-neural-style)
 - [neural-art-mini](https://github.com/pavelgonchar/neural-art-mini)

 ### 2.12 GAN
 - [DCGAN(Gluon:star:)](https://github.com/kazizzad/DCGAN-Gluon-MxNet)
 - [**CycleGAN**](https://github.com/leocvml/CycleGAN-gluon-mxnet)

 ### 2.13 MRI & DTI
 - [Chest-XRay](https://github.com/kperkins411/MXNet-Chest-XRay-Evaluation)
 - [LUCAD](https://github.com/HPI-DeepLearning/LUCAD)

 ### 2.14 Misc
 - [VisualBackProp](https://github.com/Bartzi/visual-backprop-mxnet)
 - VQA [[sym]](https://github.com/shiyangdaisy23/mxnet-vqa) [[gluon]](https://github.com/shiyangdaisy23/vqa-mxnet-gluon)
 - [Hierarchical Question-Imagee Co-Attention](https://github.com/WillSuen/VQA)
 - [text2image(Gluon:star:)](https://github.com/dbsheta/text2image)
 - [Traffic sign classification](https://github.com/sookinoby/mxnet-ccn-samples)
 - [cicada classification](https://github.com/dokechin/cicada_shell)
 - [geometric-matching](https://github.com/x007dwd/geometric-matching-mxnet)
 - [Loss Surfaces](https://github.com/nicklhy/cnn_loss_surface)
 - [Class Activation Mapping](https://github.com/nicklhy/CAM)
 - [AdversarialAutoEncoder](https://github.com/nicklhy/AdversarialAutoEncoder)
 - [Neural Image Caption](https://github.com/saicoco/mxnet_image_caption)
 - [mmd/jmmd/adaBN](https://github.com/deepinsight/transfer-mxnet)
 - [NetVlad](https://github.com/likelyzhao/NetVlad-MxNet)
 - [multilabel](https://github.com/miraclewkf/multilabel-MXNet)
 - [multi-task](https://github.com/miraclewkf/multi-task-MXNet)
 - [siamese](https://github.com/saicoco/mxnet-siamese)
 - [matchnet](https://github.com/zhengxiawu/mxnet-matchnet)
 - [DPSH](https://github.com/akturtle/DPSH)
 - [Yelp Restaurant Photo Classifacation](https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification)
 - [siamese_network_on_omniglot(Gluon:star:)](https://github.com/muchuanyun/siamese_network_on_omniglot)
 - [StrassenNets](https://github.com/mitscha/strassennets)
 - [Image Embedding Learning (Gluon:star:)](https://github.com/chaoyuaw/incubator-mxnet)
 - [DIRNet](https://github.com/HPI-DeepLearning/DIRNet/tree/master/DIRNet-mxnet)
 - [Receptive Field Tool](https://github.com/chinakook/mxnet/blob/kkmaster/python/kktools/rf.py)
 - [mxnet-videoio](https://github.com/MTCloudVision/mxnet-videoio)
 - [AudioDataLoader](https://github.com/gaurav-gireesh/AudioDataLoader)
 - [mxnet_tetris](https://github.com/sunkwei/mxnet_tetris)
 - [Visual Search (Gluon:star:)](https://github.com/ThomasDelteil/VisualSearch_MXNet)
 - [DALI](https://github.com/NVIDIA/DALI)
 - [relational-network-gluon](https://github.com/seujung/relational-network-gluon)
 - [HKO-7](https://github.com/sxjscience/HKO-7) [[weather-forecasting]](https://github.com/igloooo/weather-forecasting-mxnet)
 - [siamfc](https://github.com/forschumi/siamfc-mxnet)
 - [AdvBox](https://github.com/baidu/AdvBox)
 - [SAGE-GRAPH](https://github.com/diyang/SAGE-GRAPH-R)
 - [Memory-Aware-Synapses](https://github.com/mingzhang96/MAS-mxnet)

## <a name="NLP"></a>3. NLP
 - [**sockeye**](https://github.com/awslabs/sockeye)
 - [**gluon-nlp**(Gluon:star:)](https://github.com/dmlc/gluon-nlp)
 - [MXNMT](https://github.com/magic282/MXNMT)
 - [Char-RNN(Gluon:star:)](https://github.com/SherlockLiao/Char-RNN-Gluon)
 - [Character-level CNN Text Classification (Gluon:star:)](https://github.com/ThomasDelteil/CNN_NLP_MXNet)
 - [AC-BLSTM](https://github.com/Ldpe2G/AC-BLSTM)
 - seq2seq [[sym]](https://github.com/yoosan/mxnet-seq2seq) [[gluon]](https://github.com/ZiyueHuang/MXSeq2Seq)
 - MemN2N [[sym]](https://github.com/nicklhy/MemN2N) [[gluon]](https://github.com/fanfeifan/MemN2N-Mxnet-Gluon)
 - [Neural Programmer-Interpreters](https://github.com/Cloudyrie/npi)
 - [sequence-sampling](https://github.com/doetsch/sequence-sampling-mxnet)
 - [retrieval chatbot](https://github.com/NonvolatileMemory/baseline_for_chatbot-mxnet)
 - [cnn+Highway Net](https://github.com/wut0n9/cnn_chinese_text_classification)
 - [sentiment-analysis(Gluon:star:)](https://github.com/aws-samples/aws-sentiment-analysis-mxnet-gluon)
 - [parserChiang(Gluon:star:)](https://github.com/linmx0130/parserChiang)
 - [Neural Variational Document Model(Gluon:star:)](https://github.com/dingran/nvdm-mxnet)
 - [NER with  Bidirectional LSTM-CNNs](https://github.com/opringle/named_entity_recognition)
 - [Sequential Matching Network(Gluon:star:)](https://github.com/NonvolatileMemory/MXNET-SMN)
 - [ko_en_NMT(Gluon:star:)](https://github.com/haven-jeon/ko_en_neural_machine_translation)
 - [**Gluon Dynamic-batching**(Gluon:star:)](https://github.com/szha/mxnet-fold)
 - [translatR](https://github.com/jeremiedb/translatR)
 - [RNN-Transducer](https://github.com/HawkAaron/mxnet-transducer)
 - [Deep Biaffine Parser](https://github.com/hankcs/DeepBiaffineParserMXNet)
 - [Crepe model](https://github.com/ThomasDelteil/CNN_NLP_MXNet)
 - [EXAM](https://github.com/bcol23/EXAM-MXNet)
 - [**RegionEmbedding**](https://github.com/zhaozhengChen/RegionEmbedding)
 - [Structured-Self-Attentive-Sentence-Embedding](https://github.com/kenjewu/Structured-Self-Attentive-Sentence-Embedding)
 - [translatR](https://github.com/jeremiedb/translatR)
 - [**BERT-embedding**](https://github.com/imgarylai/bert-embedding)

## <a name="Speech"></a>4. Speech
 - [mxnet_kaldi](https://github.com/vsooda/mxnet_kaldi)
 - [**deepspeech**](https://github.com/samsungsds-rnd/deepspeech.mxnet)
 - [wavenet](https://github.com/shuokay/mxnet-wavenet) [[WaveNet-gluon]](https://github.com/seujung/WaveNet-gluon)
 - [Tacotron](https://github.com/PiSchool/mxnet-tacotron)
 - [mxnet-audio](https://github.com/chen0040/mxnet-audio)

## <a name="Time_series_forecasting"></a>5. Time series forecasting
 - [LSTNet](https://github.com/opringle/multivariate_time_series_forecasting)

## <a name="Spatiotemporal"></a>6. Spatiotemporal
 - [gluon-spaceTime](https://github.com/D-Roberts/gluon-spaceTime)

## <a name="CTR"></a>7. CTR
 - [MXNet for CTR ](https://github.com/CNevd/DeepLearning-MXNet)
 - [CDL](https://github.com/js05212/MXNet-for-CDL)
 - [SpectralLDA](https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet)
 - [DEF(Gluon:star:)](https://github.com/altosaar/deep-exponential-families-gluon)
 - [mxnet-recommender(Gluon:star:)](https://github.com/chen0040/mxnet-recommender)
 - [collaborative_filtering](https://github.com/opringle/collaborative_filtering)
 - [gluon-rank](https://github.com/opringle/gluon-rank)
 - [ncf](https://github.com/xinyu-intel/ncf_mxnet)

## <a name="DRL"></a>8. DRL
 - [DRL](https://github.com/qyxqyx/DRL)
 - [DQN(Gluon:star:)](https://github.com/kazizzad/DQN-MxNet-Gluon)
 - [Double DQN(Gluon:star:)](https://github.com/kazizzad/Double-DQN-MxNet-Gluon)
 - [openai-mxnet](https://github.com/boddmg/openai-mxnet)
 - [PPO(Gluon:star:)](https://github.com/dai-dao/PPO-Gluon)
 - [CrazyAra](https://github.com/QueensGambit/CrazyAra)

## <a name="Neuro_evolution"></a>9. Neuro Evolution
 - [galapagos_nao](https://github.com/jeffreyksmithjr/galapagos_nao)

## <a name="One_class_learning"></a>10. One Class Learning
 - [anomaly_detection](https://github.com/malykhin/anomaly_detection/blob/master/anomaly_AWS.ipynb)

## <a name="PPL"></a>11. Probabilistic Programming
 - [**MXFusion**](https://github.com/amzn/MXFusion)

## <a name="TL"></a>12. Transfer Learning
 - [**xfer**](https://github.com/amzn/xfer)

## <a name="Tools"></a>13. Tools
 ### 13.1 Converter
 - [mxnet2tf](https://github.com/vuvko/mxnet2tf)
 - [MMdnn](https://github.com/Microsoft/MMdnn)
 - [onnx-mxnet](https://github.com/onnx/onnx-mxnet)
 - [mxnet_to_onnx](https://github.com/NVIDIA/mxnet_to_onnx)
 - [R-Convert-json-to-symbol](https://github.com/Imshepherd/MxNetR-Convert-json-to-symbol)
 - [**mxnet2ncnn**](https://github.com/Tencent/ncnn/blob/28b35b8c4f3d58feaaaeaa58273b763751827aab/tools/mxnet/mxnet2ncnn.cpp)
 - [Gluon2PyTorch](https://github.com/nerox8664/gluon2pytorch)
 - [Gluon2Keras](https://github.com/nerox8664/gluon2keras)

 ### 13.2 Language Bindings
 - [mxnet.rb](https://github.com/mrkn/mxnet.rb)
 - [mxnet.csharp](https://github.com/yajiedesign/mxnet.csharp)
 - [SiaNet(csharp)](https://github.com/deepakkumar1984/SiaNet)
 - [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor)
 - [dmxnet](https://github.com/sociomantic-tsunami/dmxnet)
 - [load_op](https://github.com/DuinoDu/load_op.mxnet)
 - [MobulaOP](https://github.com/wkcn/MobulaOP)

 ### 13.3 Visualization
 - [mxbox](https://github.com/Lyken17/mxbox)
 - [mixboard](https://github.com/DrSensor/mixboard)
 - [mxflows](https://github.com/aidan-plenert-macdonald/mxflows)
 - [mxserver](https://github.com/Harmonicahappy/mxserver)
 - [VisualDL](https://github.com/PaddlePaddle/VisualDL)
 - [mxProfileParser](https://github.com/TaoLv/mxProfileParser)
 - [polyaxon](https://github.com/polyaxon/polyaxon) with [examples](https://github.com/polyaxon/polyaxon-examples/tree/master/mxnet/cifar10)
 - [Netron](https://github.com/lutzroeder/Netron)
 - [**mxboard**](https://github.com/awslabs/mxboard)
 - [CalFLOPS](https://github.com/likelyzhao/CalFLOPS-Mxnet)

 ### 13.4 Parallel and Distributed computing
 - [mxnet-rdma](https://github.com/liuchang1437/mxnet-rdma)
 - [RDMA-MXNet-ps-lite](https://github.com/ralzq01/RDMA-MXNet-ps-lite)
 - [MPIZ-MXNet](https://github.com/Shenggan/MPIZ-MXNet)
 - [MXNetOnYARN](https://github.com/Intel-bigdata/MXNetOnYARN)
 - [mxnet-operator](https://github.com/deepinsight/mxnet-operator)
 - [mxnet_on_kubernetes](https://github.com/WorldAITime/mxnet_on_kubernetes)
 - [speculative-synchronization](https://github.com/All-less/mxnet-speculative-synchronization)
 - [XLearning](https://github.com/Qihoo360/XLearning)
 - [Gluon Distributed Training (Gluon:star:)](https://mxnet.indu.ai/tutorials/distributed-training-using-mxnet)
 - [gpurelperf](https://github.com/andylamp/gpurelperf)
 - [horovod](https://github.com/horovod/horovod)
 - [**byteps**](https://github.com/bytedance/byteps)

 ### 13.5 Productivity
 - [Email Monitor MxnetTrain](https://github.com/fierceX/Email_Monitor_MxnetTrain)
 - [mxnet-oneclick](https://github.com/imistyrain/mxnet-oneclick)
 - [mxnet-finetuner](https://github.com/knjcode/mxnet-finetuner)
 - [Early-Stopping](https://github.com/kperkins411/MXNet_Demo_Early-Stopping)
 - [MXNet_Video_Trainer](https://github.com/likelyzhao/MXNet_Video_Trainer)
 - [rs_mxnet_reader](https://github.com/ChenKQ/rs_mxnet_reader)

 ### 13.6 Parameter optimizer
 - [YellowFin](https://github.com/StargazerZhu/YellowFin_MXNet)
 - [**LookaheadOptimizer**](https://github.com/wkcn/LookaheadOptimizer-mx)

 ### 13.7 Deployment
 - [Turi Create](https://github.com/apple/turicreate)
 - [MXNet-HRT](https://github.com/OAID/MXNet-HRT)
 - [Tengine](https://github.com/OAID/Tengine)
 - [Collective Knowledge](https://github.com/ctuning/ck-mxnet)
 - [flask-app-for-mxnet-img-classifier](https://github.com/XD-DENG/flask-app-for-mxnet-img-classifier)
 - [qt-mxnet](https://github.com/mjamroz/qt-mxnet)
 - [mxnet_predict_ros](https://github.com/Paspartout/mxnet_predict_ros)
 - [mxnet-lambda](https://github.com/awslabs/mxnet-lambda)
 - [openHabAI](https://github.com/JeyRunner/openHabAI)
 - ImageRecognizer [[iOS]](https://github.com/dneprDroid/ImageRecognizer-iOS) [[Android]](https://github.com/dneprDroid/ImageRecognizer-Android)
 - [MXNet to MiniFi](https://github.com/tspannhw/nvidiajetsontx1-mxnet)
 - [MXNet Model Serving ](https://github.com/yuruofeifei/mms)
 - [mxnet-model-server](https://github.com/awslabs/mxnet-model-server)
 - [tvm-mali](https://github.com/merrymercy/tvm-mali)
 - [mxnet-and-sagemaker](https://github.com/cosmincatalin/object-counting-with-mxnet-and-sagemaker)
 - [example-of-nnvm-in-cpp](https://github.com/zhangxinqian/example-of-nnvm-in-cpp)
 - [tensorly](https://github.com/tensorly/tensorly)
 - [OpenVINO](https://software.intel.com/en-us/openvino-toolkit/documentation/featured)

 ### 13.8 Other Branches
 - [ngraph-mxnet](https://github.com/NervanaSystems/ngraph-mxnet)
 - [distributedMXNet](https://github.com/TuSimple/distributedMXNet)

# Awesome MXNet [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/jtoy/awesome)

A curated list of MXNet examples, tutorials and blogs. It is inspired by awesome-caffe.

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.

## Table of Contents
- [1. Tutorials](#Tutorials)
- [2. Vision](#Vision)
- [3. NLP](#NLP)
- [4. Speech](#Speech)
- [5. CTR](#CTR)
- [6. Tools](#Tools)

============================================================================================================
## <a name="Tutorials"></a>1. Tutorials
- [Tutorial Documentation](https://mxnet.incubator.apache.org/tutorials/)
- [Gluon Tutorial Documentation](http://gluon.mxnet.io/)
- [Gluon Tutorial Documentation (Simplified Chinese)](https://zh.gluon.ai/)
- [CheatSheet](https://github.com/chinakook/Awesome-MXNet/blob/master/apache-mxnet-cheat.pdf)
- [Using MXNet](https://github.com/JONGGON/Mxnet_Tutorial)
- [Linalg examples](https://github.com/ARCambridge/MXNet_linalg_examples)

## <a name="Vision"></a>2. Vision
>> ### 2.1 Image Classification
>> - [ResNet](https://github.com/tornadomeet/ResNet)
>> - [DenseNet](https://github.com/bruinxiong/densenet.mxnet)
>> - [DPN](https://github.com/cypw/DPNs):star:
>> - [CRU-Net](https://github.com/cypw/CRU-Net)
>> - [MobileNet](https://github.com/KeyKy/mobilenet-mxnet)
>> - [ShuffleNet](https://github.com/ZiyueHuang/MXShuffleNet)
>> - [Xception](https://github.com/bruinxiong/xception.mxnet)
>> - [SqeezeNet](https://github.com/miaow1988/SqueezeNet_v1.2)
>> - [FractalNet](https://github.com/newuser-16824/mxnet-fractalnet)
>> - [BMXNet](https://github.com/hpi-xnor/BMXNet)
>> - [Self-Norm Nets](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/SelfNormNets)
>> - [Factorized-Bilinear-Network](https://github.com/lyttonhao/Factorized-Bilinear-Network):star:
>> - [DPSH](https://github.com/akturtle/DPSH)
>> - [Yelp Restaurant Photo Classifacation](https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification):star:

>> ### 2.2 Object Detection
>> - [PVANet](https://github.com/apache/incubator-mxnet/pull/7786)
>> - [SSD](https://github.com/zhreshold/mxnet-ssd)
>> - [YOLO](https://github.com/zhreshold/mxnet-yolo)
>> - [YOLO/dark2mxnet](https://github.com/bowenc0221/MXNet-YOLO)
>> - [Faster RCNN](https://github.com/precedenceguo/mx-rcnn)
>> - [R-FCN](https://github.com/msracver/Deformable-ConvNets)
>> - [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)
>> - [Deformable-ConvNets+SoftNMS](https://github.com/bharatsingh430/Deformable-ConvNets):star:
>> - [SSD+Focal Loss](https://github.com/eldercrow/focal_loss_mxnet_ssd)
>> - [Faster RCNN+Focal Loss](https://github.com/unsky/focal-loss)
>> - [RetinaNet](https://github.com/unsky/RetinaNet)
>> - [SqueezeDet](https://github.com/alvinwan/squeezeDetMX)
>> - [IOULoss](https://github.com/wcj-Ford/IOULoss)

>> ### 2.3 Image Segmentation
>> - [Mask R-CNN+FPN](https://github.com/TuSimple/mx-maskrcnn)
>> - [Mask R-CNN](https://github.com/xilaili/maskrcnn.mxnet)
>> - [FCIS](https://github.com/msracver/FCIS)
>> - [ResNet-38](https://github.com/itijyou/ademxapp)
>> - [Deeplab v2](https://github.com/buptweixin/mxnet-deeplab)
>> - [U-Net](https://github.com/chinakook/U-Net)
>> - [U-Net (kaggle dstl)](https://github.com/u1234x1234/kaggle-dstl-satellite-imagery-feature-detection)
>> - [Segnet](https://github.com/solin319/incubator-mxnet/tree/solin-patch-segnet)
>> - [FCN-ASPP](https://github.com/ComeOnGetMe/FCN-ASPP-with-uncertainty)

>> ### 2.4 Video Recognition and Object Detection
>> - [Deep Feature Flow](https://github.com/msracver/Deep-Feature-Flow):star:
>> - [Flow-Guided Feature Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation):star:
>> - [st-resnet](https://github.com/jay1204/st-resnet)

>> ### 2.5 Face and Human releated
>> - [MTCNN](https://github.com/Seanlinx/mtcnn)
>> - [MTCNN (original detector)](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
>> - [MXNet Face](https://github.com/tornadomeet/mxnet-face)
>> - [Tiny Face](https://github.com/chinakook/hr101_mxnet)
>> - [FaceDetection-ConvNet-3D](https://github.com/tfwu/FaceDetection-ConvNet-3D)
>> - [VanillaCNN](https://github.com/flyingzhao/mxnet_VanillaCNN)
>> - [DeepID v1](https://github.com/AihahaFox/deepid-mxnet)
>> - [Head Pose](https://github.com/LaoDar/cnn_head_pose_estimator)
>> - [Triple Loss](https://github.com/xlvector/learning-dl/tree/master/mxnet/triple-loss)
>> - [Center Loss](https://github.com/pangyupo/mxnet_center_loss)
>> - [Large-Margin Softmax Loss](https://github.com/luoyetx/mx-lsoftmax)
>> - [Range Loss](https://github.com/ShownX/mxnet-rangeloss)
>> - [Convolutional Sketch Inversion](https://github.com/VinniaKemala/sketch-inversion)
>> - [Convolutional Pose Machines](https://github.com/li-haoran/mxnet-convolutional_pose_machines_Testing)
>> - [OpenPose](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation)
>> - [Face68Pts](https://github.com/LaoDar/mxnet_cnn_face68pts)
>> - [Dynamic pose estimation](https://github.com/gengshan-y/dyn_pose)
>> - [LSTM for HAR](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition)
>> - [C3D](https://github.com/JaggerYoung/C3D-mxnet)
>> - [DeepHumanPrediction](https://github.com/JONGGON/DeepHumanPrediction)
>> - [SphereFace/FocalLoss/CenterLoss/KNNLoss(CUDA)](https://github.com/deepearthgo/Cuda-Mxnet)
>> - [SphereFace/KNNLoss(Python)](https://github.com/deepearthgo/Python-Mxnet)

>> ### 2.6 Image Super-resolution
>> - [SRCNN](https://github.com/Codersadis/SRCNN-MXNET)
>> - [SuperResolutionCNN](https://github.com/galad-loth/SuperResolutionCNN)

>> ### 2.7 OCR
>> - [STN OCR](https://github.com/Bartzi/stn-ocr)
>> - [Plate Recognition (Chinese)](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)
>> - [mxnet-cnn-plate-recognition](https://github.com/huxiaoman7/mxnet-cnn-plate-recognition)
>> - [crnn](https://github.com/xinghedyc/mxnet-cnn-lstm-ctc-ocr)
>> - [crnn (with Chinese Support)](https://github.com/novioleo/crnn.mxnet)
>> - [CNN-LSTM-CTC](https://github.com/oyxhust/CNN-LSTM-CTC-text-recognition)
>> - [SSD Text Detection](https://github.com/oyxhust/ssd-text_detection)
>> - [cnnbilstm](https://github.com/deepinsight/cnnbilstm-mxnet)

>> ### 2.8 Images Generation
>> - [pix2pix](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/Pix2Pix)
>> - [Neural-Style-MMD](https://github.com/lyttonhao/Neural-Style-MMD):star:
>> - [MSG-Net(Gluon)](https://github.com/zhanghang1989/MXNet-Gluon-Style-Transfer)
>> - [fast-style-transfer](https://github.com/SineYuan/mxnet-fast-neural-style)

>> ### 2.9 GAN

>> ### 2.10 MRI&DTI
>> - [Chest-XRay](https://github.com/kperkins411/MXNet-Chest-XRay-Evaluation)

>> ### 2.11 Misc
>> - [VisualBackProp](https://github.com/Bartzi/visual-backprop-mxnet)
>> - [VQA](https://github.com/shiyangdaisy23/mxnet-vqa)
>> - [VQA(Gluon)](https://github.com/shiyangdaisy23/vqa-mxnet-gluon)
>> - [Traffic sign classification](https://github.com/sookinoby/mxnet-ccn-samples)
>> - [cicada classification](https://github.com/dokechin/cicada_shell)
>> - [Loss Surfaces](https://github.com/nicklhy/cnn_loss_surface)
>> - [CAM](https://github.com/nicklhy/CAM)
>> - [AdversarialAutoEncoder](https://github.com/nicklhy/AdversarialAutoEncoder)
>> - [Neural Image Caption](https://github.com/saicoco/mxnet_image_caption)
>> - [mmd/jmmd/adaBN](https://github.com/deepinsight/transfer-mxnet)
>> - [NetVlad](https://github.com/likelyzhao/NetVlad-MxNet)

## <a name="NLP"></a>3. NLP
>> - [sockeye](https://github.com/awslabs/sockeye)
>> - [MXNMT](https://github.com/magic282/MXNMT)
>> - [Char-RNN(Gluon)](https://github.com/SherlockLiao/Char-RNN-Gluon)
>> - [AC-BLSTM](https://github.com/Ldpe2G/AC-BLSTM)
>> - [MXSeq2Seq(Gluon)](https://github.com/ZiyueHuang/MXSeq2Seq)
>> - [MemN2N](https://github.com/nicklhy/MemN2N)
>> - [Neural Programmer-Interpreters](https://github.com/Cloudyrie/npi)
>> - [sequence-sampling](https://github.com/doetsch/sequence-sampling-mxnet)

## <a name="Speech"></a>4. Speech
>> - [deepspeech](https://github.com/samsungsds-rnd/deepspeech.mxnet)
>> - [wavenet](https://github.com/shuokay/mxnet-wavenet)

## <a name="Speech"></a>5. CTR
>> - [MXNet for CTR ](https://github.com/CNevd/DeepLearning-MXNet)
>> - [CDL](https://github.com/js05212/MXNet-for-CDL)
>> - [SpectralLDA](https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet)

## <a name="Speech"></a>6. Tools
>> - [YellowFin](https://github.com/StargazerZhu/YellowFin_MXNet)
>> - [MXNet Model Serving ](https://github.com/yuruofeifei/mms)
>> - [Early-Stopping](https://github.com/kperkins411/MXNet_Demo_Early-Stopping)
>> - [CaffeTranslator](https://github.com/indhub/CaffeTranslator)
>> - [MXNetOnACL](https://github.com/OAID/MXNetOnACL)

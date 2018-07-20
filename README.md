# Awesome MXNet(Beta) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/jtoy/awesome)

A curated list of MXNet examples, tutorials and blogs. It is inspired by awesome-caffe.

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.

## Table of Contents
- [1. Tutorials and Resources](#Tutorials)
- [2. Vision](#Vision)
- [3. NLP](#NLP)
- [4. Speech](#Speech)
- [5. Time series forecasting](#Time_series_forecasting)
- [6. CTR](#CTR)
- [7. DRL](#DRL)
- [8. Neuro Evolution](#Neuro_evolution)
- [9. Tools](#Tools)

________________


## <a name="Tutorials and Resources"></a>1. Tutorials and Resources
- [Documents](https://mxnet.incubator.apache.org/) [[site]](https://github.com/apache/incubator-mxnet-site)
- Tutorial Documentation [[English]](https://mxnet.incubator.apache.org/tutorials/) [[Chinese]](https://github.com/wangx404/symbol_coding_tutorials_of_MXNet)
- Gluon Tutorial Documentation [[English]](http://gluon.mxnet.io/) [[Chinese]](https://zh.gluon.ai/)
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
>> ### 2.1 Image Classification
>> - ResNet [[sym]](https://github.com/tornadomeet/ResNet) [[gluon]](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/resnet.py) [[v1b]](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py)
>> - DenseNet [[sym]](https://github.com/bruinxiong/densenet.mxnet) [[gluon]](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/densenet.py)
>> - SENet [[sym]](https://github.com/bruinxiong/SENet.mxnet) [[gluon]](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/se_resnet.py) [[caffe]](https://github.com/IIMarch/SENet-mxnet)
>> - MobileNet [[gluon v1/v2]](/home/dingkou/dev/mx/python/mxnet/gluon/model_zoo/vision/mobilenet.py) [[sym v2]](https://github.com/liangfu/mxnet-mobilenet-v2)
>> - Xception [[sym]](https://github.com/bruinxiong/xception.mxnet) [[Keras]](https://github.com/u1234x1234/mxnet-xception)
>> - [DPN](https://github.com/cypw/DPNs)
>> - [CapsNet](https://github.com/Soonhwan-Kwon/capsnet.mxnet)
>> - [NASNet-A(Gluon:star:)](https://github.com/qingzhouzhen/incubator-mxnet/blob/nasnet/python/mxnet/gluon/model_zoo/vision/nasnet.py)
>> - [CRU-Net](https://github.com/cypw/CRU-Net)
>> - [ShuffleNet](https://github.com/ZiyueHuang/MXShuffleNet)
>> - [**IGCV3**](https://github.com/homles11/IGCV3)
>> - [SqeezeNet](https://github.com/miaow1988/SqueezeNet_v1.2)
>> - [FractalNet](https://github.com/newuser-16824/mxnet-fractalnet)
>> - [BMXNet](https://github.com/hpi-xnor/BMXNet)
>> - [fusenet](https://github.com/zlmzju/fusenet)
>> - [Self-Norm Nets](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/SelfNormNets)
>> - [Factorized-Bilinear-Network](https://github.com/lyttonhao/Factorized-Bilinear-Network)
>> - [AOGNet](https://github.com/xilaili/AOGNet)
>> - [NonLocal+SENet](https://github.com/WillSuen/NonLocalandSEnet)
>> - [mixup](https://github.com/unsky/mixup)
>> - [sparse-structure-selection](https://github.com/TuSimple/sparse-structure-selection)
>> - [neuron-selectivity-transfer](https://github.com/TuSimple/neuron-selectivity-transfer)
>> - [L-GM-Loss](https://github.com/LeeJuly30/L-GM-Loss-For-Gluon)
>> - [**CoordConv**](https://github.com/ZwX1616/mxnet-CoordConv)
>> - 3rdparty Resnet/Resnext/Inception/Xception/Air/DPN/SENet [pretrained models](https://github.com/soeaver/mxnet-model) 

>> ### 2.2 Object Detection
>> - [PVANet](https://github.com/apache/incubator-mxnet/pull/7786)
>> - SSD [[Origin]](https://github.com/zhreshold/mxnet-ssd) [[Focal Loss]](https://github.com/eldercrow/focal_loss_mxnet_ssd) [[FPN]](https://github.com/zunzhumu/ssd) [[DSSD/TDM]](https://github.com/MTCloudVision/mxnet-dssd) [[RetinaNet]](https://github.com/jkznst/RetinaNet-mxnet) [[RefineDet]](https://github.com/MTCloudVision/RefineDet-Mxnet)
>> - YOLO [[sym v1/v2]](https://github.com/zhreshold/mxnet-yolo) [[darknet]](https://github.com/bowenc0221/MXNet-YOLO) [[gluon]](https://github.com/MashiMaroLjc/YOLO) [[v3]](https://github.com/Fermes/yolov3-mxnet)
>> - Faster RCNN [[Origin]](https://github.com/precedenceguo/mx-rcnn) [[gluon]](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo/faster_rcnn) [[ya_mxdet]](https://github.com/linmx0130/ya_mxdet) [[Focal Loss]](https://github.com/unsky/focal-loss) [[Light-Head]](https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784)
>> - [**Deformable-ConvNets**](https://github.com/msracver/Deformable-ConvNets) with Faster RCNN/R-FCN/FPN/SoftNMS and Deeplab
>> - [**Relation-Networks**](https://github.com/msracver/Relation-Networks-for-Object-Detection) with FPN
>> - [FCIS](https://github.com/msracver/FCIS)
>> - [Mask R-CNN](https://github.com/TuSimple/mx-maskrcnn)
>> - [SqueezeDet](https://github.com/alvinwan/squeezeDetMX)
>> - [IOULoss](https://github.com/wcj-Ford/IOULoss)
>> - [FocalLoss(CUDA)](https://github.com/yuantangliang/softmaxfocalloss)
>> - [dspnet](https://github.com/liangfu/dspnet)
>> - [**Faster_RCNN_for_DOTA**](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA)
>> - [cascade-rcnn-gluon(Gluon:star:)](https://github.com/lizhenbang56/cascade-rcnn-gluon)
>> - [**SNIPER**](https://github.com/mahyarnajibi/SNIPER) with R-FCN-3K and SSH Face Detector

>> ### 2.3 Image Segmentation
>> - [FCN](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/fcn.py)
>> - Deeplab [[v2]](https://github.com/buptweixin/mxnet-deeplab) [[gluon]](https://github.com/zehaochen19/segmentation_gluon) [[v3+]](https://github.com/duducheng/deeplabv3p_gluon)
>> - U-Net [[gluon]](https://github.com/chinakook/U-Net) [[kaggle dstl]](https://github.com/u1234x1234/kaggle-dstl-satellite-imagery-feature-detection)
>> - [SegNet](https://github.com/solin319/incubator-mxnet/tree/solin-patch-segnet)
>> - [PSPNet](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/pspnet.py) with [SyncBN](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/syncbn.py)
>> - [DUC](https://github.com/TuSimple/TuSimple-DUC)
>> - [ResNet-38](https://github.com/itijyou/ademxapp)
>> - [SEC](https://github.com/ascust/SEC-MXNet)

>> ### 2.4 Video Recognition and Object Detection
>> - [Deep Feature Flow](https://github.com/msracver/Deep-Feature-Flow)
>> - [Flow-Guided Feature Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation)
>> - [st-resnet](https://github.com/jay1204/st-resnet)

>> ### 2.5 Face Detection and Recognition
>> - [MXNet Face](https://github.com/tornadomeet/mxnet-face)
>> - MTCNN [[w/ train]](https://github.com/Seanlinx/mtcnn) [[caffe]](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
>> - Tiny Face [[w/ train]](https://github.com/IIMarch/tiny-face-mxnet) [[matconvnet]](https://github.com/chinakook/hr101_mxnet)
>> - [S3FD](https://github.com/zunzhumu/S3FD)
>> - [FaceDetection-ConvNet-3D](https://github.com/tfwu/FaceDetection-ConvNet-3D)
>> - [DeepID v1](https://github.com/AihahaFox/deepid-mxnet)
>> - Range Loss [[CustomOp]](https://github.com/ShownX/mxnet-rangeloss) [[gluon]](https://github.com/LeeJuly30/RangeLoss-For-Gluno)
>> - [Convolutional Sketch Inversion](https://github.com/VinniaKemala/sketch-inversion)
>> - [Face68Pts](https://github.com/LaoDar/mxnet_cnn_face68pts)
>> - [DCGAN face generation(Gluon:star:)](https://github.com/dbsheta/dcgan_face_generation)
>> - [**InsightFace**](https://github.com/deepinsight/insightface)
>> - [LightCNN](https://github.com/ly-atdawn/LightCNN-mxnet)
>> - [E2FAR](https://github.com/ShownX/mxnet-E2FAR)
>> - [FacialLandmark](https://github.com/BobLiu20/FacialLandmark_MXNet)
>> - [batch_hard_triplet_loss](https://github.com/IcewineChen/mxnet-batch_hard_triplet_loss)

>> ### 2.6 ReID
>> - [rl-multishot-reid](https://github.com/TuSimple/rl-multishot-reid)
>> - [DarkRank](https://github.com/TuSimple/DarkRank)

>> ### 2.7 Human Analyzation and Activity Recognition
>> - [Head Pose](https://github.com/LaoDar/cnn_head_pose_estimator)
>> - [Convolutional Pose Machines](https://github.com/li-haoran/mxnet-convolutional_pose_machines_Testing)
>> - [Realtime Multi-Person Pose Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation)
>> - [OpenPose](https://github.com/kohillyang/mx-openpose)
>> - [Dynamic pose estimation](https://github.com/gengshan-y/dyn_pose)
>> - [LSTM for HAR](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition)
>> - [C3D](https://github.com/JaggerYoung/C3D-mxnet)
>> - [P3D](https://github.com/IIMarch/pseudo-3d-residual-networks-mxnet)
>> - [DeepHumanPrediction](https://github.com/JONGGON/DeepHumanPrediction)
>> - [Reinspect](https://github.com/NoListen/mxnet-reinspect)
>> - [COCO Human keypoint](https://github.com/wangsr126/mxnet-pose)
>> - [**R2Plus1D**](https://github.com/starsdeep/R2Plus1D-MXNet)
>> - [**CSRNet**](https://github.com/wkcn/CSRNet-mx)

>> ### 2.8 Image Super-resolution
>> - SRCNN [[1]](https://github.com/Codersadis/SRCNN-MXNET) [[2]](https://github.com/galad-loth/SuperResolutionCNN)
>> - [**Super-Resolution-Zoo**](https://github.com/WolframRhodium/Super-Resolution-Zoo) MXNet pretrained models for super resolution, denoising and deblocking

>> ### 2.9 OCR
>> - [STN OCR](https://github.com/Bartzi/stn-ocr)
>> - [SSD Text Detection](https://github.com/oyxhust/ssd-text_detection)
>> - [EAST](https://github.com/wangpan8154/east-text-detection-with-mxnet/tree/1a63083d69954e7c1c7ac277cf6b8ed5af4ec770)
>> - [**CTPN.mxnet**](https://github.com/chinakook/CTPN.mxnet)
>> - crnn [[Chinese]](https://github.com/diaomin/crnn-mxnet-chinese-text-recognition) [[gluon]](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC) [[insightocr]](https://github.com/deepinsight/insightocr)


>> ### 2.10 Point cloud & 3D
>> - [mx-pointnet](https://github.com/Zehaos/mx-pointnet)
>> - [PointCNN.MX](https://github.com/chinakook/PointCNN.MX)
>> - [RC3D](https://github.com/likelyzhao/MxNet-RC3D/blob/master/RC3D/symbols/RC3D.py)

>> ### 2.11 Images Generation
>> - [pix2pix](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/Pix2Pix)
>> - [Image colorization](https://github.com/skirdey/mxnet-pix2pix)
>> - [Neural-Style-MMD](https://github.com/lyttonhao/Neural-Style-MMD)
>> - [MSG-Net(Gluon:star:)](https://github.com/zhanghang1989/MXNet-Gluon-Style-Transfer)
>> - [fast-style-transfer](https://github.com/SineYuan/mxnet-fast-neural-style)
>> - [neural-art-mini](https://github.com/pavelgonchar/neural-art-mini)

>> ### 2.12 GAN
>> - [DCGAN(Gluon:star:)](https://github.com/kazizzad/DCGAN-Gluon-MxNet)

>> ### 2.13 MRI & DTI
>> - [Chest-XRay](https://github.com/kperkins411/MXNet-Chest-XRay-Evaluation)
>> - [LUCAD](https://github.com/HPI-DeepLearning/LUCAD)

>> ### 2.14 Misc
>> - [VisualBackProp](https://github.com/Bartzi/visual-backprop-mxnet)
>> - VQA [[sym]](https://github.com/shiyangdaisy23/mxnet-vqa) [[gluon]](https://github.com/shiyangdaisy23/vqa-mxnet-gluon)
>> - [Hierarchical Question-Imagee Co-Attention](https://github.com/WillSuen/VQA)
>> - [text2image(Gluon:star:)](https://github.com/dbsheta/text2image)
>> - [Traffic sign classification](https://github.com/sookinoby/mxnet-ccn-samples)
>> - [cicada classification](https://github.com/dokechin/cicada_shell)
>> - [geometric-matching](https://github.com/x007dwd/geometric-matching-mxnet)
>> - [Loss Surfaces](https://github.com/nicklhy/cnn_loss_surface)
>> - [Class Activation Mapping](https://github.com/nicklhy/CAM)
>> - [AdversarialAutoEncoder](https://github.com/nicklhy/AdversarialAutoEncoder)
>> - [Neural Image Caption](https://github.com/saicoco/mxnet_image_caption)
>> - [mmd/jmmd/adaBN](https://github.com/deepinsight/transfer-mxnet)
>> - [NetVlad](https://github.com/likelyzhao/NetVlad-MxNet)
>> - [multilabel](https://github.com/miraclewkf/multilabel-MXNet)
>> - [multi-task](https://github.com/miraclewkf/multi-task-MXNet)
>> - [siamese](https://github.com/saicoco/mxnet-siamese)
>> - [matchnet](https://github.com/zhengxiawu/mxnet-matchnet)
>> - [DPSH](https://github.com/akturtle/DPSH)
>> - [Yelp Restaurant Photo Classifacation](https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification)
>> - [siamese_network_on_omniglot(Gluon:star:)](https://github.com/muchuanyun/siamese_network_on_omniglot)
>> - [StrassenNets](https://github.com/mitscha/strassennets)
>> - [Image Embedding Learning (Gluon:star:)](https://github.com/chaoyuaw/incubator-mxnet)
>> - [DIRNet](https://github.com/HPI-DeepLearning/DIRNet/tree/master/DIRNet-mxnet)
>> - [Receptive Field Tool](https://github.com/chinakook/mxnet/blob/kkmaster/python/kktools/rf.py)
>> - [mxnet-videoio](https://github.com/MTCloudVision/mxnet-videoio)
>> - [mxnet_tetris](https://github.com/sunkwei/mxnet_tetris)
>> - [Visual Search (Gluon:star:)](https://github.com/ThomasDelteil/VisualSearch_MXNet)
>> - [**DALI**](https://github.com/NVIDIA/DALI)

## <a name="NLP"></a>3. NLP
>> - [**sockeye**](https://github.com/awslabs/sockeye)
>> - [MXNMT](https://github.com/magic282/MXNMT)
>> - [Char-RNN(Gluon:star:)](https://github.com/SherlockLiao/Char-RNN-Gluon)
>> - [Character-level CNN Text Classification (Gluon:star:)](https://github.com/ThomasDelteil/CNN_NLP_MXNet)
>> - [AC-BLSTM](https://github.com/Ldpe2G/AC-BLSTM)
>> - seq2seq [[sym]](https://github.com/yoosan/mxnet-seq2seq) [[gluon]](https://github.com/ZiyueHuang/MXSeq2Seq)
>> - MemN2N [[sym]](https://github.com/nicklhy/MemN2N) [[gluon]](https://github.com/fanfeifan/MemN2N-Mxnet-Gluon)
>> - [Neural Programmer-Interpreters](https://github.com/Cloudyrie/npi)
>> - [sequence-sampling](https://github.com/doetsch/sequence-sampling-mxnet)
>> - [retrieval chatbot](https://github.com/NonvolatileMemory/baseline_for_chatbot-mxnet)
>> - [multi-attention(Gluon:star:)](https://github.com/danache/multi-attention-in-mxnet)
>> - [cnn+Highway Net](https://github.com/wut0n9/cnn_chinese_text_classification)
>> - [sentiment-analysis(Gluon:star:)](https://github.com/aws-samples/aws-sentiment-analysis-mxnet-gluon)
>> - [parserChiang(Gluon:star:)](https://github.com/linmx0130/parserChiang)
>> - [Neural Variational Document Model(Gluon:star:)](https://github.com/dingran/nvdm-mxnet)
>> - [NER with  Bidirectional LSTM-CNNs](https://github.com/opringle/named_entity_recognition)
>> - [Sequential Matching Network(Gluon:star:)](https://github.com/NonvolatileMemory/MXNET-SMN)
>> - [ko_en_NMT(Gluon:star:)](https://github.com/haven-jeon/ko_en_neural_machine_translation)
>> - [**Gluon Dynamic-batching**(Gluon:star:)](https://github.com/szha/mxnet-fold)
>> - [translatR](https://github.com/jeremiedb/translatR)
>> - [RNN-Transducer](https://github.com/HawkAaron/mxnet-transducer)
>> - [**gluon-nlp**(Gluon:star:)](https://github.com/dmlc/gluon-nlp)

## <a name="Speech"></a>4. Speech
>> - [mxnet_kaldi](https://github.com/vsooda/mxnet_kaldi)
>> - [**deepspeech**](https://github.com/samsungsds-rnd/deepspeech.mxnet)
>> - [wavenet](https://github.com/shuokay/mxnet-wavenet)
>> - [Tacotron](https://github.com/PiSchool/mxnet-tacotron)
>> - [mxnet-audio](https://github.com/chen0040/mxnet-audio)

## <a name="Time_series_forecasting"></a>5. Time series forecasting
>> - [LSTNet](https://github.com/opringle/multivariate_time_series_forecasting)

## <a name="CTR"></a>6. CTR
>> - [MXNet for CTR ](https://github.com/CNevd/DeepLearning-MXNet)
>> - [CDL](https://github.com/js05212/MXNet-for-CDL)
>> - [SpectralLDA](https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet)
>> - [DEF(Gluon:star:)](https://github.com/altosaar/deep-exponential-families-gluon)
>> - [mxnet-recommender(Gluon:star:)](https://github.com/chen0040/mxnet-recommender)
>> - [collaborative_filtering](https://github.com/opringle/collaborative_filtering)

## <a name="DRL"></a>7. DRL
>> - [DRL](https://github.com/qyxqyx/DRL)
>> - [DQN(Gluon:star:)](https://github.com/kazizzad/DQN-MxNet-Gluon)
>> - [Double DQN(Gluon:star:)](https://github.com/kazizzad/Double-DQN-MxNet-Gluon)
>> - [openai-mxnet](https://github.com/boddmg/openai-mxnet)
>> - [PPO(Gluon:star:)](https://github.com/dai-dao/PPO-Gluon)

## <a name="Neuro_evolution"></a>8. Neuro Evolution
>> - [galapagos_nao](https://github.com/jeffreyksmithjr/galapagos_nao)

## <a name="Tools"></a>9. Tools
>> ### 9.1 Converter
>> - [mxnet2tf](https://github.com/vuvko/mxnet2tf)
>> - [MXNetToMXNet](https://github.com/IIMarch/MXNetToMXNet)
>> - [MMdnn](https://github.com/Microsoft/MMdnn)
>> - [onnx-mxnet](https://github.com/onnx/onnx-mxnet)
>> - [mxnet_to_onnx](https://github.com/NVIDIA/mxnet_to_onnx)
>> - [R-Convert-json-to-symbol](https://github.com/Imshepherd/MxNetR-Convert-json-to-symbol)

>> ### 9.2 Language Bindings
>> - [mxnet.rb](https://github.com/mrkn/mxnet.rb)
>> - [mxnet.csharp](https://github.com/yajiedesign/mxnet.csharp)
>> - [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor)
>> - [dmxnet](https://github.com/sociomantic-tsunami/dmxnet)
>> - [load_op](https://github.com/DuinoDu/load_op.mxnet)
>> - [MobulaOP](https://github.com/wkcn/MobulaOP)

>> ### 9.3 Visualization
>> - [mxbox](https://github.com/Lyken17/mxbox)
>> - [mixboard](https://github.com/DrSensor/mixboard)
>> - [mxflows](https://github.com/aidan-plenert-macdonald/mxflows)
>> - [mxserver](https://github.com/Harmonicahappy/mxserver)
>> - [VisualDL](https://github.com/PaddlePaddle/VisualDL)
>> - [mxProfileParser](https://github.com/TaoLv/mxProfileParser)
>> - [polyaxon](https://github.com/polyaxon/polyaxon) with [examples](https://github.com/polyaxon/polyaxon-examples/tree/master/mxnet/cifar10)
>> - [Netron](https://github.com/lutzroeder/Netron)
>> - [**mxboard**](https://github.com/awslabs/mxboard)

>> ### 9.4 Parallel and Distributed computing
>> - [mxnet-rdma](https://github.com/liuchang1437/mxnet-rdma)
>> - [RDMA-MXNet-ps-lite](https://github.com/ralzq01/RDMA-MXNet-ps-lite)
>> - [MPIZ-MXNet](https://github.com/Shenggan/MPIZ-MXNet)
>> - [MXNetOnYARN](https://github.com/Intel-bigdata/MXNetOnYARN)
>> - [mxnet-operator](https://github.com/deepinsight/mxnet-operator)
>> - [mxnet_on_kubernetes](https://github.com/WorldAITime/mxnet_on_kubernetes)
>> - [speculative-synchronization](https://github.com/All-less/mxnet-speculative-synchronization)
>> - [XLearning](https://github.com/Qihoo360/XLearning)
>> - [Gluon Distributed Training (Gluon:star:)](https://mxnet.indu.ai/tutorials/distributed-training-using-mxnet)
>> - [gpurelperf](https://github.com/andylamp/gpurelperf)

>> ### 9.5 Productivity
>> - [Email Monitor MxnetTrain](https://github.com/fierceX/Email_Monitor_MxnetTrain)
>> - [mxnet-oneclick](https://github.com/imistyrain/mxnet-oneclick)
>> - [mxnet-finetuner](https://github.com/knjcode/mxnet-finetuner)
>> - [Early-Stopping](https://github.com/kperkins411/MXNet_Demo_Early-Stopping)
>> - [MXNet_Video_Trainer](https://github.com/likelyzhao/MXNet_Video_Trainer)
>> - [rs_mxnet_reader](https://github.com/ChenKQ/rs_mxnet_reader)

>> ### 9.6 Parameter optimizer
>> - [YellowFin](https://github.com/StargazerZhu/YellowFin_MXNet)

>> ### 9.7 Deployment
>> - [Turi Create](https://github.com/apple/turicreate)
>> - [MXNet-HRT](https://github.com/OAID/MXNet-HRT)
>> - [Tengine](https://github.com/OAID/Tengine)
>> - [Collective Knowledge](https://github.com/ctuning/ck-mxnet)
>> - [flask-app-for-mxnet-img-classifier](https://github.com/XD-DENG/flask-app-for-mxnet-img-classifier)
>> - [qt-mxnet](https://github.com/mjamroz/qt-mxnet)
>> - [mxnet_predict_ros](https://github.com/Paspartout/mxnet_predict_ros)
>> - [mxnet-lambda](https://github.com/awslabs/mxnet-lambda)
>> - [openHabAI](https://github.com/JeyRunner/openHabAI)
>> - ImageRecognizer [[iOS]](https://github.com/dneprDroid/ImageRecognizer-iOS) [[Android]](https://github.com/dneprDroid/ImageRecognizer-Android)
>> - [MXNet to MiniFi](https://github.com/tspannhw/nvidiajetsontx1-mxnet)
>> - [MXNet Model Serving ](https://github.com/yuruofeifei/mms)
>> - [mxnet-model-server](https://github.com/awslabs/mxnet-model-server)
>> - [tvm-mali](https://github.com/merrymercy/tvm-mali)
>> - [mxnet-and-sagemaker](https://github.com/cosmincatalin/object-counting-with-mxnet-and-sagemaker)
>> - [example-of-nnvm-in-cpp](https://github.com/zhangxinqian/example-of-nnvm-in-cpp)

>> ### 9.8 Other Branches
>> - [ngraph-mxnet](https://github.com/NervanaSystems/ngraph-mxnet)
>> - [distributedMXNet](https://github.com/TuSimple/distributedMXNet)

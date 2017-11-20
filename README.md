# Awesome MXNet(Beta) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/jtoy/awesome)

A curated list of MXNet examples, tutorials and blogs. It is inspired by awesome-caffe.

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.

## Table of Contents
- [1. Tutorials](#Tutorials)
- [2. Vision](#Vision)
- [3. NLP](#NLP)
- [4. Speech](#Speech)
- [5. CTR](#CTR)
- [6. DRL](#DRL)
- [7. Tools](#Tools)

============================================================================================================
## <a name="Tutorials"></a>1. Tutorials
- [incubator-mxnet-site](https://github.com/apache/incubator-mxnet-site)
- [Tutorial Documentation](https://mxnet.incubator.apache.org/tutorials/)
- [Gluon Tutorial Documentation](http://gluon.mxnet.io/)
- [Gluon Tutorial Documentation (Simplified Chinese)](https://zh.gluon.ai/)
- [Gluon Api](https://github.com/gluon-api/gluon-api)
- [CheatSheet](https://github.com/chinakook/Awesome-MXNet/blob/master/apache-mxnet-cheat.pdf)
- [Using MXNet](https://github.com/JONGGON/Mxnet_Tutorial)
- [TVM Documentation](http://docs.tvmlang.org/)
- [NNVM Documentation](http://nnvm.tvmlang.org/)
- [Linalg examples](https://github.com/ARCambridge/MXNet_linalg_examples)

## <a name="Vision"></a>2. Vision
>> ### 2.1 Image Classification
>> - [ResNet](https://github.com/tornadomeet/ResNet)
>> - [DenseNet](https://github.com/bruinxiong/densenet.mxnet)
>> - [DPN](https://github.com/cypw/DPNs)
>> - [SENet](https://github.com/bruinxiong/SENet.mxnet)
>> - [CapsNet(Gluon:star:)](https://github.com/AaronLeong/CapsNet_Mxnet)
>> - [capsules(Gluon:star:)](https://github.com/mssmkmr/capsules)
>> - [CRU-Net](https://github.com/cypw/CRU-Net)
>> - [MobileNet](https://github.com/KeyKy/mobilenet-mxnet)
>> - [ShuffleNet](https://github.com/ZiyueHuang/MXShuffleNet)
>> - [Xception](https://github.com/bruinxiong/xception.mxnet)
>> - [Xception+Keras2MXNet](https://github.com/u1234x1234/mxnet-xception)
>> - [SqeezeNet](https://github.com/miaow1988/SqueezeNet_v1.2)
>> - [FractalNet](https://github.com/newuser-16824/mxnet-fractalnet)
>> - [BMXNet](https://github.com/hpi-xnor/BMXNet)
>> - [fusenet](https://github.com/zlmzju/fusenet)
>> - [Self-Norm Nets](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/SelfNormNets)
>> - [Factorized-Bilinear-Network](https://github.com/lyttonhao/Factorized-Bilinear-Network)

>> ### 2.2 Object Detection
>> - [PVANet](https://github.com/apache/incubator-mxnet/pull/7786)
>> - [SSD](https://github.com/zhreshold/mxnet-ssd)
>> - [YOLO](https://github.com/zhreshold/mxnet-yolo)
>> - [YOLO/dark2mxnet](https://github.com/bowenc0221/MXNet-YOLO)
>> - [Faster RCNN](https://github.com/precedenceguo/mx-rcnn)
>> - [Faster RCNN(Gluon:star:)](https://github.com/linmx0130/ya_mxdet)
>> - [R-FCN](https://github.com/msracver/Deformable-ConvNets)
>> - [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)
>> - [Deformable-ConvNets+SoftNMS](https://github.com/bharatsingh430/Deformable-ConvNets)
>> - [SSD+Focal Loss](https://github.com/eldercrow/focal_loss_mxnet_ssd)
>> - [Faster RCNN+Focal Loss](https://github.com/unsky/focal-loss)
>> - [RetinaNet](https://github.com/unsky/RetinaNet)
>> - [SqueezeDet](https://github.com/alvinwan/squeezeDetMX)
>> - [IOULoss](https://github.com/wcj-Ford/IOULoss)
>> - [FocalLoss(CUDA)](https://github.com/yuantangliang/softmaxfocalloss)

>> ### 2.3 Image Segmentation
>> - [Mask R-CNN+FPN](https://github.com/TuSimple/mx-maskrcnn)
>> - [Mask R-CNN](https://github.com/xilaili/maskrcnn.mxnet)
>> - [DUC](https://github.com/TuSimple/TuSimple-DUC)
>> - [FCIS](https://github.com/msracver/FCIS)
>> - [ResNet-38](https://github.com/itijyou/ademxapp)
>> - [Deeplab v2](https://github.com/buptweixin/mxnet-deeplab)
>> - [Deeplab(Gluon:star:)](https://github.com/zehaochen19/segmentation_gluon)
>> - [U-Net(Gluon:star:)](https://github.com/chinakook/U-Net)
>> - [U-Net(kaggle dstl)](https://github.com/u1234x1234/kaggle-dstl-satellite-imagery-feature-detection)
>> - [Segnet](https://github.com/solin319/incubator-mxnet/tree/solin-patch-segnet)
>> - [FCN-ASPP](https://github.com/ComeOnGetMe/FCN-ASPP-with-uncertainty)
>> - [GluonSeg(Gluon:star:)](https://github.com/aurora95/GluonSeg)

>> ### 2.4 Video Recognition and Object Detection
>> - [Deep Feature Flow](https://github.com/msracver/Deep-Feature-Flow)
>> - [Flow-Guided Feature Aggregation](https://github.com/msracver/Flow-Guided-Feature-Aggregation)
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
>> - [Center Loss(Gluon:star:)](https://github.com/ShownX/mxnet-center-loss)
>> - [Large-Margin Softmax Loss](https://github.com/luoyetx/mx-lsoftmax)
>> - [Range Loss](https://github.com/ShownX/mxnet-rangeloss)
>> - [Convolutional Sketch Inversion](https://github.com/VinniaKemala/sketch-inversion)
>> - [Convolutional Pose Machines](https://github.com/li-haoran/mxnet-convolutional_pose_machines_Testing)
>> - [Realtime Multi-Person Pose Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation)
>> - [OpenPose](https://github.com/kohillyang/mx-openpose)
>> - [Face68Pts](https://github.com/LaoDar/mxnet_cnn_face68pts)
>> - [Dynamic pose estimation](https://github.com/gengshan-y/dyn_pose)
>> - [LSTM for HAR](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition)
>> - [C3D](https://github.com/JaggerYoung/C3D-mxnet)
>> - [DeepHumanPrediction](https://github.com/JONGGON/DeepHumanPrediction)
>> - [SphereFace/FocalLoss/CenterLoss/KNNLoss(CUDA)](https://github.com/deepearthgo/Cuda-Mxnet)
>> - [SphereFace/KNNLoss(Python)](https://github.com/deepearthgo/Python-Mxnet)
>> - [DCGAN face generation(Gluon:star:)](https://github.com/dbsheta/dcgan_face_generation)

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
>> - [Image colorization](https://github.com/skirdey/mxnet-pix2pix)
>> - [Neural-Style-MMD](https://github.com/lyttonhao/Neural-Style-MMD)
>> - [MSG-Net(Gluon:star:)](https://github.com/zhanghang1989/MXNet-Gluon-Style-Transfer)
>> - [fast-style-transfer](https://github.com/SineYuan/mxnet-fast-neural-style)
>> - [neural-art-mini](https://github.com/pavelgonchar/neural-art-mini)

>> ### 2.9 GAN
>> - [DCGAN(Gluon:star:)](https://github.com/kazizzad/DCGAN-Gluon-MxNet)

>> ### 2.10 MRI&DTI
>> - [Chest-XRay](https://github.com/kperkins411/MXNet-Chest-XRay-Evaluation)

>> ### 2.11 Misc
>> - [VisualBackProp](https://github.com/Bartzi/visual-backprop-mxnet)
>> - [VQA](https://github.com/shiyangdaisy23/mxnet-vqa)
>> - [VQA(Gluon:star:)](https://github.com/shiyangdaisy23/vqa-mxnet-gluon)
>> - [text2image(Gluon:star:)](https://github.com/dbsheta/text2image)
>> - [Traffic sign classification](https://github.com/sookinoby/mxnet-ccn-samples)
>> - [cicada classification](https://github.com/dokechin/cicada_shell)
>> - [geometric-matching](https://github.com/x007dwd/geometric-matching-mxnet)
>> - [Loss Surfaces](https://github.com/nicklhy/cnn_loss_surface)
>> - [CAM](https://github.com/nicklhy/CAM)
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

## <a name="NLP"></a>3. NLP
>> - [sockeye](https://github.com/awslabs/sockeye)
>> - [MXNMT](https://github.com/magic282/MXNMT)
>> - [Char-RNN(Gluon:star:)](https://github.com/SherlockLiao/Char-RNN-Gluon)
>> - [AC-BLSTM](https://github.com/Ldpe2G/AC-BLSTM)
>> - [mxnet-seq2seq](https://github.com/yoosan/mxnet-seq2seq)
>> - [MXSeq2Seq(Gluon:star:)](https://github.com/ZiyueHuang/MXSeq2Seq)
>> - [MemN2N](https://github.com/nicklhy/MemN2N)
>> - [Neural Programmer-Interpreters](https://github.com/Cloudyrie/npi)
>> - [sequence-sampling](https://github.com/doetsch/sequence-sampling-mxnet)
>> - [retrieval chatbot](https://github.com/NonvolatileMemory/baseline_for_chatbot-mxnet)
>> - [multi-attention(Gluon:star:)](https://github.com/danache/multi-attention-in-mxnet)
>> - [cnn+Highway Net](https://github.com/wut0n9/cnn_chinese_text_classification)

## <a name="Speech"></a>4. Speech
>> - [deepspeech](https://github.com/samsungsds-rnd/deepspeech.mxnet)
>> - [wavenet](https://github.com/shuokay/mxnet-wavenet)

## <a name="CTR"></a>5. CTR
>> - [MXNet for CTR ](https://github.com/CNevd/DeepLearning-MXNet)
>> - [CDL](https://github.com/js05212/MXNet-for-CDL)
>> - [SpectralLDA](https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet)
>> - [DEF(Gluon:star:)](https://github.com/altosaar/deep-exponential-families-gluon)

## <a name="DRL"></a>6. DRL
>> - [DRL](https://github.com/qyxqyx/DRL)
>> - [DQN(Gluon:star:)](https://github.com/kazizzad/DQN-MxNet-Gluon)
>> - [Double DQN(Gluon:star:)](https://github.com/kazizzad/Double-DQN-MxNet-Gluon)
>> - [openai-mxnet](https://github.com/boddmg/openai-mxnet)

## <a name="Tools"></a>7. Tools
>> ### 7.1 Converter
>> - [CaffeTranslator](https://github.com/indhub/CaffeTranslator)
>> - [mxnet2tf](https://github.com/vuvko/mxnet2tf)
>> - [MXNetToMXNet](https://github.com/IIMarch/MXNetToMXNet)

>> ### 7.2 Language Bindings
>> - [mxnet.rb](https://github.com/mrkn/mxnet.rb)
>> - [mxnet.csharp](https://github.com/yajiedesign/mxnet.csharp)
>> - [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor)
>> - [dmxnet](https://github.com/sociomantic-tsunami/dmxnet)

>> ### 7.3 Visualization
>> - [mxbox](https://github.com/Lyken17/mxbox)
>> - [mixboard](https://github.com/DrSensor/mixboard)
>> - [mxflows](https://github.com/aidan-plenert-macdonald/mxflows)

>> ### 7.4 Parallel and Distributed computing
>> - [mxnet-rdma](https://github.com/liuchang1437/mxnet-rdma)
>> - [RDMA-MXNet-ps-lite](https://github.com/ralzq01/RDMA-MXNet-ps-lite)
>> - [MPIZ-MXNet](https://github.com/Shenggan/MPIZ-MXNet)
>> - [MXNetOnYARN](https://github.com/Intel-bigdata/MXNetOnYARN)
>> - [mxnet-operator](https://github.com/deepinsight/mxnet-operator)
>> - [mxnet_on_kubernetes](https://github.com/WorldAITime/mxnet_on_kubernetes)

>> ### 7.5 Productivity
>> - [MXNet Model Serving ](https://github.com/yuruofeifei/mms)
>> - [Email Monitor MxnetTrain](https://github.com/fierceX/Email_Monitor_MxnetTrain)
>> - [mxnet-oneclick](https://github.com/imistyrain/mxnet-oneclick)
>> - [mxnet-finetuner](https://github.com/knjcode/mxnet-finetuner)
>> - [Early-Stopping](https://github.com/kperkins411/MXNet_Demo_Early-Stopping)
>> - [MXNet_Video_Trainer](https://github.com/likelyzhao/MXNet_Video_Trainer)
>> - [rs_mxnet_reader](https://github.com/ChenKQ/rs_mxnet_reader)

>> ### 7.6 Parameter optimizer
>> - [YellowFin](https://github.com/StargazerZhu/YellowFin_MXNet)

>> ### 7.7 Deployment
>> - [onnx-mxnet](https://github.com/onnx/onnx-mxnet)
>> - [MXNetOnACL](https://github.com/OAID/MXNetOnACL)
>> - [Collective Knowledge](https://github.com/ctuning/ck-mxnet)
>> - [flask-app-for-mxnet-img-classifier](https://github.com/XD-DENG/flask-app-for-mxnet-img-classifier)
>> - [qt-mxnet](https://github.com/mjamroz/qt-mxnet)
>> - [mxnet_predict_ros](https://github.com/Paspartout/mxnet_predict_ros)
>> - [mxnet-lambda](https://github.com/awslabs/mxnet-lambda)
>> - [openHabAI](https://github.com/JeyRunner/openHabAI)
>> - [ImageRecognizer-iOS](https://github.com/dneprDroid/ImageRecognizer-iOS)
>> - [ImageRecognizer-Android](https://github.com/dneprDroid/ImageRecognizer-Android)
>> - [MXNet to MiniFi](https://github.com/tspannhw/nvidiajetsontx1-mxnet)

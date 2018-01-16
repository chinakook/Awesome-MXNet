# Awesome MXNet(Beta) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/jtoy/awesome)

A curated list of MXNet examples, tutorials and blogs. It is inspired by awesome-caffe.

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.

## Table of Contents
- [1. Tutorials](#Tutorials)
- [2. Vision](#Vision)
- [3. NLP](#NLP)
- [4. Speech](#Speech)
- [5. Time series forecasting](#Time_series_forecasting)
- [6. CTR](#CTR)
- [7. DRL](#DRL)
- [8. Tools](#Tools)

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
- [NNVM Vison Demo](https://github.com/masahi/nnvm-vision-demo)

## <a name="Vision"></a>2. Vision
>> ### 2.1 Image Classification
>> - [ResNet](https://github.com/tornadomeet/ResNet)
>> - [DenseNet](https://github.com/bruinxiong/densenet.mxnet)
>> - [DPN](https://github.com/cypw/DPNs)
>> - [SENet](https://github.com/bruinxiong/SENet.mxnet)
>> - [SENet(from Caffe)](https://github.com/IIMarch/SENet-mxnet)
>> - [CapsNet](https://github.com/Soonhwan-Kwon/capsnet.mxnet)
>> - [NASNet-A(Gluon:star:)](https://github.com/qingzhouzhen/incubator-mxnet/blob/nasnet/python/mxnet/gluon/model_zoo/vision/nasnet.py)
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
>> - [AOGNet](https://github.com/xilaili/AOGNet)
>> - [mixup](https://github.com/unsky/mixup)
>> - [mxnet-model](https://github.com/soeaver/mxnet-model)

>> ### 2.2 Object Detection
>> - [PVANet](https://github.com/apache/incubator-mxnet/pull/7786)
>> - [SSD](https://github.com/zhreshold/mxnet-ssd)
>> - [YOLO](https://github.com/zhreshold/mxnet-yolo)
>> - [YOLO(Gluon:star:)](https://github.com/MashiMaroLjc/YOLO)
>> - [YOLO/dark2mxnet](https://github.com/bowenc0221/MXNet-YOLO)
>> - [Faster RCNN](https://github.com/precedenceguo/mx-rcnn)
>> - [Faster RCNN(Gluon:star:)](https://github.com/linmx0130/ya_mxdet)
>> - [Faster RCNN+Deeplab+R-FCN+Deformable-ConvNets+FPN+SoftNMS](https://github.com/msracver/Deformable-ConvNets)
>> - [FCIS](https://github.com/msracver/FCIS)
>> - [Mask R-CNN](https://github.com/TuSimple/mx-maskrcnn)
>> - [SSD+Focal Loss](https://github.com/eldercrow/focal_loss_mxnet_ssd)
>> - [Faster RCNN+Focal Loss](https://github.com/unsky/focal-loss)
>> - [RetinaNet](https://github.com/unsky/RetinaNet)
>> - [SqueezeDet](https://github.com/alvinwan/squeezeDetMX)
>> - [IOULoss](https://github.com/wcj-Ford/IOULoss)
>> - [FocalLoss(CUDA)](https://github.com/yuantangliang/softmaxfocalloss)
>> - [Light-Head R-CNN](https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784)

>> ### 2.3 Image Segmentation
>> - [DUC](https://github.com/TuSimple/TuSimple-DUC)
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
>> - [P3D](https://github.com/IIMarch/pseudo-3d-residual-networks-mxnet)
>> - [InsightFace](https://github.com/deepinsight/insightface)
>> - [rl-multishot-reid](https://github.com/TuSimple/rl-multishot-reid)
>> - [LightCNN](https://github.com/ly-atdawn/LightCNN-mxnet)

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

>> ### 2.8 Point cloud & 3D
>> - [mx-pointnet](https://github.com/Zehaos/mx-pointnet)

>> ### 2.9 Images Generation
>> - [pix2pix](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/Pix2Pix)
>> - [Image colorization](https://github.com/skirdey/mxnet-pix2pix)
>> - [Neural-Style-MMD](https://github.com/lyttonhao/Neural-Style-MMD)
>> - [MSG-Net(Gluon:star:)](https://github.com/zhanghang1989/MXNet-Gluon-Style-Transfer)
>> - [fast-style-transfer](https://github.com/SineYuan/mxnet-fast-neural-style)
>> - [neural-art-mini](https://github.com/pavelgonchar/neural-art-mini)

>> ### 2.10 GAN
>> - [DCGAN(Gluon:star:)](https://github.com/kazizzad/DCGAN-Gluon-MxNet)

>> ### 2.11 MRI & DTI
>> - [Chest-XRay](https://github.com/kperkins411/MXNet-Chest-XRay-Evaluation)
>> - [LUCAD](https://github.com/HPI-DeepLearning/LUCAD)

>> ### 2.12 Misc
>> - [VisualBackProp](https://github.com/Bartzi/visual-backprop-mxnet)
>> - [VQA](https://github.com/shiyangdaisy23/mxnet-vqa)
>> - [VQA(Gluon:star:)](https://github.com/shiyangdaisy23/vqa-mxnet-gluon)
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
>> - [Image Embedding Learning(Gluon:star:)](https://github.com/chaoyuaw/incubator-mxnet)
>> - [DIRNet](https://github.com/HPI-DeepLearning/DIRNet/tree/master/DIRNet-mxnet)

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
>> - [sentiment-analysis(Gluon:star:)](https://github.com/aws-samples/aws-sentiment-analysis-mxnet-gluon)
>> - [parserChiang(Gluon:star:)](https://github.com/linmx0130/parserChiang)
>> - [Neural Variational Document Model(Gluon:star:)](https://github.com/dingran/nvdm-mxnet)
>> - [NER with  Bidirectional LSTM-CNNs](https://github.com/opringle/named_entity_recognition)
>> - [Sequential Matching Network(Gluon:star:)](https://github.com/NonvolatileMemory/MXNET-SMN)

## <a name="Speech"></a>4. Speech
>> - [deepspeech](https://github.com/samsungsds-rnd/deepspeech.mxnet)
>> - [wavenet](https://github.com/shuokay/mxnet-wavenet)
>> - [openspeech(Gluon:star:)](https://github.com/awslabs/openspeech)
>> - [Tacotron](https://github.com/PiSchool/mxnet-tacotron)

## <a name="Time_series_forecasting"></a>5. Time series forecasting
>> - [LSTNet](https://github.com/opringle/multivariate_time_series_forecasting)

## <a name="CTR"></a>6. CTR
>> - [MXNet for CTR ](https://github.com/CNevd/DeepLearning-MXNet)
>> - [CDL](https://github.com/js05212/MXNet-for-CDL)
>> - [SpectralLDA](https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet)
>> - [DEF(Gluon:star:)](https://github.com/altosaar/deep-exponential-families-gluon)

## <a name="DRL"></a>7. DRL
>> - [DRL](https://github.com/qyxqyx/DRL)
>> - [DQN(Gluon:star:)](https://github.com/kazizzad/DQN-MxNet-Gluon)
>> - [Double DQN(Gluon:star:)](https://github.com/kazizzad/Double-DQN-MxNet-Gluon)
>> - [openai-mxnet](https://github.com/boddmg/openai-mxnet)
>> - [PPO(Gluon:star:)](https://github.com/dai-dao/PPO-Gluon)

## <a name="Tools"></a>8. Tools
>> ### 8.1 Converter
>> - [CaffeTranslator](https://github.com/indhub/CaffeTranslator)
>> - [mxnet2tf](https://github.com/vuvko/mxnet2tf)
>> - [MXNetToMXNet](https://github.com/IIMarch/MXNetToMXNet)
>> - [MMdnn](https://github.com/Microsoft/MMdnn)
>> - [onnx-mxnet](https://github.com/onnx/onnx-mxnet)
>> - [mxnet_to_onnx](https://github.com/NVIDIA/mxnet_to_onnx)
>> - [R-Convert-json-to-symbol](https://github.com/Imshepherd/MxNetR-Convert-json-to-symbol)

>> ### 8.2 Language Bindings
>> - [mxnet.rb](https://github.com/mrkn/mxnet.rb)
>> - [mxnet.csharp](https://github.com/yajiedesign/mxnet.csharp)
>> - [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor)
>> - [dmxnet](https://github.com/sociomantic-tsunami/dmxnet)

>> ### 8.3 Visualization
>> - [mxbox](https://github.com/Lyken17/mxbox)
>> - [mixboard](https://github.com/DrSensor/mixboard)
>> - [mxflows](https://github.com/aidan-plenert-macdonald/mxflows)
>> - [mxboard](https://github.com/Harmonicahappy/mxboard)
>> - [VisualDL](https://github.com/PaddlePaddle/VisualDL)

>> ### 8.4 Parallel and Distributed computing
>> - [mxnet-rdma](https://github.com/liuchang1437/mxnet-rdma)
>> - [RDMA-MXNet-ps-lite](https://github.com/ralzq01/RDMA-MXNet-ps-lite)
>> - [MPIZ-MXNet](https://github.com/Shenggan/MPIZ-MXNet)
>> - [MXNetOnYARN](https://github.com/Intel-bigdata/MXNetOnYARN)
>> - [mxnet-operator](https://github.com/deepinsight/mxnet-operator)
>> - [mxnet_on_kubernetes](https://github.com/WorldAITime/mxnet_on_kubernetes)
>> - [speculative-synchronization](https://github.com/All-less/mxnet-speculative-synchronization)

>> ### 8.5 Productivity
>> - [Email Monitor MxnetTrain](https://github.com/fierceX/Email_Monitor_MxnetTrain)
>> - [mxnet-oneclick](https://github.com/imistyrain/mxnet-oneclick)
>> - [mxnet-finetuner](https://github.com/knjcode/mxnet-finetuner)
>> - [Early-Stopping](https://github.com/kperkins411/MXNet_Demo_Early-Stopping)
>> - [MXNet_Video_Trainer](https://github.com/likelyzhao/MXNet_Video_Trainer)
>> - [rs_mxnet_reader](https://github.com/ChenKQ/rs_mxnet_reader)

>> ### 8.6 Parameter optimizer
>> - [YellowFin](https://github.com/StargazerZhu/YellowFin_MXNet)

>> ### 8.7 Deployment
>> - [Turi Create](https://github.com/apple/turicreate)
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
>> - [MXNet Model Serving ](https://github.com/yuruofeifei/mms)
>> - [mxnet-model-server](https://github.com/awslabs/mxnet-model-server)

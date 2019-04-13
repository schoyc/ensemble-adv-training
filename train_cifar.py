from train_cifar10_resnet import *

# ResNet44 v1
model1, model1_name = train_resnet(version=1, n=7)
# ResNet56 v2
model2, model2_name = train_resnet(version=2, n=6)
# ResNet110 v2
model3, model3_name = train_resnet(version=2, n=12)

# ERA-SESSION8
## Goals:
 - Experiment with different normalization methods like BatchNorm, GroupNorm, LayerNorm
 - Try to add skip connections and understand how it benefits overall results
 - Keep the model params under 50k and achieve accuracy of >= 70% under 20 epochs

## Results:
Normalization  |   Train Accuracy  |   Test Accuracy
----------------------------------------------------
BatchNorm      |   77.30%          |   76.24%
GroupNorm      |    75.09%         |   75.26%
LayerNorm      |    75.26%         |    72.04%

## BatchNorm:
### Model Summary:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
            Conv2d-4           [-1, 32, 32, 32]           4,608
              ReLU-5           [-1, 32, 32, 32]               0
       BatchNorm2d-6           [-1, 32, 32, 32]              64
            Conv2d-7           [-1, 16, 32, 32]             512
              ReLU-8           [-1, 16, 32, 32]               0
         MaxPool2d-9           [-1, 16, 16, 16]               0
           Conv2d-10           [-1, 16, 16, 16]           2,304
             ReLU-11           [-1, 16, 16, 16]               0
      BatchNorm2d-12           [-1, 16, 16, 16]              32
           Conv2d-13           [-1, 16, 16, 16]           2,304
             ReLU-14           [-1, 16, 16, 16]               0
      BatchNorm2d-15           [-1, 16, 16, 16]              32
           Conv2d-16           [-1, 32, 16, 16]           4,608
             ReLU-17           [-1, 32, 16, 16]               0
      BatchNorm2d-18           [-1, 32, 16, 16]              64
           Conv2d-19           [-1, 16, 16, 16]             512
             ReLU-20           [-1, 16, 16, 16]               0
        MaxPool2d-21             [-1, 16, 8, 8]               0
           Conv2d-22             [-1, 16, 8, 8]           2,304
             ReLU-23             [-1, 16, 8, 8]               0
      BatchNorm2d-24             [-1, 16, 8, 8]              32
           Conv2d-25             [-1, 16, 8, 8]           2,304
             ReLU-26             [-1, 16, 8, 8]               0
      BatchNorm2d-27             [-1, 16, 8, 8]              32
           Conv2d-28             [-1, 16, 8, 8]           2,304
             ReLU-29             [-1, 16, 8, 8]               0
      BatchNorm2d-30             [-1, 16, 8, 8]              32
AdaptiveAvgPool2d-31             [-1, 16, 1, 1]               0
           Conv2d-32             [-1, 10, 1, 1]             160
================================================================
Total params: 22,672
Trainable params: 22,672
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.92
Params size (MB): 0.09
Estimated Total Size (MB): 2.02
----------------------------------------------------------------
```
### Metric Curves
![image](https://github.com/RaviNaik/ERA-SESSION8/assets/23289802/302766c7-7a45-4eb0-86c7-279d43053c4f)

### Misclassified Images
![image](https://github.com/RaviNaik/ERA-SESSION8/assets/23289802/7a0c384c-58d9-4afc-b41b-6e07023f1a7b)

## GroupNorm:
### Metric Curves
![image](https://github.com/RaviNaik/ERA-SESSION8/assets/23289802/79c21b6b-741a-40a9-8ea1-ccd7a590329e)

### Misclassified Images
![image](https://github.com/RaviNaik/ERA-SESSION8/assets/23289802/5c5b1503-45bb-444b-85ea-c3bf955b7c77)


## LayerNorm:
### Metric Curves
![image](https://github.com/RaviNaik/ERA-SESSION8/assets/23289802/a03741b4-41de-4045-b248-3e9b3df585dd)

### Misclassified Images
![image](https://github.com/RaviNaik/ERA-SESSION8/assets/23289802/a371c138-f725-42be-bfc4-c2bdb7f449af)



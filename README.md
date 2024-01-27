### Convolutional Neural Network for Malaria Diagnosis.

CNN is using Binary Classification to indentify infected blood cells.

### *Model: "sequential"*

| Layer (type)                   | Output Shape         | Param # |
|--------------------------------|----------------------|---------|
| conv2d_2 (Conv2D)              | (None, 220, 220, 6)  | 456     |
| batch_normalization            | (None, 220, 220, 6)  | 24      |
| max_pooling2d_2 (MaxPooling2D) | (None, 110, 110, 6)  | 0       |
| conv2d_2 (Conv2D)              | (None, 106, 106, 16) | 2416    |
| batch_normalization            | (None, 106, 106, 16) | 64      |
| max_pooling2d_2 (MaxPooling2D) | (None, 53, 53, 16)   | 0       |
| flatten_1 (Flatten)            | (None, 44944)        | 0       |
| dense_3 (Dense)                | (None, 100)          | 4494500 |
| batch_normalization            | (None, 100)          | 400     |
| dense_4 (Dense)                | (None, 10)           | 1010    |
| batch_normalization            | (None, 10)           | 40      |
| dense_5 (Dense)                | (None, 1)            | 11      |

                                                
=================================================================
Total params: 4498404 (17.16 MB) \
Trainable params: 4498404 (17.16 MB) \
Non-trainable params: 0 (0.00 Byte) 
_________________________________________________________________


### Dataset
Malaria dataset provides labeled images.
All images are labeled either infected with malaria parasite or not.

![dataset](/Images/Malaria_dataset.png "Dataset used in project.")

---

### Model training.
Model was trained in 50 epochs.
With results:
* loss on training dataset = 0.0249
* loss on validation dataset = 0.3612
* accuracy on training dataset = 0.9923
* accuracy on validation dataset = 0.9463

![model_loss](/Images/Model_loss.png "Training loss vs validation.")
===
![model_accuracy](/Images/Model_accuracy.png "Training accuracy vs validation.")
===
***
### Model evaluation.

Model evaluation results for test dataset are:
*loss = 0.3854
*accuracy = 0.9426

As final test model was used to classify nine images.
![model_predictions](/Images/Model_predictions.png "Model predictions.")

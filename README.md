# digits-recognition-using-cnn

This is a digits recognition program using cnn.It contains two parts.First part is the trainning part.It uses tensorflow to train the cnn model.After training complete,a model file will be saved.To train it,type following command:

> python train_cnn.py

Another part is the prediction part.It load the image to be recognized and the trained model.To use it,type following command:

>python predict_cnn.py --image data/imageName 

The training dataset can be MNIST or other dataset on the internet.

Simple test result:

![Aaron Swartz](https://github.com/nicholas-tien/digits-recognition-using-cnn/blob/master/data/test_result.png)

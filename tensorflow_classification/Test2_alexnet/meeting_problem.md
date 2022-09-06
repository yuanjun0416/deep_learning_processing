# meeting_problem

### (1) model.save()和model.save_weights()的区别(tensorflow.keras, 其它的深度学习框架大同小异)  trian.py: eighty-six line
  
  * model.save保存了模型的图结构和模型的参数, 保存模型的后缀为.hdf5或.h5(查看其它的文章)
    ```
    model.save()保存了模型的图结构, 直接使用load_model()方法加载模型结构做测试, 例如:
    
    from tensorflow.keras.models import load_model
    model = load_model('my_model_.hdf5')
    ```
  * model.save_weights()只保存了模型的参数，没有保存模型的图结构，保存模型的后缀是.h5
    ```
    model.save_weights()保存的模型就稍微复杂了一些，还需要再次描述模型结构才能再次加载模型
     
    def AlexNetv1(input_shape=(im_hsize, im_wsize, 3), num_class=1000):
      input_images = layers.Input(shape=input_shape, dtype='float32')
      x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_images)
      x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)
      x = layers.MaxPool2D(pool_size=3, strides=2)(x)
                           .
                           .
                           .
      predict = layers.Softmax()(x)
      model = models.Model(input=input_images, output=predict)
      return model
      
    model = AlexNetv1()
    model.load_weights('my_model_.h5')
    ```
  

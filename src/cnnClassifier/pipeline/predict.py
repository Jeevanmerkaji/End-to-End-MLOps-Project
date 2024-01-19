import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



# class PredictionPipeline:
#     def __init__(self,filename):
#         self.filename =filename


    
#     def predict(self):
#         # load model
#         model = load_model(os.path.join("artifacts","training", "model.h5"))

#         imagename = self.filename
#         test_image = image.load_img(imagename, target_size = (224,224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = np.argmax(model.predict(test_image), axis=1)
#         print(result)

#         if result[0] == 1:
#             prediction = 'Healthy'
#             return [{ "image" : prediction}]
#         else:
#             prediction = 'Coccidiosis'
#             return [{ "image" : prediction}]


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Assuming you have 4 classes, change the number of units in the Dense layer to 4
        num_classes = 4
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'Cloudy'
            return [{"image": prediction}]
        elif result[0] == 1:
            prediction = 'Desert'
            return [{"image": prediction}]
        elif result[0] == 2:
            prediction = 'green_area'
            return [{"image": prediction}]
        else:
            prediction = 'water'
            return [{"image": prediction}]

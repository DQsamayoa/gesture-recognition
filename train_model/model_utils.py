import tensorflow.keras.applications as tf_app
from tensorflow.keras.models import Model, load_model
import os


class buildModel:

    def __init__(self):
        """
        Parameters
        ----------

        Raises
        ------
        NotImplementedError

        """

        # Create and normalize the path to store the models
        store_path = os.path.join('.', 'models')
        store_path = os.path.abspath(store_path)
        store_path = os.path.normpath(store_path)
        self.trained_model_path = os.path.join(store_path, 'trained_models')
        self.base_model_path = os.path.join(store_path, 'base_models')

        # Validating if the directories exist
        if not os.path.exists(self.trained_model_path):
            os.makedirs(self.trained_model_path)

        if not os.path.exists(self.base_model_path):
            os.makedirs(self.base_model_path)

    def get_cnn_model(self, model_name = 'inception', weights = 'imagenet'):
        """
        Parameters
        ----------
        model_name: str {'inception', 'facenet', 'resnet101', 'resnet152', 'resnet50', 'yolo'}
            Determine the CNN model to retrieve:
            - 'inception'(default): InceptionV3 architechture with imagenet weights (default)
            - 'facenet': Facenet architecture with weights
            - 'resnet101': ResNet101_v2 architechture with imagenet weights (default)
            - 'resnet152': ResNet152_v2 architechture with imagenet weights (default)
            - 'resnet50': ResNet50_v2 architechture with imagenet weights (default)
            - 'yolo': YOLOv3 architechture with weights

        weights: str, {None, 'imagenet', path_to_file}
            Determine the weights to use in the model:
            - None: Use Random initialization
            - 'imagenet' (default): Use the imagenet weights pre-trained
            - path_to_file: The path to the weights file to be loaded
        
        Raises
        ------
        NotImplementedError
        
        Returns
        -------
        """
        
        # Retrieveng architechture and weights from the web
        if model_name == 'inception':
            cnn_model = tf_app.InceptionV3(weights = weights, include_top = False)
            cnn_model.save(os.path.join(self.base_model_path, '{}_v3.h5'.format(model_name)))
        elif model_name == 'resnet101':
            cnn_model = tf_app.ResNet101V2(weights = weights, include_top = False)
            cnn_model.save(os.path.join(self.base_model_path, '{}_v2.h5'.format(model_name)))
        elif model_name == 'resnet152':
            cnn_model = tf_app.ResNet152V2(weights = weights, include_top = False)
            cnn_model.save(os.path.join(self.base_model_path, '{}_v2.h5'.format(model_name)))
        elif model_name == 'resnet50':
            cnn_model = tf_app.ResNet101V2(weights = weights, include_top = False)
            cnn_model.save(os.path.join(self.base_model_path, '{}_v2.h5'.format(model_name)))

import tensorflow.keras.applications as tf_app
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TimeDistributed, GRU, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os


class buildModel:

    def __init__(self, store_path = './models'):
        """
        Parameters
        ----------
        store_path: str, Optional
            A local path where to store the models created in the process.
            The defaul is the subdirectory ./models

        Raises
        ------

        """

        # Normalize path for the provided
        store_path = os.path.abspath(store_path)
        store_path = os.path.normpath(store_path)

        # Creating base and trainde folder for the models
        self.trained_model_path = os.path.join(store_path, 'trained_models')
        self.base_model_path = os.path.join(store_path, 'base_models')

        # Validating if the directories exist
        if not os.path.exists(self.trained_model_path):
            os.makedirs(self.trained_model_path)

        if not os.path.exists(self.base_model_path):
            os.makedirs(self.base_model_path)

        # Define dictionary with the CNN models for the transfer learning
        self.cnn_models_dict = {'inception': 'inception_v3.h5',
                                'inception_resnet': 'inception_resnet_v2.h5',
                                'resnet101': 'resnet101_v2.h5',
                                'resnet152': 'resnet152_v2.h5',
                                'resnet50': 'resnet50_v2.h5',
                                'yolo': 'yolo_v3.h5'}

    def __download_cnn_model(self, model_name = 'inception', weights = 'imagenet'):
        """ Donwload CNN models to use in the gesture-recognition architechture
        Parameters
        ----------
        model_name: str {'inception', 'inception_resnet', 'resnet101', 'resnet152', 'resnet50', 'yolo'}
            Determine the CNN model to retrieve:
            - 'inception'(default): InceptionV3 architechture with the defined weights
            - 'inception_resnet': InceptionResNetV2 architecture with the defined weights
            - 'resnet101': ResNet101_v2 architechture with the defined weights
            - 'resnet152': ResNet152_v2 architechture with the defined weights
            - 'resnet50': ResNet50_v2 architechture with the defined weights
            - 'yolo': YOLOv3 architechture with weights

        weights: str, {None, 'imagenet', path_to_file}
            Determine the weights to use in the model:
            - None: Use Random initialization
            - 'imagenet' (default): Use the imagenet weights pre-trained
            - path_to_weights: The path to the weights file to be loaded

        Raises
        ------
            NameError: If model_name is not defined in the self.cnn_models_dict dictionary.

        Returns
        -------
        Keras Model Instance
            A keras model
        """

        assert model_name in self.cnn_models_dict.keys(), NameError("No model_name implemented.")

        # Retrieveng architechture and weights from the web
        if model_name == 'inception':
            cnn_model = tf_app.InceptionV3(weights = weights, include_top = False)
        elif model_name == 'inception_resnet':
            cnn_model = tf_app.InceptionResNetV2(weights = weights, include_top = False)
        elif model_name == 'resnet101':
            cnn_model = tf_app.ResNet101V2(weights = weights, include_top = False)
        elif model_name == 'resnet152':
            cnn_model = tf_app.ResNet152V2(weights = weights, include_top = False)
        elif model_name == 'resnet50':
            cnn_model = tf_app.ResNet101V2(weights = weights, include_top = False)
        elif model_name == 'yolo':
            # TODO: implement load for yolo version
            pass
        else:
            # Flag to crontrol the previous flow
            return None

        # Save the model retrieved from the web into local path for future use
        cnn_model.save(os.path.join(self.base_model_path, self.cnn_models_dict[model_name]))

        # Flag to control the previous flow
        return cnn_model

    def __get_cnn_model(self, file_path, weights):
        """ Load CNN models to use in the gesture-recognition architechture
        Parameters
        ----------
        file_path: str
            A local file path where the CNN model is stored.

        weights: str, {None, 'imagenet', path_to_file}
            Determine the weights to use in the model:
            - None: Use Random initialization
            - 'imagenet' (default): Use the imagenet weights pre-trained
            - path_to_weights: The path to the weights file to be loaded

        Raises
        ------
            NameError: If the file does not exist.

        Returns
        -------
        Keras Model Instance
            A keras model
        """

        try:
            # Load keras model if exist
            cnn_model = load_model(file_path, compile = False)
        except:
            # If the model does not exist, try to download it.
            file_name = os.path.split(file_path)[-1]
            for key, val in self.cnn_models_dict.items():
                if val == file_name:
                    model_name = key
                    break

            cnn_model = self.__download_cnn_model(model_name, weights)

        return cnn_model

    def define_cnn_layer(self, file_path = None, model_name = 'inception', weights = 'imagenet',
                        freeze_model = True, cnn_layer = None):
        """ Create a CNN layer for the gesture-recognition architecture.
        Parameters
        ----------
        file_path: str, Optional
            A file path to retrieve the CNN model to use. If None (Default) provided, it will try to load the model from
            the predefined base path of the class

        model_name: str {'inception', 'inception_resnet', 'resnet101', 'resnet152', 'resnet50', 'yolo'}
            Determine the CNN model to use as CNN layer in the action model:
            - 'inception'(default): InceptionV3 architechture with the defined weights
            - 'inception_resnet': InceptionResNetV2 architecture with the defined weights
            - 'resnet101': ResNet101_v2 architechture with the defined weights
            - 'resnet152': ResNet152_v2 architechture with the defined weights
            - 'resnet50': ResNet50_v2 architechture with the defined weights
            - 'yolo': YOLOv3 architechture with weights

        weights: str, {None, 'imagenet', path_to_file}
            Determine the weights to use in the model:
            - None: Use Random initialization
            - 'imagenet' (Default): Use the imagenet weights pre-trained
            - path_to_weights: The path to the weights file to be loaded

        freeze_model: bool, Optional
            True (Default) if want to freeze (do not train) the cnn layer. False otherwise.

        cnn_layer: Keras recurrent layer
            A keras convolutional neural netwrok layer to use. Default is None. This parameter has higher
            precedence if is given.

        Raises
        ------

        Returns
        -------
        self
        """

        if cnn_layer is None:
            # Build file path for pre-defined models, in the case the file_path is not provided
            if file_path is None:
                # Retrieve the cnn model from the file path using the weights defined
                file_path = os.path.join(self.base_model_path, model_name)

                # Validate if the model_name is for the file or the cnn model
                if not os.path.exists(file_path):
                    file_path = os.path.join(self.base_model_path, self.cnn_models_dict[model_name])

            cnn_layer = self.__get_cnn_model(file_path, weights)

        # True for not retrain the weights of the cnn model, False otherwise.
        if freeze_model:
            cnn_layer.trainable = False
        else:
            cnn_layer.trainable = True

        # Define the cnn model to use for the gesture-recognition model
        self.cnn_layer = cnn_layer

        return self

    def define_rnn_layer(self, type = 'lstm', units = 64, rnn_layer = None, **kwargs):
        """ Create RNN layer for the gesture-recognition architecture.
        Parameters
        ----------
        type: str {'lstm', 'gru'}
            Define the recurrent neural network to use:
            - 'lstm' (Default): Use a LSTM layer for the model
            - 'gru': Use a GRU layer for the model

        units: int, Optional
            A positive integer, dimensionality of the output spacae (rnn_layer argument).
            Default value is 64.

        rnn_layer: Keras recurrent layer
            A keras recurrent neural netwrok layer to use. Default is None. This parameter has higher
            precedence if is given.

        **kwargs:
            Key words arguments allowed for the LSTM or GRU layer, depending the one choosed in type.

        Raises
        ------

        Returns
        -------
        self
        """

        # Create the sequential model with the cnn_model defined
        rnn_model = Sequential()
        rnn_model.add(TimeDistributed(self.cnn_layer))

        # Create the RNN layer according with the inputs
        if rnn_layer is None:
            if type == 'lstm':
                rnn_layer = LSTM(units, **kwargs)
            elif type == 'gru':
                rnn_layer = GRU(units, **kwargs)

        # Add the RNN layer to the sequential model
        rnn_model.add(rnn_layer)

        # Define the cnn-lstm model for the gesture-recognition
        self.rnn_model = rnn_model

        return self

    def define_dense_layer(self, units = 64 dense_layer = None, **kwargs):
        """ Create a dense layer for the gesture-recognition architecture.
        Parameters
        ----------

        units: int, Optional
            A positive integer, dimensionality of the output spacae (rnn_layer argument).
            Default value is 64.

        dense_layer: Keras dense layer
            A keras recurrent neural netwrok layer to use. Default is None. This parameter has higher
            precedence if is given. This layer should not have the decision layer (softmax or sigmoid).

        **kwargs:
            Key words arguments allowed for the Dense layer.

        Raises
        ------

        Returns
        -------
        self
        """

        # Create a dense layer if None is provided
        if dense_layer is None:
            dense_layer = Dense(units, **kwargs)

        # Add the dense layer
        self.dense_layer = self.rnn_layer.add(dense_layer)

        # Create the decision layer for the output.
        decision_layer = Dense(number_categories, activation = 'softmax')
        self.dense_layer.add(decision_layer)

        return self.dense_layer

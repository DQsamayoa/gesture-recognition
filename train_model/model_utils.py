import tensorflow.keras.applications as tf_app
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TimeDistributed, GRU, LSTM, Dense, GlobalMaxPool2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from keras_video import VideoFrameGenerator
import os
import glob


class buildModel:

    def __init__(self, number_categories, store_path = './models'):
        """
        Parameters
        ----------
        number_categories: int
            A positive integer with the number of categories to classify.

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

        self.number_categories = number_categories

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
            conv_net = tf_app.InceptionV3
        elif model_name == 'inception_resnet':
            conv_net = tf_app.InceptionResNetV2
        elif model_name == 'resnet101':
            conv_net = tf_app.ResNet101V2
        elif model_name == 'resnet152':
            conv_net = tf_app.ResNet152V2
        elif model_name == 'resnet50':
            conv_net = tf_app.ResNet101V2
        elif model_name == 'yolo':
            # TODO: implement load for yolo version
            pass
        else:
            # Flag to crontrol the previous flow
            return None

        cnn_model = conv_net(weights = weights, include_top = False, input_shape = self.input_shape[1:])

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

        self.cnn_inputs = cnn_layer.inputs[0].shape[1:]

        # Save the boolean value for freeze cnn model and save the cnn model
        self.freeze_cnn_layer = freeze_model

        # Define the cnn model to use for the gesture-recognition model
        self.cnn_layer = Sequential([cnn_layer, GlobalMaxPool2D()])

        return self

    def define_rnn_layer(self, time_steps, type = 'lstm', units = 64, rnn_layer = None, **kwargs):
        """ Create RNN layer for the gesture-recognition architecture.
        Parameters
        ----------
        time_steps: int
            The number of observations to use in the recurrent network.

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

        # Define the model inputs considering the time steps for the recurrent network
        self.input_shape = (time_steps, ) + self.cnn_inputs

        # Create the sequential model with the cnn_model defined
        rnn_model = Sequential()
        rnn_model.add(TimeDistributed(self.cnn_layer, input_shape = self.input_shape))

        # Whether freeze or not the cnn layer:
        # freeze = True -> trainable = False
        rnn_model.layers[-1].trainable = not self.freeze_cnn_layer

        # Create the RNN layer according with the inputs
        if rnn_layer is None:
            if type == 'lstm':
                rnn_layer = LSTM(units, **kwargs)
            elif type == 'gru':
                rnn_layer = GRU(units, **kwargs)

        # Add the RNN layer to the sequential model
        rnn_model.add(rnn_layer)

        # Define the cnn-lstm model for the gesture-recognition
        self.rnn_layer = rnn_model

        return self

    def define_dense_layer(self, units_list = [128, 64], activation = 'relu', **kwargs):
        """ Create a dense layer for the gesture-recognition architecture.
        Parameters
        ----------

        units_list: list [int], Optional
            A list of positive integer, one for each hidden dense layer to add
            to the output rnn_layer. Default value is [128, 64].

        activation: str, Optional
            Activation function. Default 'relu'.

        **kwargs:
            Key words arguments allowed for the Dense layer.

        Raises
        ------
            Assert for units_list is a list of integers with at least one element

        Returns
        -------
        self
        """

        assert isinstance(units_list, list), 'Error: a list of integeres needed'
        assert len(units_list) > 0, 'Error: a list with at least one integer should be provided'
        assert isinstance(units_list[0], int), 'Error: a list of integeres needed'

         # Assign the model variable
        model = self.rnn_layer

        # Add the dense layer(s)
        for units in units_list:
            dense_layer = Dense(units, activation = activation, **kwargs)
            model.add(dense_layer)

        # Create the decision layer for the output.
        model.add(Dense(self.number_categories, activation = 'softmax'))
        model.compile()

        self.model = model

        return self


class buildDataset:

    def __init__(self, folder_path, video_ext = 'avi'):
        """
        Parameters
        ----------
        folder_path: str,
            A local path where the videos are stored. The structure for the folder
        should be as follow:
            folder_path
            +-- folder_category_01
            |   +-- file_videoA
            |   +-- file_videoB
            |   +-- ...
            +-- folder_category_02
            |   +-- file_videoA
            |   +-- file_videoB
            |   +-- ...
            +-- folder_category_03
            |   +-- file_videoA
            |   +-- file_videoB
            |   +-- ...
            +-- ...
        the name of the folder categories will be used for label, but the name for
        the video files does not matter.

        video_ext: str, optional
            The video extension of the file. Default value 'avi'.

        Raises
        ------

        """


        # Use sub directories names as categories/classes
        videos_path = glob.glob(os.path.join(folder_path, '*'))
        categories = [i.split(os.path.sep)[1] for i in videos_path]
        categories.sort()
        self.categories = categories

        # Create the glob pattern to extract the frames from videos
        self.glob_pattern = os.path.join(folder_path, '{classname}', '*.' + str(video_ext))

    def create_train_dataset(self, model, prop_val_dataset = 0.33, do_data_aug = True):
        """ Retrieve the train, validation and test datasets

        model: buildModel class
            A keras model created by buildModel class.

        prop_val_dataset: float, Optional
            A float value between 0 and 1 to split the training set into train and
        validation. Default value 0.33.

        do_data_aug: boolean, Optional
            Whether or not do data augmentation to the frames. Default value True.

        """

        _, time_step, *size, channels =  model.input_shape
        self.size = tuple(size)
        self.channels = channels
        self.time_step = time_step

        # for data augmentation
        if do_data_aug:
            data_aug = keras.preprocessing.image.ImageDataGenerator(
                zoom_range=.1,
                horizontal_flip = True,
                rotation_range = 8,
                width_shift_range = .2,
                height_shift_range = .2)
        else:
            data_aug = None

        # Create video frame generator
        train_dataset = VideoFrameGenerator(
            classes = self.categories,
            glob_pattern = self.glob_pattern,
            nb_frames = self.time_step,
            split_val = prop_val_dataset,
            shuffle = True,
            batch_size = None,
            target_shape = self.size,
            nb_channel = self.channels,
            transformation = data_aug,
            use_frame_cache = True)

        # Create
        val_dataset = train_dataset.get_validation_generator()

        return train_dataset, val_dataset

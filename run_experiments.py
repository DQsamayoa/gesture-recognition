from train_model.model_utils import buildModel
from train_model.model_utils import buildDataset
from train_model.model_utils import Experiments
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

# Defining variables
NUMBER_CATEGORIES = 7

# Creating the general class
build_class = buildModel(number_categories = NUMBER_CATEGORIES)

cnn_models = ['inception', 'inception_resnet', 'resnet101', 'resnet152', 'resnet50']

for cnn_model_name in cnn_models:
    print("Starting with " + cnn_model_name + "...")

    # Create a model
    base_model = build_class \
                .define_cnn_layer(model_name = cnn_model_name) \
                .define_rnn_layer(time_steps = 25) \
                .define_dense_layer([128, 128, 64, 64], kernel_regularizer = L2(l2 = 0.01))

    # Retriving dataset
    dataset = buildDataset(folder_path = 'videos', video_ext = 'avi')
    train, val = dataset.create_train_dataset(model = base_model.model, preprocess_input = base_model.preprocess_input)

    # Training the model
    my_experiment = Experiments(cnn_model, my_model, train, val, optimizer = Adam(0.01))
    new_experiment = my_experiment.train_model(cnn_model_name + 'v1.0', epochs = 50, checkpoint_path = cnn_model_name)

    print("Finishing hard training for "  + cnn_model_name)
    print("Starting fine tunning training...")

    # Fine tunning model
    fine_experiment = new_experiment.fine_tunning_model(50, optimizer = Adam(1e-5))
    fine_tunning_experiment = fine_experiment.train_model(cnn_model + 'v1.1', epochs = 5, checkpoint_path = cnn_model)

    print("Finished fine tunning training for "  + cnn_model_name)

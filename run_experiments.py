from train_models.model_utils import buildModel
from train_model.model_utils import buildDataset
from train_model.model_utils import Experiments

# Defining variables
NUMBER_CATEGORIES = 7


# Creating the general class
build_class = buildModel(number_categories = NUMBER_CATEGORIES)

# Create a model
base_model = build_class \
            .define_cnn_layer() \
            .define_rnn_layer(time_steps = 5) \
            .define_dense_layer()

my_model = base_model.model

# Retriving dataset
dataset = buildDataset(folder_path = 'videos', video_ext = 'avi')
train, val = dataset.create_train_dataset(model = my_model)

# Training the model
my_experiment = Experiments('test_v1.0.0', my_model, train, val)
new_experiment = my_experiment.train_model('p1.0', epochs = 5)

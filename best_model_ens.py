import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.preprocessing import Normalizer
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.callbacks import TensorBoard
import datetime
from sklearn import metrics

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs")




plt.clf()

plt.style.use("ggplot")

# put your own path to the data root directory (see example in `Data architecture` section)
data_dir = Path("C:/Users/ulyss/modele_ML/ENS_Challenge")

# load the training and testing data sets
train_features_dir = data_dir / "train_input" / "moco_features"
test_features_dir = data_dir / "test_input" / "moco_features"
df_train = pd.read_csv(data_dir  / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(data_dir  / "supplementary_data" / "test_metadata.csv")

# concatenate y_train and df_train
y_train = pd.read_csv(data_dir  / "train_output.csv")
df_train = df_train.merge(y_train, on="Sample ID")

print(f"Training data dimensions: {df_train.shape}")  # (344, 4)
df_train.head()


X_train = []
y_train = []
centers_train = []
patients_train = []
coordinates = []

for sample, label, center, patient in tqdm(
    df_train[["Sample ID", "Target", "Center ID", "Patient ID"]].values
):
    # load the coordinates and features (1000, 3+2048)
    _features = np.load(train_features_dir / sample)
    # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
    # and the MoCo V2 features
    coordinate, features = _features[:, :3], _features[:, :]  # Ks

    features1= np.insert(features,0,int(center[2]), axis=1)
    X_train.append(features1)
    y_train.append(label)
    # coordinates.append(coordinate)
    centers_train.append(center)
    # patients_train.append(patient)

# convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
# centers_train = np.array(centers_train)
# patients_train = np.array(patients_train)


X_train.shape
X_train=X_train.reshape((344*1000,2052))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train=X_train.reshape((344,1000,2052))

labels_count = pd.DataFrame(y_train).value_counts()



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=23)
del X_train
# x_train, x_test1, y_train, y_test1 = train_test_split(x_train, y_train, test_size=0.0, random_state=20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(np.count_nonzero(y_test == 0))

x_train = list(np.einsum("ij...->ji...", x_train))
x_test = list(np.einsum("ij...->ji...", x_test))
# x_test1 = list(np.einsum("ij...->ji...", x_test1))
# y_test1 = y_test1[:, np.newaxis]
y_train = y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]

POSITIVE_CLASS = 1
BAG_COUNT = x_train[0].shape[0]
VAL_BAG_COUNT = x_test[0].shape[0]
BAG_SIZE = len(x_train)
ENSEMBLE_AVG_COUNT = 1


class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)

from tensorflow.keras.layers import LayerNormalization
import keras.backend as k

def create_model(instance_shape):
    # Extract features from inputs.
    inputs, embeddings = [], []
  #  shared_dense_layer_1 = layers.Dense(256, activation="relu")
  #  shared_dense_layer_2 = layers.Dense(128, activation="relu")
    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)

   #     dense_1 = shared_dense_layer_1(inp)
   #     dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
  #      embeddings.append(dense_2)
        embeddings.append(inp)
    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=32,
        kernel_regularizer=keras.regularizers.l2(0.1),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i][:,4:]]) for i in range(len(alpha))
    ]
    #Concatenate layers.
    #concat = layers.concatenate(multiply_layers, axis=1)
    concat = layers.add(multiply_layers)
    concat=LayerNormalization(epsilon=1e-6)(concat)
    #concat = layers.Dropout(rate=0.1)(concat)
    concat = layers.concatenate([concat, embeddings[0][:,0:2]], axis=1)
    concat= layers.Dense(300, activation="tanh")(concat)
    concat= layers.Dense(400, activation="relu")(concat)
    # Classification output node.
    output = layers.Dense(1, activation="sigmoid")(concat)

    return keras.Model(inputs, output)
def compute_class_weights(labels):

    # Count number of postive and negative bags.
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count

    # Build class weight dictionary.
    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }



def train(train_data, train_labels, val_data, val_labels, model):

    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Take the file name from the wrapper.
    file_path = "C:/Users/ulyss/modele_ML/ENS_Challenge"

    # Initialize model checkpoint callback.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialize early stopping callback.
    # The model performance is monitored across the validation data and stops training
    # when the generalization error cease to decrease.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, mode="min"
    )

    # Compile model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.01), loss="binary_crossentropy", metrics=["AUC"],
    )


    # Fit model.
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=15,
        class_weight=compute_class_weights(train_labels),
        batch_size=1,
        callbacks=[early_stopping, model_checkpoint,tensorboard_callback],
        verbose=1
    )

    # Load best weights.
    model.load_weights(file_path)

    return model


# Building model(s).
instance_shape = x_train[0][0].shape
models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

# Show single model architecture.
# print(models[0].summary())

# Training model(s).
trained_models = [
    train(x_train, y_train, x_test, y_test, model)
    for model in (models)
]


del x_train
del y_train

model = trained_models[0]
#model.evaluate(x_test, y_test)
pred = model.predict(x_test)

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test[:, 0], pred[:,0])
print("this is auc",auc)

from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_test[:, 0], pred[:,0])
auc = metrics.roc_auc_score(y_test[:, 0], pred[:,0])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

X_test = []


# load the data from `df_test` (~ 1 minute)
for sample in tqdm(df_test["Sample ID"].values):
    _features = np.load(test_features_dir / sample)
    coordinates, features = _features[:, :3], _features[:, :]
    features1= np.insert(features,0,5, axis=1)
    X_test.append(features1)

X_test = np.array(X_test)


X_test.shape
X_test=X_test.reshape((149*1000,2052))
X_test = scaler.transform(X_test)
X_test=X_test.reshape((149,1000,2052))
X_test = list(np.einsum("ij...->ji...", X_test))
model = trained_models[0]
pred_test1 = model.predict(X_test)[:,0]

pred_test2 = model.predict(X_test)[:,0]

pred_test3 = model.predict(X_test)[:,0]



for i in range(149):
  for j in range(1000):
   X_test[j][i][0]=5
pred_test3 = model.predict(X_test)[:,0]
del X_test
pred_test1 = pred_test1*(1/3) + pred_test2*(1/3) + pred_test3*(1/3)

submission = pd.DataFrame(
    {"Sample ID": df_test["Sample ID"].values, "Target": pred_test1}
).sort_values(
    "Sample ID"
)  # extra step to sort the sample IDs

# sanity checks
assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
assert submission.shape == (149, 2), "Your submission file must be of shape (149, 2)"
assert list(submission.columns) == [
    "Sample ID",
    "Target",
]

# save the submission as a csv file
submission.to_csv(data_dir /"benchmark_test_output_ulysse.csv", index=None)
submission.head()



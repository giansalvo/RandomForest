# adapted from https://www.tensorflow.org/decision_forests/tutorials/beginner_colab?hl=en

import tensorflow as tf
# Check the version of TensorFlow Decision Forests
print("Found TensorFlow v." + tf.__version__)

import tensorflow_decision_forests as tfdf
# Check the version of TensorFlow Decision Forests
print("Found TensorFlow Decision Forests v" + tfdf.__version__)

import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("./penguins.csv")

# Display the first 3 examples.
dataset_df.head(3)

# Encode the categorical labels as integers.
#
# Details:
# This stage is necessary if your classification label is represented as a
# string since Keras expects integer classification labels.
# When using `pd_dataframe_to_tf_dataset` (see below), this step can be skipped.

# Name of the label column.
label = "species"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)

# Split the dataset into a training and a testing dataset.

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

# And finally, convert the pandas dataframe (pd.Dataframe) into tensorflow datasets (tf.data.Dataset):
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(x=train_ds)



# Let's evaluate our model on the test dataset.

model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)

print("Evaluation results:")
for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

# Export the model to the SavedModel format for later re-use
model_1.save("./my_saved_model")

# plot the model
model_1.summary()



# The input features
print(model_1.make_inspector().features())

# The feature importances
print(model_1.make_inspector().variable_importances())



print(model_1.make_inspector().evaluation())

print(model_1.make_inspector().training_logs())

# Let's plot it:
print("Plot logs...")
logs = model_1.make_inspector().training_logs()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")
plt.savefig("logs.png")
plt.close()

print("Program end.")
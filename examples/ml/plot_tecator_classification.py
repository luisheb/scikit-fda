"""
Spectometric data: Classifying high vs. low fat meat using derivatives and
scalar features
==========

Binary classification of meat samples as high or low-fat based on their
infrared spectrometric curves and associated scalar features.
"""

# Author: Luis Hebrero Garicano
# License: BSD 3-Clause

# %%
# In this example, we classify meat samples into two categories: **low fat**
# (less than 20% fat content) and **high fat** (20% or more) using:
# - The original spectrometric infrared curves (functional data)
# - Their first and second derivatives
# - Scalar covariates: water and protein content
# 
# These data come from the Tecator dataset, a popular benchmark in functional
# data analysis.

# %%
from skfda.datasets import fetch_tecator

X_df, y_df = fetch_tecator(return_X_y=True, as_frame=True)

fd = X_df.iloc[:, 0].array
fat = y_df["fat"].to_numpy()
water = y_df["water"].to_numpy()
protein = y_df["protein"].to_numpy()

# %%
# We define the target labels for classification. Samples with less than 20%
# fat are labeled as "low fat", and the rest as "high fat".

# %%
import numpy as np

fat_threshold = 20
low_fat = fat < fat_threshold
labels = np.where(low_fat, "low fat", "high fat")
y = np.where(low_fat, 0, 1)  # 0 = low fat, 1 = high fat (binary label)

# %%
# We plot the original spectrometric curves, grouped by fat content.

# %%
import matplotlib.pyplot as plt

colors = {
    "high fat": "red",
    "low fat": "blue",
}

fd.plot(
    group=labels,
    group_colors=colors,
    linewidth=0.5,
    alpha=0.7,
    legend=True,
)

# %%
# We also visualize the first and second derivatives.
#
# In spectrometric data, much of the relevant discriminative information lies
# not in the absolute levels of absorbance but in the shape and variation of
# the spectra. As highlighted in Ferraty and Vieu (Chapter 7), derivatives
# often capture subtle local features, such as slope changes and inflection
# points, that are associated with underlying chemical properties. In our case,
# the first and second derivatives help emphasize these local variations,
# making it easier to distinguish between samples with different fat content.

# %%
fd_1st = fd.derivative()
fd_2nd = fd.derivative(order=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

fd_1st.plot(
    group=labels,
    group_colors=colors,
    linewidth=0.5,
    alpha=0.7,
    legend=True,
    axes=ax1,
)
ax1.set_title("First Derivative")

fd_2nd.plot(
    group=labels,
    group_colors=colors,
    linewidth=0.5,
    alpha=0.7,
    legend=True,
    axes=ax2,
)
ax2.set_title("Second Derivative")

fig.tight_layout()
fig

# %%
# We also inspect the scalar features (water and protein percentages), which
# may help in distinguishing the two fat groups.

# %%
import pandas as pd

df_scalar = pd.DataFrame(
    {
        "water": water,
        "protein": protein,
        "fat_group": labels,
    },
)

print(df_scalar.groupby("fat_group").describe())

# %%
# Train-test split and assembling the full dataset
# ------------------------------------------------
# We now divide our dataset into training and testing partitions. We split all
# functional components (original spectrum, first derivative, second
# derivative), as well as the scalar variables (water and protein), in a
# stratified fashion to preserve the proportion of low- and high-fat samples.

# %%
from sklearn.model_selection import train_test_split

(
    fd_train,
    fd_test,
    fd_1st_train,
    fd_1st_test,
    fd_2nd_train,
    fd_2nd_test,
    water_train,
    water_test,
    protein_train,
    protein_test,
    y_train,
    y_test,
) = train_test_split(
    fd,
    fd_1st,
    fd_2nd,
    water,
    protein,
    y,
    test_size=0.25,
    stratify=y,
    random_state=27,
)

# %%
# Scaling
# -------
#
# When working with vector-valued functional data, particularly when combining
# a function with its derivatives, applying an appropriate scaling procedure is
# crucial to ensure that each component contributes meaningfully to any
# distance-based analysis.
#
# This need becomes especially clear when incorporating derivatives. As is
# well known in functional data analysis, differentiation tends to amplify
# local fluctuations and reduce the range of the signal, significantly altering
# its scale. Even if all components, such as a function and its first and
# second derivatives, are derived from the same underlying process, their norms
# can differ by several orders of magnitude. If left unscaled, this imbalance
# will cause the more variable components to dominate the overall distance
# computation.
#
# For example, in our dataset, the original function ranges between 2 and 5,
# the first derivative varies between approximately -0.02 and 0.05, and the
# second derivative oscillates between -0.004 and 0.003. Despite originating
# from the same phenomenon, the second derivative's much smaller range and
# higher oscillatory behavior would disproportionately influence distance-based
# methods such as k-nearest neighbors or clustering unless scaling is applied
# beforehand.
#
# To address this, we apply **Root Mean Square (RMS) L2 norm** scaling. This
# method rescales each component by the square root of its average squared
# norm across the sample, equalizing their overall contribution.

# %%
from skfda.exploratory.stats import root_mean_square_l2

# Rescale train functions
scale_train_fd = root_mean_square_l2(fd_train)
scale_train_fd_1st = root_mean_square_l2(fd_1st_train)
scale_train_fd_2nd = root_mean_square_l2(fd_2nd_train)

fd_train_scaled = fd_train / scale_train_fd
fd_1st_train_scaled = fd_1st_train / scale_train_fd_1st
fd_2nd_train_scaled = fd_2nd_train / scale_train_fd_2nd

# Rescale test functions
scale_test_fd = root_mean_square_l2(fd_test)
scale_test_fd_1st = root_mean_square_l2(fd_1st_test)
scale_test_fd_2nd = root_mean_square_l2(fd_2nd_test)

fd_test_scaled = fd_test / scale_test_fd
fd_1st_test_scaled = fd_1st_test / scale_test_fd_1st
fd_2nd_test_scaled = fd_2nd_test / scale_test_fd_2nd

# %%
# After applying RMS L2 scaling, the magnitudes of all three components
# (the original curves, the first derivative, and the second derivative)
# are now brought to a comparable range.

# %%
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

fd_train_scaled.plot(
    group=labels,
    group_colors=colors,
    linewidth=0.5,
    alpha=0.7,
    legend=True,
    axes=ax1,
)
fd_1st_train_scaled.plot(
    group=labels,
    group_colors=colors,
    linewidth=0.5,
    alpha=0.7,
    legend=True,
    axes=ax2,
)
fd_2nd_train_scaled.plot(
    group=labels,
    group_colors=colors,
    linewidth=0.5,
    alpha=0.7,
    legend=True,
    axes=ax3,
)

# %%
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

scaler = SklearnStandardScaler()
scalar_features_train = scaler.fit_transform(
    np.column_stack([water_train, protein_train]),
)
scalar_features_test = scaler.fit_transform(
    np.column_stack([water_test, protein_test]),
)

# %%
# Now we combine all inputs into  pandas DataFrames, which allow us to work
# with mixed data. Each row corresponds to a sample,
# and the columns include the three functional components and two scalar ones.

# %%
from skfda.exploratory.visualization.representation import plot_mixed_data

X_train = pd.DataFrame(
    {
        "original": fd_train_scaled,
        "derivative": fd_1st_train_scaled,
        "second_derivative": fd_2nd_train_scaled,
        "water": scalar_features_train[:, 0],
        "protein": scalar_features_train[:, 1],
    },
)

X_test = pd.DataFrame(
    {
        "original": fd_test_scaled,
        "derivative": fd_1st_test_scaled,
        "second_derivative": fd_2nd_test_scaled,
        "water": scalar_features_test[:, 0],
        "protein": scalar_features_test[:, 1],
    },
)

plot_mixed_data(X_train)
# %%
# Distance metric definition
# --------------------------
# We define a product metric that combines appropriate distances for each
# column.
#
# For the functional components we use the $L^2$ distance, and for the scalar
# features we use the standard Euclidean distance.

# %%
from skfda.misc.metrics import PProductMetric, default_metric, l2_distance
from skfda.representation import FData
from skfda.typing._numpy import NDArrayFloat

metric: PProductMetric[pd.DataFrame, FData|NDArrayFloat]= PProductMetric(
    p=2,
    metrics=[
        l2_distance,  # original spectra
        l2_distance,  # first derivative
        l2_distance,  # second derivative
        default_metric,  # water
        default_metric,  # protein
    ],
)

# %%
# Model fitting and evaluation
# ----------------------------
# We now train the K-nearest neighbors classifier using the product metric and
# evaluate its performance on the test set.

# %%
from skfda.ml.classification import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
knn.fit(X_train, y_train)

# %%
# Finally, we predict and evaluate the accuracy of the model.

# %%
from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2%}")

# %%
# Comparison with the model only using the original spectrometric infrared
# curves

# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(fd_train, y_train)

y_pred = knn.predict(fd_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2%}")

# %%
# This example highlights the value of incorporating multiple components in
# functional classification models. By combining the original spectrometric
# curves with their first and second derivatives, as well as scalar features
# like water and protein content, we significantly improved the classifier's
# performance. The accuracy increased from 81% using only the raw curves to
# over 98% when integrating all components, an improvement of nearly 17%.



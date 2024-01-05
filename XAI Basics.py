# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:56:02 2024

@author: S.M.H Mousavi
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
# Train a Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)



# XAI Feature Importances
importances = rf.feature_importances_
# Print feature importances
print(f"Feature Importances")
for index, (name, importance) in enumerate(zip(feature_names, importances)):
    print(f"{index}. {name}: {importance:.4f}")
# Create a figure for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(y=feature_names, x=importances, palette="viridis")
plt.title('Feature Importances in Iris Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Importance', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# XAI SHAP
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Create a SHAP explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
# Summarize the SHAP values for each feature
shap_sum = np.abs(shap_values[0]).mean(axis=0)
# Print SHAP values for each feature
print(f"SHAP")
feature_names = iris.feature_names
for index, (name, shap_value) in enumerate(zip(feature_names, shap_sum)):
    print(f"{index}. {name}: {shap_value:.4f}")
# Create a figure for SHAP values
plt.figure(figsize=(10, 6))
sns.barplot(y=feature_names, x=shap_sum, palette="rocket")
plt.title('SHAP Values for Iris Dataset Features', fontsize=16, fontweight='bold')
plt.xlabel('Mean(|SHAP Value|)', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# XAI Surrogate
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Train a surrogate decision tree model
surrogate = DecisionTreeClassifier(max_depth=3).fit(X_train, rf.predict(X_train))
# Get feature importances from the surrogate model
surrogate_importances = surrogate.feature_importances_
feature_names = iris.feature_names
print(f"Surrogate")
for index, (name, importance) in enumerate(zip(feature_names, surrogate_importances)):
    print(f"{index}. {name}: {importance:.4f}")
# Plot the surrogate model
plt.figure(figsize=(20, 10))
plot_tree(surrogate, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Surrogate Decision Tree for Random Forest Model', fontsize=16, fontweight='bold')
plt.show()



# XAI Permutation Importance
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
# Assuming the RandomForest model 'rf' and the test data 'X_test', 'y_test' are already defined
# Compute permutation importance
perm_importance = permutation_importance(rf, X_test, y_test)
# Get sorted indices
sorted_idx = perm_importance.importances_mean.argsort()
# Convert feature names to a list for proper indexing
feature_names_sorted = [iris.feature_names[i] for i in sorted_idx]
print(f"Permutation Importance")
for i in sorted_idx:
    feature_name = iris.feature_names[i]
    importance_value = perm_importance.importances_mean[i]
    print(f"{i}. {feature_name}: {importance_value:.4f}")
# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=perm_importance.importances_mean[sorted_idx], y=feature_names_sorted, palette="viridis")
plt.xlabel("Permutation Importance", fontsize=14, fontweight='bold')
plt.ylabel("Features", fontsize=14, fontweight='bold')
plt.title("Permutation Importance of Features in Iris Dataset", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# XAI LIME
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
# Create the LIME explainer
explainer = LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, mode='classification')
# Select an instance to explain
instance_idx = 1
instance = X_test[instance_idx]
# Initialize an array to store feature weights for each class
feature_weights = np.zeros((len(iris.target_names), len(iris.feature_names)))
# Explain the prediction of this instance for each class and aggregate the weights
for class_idx in range(len(iris.target_names)):
    explanation = explainer.explain_instance(instance, rf.predict_proba, num_features=len(feature_names), labels=[class_idx])
    exp_map = explanation.as_map()
    
    # Aggregate weights
    for feature, weight in exp_map[class_idx]:
        feature_weights[class_idx, feature] += weight
# Average the feature weights across classes
average_feature_weights = np.mean(feature_weights, axis=0)
print(f"LIME")
# Print each feature's name, index, and corresponding average value
for index, (name, weight) in enumerate(zip(iris.feature_names, average_feature_weights)):
    print(f"{index}. {name}: {weight:.4f}")
# Plot the aggregated weights
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(average_feature_weights)
plt.barh(np.array(iris.feature_names)[sorted_idx], average_feature_weights[sorted_idx])
plt.xlabel('Average Feature Contribution', fontsize=14, fontweight='bold')
plt.title('Aggregated LIME Feature Contributions Across All Classes', fontsize=14, fontweight='bold')
plt.show()




# XAI Anchor
from alibi.explainers import AnchorTabular
# Create an AnchorTabular explainer
explainer = AnchorTabular(rf.predict, feature_names=iris.feature_names)
# Fit the explainer
explainer.fit(X_train)
# Explain an instance
instance = X_test[0]
explanation = explainer.explain(instance, threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
import pandas as pd
# Extract the features that are part of the anchor
anchor_features = set()
for feature in explanation.anchor:
    feature_name = feature.split('=')[0].strip()
    anchor_features.add(feature_name)
# Print detailed information about anchor features
print(f"Anchor")
for feature_name, feature_value in zip(iris.feature_names, instance):
    anchor_status = 'Anchor' if feature_name in anchor_features else 'Not Anchor'
    print(f"Feature: {feature_name}, Value: {feature_value}, Status: {anchor_status}")
# Create a DataFrame for plotting
df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Value': instance,
    'In Anchor': ['Anchor' if feature in anchor_features else 'Not Anchor' for feature in iris.feature_names]
})
# Create a horizontal bar plot with Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Value', y='Feature', hue='In Anchor', data=df, dodge=False, palette=['salmon', 'lightblue'])
plt.title('Anchor Explanation for a Test Instance', fontsize=16, fontweight='bold')
plt.xlabel('Feature Value', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=14, fontweight='bold')
plt.legend(title='In Anchor', loc='lower right')
plt.show()


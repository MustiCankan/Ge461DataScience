import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('falldetection_dataset.csv',header=None)

# Droping the first column 
df = df.drop(columns=[0])
df.shape

dataset_y = df[1]

dataset_x = df.drop(columns=[1])

dataset_y_changed = dataset_y.replace({"F":1,"NF":0})

features_dataset_mean = dataset_x.mean(axis=0)
features_dataset_std = dataset_x.std(axis=0)
X_centered =( dataset_x - features_dataset_mean) / features_dataset_std

# Extract N =  2 
pca_kmeans = PCA(n_components=2)
pca_x = pca_kmeans.fit_transform(X_centered)

# how much variance do PC1 and PC2 
ratios = pca_kmeans.explained_variance_ratio_

print(f"PC1 explains {ratios[0] * 100:.2f}% of the total variance")
print(f"PC2 explains {ratios[1] * 100:.2f}% of the total variance")
print(f"Cumulative (PC1 + PC2): {ratios.sum() * 100:.2f}% of the total variance")

# giving a larger plot


plt.figure(figsize=(8, 6))

plt.scatter(pca_x[:, 0], pca_x[:, 1],
            c=dataset_y_changed,
            cmap='plasma')

# giving a larger plot
plt.figure(figsize=(12, 12))

plt.scatter(pca_x[:, 0], pca_x[:, 1],
            c=dataset_y_changed,
            cmap='plasma')

# labeling x and y axes
plt.title("PCA1 versus PCA2")
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


inertias = []
silhouettes = []
Ks = range(2, 7)

for k in Ks:
    km = KMeans(n_clusters=k, random_state=0).fit(pca_x)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(pca_x, km.labels_))

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(Ks, inertias, '-o');      ax[0].set(title="Elbow: inertia vs k", xlabel="k", ylabel="inertia")
ax[1].plot(Ks, silhouettes, '-o');  ax[1].set(title="Silhouette vs k", xlabel="k", ylabel="silhouette score")
plt.show()


km2   = KMeans(n_clusters=2, random_state=42, n_init=10)
cl2   = km2.fit_predict(pca_x)          # cluster memberships (0 or 1)
labels_true = dataset_y                         # your 'F' / 'NF' vector

cluster_to_label = {}
for cid in np.unique(cl2):
    majority = pd.Series(labels_true[cl2 == cid]).mode()[0]
    cluster_to_label[cid] = majority

labels_pred = np.vectorize(cluster_to_label.get)(cl2)


overall_acc = accuracy_score(labels_true, labels_pred) * 100
print(f"Overall overlap = {overall_acc:.2f} %")


cm = confusion_matrix(labels_true, labels_pred, labels=['F', 'NF'])
print("\nConfusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(cm, index=['F', 'NF'], columns=['F', 'NF']))

for cid in np.unique(cl2):
    mask      = cl2 == cid
    purity    = (labels_true[mask] == labels_pred[mask]).mean() * 100
    print(f"Cluster {cid}: purity = {purity:.2f} % "
          f"({mask.sum()} samples, majority = '{cluster_to_label[cid]}')")


marker_dict = {'F': 'o', 'NF': 's'}   # fall = circle, non-fall = square
color_dict  = {0: 'C0', 1: 'C1'}      # cluster ID → colour

plt.figure(figsize=(8, 8))
for cid in np.unique(cl2):
    for lab in ['F', 'NF']:
        mask = (cl2 == cid) & (labels_true == lab)
        plt.scatter(pca_x[mask, 0], pca_x[mask, 1],
                    c=color_dict[cid],
                    marker=marker_dict[lab],
                    edgecolor='k', alpha=0.7,
                    label=f'cluster {cid} – {lab}')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA scatter: colour = cluster, shape = true label')
# de-duplicate identical legend entries
handles, labels = plt.gca().get_legend_handles_labels()
uniq = dict(zip(labels, handles))
plt.legend(uniq.values(), uniq.keys())
plt.tight_layout()
plt.show()

m = dataset_x.shape[0]
train_size = 0.7* m

validation_size = test_size = 0.15*m 


print("Train Size" ,train_size)
print("Validation Size" ,validation_size)
print("Test Size" ,test_size)



X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_centered, dataset_y_changed, test_size=0.30, random_state=42, stratify=dataset_y_changed
)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=22, stratify=y_tmp
)

print("Sizes:",  X_train.shape, X_val.shape, X_test.shape)

# Hyperparameters
svm_param_grid = {
    'C':       [0.1, 1],
    'kernel':  ['rbf'],
    'gamma':   ['scale','auto']
}
mlp_param_grid = {
    'hidden_layer_sizes': [(2,), (10,)],
    'activation': ['logistic','relu'],
    'alpha':              [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-3, 1e-2],
    
}

def tune_model(ModelClass, param_grid, X_tr, y_tr, X_va, y_va):
    records = []
    for params in (dict(zip(param_grid, v)) for v in 
                   __import__('itertools').product(*param_grid.values())):
        model = ModelClass(**params, random_state=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        records.append({**params, 'val_acc': acc})
    return pd.DataFrame.from_records(records)


svm_results = tune_model(SVC,  svm_param_grid, X_train, y_train, X_val, y_val)
mlp_results = tune_model(MLPClassifier, mlp_param_grid, X_train, y_train, X_val, y_val)

print("Top SVM configs:\n", svm_results.sort_values('val_acc', ascending=False).head())
print("Top MLP configs:\n", mlp_results.sort_values('val_acc', ascending=False))


best_svm = svm_results.loc[svm_results['val_acc'].idxmax()].to_dict()
best_mlp = mlp_results.loc[mlp_results['val_acc'].idxmax()].to_dict()


best_svm_model = SVC(**{k:best_svm[k] for k in svm_param_grid}, random_state=0)
best_mlp_model = MLPClassifier(**{k:best_mlp[k] for k in mlp_param_grid}, random_state=0)


best_svm_model.fit(X_train, y_train)
best_mlp_model.fit(X_train, y_train)


for name, model in [('SVM', best_svm_model), ('MLP', best_mlp_model)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} on TEST ===")
    print(f"Accuracy: {acc:.2%}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

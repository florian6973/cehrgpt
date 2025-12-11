




import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# base_folder_train = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_train'
# base_folder_test = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_test'

base_folder_train = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_train_mia_20_f'
base_folder_test = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_test_mia_20_f'



train_features = os.path.join(base_folder_train, 'features_without_label/train_features')
test_features = os.path.join(base_folder_test, 'features_without_label/train_features')

losses_train = np.load(os.path.join(base_folder_train, "losses-test.npy"))
all_losses_train = np.load(os.path.join(base_folder_train, "all_losses-test.npy"))
losses_test = np.load(os.path.join(base_folder_test, "losses-test.npy"))
all_losses_test = np.load(os.path.join(base_folder_test, "all_losses-test.npy"))

print(losses_train.shape)
print(losses_test.shape)

all_losses = np.concatenate([losses_train, losses_test])
losses_1 = np.concatenate([all_losses_train[:, 0], all_losses_test[:, 0]])
losses_2 = np.concatenate([all_losses_train[:, 1], all_losses_test[:, 1]])
losses_3 = np.concatenate([all_losses_train[:, 2], all_losses_test[:, 2]])

print(losses_1.shape)
print(losses_2.shape)
print(losses_3.shape)




labels = np.concatenate([np.zeros(losses_train.shape[0]), np.ones(losses_test.shape[0])])

# compute auroc
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(labels, all_losses)
print(auroc)

auroc_1 = roc_auc_score(labels, losses_1)
auroc_2 = roc_auc_score(labels, losses_2)
auroc_3 = roc_auc_score(labels, losses_3)

print(auroc_1)
print(auroc_2)
print(auroc_3)

# train ml model with features losses_1, losses_2, losses_3
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import train_test_split

losses_1 = losses_1.reshape(-1, 1)
losses_2 = losses_2.reshape(-1, 1)
losses_3 = losses_3.reshape(-1, 1)

features = np.concatenate([losses_1, losses_2, losses_3], axis=1)
print(features.shape)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probas = clf.predict_proba(X_test)
print(predictions)
print(accuracy_score(y_test, predictions))
# print(roc_auc_score(y_test, predictions))
print(roc_auc_score(y_test, probas[:, 1]))

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
roc_auc = auc(fpr, tpr)

# plot roc curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve_mia_loss.png')





exit()




def load_features(features_path):
    dfs = []
    for file in tqdm(sorted(glob.glob(os.path.join(features_path, '*.parquet')))[:100]):
        df = pd.read_parquet(file)
        dfs.append(df)
    if len(dfs) > 1:
        return np.array(pd.concat(dfs)['features'].values.tolist())
    else:
        return np.zeros((0,))

train_features = load_features(train_features)
test_features = load_features(test_features)


print(train_features.shape)
print(test_features.shape)

train_features = train_features[:10000]
test_features = test_features[:10000]

dataset = np.concatenate([train_features, test_features])
labels = np.concatenate([np.zeros(train_features.shape[0]), np.ones(test_features.shape[0])])

# print(train_features)
# print(test_features)

if train_features.shape[0] > 0 and test_features.shape[0] > 0:
        

    # train knn classifer on 10000 patient each
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import RandomForestClassifier


    train_set, test_set, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42, stratify=labels)

    # scaler = StandardScaler()
    # train_set = scaler.fit_transform(train_set)
    # test_set = scaler.transform(test_set)

    # train knn classifer on 10000 patient each

    # clf = KNeighborsClassifier(n_neighbors=5)
    clf = KNeighborsClassifier()
    # MLP?
    # clf = RandomForestClassifier()
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_set, train_labels)

    predictions = clf.predict(test_set)

    print(predictions)

    # display cm
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    print(accuracy_score(test_labels, predictions))
    print(roc_auc_score(test_labels, predictions))

    # display roc curve
    fpr, tpr, thresholds = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_20.png')
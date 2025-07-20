




import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# base_folder_train = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_train'
# base_folder_test = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_test'

base_folder_train = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_train_mia_20_e'
base_folder_test = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_test_mia_20_e'



train_features = os.path.join(base_folder_train, 'features_without_label/train_features')
test_features = os.path.join(base_folder_test, 'features_without_label/train_features')

losses_train = np.load(os.path.join(base_folder_train, "losses-test.npy"))
losses_test = np.load(os.path.join(base_folder_test, "losses-test.npy"))

print(losses_train.shape)
print(losses_test.shape)

all_losses = np.concatenate([losses_train, losses_test])
labels = np.concatenate([np.zeros(losses_train.shape[0]), np.ones(losses_test.shape[0])])

# compute auroc
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(labels, all_losses)
print(auroc)
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
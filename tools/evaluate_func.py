import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import utils


def svm(train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("acc train SVM: {}".format(acc_train))
    print("acc test SVM: {}".format(acc_test))
    return acc_train, acc_test


def knn(k, train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN

    """
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc


def nearsub(n_comp, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.

    Options:
        n_comp (int): number of components for PCA or SVD

    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1  # should be correct most of the time
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(),
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=n_comp).fit(features_sort[j])
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)

        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_svd
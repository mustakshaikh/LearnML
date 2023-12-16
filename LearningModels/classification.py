import sys

assert sys.version_info >= (3, 7)

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score,f1_score,roc_curve, precision_recall_curve,roc_auc_score


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
#print(X.shape)

some_digit = X[0]
plot_digit(some_digit)
plt.show()
print(y[0])
#Splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
pred = sgd_clf.predict([some_digit])
print(pred)

score_arr=cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(score_arr)

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))  # prints False: no 5s detected
cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")

skfolds = StratifiedKFold(n_splits=3)  # add shuffle=True if the dataset is
                                       # not already shuffled
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))  # prints 0.95035, 0.96035, and 0.9604

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

prec = precision_score(y_train_5, y_train_pred)
recall=recall_score(y_train_5, y_train_pred)
f1=f1_score(y_train_5, y_train_pred)
print("pricision "+ str(prec))
print("recall " + str(recall))
print("F1 " + str(f1))
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
print(" Score of all the y values " + str(y_scores))
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
# beautify the figure: add grid, legend, axis, labels, and circles
plt.show()

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
# beautify the figure: add labels, grid, legend, arrow, and text
plt.show()

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
print("Threshold for 90 precision " + str(threshold_for_90_precision))

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

print(precision_score(y_train_5, y_train_pred_90))
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print("Recall at 90 precision " + str(recall_at_90_precision))

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
print("fpr,tpr,thresholds",fpr,tpr,thresholds)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
[...]  # beautify the figure: add labels, grid, legend, arrow, and text
plt.show()
print(roc_auc_score(y_train_5, y_scores))

##---------------------------------------------------------
# Multiclass Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


features = ["age","gender","height","weight", "ap_hi","ap_lo","gluc","cholesterol", "smoke", "active","alco"]
X = df.loc[:, features]
y = df.loc[:, ["cardio"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

clf = RandomForestClassifier(n_estimators = 75, max_depth=15)
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_train)

print("ACCURACY : ", metrics.accuracy_score(y_train, y_pred))

cm = confusion_matrix(y_test,y_pred)
cm

feature_imp = pd.Series(clf.feature_importances_, index = features).sort_values(ascending = False)
print (feature_imp)
 


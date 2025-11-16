import joblib
from sklearn.tree import DecisionTreeClassifier
from configurations import X_train_y_train


X_train,y_train=X_train_y_train()


tree_model=DecisionTreeClassifier()

tree_model.fit(X_train,y_train)

joblib.dump(tree_model,"models/dt_model.pkl")



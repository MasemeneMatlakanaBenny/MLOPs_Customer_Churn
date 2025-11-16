import joblib
from sklearn.linear_model import LogisticRegression
from configurations import X_train_y_train


X_train,y_train=X_train_y_train()


model=LogisticRegression(solver="liblinear")
model.fit(X_train,y_train)

joblib.dump(model,"models/log_model.pkl")

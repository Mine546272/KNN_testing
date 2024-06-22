import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data = pd.DataFrame({
    'like': [162, 244, 131, 151, 172, 210, 107, 214 , 162, 250],
    'profile': [35, 155, 62, 77, 237, 61, 45, 57, 35, 39],
    'species': ['quarter 1', 'quarter 2', 'quarter 1', 'quarter 2', 'quarter 3', 'quarter 3', 'quarter 1', 'quarter 2', 'quarter 1', 'quarter 3']
})


X = data[['like', 'profile']]
y = data['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)   

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

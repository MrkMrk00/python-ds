import pandas as pd 
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report 
from multiprocessing import Process

df: pd.DataFrame = pd.read_pickle('data.pkl')

# Filter out only columns defined as important from .tsv file
important_features = [ col.split(' ')[0] 
    for col 
    in pd.read_csv('important.tsv', sep='\t').columns
]
df = df[['biome', *important_features[1:]]]

# Replace all NA values with 0.0
df.replace(pd.NA, 0.0, inplace=False)

# Add a new derived column sum_all - sum of all numeric columns (feature values)
df['sum_all'] = df.sum(axis=1, numeric_only=True)

# Filter out rows with sum_all <= 0.0 - no feature was found in the sample
df = df[df['sum_all'] > 0.0]

target = df['biome']
features = df.drop(columns=['biome'])

X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.DataFrame
y_test: pd.DataFrame

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, 
                                                             target,
                                                             test_size=0.2,
                                                             random_state=69420) # type: ignore

def train_and_predict_svm():
    model = SVC()
    model.fit(X=X_train, y=y_train)
    
    y_predict = model.predict(X=X_test)

    with open('result-svm.txt', 'w') as f:
        f.write(str(classification_report(y_test, y_predict)))


def train_and_predict_decision_tree():
    model = DecisionTreeClassifier()
    model.fit(X=X_train, y=y_train)
    
    y_predict = model.predict(X=X_test)
    with open('result-decision-tree.txt', 'w') as f:
        f.write(str(classification_report(y_test, y_predict)))


Process(target=train_and_predict_svm).start()
Process(target=train_and_predict_decision_tree).start()



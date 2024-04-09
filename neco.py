import pandas as pd 
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

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

class Model:
    def __init__(self, classifier, grid_search_params):
        self._classifier = classifier
        self._grid_search_params = grid_search_params

    def train_and_predict(self, 
                          X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: pd.DataFrame,
                          y_test: pd.DataFrame):
        
        self._classifier.fit(X=X_train, y=y_train)
        y_predict = self._classifier.predict(X=X_test)

        file_name = self._get_name() + '-classification_report.txt'
        with open(file_name, 'w') as f:
            f.write(str(classification_report(y_test, y_predict)))

    def _get_name(self) -> str:
        return self._classifier.__class__.__name__.lower()

    def grid_search(self, **kwargs):
        print(f'Performing grid search on {self._get_name()}...')
        grid_search = sklearn.model_selection.GridSearchCV(
            self._classifier, 
            self._grid_search_params, 
            **kwargs,
        )

        grid_search.fit(X_train, y_train)
        file_name = self._get_name() + '-grid_search.txt'

        with open(file_name, 'w') as f:
            f.write(str(grid_search.best_params_))


svm = Model(SVC(), { 
    'C': [0.5, 1], 
    'kernel': ['rbf'],
})

# svm.train_and_predict(
#     X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test
# )

# svm.grid_search() to je pomalý :/

# From GridSearchCV: criterion='log_loss', splitter='best'
tree = Model(DecisionTreeClassifier(criterion='log_loss', splitter='best'), {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
})

# tree.grid_search(n_jobs=6)
tree.train_and_predict(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)


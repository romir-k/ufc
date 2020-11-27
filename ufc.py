# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# import dataset
fights = pd.read_csv('/home/romir/Downloads/ufc-master.csv')

# process all categorical variables
fights["finish"] = str(fights["finish"])
fights["finish_details"] = str(fights["finish_details"])
label = LabelEncoder()
fights["R_Stance"] = label.fit_transform(fights["R_Stance"])
fights["B_Stance"] = label.fit_transform(fights["B_Stance"])
fights["gender"] = label.fit_transform(fights["gender"])
fights["Winner"] = label.fit_transform(fights["Winner"])
fights["better_rank"] = label.fit_transform(fights["better_rank"])
fights["weight_class"] = label.fit_transform(fights["weight_class"])
fights["finish"] = label.fit_transform(fights["finish"])
fights["finish_details"] = label.fit_transform(fights["finish_details"])
fights["country"] = label.fit_transform(fights["country"])
fights["location"] = label.fit_transform(fights["location"])
fights["R_fighter"] = label.fit_transform(fights["R_fighter"])
fights["B_fighter"] = label.fit_transform(fights["B_fighter"])
fights["date"] = label.fit_transform(fights["date"])

# replace missing values with median value
fights = fights.fillna(0)

# split dataset into training set and testing set
out = fights.Winner
inp = fights.drop(columns=['Winner', 'finish_round_time'])
x_train, x_test, y_train, y_test = train_test_split(inp, out, test_size=0.2)

# train model
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))]
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

"""
Griffin Miller
This program develops machine learning models to predict if a ball put in
play off of a given pitcher will result in a hit or an out. It also evaluates
each models ROC-AUC score and plots the curve.
"""
from matplotlib import pyplot as plt
from pybaseball import statcast_pitcher
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Tyler Glasnow MLBAM ID is 607192
# Yu Darvish MLBAM ID is 506433


def get_player_stats(id):
    """
    Takes pitcher id as paramter and retrieves all pitches thrown by that
    pitcher in the 2019 season.
    Reduces the dataframe to contain only the columns we want for our
    analysis and only the pitches that are put in play.
    returns condensed dataframe.
    """
    data = statcast_pitcher('2019-03-28', '2019-09-29', id)
    df = data[['pitch_type', 'release_speed', 'release_spin_rate',
               'if_fielding_alignment', 'launch_angle', 'launch_speed',
               'hc_x', 'hc_y', 'stand', 'type', 'events']]

    new_df = df[df['type'] == 'X']
    return new_df


def clean(df):
    """
    Takes in the condensed dataframe as a paramater.
    Creates new hitting column that holds 1 for a hit, 0 for an out.
    Creates a new column indicating if its a right handed hitter or not.
    returns the finalized dataframe.
    """
    df['RH'] = df['stand'].apply(lambda x: 1 if x == 'R' else 0)

    df['hit'] = df['events'].apply(lambda x: 1 if x in
                                   ('single', 'double', 'triple'
                                    'home_run') else 0)

    return df


def split_data(df):
    """
    Takes in the clean and finalized dataframe. Splits the data into
    test and training sets and scales the features. returns the different
    sets of data.
    """
    # drop any instances that have missing values
    df = df.dropna()

    # define features
    features = df[['pitch_type', 'release_speed', 'release_spin_rate',
                   'if_fielding_alignment', 'launch_angle', 'launch_speed',
                   'hc_x', 'hc_y', 'stand', 'type', 'RH']]

    # make dummies for categorical features
    features = pd.get_dummies(features)

    # define label
    label = df['hit']

    # split data into test and training
    features_train, features_test, label_train, label_test = \
        train_test_split(features, label, test_size=0.3)

    standard = StandardScaler()

    features_train = standard.fit_transform(features_train)
    features_test = standard.transform(features_test)

    return features_train, features_test, label_train, label_test


def create_tree(f_train, f_test, l_train, l_test):
    """
    Creates a decision tree model and measures its accuracies.
    returns the model object.
    """
    # initialize model
    model = DecisionTreeClassifier(max_depth=2)

    # train it on training data
    model.fit(f_train, l_train)

    # gather the model's predictions for train
    train_predictions = model.predict(f_train)

    # gather the model's predictions for test
    test_predictions = model.predict(f_test)

    # calculate accuaracy of train
    print('Tree Train Accuracy: ', accuracy_score(l_train, train_predictions))

    # calculate accuracy of test
    print('Tree Test Accuracy: ', accuracy_score(l_test, test_predictions))

    return model


def create_regression(f_train, f_test, l_train, l_test):
    """
    Creates a logistic regression model and measures its accuracies.
    returns the model object.
    """
    model = LogisticRegression()

    # train it on training data
    model.fit(f_train, l_train)

    # gather the model's predictions for train
    train_predictions = model.predict(f_train)

    # gather the model's predictions for test
    test_predictions = model.predict(f_test)

    # calculate accuaracy of train
    print('Reg Train Accuracy: ', accuracy_score(l_train, train_predictions))

    # calculate accuracy of test
    print('Reg Test Accuracy: ', accuracy_score(l_test, test_predictions))

    return model


def plot_roc_auc(tree, reg, f_test, l_test):
    """
    This function takes in both ML models as well as the test set's
    features and labels. It calculates the probabilites of each instance
    in each model and then calculates its AUC score. Additionally, it plots
    the ROC curve of each model to compare visually.
    """

    # Calculate probabilites for each model
    s_probs = [0 for _ in range(len(l_test))]
    tree_probs = tree.predict_proba(f_test)
    reg_probs = reg.predict_proba(f_test)

    # Only keep postive outcomes
    tree_probs = tree_probs[:, 1]
    reg_probs = reg_probs[:, 1]

    # Calculate AUC value for each model and 0.5
    s_auc = roc_auc_score(l_test, s_probs)
    tree_auc = roc_auc_score(l_test, tree_probs)
    reg_auc = roc_auc_score(l_test, reg_probs)

    print(f'Random prediction AUC: {s_auc:.3f}')
    print(f'Decision Tree AUC: {tree_auc:.3f}')
    print(f'Logistic Regression AUC: {reg_auc:3f}')

    # calculate FPR & TPRs
    s_fpr, s_tpr, s_thresh = roc_curve(l_test, s_probs)
    tree_fpr, tree_tpr, tree_thresh = roc_curve(l_test, tree_probs)
    reg_fpr, reg_tpr, reg_thresh = roc_curve(l_test, reg_probs)

    # plot the curves
    plt.plot(s_fpr, s_tpr, linestyle='--', label='Random Prediction')
    plt.plot(tree_fpr, tree_tpr, linestyle='solid', label='Decision Tree AUC')
    plt.plot(reg_fpr, reg_tpr, linestyle='dotted', label='Regression AUC')

    plt.title('ROC Plot')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('Baseball_AUC.png')


def main():
    data = get_player_stats(506433)
    clean_data = clean(data)
    f_train, f_test, l_train, l_test = split_data(clean_data)
    tree = create_tree(f_train, f_test, l_train, l_test)
    reg = create_regression(f_train, f_test, l_train, l_test)
    plot_roc_auc(tree, reg, f_test, l_test)


if __name__ == '__main__':
    main()

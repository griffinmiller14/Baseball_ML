"""
Griffin Miller
This program develops a machine learning model to predict if a ball put in
play off of a given pitcher will result in a hit or an out.
"""
from pybaseball import statcast_pitcher
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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


def create_model(df):
    """
    Takes in the clean and finalized dataframe. Trains a model to predict
    whether or not a ball put into the field of play will result in a hit
    or an out. The model is trainined on the given features and labels in
    the training set and the models accuracy is measured on the test set.
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

    # initialize model
    model = DecisionTreeClassifier(max_depth=2)

    # train it on training data
    model.fit(features_train, label_train)

    # gather the model's predictions for train
    train_predictions = model.predict(features_train)

    # gather the model's predictions for test
    test_predictions = model.predict(features_test)

    # calculate accuarcy of train
    print('Accuracy: ', accuracy_score(label_train, train_predictions))

    # calculate accuracy of test
    print('Accuracy: ', accuracy_score(label_test, test_predictions))


def main():
    data = get_player_stats(506433)
    clean_data = clean(data)
    create_model(clean_data)


if __name__ == '__main__':
    main()

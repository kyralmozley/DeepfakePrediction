'''
Moving out of Jupyter
- Preprocess data
- Train NN, Train Classifier
- Test
'''

### Imports
from imports import *
from createGraphs import *
from training import *
import createDataset

def cleanFiles(files):
    df = pd.DataFrame(columns=landmark_cols)
    # NEED TO TEST THIS LINE ON PC
    for file in files:
        df = df.append(createDataset.get_visual_features(file)[landmark_cols])

    return df
    


def predictData(df):
    train = TrainModel()
    labelPresent=False

    df = df.sort_values(by=['filename', 'frame']).reset_index()
    df.dropna(inplace=True)

    #NN input
    X = df[landmark_cols].to_numpy()

    #check if file had labels, if so can provide metrics
    if 'label' in df.columns:
        labelPresent = True
        # y1 is groundtruth for NN
        y_1 = df['label'].to_numpy()
        df['binary_label'] = np.vectorize(LABELS.get)(y_1)
        # y2 is groundtruth for classifier 
        y_2 = np.array(list(df.groupby(['filename']).binary_label.max()))

    #NN
    nn_model = load_model('model.h5')
    try:
        assert(X.shape[1] ==train.FEATURE_VECTOR_LEN)
    except AssertionError as msg:
        print("Error in dimensions between x,y in data")
        print("NN input shape expects:{}, Received:{}".format(train.FEATURE_VECTOR_LEN, X.shape[1]))
    
        
    predicted = nn_model.predict(X, verbose=1)
    df['prediction'] = predicted[:,1]

    #Classifier input
    classifier_input = train.getClassifierInput(df.groupby('filename').prediction)

    try:
        assert(classifier_input.shape[1] ==train.CLASSIFIER_FEATURES)
    except AssertionError as msg:
        print("Error in dimensions between x,y in data")
        print("Classifier input shape expects:{}, Received:{}".format(train.CLASSIFIER_FEATURES, classifier_input.shape[1]))

    knn_classifier = pickle.load(open("classifier.pkl", "rb"))

    y_pred = knn_classifier.predict(classifier_input)

    if labelPresent:
        print('Classifier Accuracy Score: ', accuracy_score(y_2, y_pred))
        pprint.pprint(metrics.classification_report(y_2, y_pred, digits=3))
    
    return y_pred

if __name__ == '__main__':
    '''
    #given array of files
    files = ['cleaned_data_real.csv', 'cleaned_data_fake.csv']
    df = pd.concat(pd.read_csv(f, index_col=['filename', 'frame']) for f in files)
    predictData(df)
    trainValidate()
    '''
    trainValidate()
    

'''
Moving out of Jupyter
- Preprocess data
- Train NN, Train Classifier
- Test
'''

### Imports
from imports import *
from createGraphs import *


landmark_cols = ['landmark_{}'.format(i) for i in range(68)]
LABELS = {"REAL":0, "FAKE":1}

class TrainModel:
    FEATURE_VECTOR_LEN = 68
    CLASSIFIER_FEATURES = 8
    NUM_CLASSES = 2

    nn_model = None
    knn_classifier = None

    def __init__(self) -> None:
        self.X_nn, self.y_nn = [], []
        self.X_classifier, self.y_classifier = [],[]
        self.X_val, self.y_val = [],[]

    def process(self, files, split_1, split_2):
        self.files = files
        print(" Processing Files {} ".format(self.files).center(20, '*'))
        print(self.files)
        # Read in data
        df = pd.concat(pd.read_csv(f, index_col=['filename', 'frame']) for f in self.files)
        df = df.sort_values(by=['filename', 'frame']).reset_index()
        df.dropna(inplace=True)
        # Split data
        # Split 1 = NN training split
        # Split 2 = Classifier split
        # 1-(Split1+Split2) = Validation split
        try:
            assert(split_1+split_2 < 1)
        except AssertionError as msg:
            print("Splits must add up to be less than 1")

        filenames = df.groupby('filename').size()
        total_files = len(filenames)
        nn_split = filenames.index[int(split_1*total_files)]
        classifier_split = filenames.index[int((split_1+split_2)*total_files)]

        split_index_1 = df.loc[(df['filename']==nn_split)].head(1).index[0]
        split_index_2 = df.loc[(df['filename']==classifier_split)].head(1).index[0]

        df_mlp = df[:split_index_1]
        df_mlp_predict = df[split_index_1:split_index_2:]
        df_val = df[split_index_2::]

        self.X_nn = df_mlp[landmark_cols].to_numpy()
        self.X_classifier = df_mlp_predict[landmark_cols].to_numpy()
        self.X_val = df_val[landmark_cols].to_numpy()

        y_nn = df_mlp['label'].to_numpy()
        self.y_nn=np.vectorize(LABELS.get)(y_nn)

        y_classifier = df_mlp_predict['label'].to_numpy()
        df_mlp_predict['binary_label'] = np.vectorize(LABELS.get)(y_classifier)
        self.y_classifier = np.array(list(df_mlp_predict.groupby(['filename']).binary_label.max()))

        y_val = df_val['label'].to_numpy()
        df_val['binary_label'] = np.vectorize(LABELS.get)(y_val)
        self.y_val = np.array(list(df_val.groupby(['filename']).binary_label.max())) 
        return(df_mlp, df_mlp_predict, df_val) 
 
    def createNN(self, lossFunction, epochs, batchsize):
        print(" Training MLP ".center(20, '*'))
        try:
            assert(self.X_nn.shape[0] ==self.y_nn.shape[0])
        except AssertionError as msg:
            print("Error in dimensions between x,y in data")      
       
        input_shape = (self.FEATURE_VECTOR_LEN, )
        Y_train = to_categorical(self.y_nn, self.NUM_CLASSES)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)
        
        # Define Model
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))

        model.compile(loss=lossFunction, optimizer='adam', metrics=['accuracy'])
        model.fit(self.X_nn, Y_train, epochs=epochs, batch_size=batchsize, verbose=1, validation_split=0.2)
        print(" Saving MLP ".center(20, '*'))
        model.save('model.h5')
        print(" Saved MLP ".center(20, '*'))

        try:
            reconstructed_model = load_model("model.h5")
        except:
            print('Failed to save model correctly (cannot reload).')

        np.testing.assert_allclose(
            model.predict(self.X_val), reconstructed_model.predict(self.X_val)
        )

        self.nn_model = model
        print("Done Training Network ".center(20, '*'))

    def trainClassifier(self, df_mlp_predict):
        print(" Training Classifier ".center(20, '*'))
        if self.nn_model is None:
            self.nn_model = load_model('model.h5')
        predicted = self.nn_model.predict(self.X_classifier, verbose=1)
        
        df_mlp_predict['prediction'] = predicted[:,1]
        print("Accuracy Score for Neural Net: ",accuracy_score(df_mlp_predict.binary_label, np.round(predicted[:,1], 0)))
        
        classifier_input = self.getClassifierInput(df_mlp_predict.groupby('filename').prediction)
        try:
            assert(classifier_input.shape[0] ==self.y_classifier.shape[0])
        except AssertionError as msg:
            print("Error in dimensions between x,y in data")

        classifier = KNeighborsClassifier(n_neighbors=50)
        classifier.fit(classifier_input, self.y_classifier)
        self.knn_classifier = classifier
        pickle.dump(classifier, open('classifier.pkl', 'wb'))
        print("Done Fitting Training Classifier ".center(20, '*'))
        return df_mlp_predict

    def validate(self, df_val):
        if self.nn_model is None:
            self.nn_model = load_model('model.h5')
        try:
            assert(self.X_val.shape[1] ==self.X_nn.shape[1])
        except AssertionError as msg:
            print("Error in dimensions between x,y in data")
            print("NN input shape expects:{}, Received:{}".format(self.X_val.shape[1], self.X_nn.shape[1]))
       
            
        val_predicted = self.nn_model.predict(self.X_val, verbose=1)
        df_val['prediction'] = val_predicted[:,1]
        classifier_input = self.getClassifierInput(df_val.groupby('filename').prediction)

        try:
            assert(classifier_input.shape[1] ==self.CLASSIFIER_FEATURES)
        except AssertionError as msg:
            print("Error in dimensions between x,y in data")
            print("Classifier input shape expects:{}, Received:{}".format(self.CLASSIFIER_FEATURES, classifier_input.shape[1]))

        if self.knn_classifier is None:
            self.knn_classifier = pickle.load(open("classifier.pkl", "rb"))

        y_pred = self.knn_classifier.predict(classifier_input)
        return y_pred
       
    def getClassifierInput(self, g):
        #Take in a group, return numpy
        # CLASSIFIER_FEATURES = 8
        min_pred = g.agg('min')
        max_pred = g.agg('max')
        stddev_pred = g.agg('std')
        var_pred = g.agg('var')
        mean_pred = g.agg('mean')
        median_pred = g.agg('median')
        range_pred = max_pred - min_pred
        q25 = g.quantile(0.25)
        q75 = g.quantile(0.75)
        iqr_pred = q75-q25

        classifier_input = pd.DataFrame({
            'Min': min_pred, 
            'Max': max_pred,
            'Std' : stddev_pred,
            'Variance' : var_pred, 
            'Mean' : mean_pred, 
            'Median' : median_pred,
            'Range' : range_pred,
            'IQR' : iqr_pred
        })
        self.CLASSIFIER_FEATURES = len(classifier_input.columns)

        return classifier_input.to_numpy()

def trainValidate():
    path = 'deepfakePrediction/'
    all_files = glob.glob(path+'cleaned_data_*.csv') #glob.glob(path+'/*.csv')+glob.glob(path+'/real/*.csv')
    train = TrainModel()
    print(all_files)
    d1, d2, d3 = train.process(all_files, 0.4,0.3)
    #train.createNN('msle', 5, 64)
    d2 = train.trainClassifier(d2)
    #plotBoxPlot(d2, 'mean')
    y_pred = train.validate(d3)
    acc = accuracy_score(train.y_val, y_pred)
    print('Validation Accuracy Score: ', acc)
    pprint.pprint(metrics.classification_report(train.y_val, y_pred, digits=3))
    return acc


        








import os
import json
import csv
import pandas as pd
from pprint import pprint

def get_visual_features(file):
    # Remove old files
    os.system('rm -rf processed')
    os.system('rm -rf video.mp4')

    # Create mp4 file
    os.system('ffmpeg -y -loglevel info -i {} video.mp4'.format(file))

    # Use OpenFace2.0 to extract features
    os.system('~/opencv-4.1.0/build/OpenBLAS/OpenFace/build/bin/FeatureExtraction -f {} -out_dir processed'.format('video.mp4'))
    
    # Read libraries outputed csv for file, returns 714 columns
    df = pd.read_csv('processed/video.csv', header=0)

    return df




def main(path, part):
    #List all the files in the dfdc_train_part folder
    files = os.listdir(path)

    #Create a csv file to store the output in, provide the headers
    with open('dataset/train_dataset_{}.csv'.format(part), 'w') as f:
        write = csv.writer(f)
        write.writerow(pd.read_csv('columns.csv'))

    #Open the json file with the REAL/FAKE labels in the dataset
    json_data = open(path+'/metadata.json')
    labels = json.load(json_data)
    
    df = pd.DataFrame()

    #Store filename and its respective label in dataframe
    for f in files:
        if f.__contains__('mp4'):
            df = df.append({'file': f, 'label':labels[f]['label']}, ignore_index=True)
    json_data.close()

    # For each file, get the features
    for index, row in df.iterrows():
        file = row['file']
        label = row['label']

        # Call visual features method (uses OpenFace 2.0)
        df_visual_features = get_visual_features(path+'/'+file)
        pprint(df_visual_features)
        data = df_visual_features.copy()
        data.insert(loc=0, column='filename', value=file)
        data.insert(loc=len(data.columns), column='label', value=label)
        data.to_csv('dataset/train_dataset_{}.csv'.format(part), mode='a', index=False, header=False)




if __name__ == '__main__':
    path = '/media/kyralm/68d88bad-cf04-4698-adb1-0e3035052da9/home/kyra/Desktop/train'

    for i in range(16,43):
        folder = path+'/dfdc_train_part_{}'.format(i)
        main(folder,i)
    '''
    part = 0
    path = path+'/dfdc_train_part_{}'.format(part)
    main(path, part)
    '''

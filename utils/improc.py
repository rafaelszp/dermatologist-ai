import numpy as np
import keras
from os import walk,path
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class SkinCancerImageGenerator(keras.utils.Sequence):

    def __init__(self,files_dir,batch_size=20, dim=(32,32), n_channels=3,
                 n_classes=3):
        self.files_dir = files_dir
        self.dim = dim
        self.batch_size = batch_size
        self.list_files,self.labels = self.__generate_list_files()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = True
        self.on_epoch_end()

    def __generate_list_files(self):
        """Generates list of IDs from directory provided"""
        files = []
        labels = []
        label_encoder = LabelEncoder()
        for (dirpath, dirnames, filenames) in walk(self.files_dir):
            for name in filenames:
                files.append(path.join(dirpath, name))
                m = re.search(r'\w+$',dirpath)
                labels.append(m.group(0))
        files = np.array(files)
        labels = np.array(labels)
        integer_encoded = label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        labels = onehot_encoder.fit_transform(integer_encoded)
        labels_dict = {}
        for file,label in zip(files,labels):
            labels_dict[file] = label

        return files,labels_dict

    #1 Ler diretório de images
    #2 Para cada imagem gerar ndarray da imagem e outro ndarray de label

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_files_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_files_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            X[i,]=np.array(i)

            # Store class
            y[i] = self.labels[ID]

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)



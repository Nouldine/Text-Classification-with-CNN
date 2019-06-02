import numpy as np
import os
import matplotlib.pyplot as plt


os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.utils import plot_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from keras.utils import plot_model


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

categories = ['athletics',
              'cricket',
              'football',
              'rugby',
              'tennis'
        ]

print("START")

def  CNN_procedures():

    # Laoding the data set - training data.
    corpus = load_files( container_path='bbcsport', description=None, load_content=True, encoding='utf-8', categories = categories, shuffle=True, decode_error='ignore', random_state = 42 )

    # You can check the target names ( categories ) 

    print( corpus.target_names )
    print( len( corpus.data) )

    texts = []

    labels = corpus.target
    print('labels', labels )
    texts = corpus.data

    tokenizer = Tokenizer( nb_words = MAX_NB_WORDS )
    tokenizer.fit_on_texts( texts )
    sequences = tokenizer.texts_to_sequences( texts )

    words_index =  tokenizer.word_index

    print('Found %s unique tokens.' % len(words_index))

    data = pad_sequences( sequences, maxlen=MAX_SEQUENCE_LENGTH )

    labels = to_categorical( np.asarray(labels))

    print("Shape of data tensor:", data.shape )
    print("Shape of label tensor:",  labels.shape )

    indices = np.arange( data.shape[ 0 ] )
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    nb_validations_samples = int( VALIDATION_SPLIT * data.shape[ 0 ] )

    print( nb_validations_samples )


    x_train = data[:-nb_validations_samples]
    y_train = labels[:-nb_validations_samples]
    x_val = data[-nb_validations_samples:]
    y_val = labels[-nb_validations_samples:]


    print('Number of positive and negative in training and validation set')

    print( y_train.sum( axis = 0 ))
    print( y_val.sum( axis = 0))

    GLOVE_DIR = "glove.6B"

    embeddings_index = {}

    glove_file = open( os.path.join( GLOVE_DIR, 'glove.6B.100d.txt' ))


    for line in glove_file:

        values = line.split()
        word = values[ 0 ]
        coefs = np.asarray( values[ 1: ], dtype = 'float32' )
        embeddings_index[ word ] = coefs

    glove_file.close()

    embedding_matrix = np.random.random(  ( len(words_index) + 1, EMBEDDING_DIM ) )

    for word, i in words_index.items():

        embedding_vector = embeddings_index.get(word)
    
        if embedding_vector is not None:

        # Words not  found in embedding index will be all-zeros
        
            embedding_matrix[ i ] = embedding_vector

    embedding_layer = Embedding( len( words_index ) + 1,  EMBEDDING_DIM, 
        weights = [ embedding_matrix ],
        input_length = MAX_SEQUENCE_LENGTH,
        trainable = True )

    sequence_input = Input( shape = ( MAX_SEQUENCE_LENGTH, ), dtype='int32')
    embedded_sequences =  embedding_layer( sequence_input )


    l_cov1 = Conv1D( 128, 5, activation = 'relu')( embedded_sequences )
    l_pool1 = MaxPooling1D( 5 )( l_cov1 )
    l_cov2 = Conv1D( 128, 5, activation = 'relu' )( l_pool1 )
    l_pool2 = MaxPooling1D( 5 )( l_cov2 )
    l_cov3 = Conv1D( 128, 5, activation = 'relu')( l_pool2 )
    l_pool3 = MaxPooling1D( 35 )( l_cov3 )  # gobal max pooling
    l_flat = Flatten()( l_pool3 )
    l_dense = Dense( 128, activation = 'relu' )( l_flat )
    preds = Dense( 5,  activation = 'softmax' )( l_dense )

    model = Model( sequence_input, preds )
    model.compile( loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = ['acc'] )

    print("Model fitting -  more complex convolutiona neural network")

    plot_model( model, to_file = 'model_static.png' )

    model.summary()
    history = model.fit( x_train, y_train, validation_data=( x_val, y_val ), epochs = 10, 
            batch_size = 50 )


    # Plot training & valition accuracy values
    plt.plot( history.history['acc'] ) # accuracy
    plt.plot( history.history['val_acc'] ) # validation accuracy
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch') # Number of iteration 
    plt.legend(['accuracy', 'validation accuracy'],  loc = 'upper left' )
    
    plt.show()

    # Plot training & validation loss values
    plt.plot( history.history['loss'] ) #  Training loss
    plt.plot( history.history['val_loss'] ) # validation loss
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'validation loss'], loc = 'upper left')

    plt.show()


CNN_procedures()




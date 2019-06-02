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


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

categories = ['alt.atheism',
        'comp.windows.x',
        'rec.sport.hockey',
        'soc.religion.christian',
        'comp.graphics',
        'misc.forsale',
        'sci.crypt',
        'talk.politics.guns',
        'comp.os.ms-windows.misc',
        'rec.autos',
        'sci.electronics',
        'talk.politics.mideast',
        'comp.sys.ibm.pc.hardware',
        'rec.motorcycles',
        'sci.med',
        'talk.politics.misc',
        'comp.sys.mac.hardware',
        'rec.sport.baseball',
        'sci.space',
        'talk.religion.misc'
        ]

print("START")

def  CNN_procedures():

    # Laoding the data set - training data.
    newsgroups_train = fetch_20newsgroups( subset='train', shuffle=True, categories = categories,)

    # You can check the target names ( categories ) 

    print( newsgroups_train.target_names )
    print( len(newsgroups_train.data) )

    texts = []

    labels = newsgroups_train.target
    print('labels', labels )
    texts = newsgroups_train.data

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


    # Complex  convulotional approach
    convs = []
    filter_sizes = [ 3, 4, 5 ]

    sequence_input = Input( shape = ( MAX_SEQUENCE_LENGTH, ), dtype='int32')
    embedded_sequences =  embedding_layer( sequence_input )


    for filter_size in filter_sizes:

        l_conv = Conv1D( nb_filter = 128, filter_length = fsz, activation='relu')(embedded_sequences )
        l_pool = MaxPooling1D( 5 )( l_conv )

        convs.append( l_pool )


    l_merge = Concatenate( axis = 1 )( convs )
    l_cov1 = Conv1D( 128, 5, activation = 'relu')( l_merge )
    l_pool1 = MaxPooling1D( 5 )( l_cov1 )
    l_cov2 = Conv1D( 128, 5, activation = 'relu')( l_pool1 )
    l_pool2 = MaxPooling1D( 30 )( l_cov2 )
    l_flat = Flatten()( l_pool2 )
    l_dense = Dense( 128, activation = 'relu' )( l_flat )
    preds = Dense( 20, activation = 'softmax')( l_dense )

    model = Model( sequence_input, preds )
    model.compile( loss = 'categorical_crossentropy',
        optimizer = 'rmsprop',
        metrics = ['acc'] )

    print("Model fitting -  more complex convolutiona neural network")

    model.summary()
    history = model.fit( x_train, y_train, validation_data=( x_val, y_val ), epochs = 10, 
            batch_size = 50 )

    #plot_model( model, to_file = 'model.png' )

    # Plot training & valition accuracy values
    plt.plot( history.history['acc'] ) # accuracy
    plt.plot( history.history['val_acc'] ) # validation accuracy
    plt.title('Model accuracy CNN-non-static')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch') # Number of iteration 
    plt.legend(['accuracy', 'validation accuracy'],  loc = 'upper left' )
    plt.show()


    # Plot training & validation loss values
    plt.plot( history.history['loss'] ) #  Training loss
    plt.plot( history.history['val_loss'] ) # validation loss
    plt.title('Model loss CNN-non-static')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'validation loss'], loc = 'upper left')

    plt.show()


CNN_procedures()




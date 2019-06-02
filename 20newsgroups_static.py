import numpy as np
import os
import matplotlib.pyplot as plt
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

os.environ['KERAS_BACKEND'] = 'tensorflow'

print ( "STARTING PROCEDURES " )

def  cnn_procedures():

    newsgroups_train = fetch_20newsgroups( subset = 'train', shuffle = True,  categories = categories, )

    print( newsgroups_train.target_names )

    print( len(newsgroups_train.data) )

    texts = []

    labels = newsgroups_train.target

    print('labels', labels )

    texts = newsgroups_train.data

    tokenizer = Tokenizer( nb_words = MAX_NB_WORDS )

    tokenizer.fit_on_texts( texts )

    sequences = tokenizer.texts_to_sequences( texts )
    
    word_index = tokenizer.word_index

    print("Found %s unique. " % len(word_index) )

    data = pad_sequences( sequences, maxlen = MAX_SEQUENCE_LENGTH )

    labels = to_categorical( np.asarray( labels ) )

    #for i in labels: 

    #   print( i )

    print("Shape  of data tensor:", data.shape )

    print("Shape of label tensor:", labels.shape )

    indices = np.arange( data.shape[ 0 ] )
    
    np.random.shuffle( indices )

    data = data[ indices ]

    labels = labels[ indices ]

    nb_validations_samples = int( VALIDATION_SPLIT * data.shape[ 0 ] )

    x_train = data[:-nb_validations_samples]
    y_train = labels[:-nb_validations_samples]
    x_val = data[-nb_validations_samples:]
    y_val = labels[-nb_validations_samples:]

    print('Number of positive and negative reviews in training and validationn set')

    print( y_train.sum( axis = 0 ) )
    print( y_val.sum( axis = 0 ) )

    GLOVE_DIR = 'glove.6B'
    
    embeddings_index = {}
    
    glove  = open( os.path.join( GLOVE_DIR, 'glove.6B.100d.txt'))
    
    for line in glove: 

       values = line.split()
       word = values[ 0 ]
       coefs = np.asarray( values[ 1: ], dtype = 'float32' )
       embeddings_index[ word ] = coefs

    
    glove.close()

    print('Total %s word  vectors in Glove 6B 100d,' % len(embeddings_index) )

    embeddings_matrix = np.random.random( ( len( word_index )  + 1, EMBEDDING_DIM ) )

    for word, i in word_index.items():

        embedding_vector = embeddings_index.get( word )

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros
            embeddings_matrix[ i ] = embedding_vector

    

    embedding_layer = Embedding( len( word_index ) + 1, EMBEDDING_DIM, weights = [ embeddings_matrix ], input_length = MAX_SEQUENCE_LENGTH, trainable = True )

    sequence_input = Input( shape = ( MAX_SEQUENCE_LENGTH, ), dtype = 'int32' )
    embedding_sequences = embedding_layer( sequence_input )

    l_cov1 = Conv1D( 128, 5, activation = 'relu')( embedding_sequences )
    l_pool1 = MaxPooling1D( 5 )( l_cov1 )
    l_cov2 = Conv1D( 128, 5, activation = 'relu' )( l_pool1 )
    l_pool2 = MaxPooling1D( 5 )( l_cov2 )
    l_cov3 = Conv1D( 128, 5, activation = 'relu')( l_pool2 )
    l_pool3 = MaxPooling1D( 35 )( l_cov3 )  # gobal max pooling
    l_flat = Flatten()( l_pool3 )
    l_dense = Dense( 128, activation = 'relu' )( l_flat )
    preds = Dense( 20,  activation = 'softmax' )( l_dense )


    model = Model( sequence_input, preds )

    model.compile( loss = 'categorical_crossentropy',
            optimizer = 'rmsprop',
            metrics = ['acc'] 

            )
    
    model.summary()
    print( x_train )
    print( y_train )
    
    history =  model.fit( x_train, y_train, validation_data = ( x_val, y_val ), nb_epoch = 10, batch_size = 128 )


    # Plot training & valition accuracy values
    plt.plot( history.history['acc'] )
    plt.plot( history.history['val_acc'] )
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accuracy', 'validation accuracy'],  loc = 'upper left' )
    plt.show()


    # Plot training & validation loss values
    plt.plot( history.history['loss'] )
    plt.plot( history.history['val_loss'] )
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'validation loss'], loc = 'upper left')

    plt.show()
  

cnn_procedures()























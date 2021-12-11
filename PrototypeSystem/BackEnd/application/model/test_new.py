import tensorflow as tf
import numpy as np
import pandas as pd
import os
from application.model.transform_type import transform_type



def detection(path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    myrand = 71927
    np.random.seed(myrand)
    tf.random.set_seed(myrand)
    print("Random seed is:", myrand)

    WORDS_SIZE = 10000
    INPUT_SIZE = 500
    NUM_CLASSES = 2

    # test=pd.read_csv("/home/djx/project/Detection/test_150.csv",encoding='gbk')
    # read data
    test=pd.read_csv(path,encoding='utf-8')
    item = len(test)
    print(test.head())

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(list(test['FileContent']))
    print('Number of tokens: ',len(tokenizer.word_counts))
    tokenizer.num_words = WORDS_SIZE

    list_tokenized_test = tokenizer.texts_to_sequences(test['FileContent'])
    x_test = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_test,
                                     maxlen=INPUT_SIZE,
                                     padding='post')
    x_test = x_test.astype(np.int64)
    # y_test=[]
    # for col in range(1,6):
    #     y_test.append(tf.keras.utils.to_categorical(test.iloc[:,col], num_classes=NUM_CLASSES).astype(np.int64))

    # Load model
    model = tf.keras.models.load_model("application/model/model-ALL-40_6w.hdf5")
    # results = model.evaluate(x_test, y_test, batch_size=128)
    # for num in range(0,len(model.metrics_names)):
    #     print(model.metrics_names[num]+': '+str(results[num]))

    predicted = model.predict(x_test)
    pred_test = [[], [], [], [], []]

    for col in range(0, len(predicted)):
        for row in predicted[col]:
            if row[0] >= row[1]:
                pred_test[col].append(0)
            else:
                pred_test[col].append(1)

    type_list, detail_list = transform_type(pred_test[0],pred_test[1],pred_test[2],pred_test[3],pred_test[4],item)
    test['Type'] = type_list
    test['Detail'] = detail_list
    test.to_csv(path,index=None)

if __name__ == '__main__':
    detection(r"CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84_bad.csv")
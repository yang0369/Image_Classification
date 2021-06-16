import os
import yaml
import demjson
import pandas as pd
import re
import numpy 
from numpy import savetxt
import logging
import argparse
import tensorflow as tf
import transformers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import BinaryAccuracy
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

AUTOTUNE = tf.data.experimental.AUTOTUNE

logger = logging.getLogger("6estates")
logging.basicConfig(level=logging.INFO)

class DataPipeline():
    """
    DataPipe processes the input text data with general data cleaning and return the processed data for model training
    """

    def __init__(self, flag_contraction=True, flag_digits=True):
        """[initialize Datapipeline]

        Keyword Arguments:
            flag_contraction {bool} -- [if expand contractions for text] (default: {True})
            flag_digits {bool} -- [if remove digits from text] (default: {True})
        """
        self._flag_contraction = flag_contraction
        self._flag_digits = flag_digits
        logger.info("Datapipeline is initiated")
    
    @staticmethod
    def expand_contractions(word:str) -> str:
        """[expand contractions]

        Arguments:
            word {str} -- [each word in sentence]

        Returns:
            str -- [word after being expanded]
        """
        word = re.sub(r"won\'t", "will not", word)
        word = re.sub(r"can\'t", "can not", word)
        word = re.sub(r"n\'t", " not", word)
        word = re.sub(r"\'re", " are", word)
        word = re.sub(r"\'d", " would", word)
        word = re.sub(r"\'ll", " will", word)
        word = re.sub(r"\'t", " not", word)
        word = re.sub(r"\'ve", " have", word)
        word = re.sub(r"\'m", " am", word)
        return word
    
    @staticmethod
    def to_lower(word:str) -> str:
        """[convert to lower case]

        Arguments:
            word {str} -- [each word in sentence]

        Returns:
            str -- [word in lower case]
        """
        word = word.lower()
        return word

    @staticmethod
    def remove_digits(word:str) -> str:
        """[remove digit word]

        Arguments:
            word {str} -- [each word in sentence]

        Returns:
            str -- [word with no digit]
        """
        patten = '[0-9]'
        word = re.sub(patten, '', word)
        return word

    @staticmethod
    def drop_na(df:pd.DataFrame) -> pd.DataFrame:
        """[drop rows with na values]

        Arguments:
            df {dataframe} -- [input dataframe]

        Returns:
            dataframe -- [dataframe with no na values]
        """
        logger.info("performed dropping missing values")
        return df.dropna() 

    @staticmethod
    def drop_duplicates(df:pd.DataFrame) -> pd.DataFrame:
        """[drop duplicated rows when both label and sentences are the same]

        Arguments:
            df {dataframe} -- [input dataframe]

        Returns:
            dataframe -- [dataframe without duplicates]
        """
        logger.info("performed dropping duplicates")
        return df.drop_duplicates(ignore_index=True)


    @staticmethod
    def load_data(data_path:str) -> pd.DataFrame:
        """[load input data input dataframe]

        Arguments:
            input_path {str} -- [where is the data saved]
        """
        df = pd.DataFrame([demjson.decode(line) for line in open(data_path, 'r')])
        return df

    def transform_data(self, data_path:str) -> pd.DataFrame:
        """[perform one-stop tranforming for data pre-processing]

        Arguments:
            sentence {str} -- [text data that needs to be cleaned]
        """
        df = self.load_data(data_path)
        df = self.drop_na(df)
        df = self.drop_duplicates(df)
        if self._flag_contraction:
            df.loc[:, "sentence"] = df.sentence.apply(lambda x: self.expand_contractions(x))
        if self._flag_digits:
            df.loc[:, "sentence"] = df.sentence.apply(lambda x: self.remove_digits(x))
        df.loc[:, "sentence"] = df.sentence.apply(lambda x: self.to_lower(x))
        logger.info("finished transforming data")        
        return df

class Model():
    """
    Model wraper for training and evaluating BERT model 
    """

    def __init__(self, config, path):
        """[initialize the model class]

        Arguments:
            config {[dict]} -- [dictionary that passes all the required path and hyperparameters]
            path {[str]} -- [where to store the trained model]
        """
        self._model_path = path
        self._tokenizer = AutoTokenizer.from_pretrained(config["model"]["MODEL"], use_fast=True)
        self._model = TFAutoModelForSequenceClassification.from_pretrained(config["model"]["MODEL"], num_labels=config["model"]["NUM_LABELS"])
        self._batch = config["model"]["BATCH_SIZE"]
        self._learning_rate = config["model"]["LEARNING_RATE"]
        self._epochs = config["model"]["EPOCHS"]
        self._max_length = config["model"]["MAX_LENGTH"]
        self._lr_patience = config["model"]["LR_PATIENCE"]
        self._es_patience = config["model"]["ES_PATIENCE"]
        self._factor = config["model"]["FACTOR"]
        logger.info("initiating model")

    def preprocess_data(self, train, val, test):
        """[preprocess the text data for BERT model, model specific processing]

        Arguments:
            train {[pd.DataFrame]} -- [pd.DataFrame]
            val {[pd.DataFrame]} -- [pd.DataFrame]
            test {[pd.DataFrame]} -- [pd.DataFrame]

        Returns:
            [tensorflow dataset object] -- [return the tensorflow dataset by batches]
        """
        train_encodings = self._tokenizer(train.sentence.tolist(), truncation=True, padding=True, max_length = self._max_length, return_tensors="tf")
        val_encodings = self._tokenizer(val.sentence.tolist(), truncation=True, padding=True, max_length = self._max_length, return_tensors="tf")
        test_encodings = self._tokenizer(test.sentence.tolist(), truncation=True, padding=True, max_length = self._max_length, return_tensors="tf")
        # prepare tensorflow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train.label.values
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val.label.values
        ))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test.label.values
        ))
        self._steps_per_epoch = train_dataset.cardinality().numpy()//self._batch 
        self._validation_steps = val_dataset.cardinality().numpy()//self._batch 
        # shuffle and batch tf dataset
        dataset_train_tf = train_dataset.shuffle(train_dataset.cardinality().numpy()).batch(self._batch)
        dataset_val_tf = val_dataset.batch(self._batch)
        dataset_test_tf = test_dataset.batch(self._batch)
        # repeat and prefetch dataset 
        self._train_dataset = dataset_train_tf.repeat(self._epochs).prefetch(buffer_size=AUTOTUNE)
        self._test_dataset = dataset_test_tf.prefetch(buffer_size=AUTOTUNE)
        self._validation_dataset = dataset_val_tf.prefetch(buffer_size=AUTOTUNE)
        return self._train_dataset, self._test_dataset, self._validation_dataset

    def build_model(self):
        """[build up the BERT model]
        """
        # make sure all layers are trainable
        self._model.trainable = True
        """
        # freeze the BERT encoder while training the classifier
        for layer in self._model.layers:
            if layer.name == "bert":
                layer.trainable = False
        # unfreeze the last X numbers of layers of encoder for training
        for layer in self._model.layers[:]:
            if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                for idx, layer in enumerate(layer.encoder.layer):
                    if idx in [6, 7, 8, 9, 10, 11]:
                        # print(layer.name)
                        layer.trainable = True
        """
        self._model.compile(optimizer=Adam(learning_rate=self._learning_rate),
                             loss=BinaryCrossentropy(from_logits=True),
                             metrics=BinaryAccuracy()) 
    def fit_model(self):
        """[fit the dataset to the model and train model]

        Returns:
            [history object] -- [records all the train and val info]
        """
        reduce_lr = ReduceLROnPlateau(
            factor=self._factor,
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=self._lr_patience,
            min_lr=1e-08)
        early_stop = EarlyStopping(
            verbose=1,
            patience=self._es_patience,
            mode='auto',
            monitor='val_loss',
            restore_best_weights=True)
        history = self._model.fit(
            x=self._train_dataset,
            validation_data=self._validation_dataset,
            steps_per_epoch=self._steps_per_epoch,
            epochs=self._epochs,
            validation_steps=self._validation_steps,
            callbacks=[early_stop, reduce_lr])
        return history

    def evaluate_model(self):
        """[evaluate the model's performance based on test data]

        Returns:
            [float] -- [return the binary accuracy]
        """
        loss, binary_acc = self._model.evaluate(self._test_dataset)
        logger.info('Test Loss: %s', loss)
        logger.info('Test Accuracy: %s', binary_acc)
        return binary_acc

    def save_model(self):
        """[save the trained model in .h5 format]
        """
        logger.info(f"Saving model in {self._model_path}")
        self._model.save_pretrained(self._model_path)
    
    def save_predicted(self):
        """[save the predicted y labels in a csv for model performance analysis]
        """
        predicted = [int(i) for i in tf.round(tf.sigmoid(self._model.predict(self._test_dataset)["logits"])).numpy()]
        logger.info(f"predicted values for test_dataset: {predicted}")
        savetxt(os.path.join(self._model_path, "predicted.csv"), predicted, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', action='store', type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    # change to local paths
    config["dataset"]["training_path"] = "./data/train.json"
    config["dataset"]["test_path"] = "./data/test.json"
    config["dataset"]["val_path"] = "./data/dev.json"
    # set the path to store trained model, training info and logger
    path = "./model"
    store_logs = os.path.join(path, "info.log") # where we store the logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # config logger
    logging.basicConfig(
        filename=store_logs,
        filemode='w',
        format='%(asctime)s , %(name)s - %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S', 
        level=logging.INFO
    )
    # initiating logger
    logger = logging.getLogger("6estate")
    logger.info(f"parameters are:{config}")
    # initiate pipeline
    dp = DataPipeline()
    train = dp.transform_data(config["dataset"]["training_path"])
    # remove non-English sentences as noise
    train = train.drop(config["datapipe"]["drop_list"], axis=0)
    test = dp.transform_data(config["dataset"]["test_path"])
    val = dp.transform_data(config["dataset"]["val_path"])
    train_size = train.shape[0]
    val_size = val.shape[0]
    test_size = test.shape[0]
    logger.info(f"train size:{train_size}")
    logger.info(f"test size:{test_size}")
    logger.info(f"val size:{val_size}")
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass   
    train = train.head(16)
    val = val.head(16)
    test = test.head(16)
    # initiating modelling
    Bert = Model(config, path)
    Bert.preprocess_data(train, val, test)
    logger.info("Finished preprocessing input data for Bert")
    Bert.build_model()
    logger.info("Finished building Bert")
    logger.info("Starting training")
    history = Bert.fit_model()
    # save history for local plots
    pd.DataFrame.from_dict(history.history).to_csv(os.path.join(path, 'history.csv'), index=False)
    logger.info("Start testing")
    test_accuracy = Bert.evaluate_model()
    Bert.save_model()
    Bert.save_predicted()
    
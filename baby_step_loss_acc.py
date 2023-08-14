import tensorflow as tf
import numpy as np

from difficulty_measurer import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import math
from base_model import *
import os
import random
from sklearn import metrics



'''
1006 : epoch 수렴하고 나서 bucket 추가하는 .py
'''


def train_baby_step(x_train_val, y_train_val, x_test, y_test, args_dict, weighed_learning=False, weighed_idx=0):

    seed = args_dict['seed']
    curriculum_bool = args_dict['curriculum_bool']
    model_type = args_dict['model_type']
    work_dir = args_dict['work_dir']
    epochs = args_dict['epochs']
    batch_size = args_dict['batch_size']
    num_curriculum = args_dict['num_curriculum']
    bucket_criterion = args_dict['bucket_criterion']
    num_units = args_dict['num_units']
    num_patience = args_dict['num_patience']
    dataset = args_dict['dataset']
    label_num = args_dict['label_num']

    print('curriculum 갯수')
    print(num_curriculum)
    print('weighed?? === '+str(weighed_learning))
    print('weighed_idx === ' + str(weighed_idx))

    # =====================================================

    tf.keras.utils.set_random_seed(seed)
    print('set seed')

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, shuffle=True)



    classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    # plt.figure()
    # for c in classes:
    #     c_x_train = x_train[y_train == c]
    #     plt.plot(c_x_train[0], label="class " + str(c), marker='o')
    # plt.legend(loc="best")
    # #plt.show()
    # #plt.close()
    # plt.clf()

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    print('x_train.shape - after split')
    print('x_train.shape')
    print(x_train.shape)

    #num_classes = len(np.unique(y_train))
    num_classes = label_num
    print('class 갯수')
    print(num_classes)

    # fordA: 3601개의 데이터 # train.shape (3601, 500, 1)

    if curriculum_bool is False:
        print('the mode is still RANDOM')
        #random = shuffle!
        idx = np.random.permutation(len(x_train))  # 이건 주석 처리 필요
        x_train = x_train[idx]  # 난이도로 나열 하는 걸로 대체
        # x_train = np.expand_dims(x_train, 1) # bidirectional이라 ndim=3으로 맞춰주기 위해 추가
        y_train = y_train[idx]  # 난이도로 나열 하는 걸로 대체

    print(type(x_train))
    # print(x_train.shape) # (3601, 500, 1)
    print('x_train.shape[1:] - after split')
    print(x_train.shape[1:]) # (500, 1)
    print(type(y_train))
    print(y_train.shape)

    # data to be plotted
    x = range(len(x_train[0]))
    y = x_train[0]

    # plotting
    plt.title("original xtrain[0]")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y, color="green")
    #plt.show()
    plt.close()
    plt.clf()


    # diversity = score_diversity(x_train)
    # complexity = score_complexity(x_train)
    # print('가장 diversity 높은 거')
    # print(np.max(diversity))
    # print('가장 diversity 낮은 거')
    # print(np.min(diversity))
    #
    # print('가장 complexity 높은 거')
    # print(np.max(complexity))
    # print('가장 complexity 낮은 거')
    # print(np.min(complexity))
    # to_draw = np.isfinite(diversity)
    # print(to_draw)

    #
    # print('diversity 차원')
    # print(diversity.shape)
    #
    # difficulty = diversity
    if weighed_learning is False:
        # default learning
        diff_df = pd.read_csv(os.path.join(r'D:\OneDrive\대학원\연구\실험\difficulty_info',(dataset+'_diff_scaled_info.csv')))
        #diff_df = pd.read_csv(os.path.join(r'D:\OneDrive\대학원\연구\실험\difficulty_info_weighted', 'MelbournePedestrian_reweighed_custom.csv'))

    else:
        diff_df = pd.read_csv(os.path.join(r'D:\OneDrive\대학원\연구\실험\difficulty_info_weighted', (dataset + '_reweighed_'+str(weighed_idx)+'.csv')))


    diff_sum = diff_df.sort_values('sum',ascending=True)['sum']
    print(diff_sum)
    diff_sum_index = diff_sum.index.to_numpy()
    print(diff_sum_index)
    plt.hist(diff_sum, bins=50)
    plt.savefig(os.path.join(work_dir,'hist_diff.png'))
    #plt.show()
    #plt.close()
    plt.clf()


    #difficulty = diversity + complexity
    #difficulty = complexity

    print('here?')
    def plot_diff_example(x,epoch):
        plot_target = x[-1]
        name = 'hardest example of the current dataset'
        plt.figure()
        #print(range(1, len(plot_target) + 1))
        #print(plot_target)
        plt.plot(range(1, len(plot_target) + 1), plot_target, marker='o', label='loss')
        # plt.xticks(range(1,len(num_data_pts_list)+1,1))
        plt.title(name)
        # plt.show()
        plt.savefig(os.path.join(work_dir, ('hardest_in_epoch_'+str(epoch)+'.png')))
        plt.clf()
        plt.close()

    if curriculum_bool is True:
        print('the mode is still CURRICULUM ==')
        #score_indices = np.argsort(difficulty.flatten())    # 정렬 인덱스 저장 ([::-1]를 붙일 경우 내림차순)

        x_train_sort = x_train[diff_sum_index]
        y_train_sort = y_train[diff_sum_index]
        # print(x_train_sort.shape) # 정렬된 대로 name을 출력
        #print(x_train)


        # data to be plotted
        x = range(len(x_train_sort[-1]))
        y = x_train_sort[-1]

        # plotting
        plt.title("Hardest example")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, y, color="green", marker='o')
        #plt.show()
        plt.savefig(os.path.join(work_dir,'hardest example.png'))
        plt.clf()
        plt.close()

        # print(x_train_sort)
        # print(y_train_sort)
        # print(x_train_sort.shape)
        # print(y_train_sort.shape)

        # sorted data를 가지고 bucket을 섞어.
        ################################
        # x_bucket_list = np.split(x_train_sort,num_curriculum)
        # y_bucket_list = np.split(y_train_sort, num_curriculum)
        #
        # c = list(zip(x_bucket_list, y_bucket_list))
        #
        # random.shuffle(c)
        #
        # x_shuffled, y_shuffled = zip(*c)
        #
        # x_shuffled = list(x_shuffled)
        # y_shuffled = list(y_shuffled)
        # # print(a)
        # # print(b)
        #
        # x_train_sort = np.concatenate(x_shuffled, axis=0)
        # y_train_sort = np.concatenate(y_shuffled, axis=0)
        #
        # # print(x_train_sort)
        # # print(y_train_sort)
        # # print(x_train_sort.shape)
        # # print(y_train_sort.shape)
        # # print('see here')
        ################################

        # 원래 x_train - y_train 매칭과 sort한 이후에도 매칭 제대로 되었는지 확인

        x_train = x_train_sort
        y_train = y_train_sort


     # =======================================


    if model_type == 'cnn':
        print()
        model = make_cnn(input_shape=x_train.shape[1:], num_classes=num_classes)
    elif model_type == 'lstm-uni':
        model = make_lstm(x_train=x_train, num_classes=num_classes, num_units=num_units, work_dir=work_dir)
    elif model_type == 'lstm-bi':
        pass

    else:
        # raise error
        print('모델 이름이 이상함________________')


    tf.keras.utils.plot_model(model, show_shapes=True)



    ### curriculum 여기서 나누기!!!??? 나누지말기!!

    when_added=[0]

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    class CurriculumSequence(tf.keras.utils.Sequence):
        # Every Sequence must implement the __getitem__ and the __len__ methods.
        # If you want to modify your dataset between epochs you may implement on_epoch_end.
        # "The method __getitem__ should return a complete batch."

        def __init__(self, x_train, y_train, batch_size, num_curriculum):
            self.x, self.y = x_train, y_train
            self.batch_size = batch_size
            self.num_curriculum = num_curriculum
            # 전체 데이터를 물리적으로 나누지말고 index로 처리
            self.new_sample_per_epoch = int(len(x_train)/num_curriculum)
            self.current_epoch = 0
            self.num_data_pts_list = []
            if self.current_epoch == 0:
                self.start_idx = 0
                self.end_idx =self.start_idx + self.new_sample_per_epoch -1
                self.current_epoch = 1
                self.x_epoch = self.x[self.start_idx:self.end_idx]
                self.y_epoch = self.y[self.start_idx:self.end_idx]
                self.num_data_pts_list.append(self.end_idx+1)

        def __len__(self):
            # Denotes the number of batches per epoch
            #print('과연 이게 계속 반복될까?')
            #print(math.ceil(len(self.x) / self.batch_size))
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            # get batch indexes from shuffled indexes
            #print('도대체 idx가 뭐니??')
            #print(idx)
            # idx가 0인 경우도 고려해서 작성

            batch_x = self.x_epoch[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y_epoch[idx * self.batch_size:(idx + 1) * self.batch_size]
            #print(len(batch_x))
            #print(len(batch_y))
            return batch_x, batch_y

        def add_bucket(self):

            print('new_sample_per_epoch: {}'.format(self.new_sample_per_epoch))
            self.temp = self.end_idx + self.new_sample_per_epoch - 1
            if len(self.x) < self.temp:
                self.end_idx = len(self.x)
            else:
                self.end_idx = self.temp  # 끝에 부분만 늘려줌.

            plot_diff_example(self.x_epoch,self.current_epoch)
            when_added.append(self.current_epoch)


        def on_epoch_end(self):
            print('epoch {} 완료======'.format(self.current_epoch))

            self.current_epoch += 1
            # epoch의 가장 끝 데이터 skewness 계산 후 출력
            self.x_epoch = self.x[self.start_idx:self.end_idx]
            self.y_epoch = self.y[self.start_idx:self.end_idx]
            print('end index {}'.format(self.end_idx))
            print(len(self.x_epoch))

            print('새로운 epoch 데이터에서의 마지막 데이터 skewness {} ====='.format(abs(skew(self.x_epoch[-1])[0].astype(float))))


    curriculum_sequence = CurriculumSequence(x_train, y_train, batch_size, num_curriculum)

    # custom call back 만들어서 그걸 sequence에서 불러서 사용할 수 있나?

    bucket_epoch = 0
    train_til_end = True

    # Custom call back to hook into on epoch end
    class CurriculumCallback(tf.keras.callbacks.Callback):


        def __init__(self, sequence,bucket_criterion):
            self.sequence = sequence
            self.epoch_accuracy = []  # accuracy at given batch
            self.epoch_loss = []  # loss at given batch
            self.bucket_epoch = 0
        # after end of each epoch change the rotation for next epoch

        def on_train_end(self, logs=None):
            # calculate total time

            #self.log.write("\n\nTotal duration: " + str(self.train_dur) + " seconds\n")
            #self.log.write("*" * 100 + "\n")
            #self.log.close()

            #self.__save_plots(logs)

            if len(self.sequence.x_epoch) != len(self.sequence.x):
                # 학습이 끝났는데, 끝났을 때 epoch 데이터셋 길이가 전체 데이터셋의 길이와 다를 때::::
                # 이때 patience 늘려줘야 함
                # alert... 하는 방법... ?
                train_til_end = False
                print(train_til_end)

            #f1 score 저장


        def on_epoch_end(self, epoch, logs=None):
            # accuracy 확인 후 이전거랑 비교해서 몇 이상이면 그때 change dataset!!!
            print(logs)
            if self.sequence.current_epoch == 1:
                pass
            else:
                if bucket_criterion == 'loss':
                    # 처음 epoch가 아닌 경우에 비교
                    print('*****************이전 loss {}'.format(self.epoch_loss[-1]))
                    print('*****************현재 loss {}'.format(logs.get('loss')))
                    print('*****************그래서 차이는 {}'.format(abs(self.epoch_loss[-1] - logs.get('loss'))))
                    if abs(self.epoch_loss[-1] - logs.get('loss')) < 0.01:
                        # 수렴한 것으로 보고 bucket 추가
                        print(self.sequence.end_idx)
                        print(len(self.sequence.x_epoch))
                        if self.sequence.end_idx < len(self.sequence.x):
                            self.sequence.add_bucket()
                            print('데이터 추가++++++++++++++++++++++++++++++++++++++++++++++++++')


                    else:
                        # 수렴하지 않은 경우
                        self.bucket_epoch += 1
                        if self.bucket_epoch == 50:
                            # 같은 bucket으로 50번 동안 수렴하지 않았을 때!
                            print(self.sequence.end_idx)
                            print(len(self.sequence.x_epoch))
                            if self.sequence.end_idx < len(self.sequence.x):
                                self.sequence.add_bucket()
                                print('데이터 추가++++++++++++++++++++++++++++++++++++++++++++++++++')
                                self.bucket_epoch == 0
                elif bucket_criterion == 'accuracy':
                    # 처음 epoch가 아닌 경우에 비교
                    print('*****************이전 acc {}'.format(self.epoch_accuracy[-1]))
                    print('*****************현재 acc {}'.format(logs.get('sparse_categorical_accuracy')))
                    print('*****************그래서 차이는 {}'.format(abs(self.epoch_accuracy[-1] - logs.get('sparse_categorical_accuracy'))))
                    if abs(self.epoch_accuracy[-1] - logs.get('sparse_categorical_accuracy')) < 0.01:
                        # 수렴한 것으로 보고 bucket 추가
                        print(self.sequence.end_idx)
                        print(len(self.sequence.x_epoch))
                        if self.sequence.end_idx < len(self.sequence.x):
                            self.sequence.add_bucket()
                            print('데이터 추가++++++++++++++++++++++++++++++++++++++++++++++++++')
                            self.bucket_epoch == 0

                    else:
                        # 수렴하지 않은 경우
                        self.bucket_epoch += 1
                        if self.bucket_epoch ==50:
                            # 같은 bucket으로 50번 동안 수렴하지 않았을 때!
                            print(self.sequence.end_idx)
                            print(len(self.sequence.x_epoch))
                            if self.sequence.end_idx < len(self.sequence.x):
                                self.sequence.add_bucket()
                                print('데이터 추가++++++++++++++++++++++++++++++++++++++++++++++++++')
                                self.bucket_epoch == 0

                else:
                    print('bucket criterion 이 이상해=========================')
            # 모든 경우에 대하여.
            self.epoch_accuracy.append(logs.get('sparse_categorical_accuracy'))
            self.epoch_loss.append(logs.get('loss'))
            self.sequence.num_data_pts_list.append(self.sequence.end_idx + 1)

    test_history = {}
    test_history['loss'] = []
    test_history['acc'] = []
    test_history['f1_micro'] = []
    test_history['f1_macro'] = []
    test_history['f1_weighted'] = []

    pred_dir_path = os.path.join(work_dir, 'all_epoch_preds')
    os.mkdir(pred_dir_path)



    class TestCallback(tf.keras.callbacks.Callback):
        def __init__(self, x_test,y_test):
            self.x_test = x_test
            self.y_test = y_test
            self.x_test_pred = x_test.transpose(2, 0, 1).reshape(-1,x_test.shape[1])
            self.epoch_idx = 0

        def on_epoch_end(self, epoch, logs={}):
            loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
            test_history['loss'].append(loss)
            test_history['acc'].append(acc)
            print('test_history')
            print(test_history)
            print('x_test_pred_shape')
            print(self.x_test_pred.shape)
            #print(self.x_test_pred)

            current_pred = np.argmax(self.model.predict(self.x_test_pred), axis=-1)
            np.savetxt(os.path.join(pred_dir_path, (str(self.epoch_idx)+'_epoch_testset_pred.out')), current_pred, delimiter=',')
            self.epoch_idx += 1
            f1_micro = metrics.f1_score(self.y_test, current_pred, average='micro')
            f1_macro = metrics.f1_score(self.y_test, current_pred, average='macro')
            f1_weighted = metrics.f1_score(self.y_test, current_pred, average='weighted')

            test_history['f1_micro'].append(f1_micro)
            test_history['f1_macro'].append(f1_macro)
            test_history['f1_weighted'].append(f1_weighted)


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(work_dir,"best_model.h5"), save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=num_patience, verbose=1),
    ]

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(work_dir,'log.csv'), append=True, separator=',')
    test_callback = TestCallback(x_test, y_test)

    # model.compile(
    #     optimizer="adam",
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )

    print(x_train)

    if curriculum_bool is True:
        print('the mode is still CURRICULUM')

        history = model.fit(
            x=curriculum_sequence,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callbacks,CurriculumCallback(curriculum_sequence,bucket_criterion),csv_logger,TestCallback(x_test,y_test)],
            validation_data=(x_val,y_val),
            verbose=1,
            shuffle=False,
        )
    else:
        print('the mode is still RANDOM')
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callbacks,csv_logger,test_callback],
            validation_data=(x_val, y_val),
            verbose=1,
            shuffle=True,
        )

    print('num_data_pts_list')
    print(curriculum_sequence.num_data_pts_list)
    num_data_pts_list = curriculum_sequence.num_data_pts_list[1:]
    print(num_data_pts_list)
    print(len(num_data_pts_list))


    # best_model = tf.keras.models.load_model(os.path.join(work_dir,"best_model.h5"))
    tf.keras.models.save_model(model, os.path.join(work_dir,"last_model.h5"), overwrite=True, include_optimizer=True)



    #https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    #https://keras.io/examples/timeseries/timeseries_transformer_classification/



    return model, history, num_data_pts_list, test_history, train_til_end, when_added
import random
import baby_step_loss_acc
import read_data
import save_plots
import os
from datetime import date
import time
import pandas as pd
import shutil
import json
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def seed_weighed_experiment(dir_name, curriculum_bool, bucket_criterion, num_units, num_curr, num_patience, dataset, seed_continue):


    model_type = 'lstm-uni'  # cnn, lstm-uni, lstm-bi
    dataset_name = dataset



    if curriculum_bool is False:
        bucket_criterion = 'None'


    if curriculum_bool is True:
        mode = 'curriculum'
        shuffle_bool = False
    else:
        mode = 'random'
        shuffle_bool = True


    epochs = 500
    batch_size = 32
    num_curriculum = num_curr


    today = date.today()
    date_today = today.strftime("%y%m%d")
    print("today =", date_today)



    version = date_today + '_' + model_type + '_' + mode
    print('version is ::::{}'.format(version))



    # 인자 값을 map으로 제공



    x_train_val, y_train_val, x_test, y_test, label_num = read_data.read_data(dataset_name)

    diff_len = False


    # https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes

    ## LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP 시작

    higher_wd = 'D:\OneDrive\대학원\연구\실험'


    for each_weight in np.arange(0,9):

        dir_name_temp = dir_name+'_'+str(each_weight)
        experiment_dir = os.path.join(higher_wd,dir_name_temp)


        if os.path.exists(experiment_dir):
            #os.rmdir(experiment_dir) # only the directory
            if seed_continue is True:
                # 이전에 학습 데이터가 있고, 이어서 하기를 원한다면,
                print('dataset {} 학습 이미 한 걸로 나옴'.format(dataset))
                print(experiment_dir)
                pass
            else:
                # 계속 학습하지 않는다면
                print('dataset {} 학습하지 않은 걸로 나옴'.format(dataset))
                shutil.rmtree(experiment_dir)
                os.mkdir(experiment_dir)

        else:
            # 폴더 자체가 아예 없으면:
            os.mkdir(experiment_dir)


        args_dict={}
        args_dict['curriculum_bool'] = curriculum_bool
        args_dict['model_type'] = model_type
        args_dict['epochs'] = epochs
        args_dict['batch_size'] = batch_size
        args_dict['num_curriculum'] = num_curriculum
        args_dict['bucket_criterion'] = bucket_criterion
        args_dict['num_units'] = num_units
        args_dict['num_patience'] = num_patience
        args_dict['dataset'] = dataset
        args_dict['label_num'] = label_num


        random.seed(3015)
        seeds = random.sample(range(10, 4000), 10) # 10개 > 30개는 해야 검정 가능해짐.
        #seeds = [2975, 3625, 2908, 2468, 1229, 179, 251, 3551, 1881, 2974, 3987, 2656, 1600, 2015, 2116, 2028, 2022, 2376, 1391, 2145, 1168, 1077]
        print(seeds)

        all_seeds_result = pd.DataFrame(index=['model_type','num_curriculum','time_taken_min','final_epoch','train_acc','valid_acc','test_acc','test_f1_weighted'],columns=seeds)
        train_til_end_list = []
        acc_list = []
        for i in seeds:
            work_dir_name = version+'_'+str(i)
            work_dir = os.path.join(experiment_dir,work_dir_name)

            if os.path.exists(work_dir):
                # os.rmdir(work_dir) # only the directory
                if seed_continue is True:
                    # 이전에 학습 데이터가 있고, 이어서 하기를 원한다면,
                    print('seed {} 학습 이미 한 걸로 나옴'.format(i))
                    print(work_dir)
                    continue
                else:
                    # 계속 학습하지 않는다면
                    print('seed {} 학습하지 않은 걸로 나옴'.format(i))
                    shutil.rmtree(work_dir)
            else:
                # 폴더 자체가 아예 없으면:
                os.mkdir(work_dir)

            print('seed start ========={}'.format(i))
            args_dict['work_dir'] = work_dir
            args_dict['seed'] = i
            args_dict['work_dir'] = work_dir
            # args_dict['weighed_learning'] = True
            # args_dict['weighed_idx'] = each_weight



            print(args_dict)

            start = time.time()
            model, history, num_data_pts_list, test_history, train_til_end, when_added = baby_step_loss_acc.train_baby_step(x_train_val, y_train_val, x_test, y_test, args_dict, weighed_learning=True,weighed_idx=each_weight) # 이제는 이게 best model 이 아님
            train_til_end_list.append(train_til_end)
            end = time.time()
            time_taken = round((end - start)/60,3)
            print("The time of execution of above program is :", round((end - start),3), "s / ", time_taken, "m")

            # seed 결과 저장용
            #test_pred = model.predict(x_test)
            test_pred = np.argmax(model.predict(x_test),axis=-1)
            np.savetxt(os.path.join(work_dir, 'testset_pred.out'), test_pred, delimiter=',')
            np.savetxt(os.path.join(work_dir, 'test_labels.out'), y_test, delimiter=',')
            f1_all = metrics.f1_score(y_test,test_pred,average=None) # sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'
            f1_micro = metrics.f1_score(y_test, test_pred, average='micro')
            f1_macro = metrics.f1_score(y_test, test_pred, average='macro')
            f1_weighted = metrics.f1_score(y_test, test_pred, average='weighted')

            cf_matrix = metrics.confusion_matrix(y_test, test_pred) #sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None

            group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(label_num, label_num)
            print(labels)
            sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='rocket_r')
            plt.savefig(os.path.join(work_dir, 'confusion_matrix.png'))

            with open(os.path.join(work_dir, 'testset_f1.txt'), 'w') as file:
                file.write('f1_all : {}\n'.format(f1_all))
                file.write('f1_micro : {}\n'.format(f1_micro))
                file.write('f1_macro : {}\n'.format(f1_macro))
                file.write('f1_weighted : {}\n'.format(f1_weighted))
                file.write('cf_matrix : \n{}\n'.format(cf_matrix))

            # 전체 결과 저장 용
            test_loss, test_acc = model.evaluate(x_test, y_test)
            acc_list.append(test_acc)

            print("Test accuracy", test_acc)
            print("Test loss", test_loss)

            save_plots.plot_history(history, "sparse_categorical_accuracy", work_dir) # def plot_history(history, metric, work_dir):
            save_plots.plot_history(history, "loss", work_dir) # def plot_history(history, metric, work_dir):
            save_plots.plot_d_pts(work_dir, num_data_pts_list) # def plot_d_pts(work_dir, num_data_pts_list):
            save_plots.plot_test(work_dir,test_history)

            all_seeds_result.at['model_type',i] = model_type
            all_seeds_result.at['num_curriculum',i] = num_curriculum
            all_seeds_result.at['time_taken_min',i] = time_taken
            all_seeds_result.at['final_epoch',i] = len(history.history["sparse_categorical_accuracy"])
            all_seeds_result.at['train_acc',i] = history.history["sparse_categorical_accuracy"][-1]
            all_seeds_result.at['valid_acc',i] = history.history["val_sparse_categorical_accuracy"][-1]
            all_seeds_result.at['test_acc',i] = test_history['acc'][-1]
            all_seeds_result.at['train_loss',i] = history.history["loss"][-1]
            all_seeds_result.at['valid_loss',i] = history.history["val_loss"][-1]
            all_seeds_result.at['test_loss',i] = test_history['loss'][-1]
            all_seeds_result.at['test_f1_weighted', i] = test_history['f1_weighted'][-1]

            # 시드별로 결과도 저장(혹시 중간에 멈춰질 경우를 대비해서)

            with open(os.path.join(work_dir, 'seed_result.txt'), 'w') as file:
                file.write('seed: {}\r\n'.format(i))
                file.write('model_type: {}\r\n'.format(model_type))
                file.write('num_curriculum: {}\r\n'.format(num_curriculum))
                file.write('time_taken_min: {}\r\n'.format(time_taken))
                file.write('final_epoch: {}\r\n'.format(len(history.history["sparse_categorical_accuracy"])))
                file.write('train_acc: {}\r\n'.format(history.history["sparse_categorical_accuracy"][-1]))
                file.write('valid_acc: {}\r\n'.format(history.history["val_sparse_categorical_accuracy"][-1]))
                file.write('test_acc: {}\r\n'.format(test_history['acc'][-1]))
                file.write('f1_micro: {}\r\n'.format(test_history['f1_micro'][-1]))
                file.write('f1_macro: {}\r\n'.format(test_history['f1_macro'][-1]))
                file.write('f1_weighted: {}\r\n'.format(test_history['f1_weighted'][-1]))
                file.write('train_loss: {}\r\n'.format(history.history["loss"][-1]))
                file.write('valid_loss: {}\r\n'.format(history.history["val_loss"][-1]))
                file.write('test_loss: {}\r\n'.format(test_history['loss'][-1]))
                file.write('when_added: {}\r\n'.format(when_added))

            with open(os.path.join(work_dir, 'testset_history.txt'), 'w') as file:
                file.write('loss')
                file.write(str(test_history['loss'])+'\r\n')
                file.write('acc')
                file.write(str(test_history['acc'])+'\r\n')
                file.write('f1_micro')
                file.write(str(test_history['f1_micro'])+'\r\n')
                file.write('f1_macro')
                file.write(str(test_history['f1_macro'])+'\r\n')
                file.write('f1_weighted')
                file.write(str(test_history['f1_weighted']))

        # 결과 저장
        all_seeds_result.to_csv(os.path.join(experiment_dir,'all_seeds_result.csv'))

        # 파라미터도 저장
        with open(os.path.join(experiment_dir,'args.txt'), 'w', encoding='utf8') as file:
            file.write(json.dumps(args_dict,ensure_ascii=False))

        avg_acc = sum(acc_list)/len(acc_list)


import numpy as np
import os

def read_data(dataset_name):


    data_dir = r'D:\OneDrive\대학원\연구\실험\_datasets'

    x_train_val = np.genfromtxt(os.path.join(data_dir, dataset_name, (dataset_name+'_TRAIN.txt')), dtype=float)[:, 1:]
    y_train_val = np.genfromtxt(os.path.join(data_dir, dataset_name, (dataset_name + '_TRAIN.txt')),  dtype=float)[:, 0]
    print(np.unique(y_train_val))


    print(x_train_val.shape)
    print(x_train_val[0])

    print(y_train_val.shape)
    print(y_train_val[1])

    x_test = np.genfromtxt(os.path.join(data_dir, dataset_name, (dataset_name + '_TEST.txt')), dtype=float)[:, 1:]
    y_test = np.genfromtxt(os.path.join(data_dir, dataset_name, (dataset_name + '_TEST.txt')), dtype=float)[:, 0]
    print(np.unique(y_test))

    print(x_test.shape)
    print(x_test[0])

    print(y_test.shape)
    print(y_test[1])

    if y_train_val.dtype is np.dtype(np.float):
        print('y가 정수가 아님==')
        print(y_train_val.dtype)
        y_train_val = y_train_val.astype(int)
        y_test = y_test.astype(int)
        print(np.unique(y_test))


    # 0부터 label 값이 시작 안되는 경우
    min_y = np.min(y_train_val)
    print('label 최소값{}'.format(min_y))
    if min_y == -1:
        y_train_val = y_train_val + 1
        y_test = y_test + 1
    elif min_y != 0:
        y_train_val = y_train_val - 1
        y_test = y_test - 1
        print(y_train_val.shape)
        print(np.unique(y_train_val))
        print(y_test.shape)
        print(np.unique(y_test))
    else:
        print(min_y)

    min_y = np.min(y_train_val)

    if min_y != 0:
        raise Exception("y 라벨값 최솟값이 -1도 아니고 0도 아님,,,,,,,,,")

    label_num = len(np.unique(y_train_val))
    print(label_num)
    print(list(np.unique(y_train_val)))
    print(list(range(label_num)))

    if dataset_name == 'FordA':
        y_train_val[y_train_val==2] = 1
        y_test[y_test==2] = 1
    elif dataset_name == 'FordB':
        y_train_val[y_train_val == 2] = 1
        y_test[y_test == 2] = 1
    elif dataset_name == 'Wafer':
        y_train_val[y_train_val == 2] = 1
        y_test[y_test == 2] = 1


    if list(np.unique(y_train_val))!=list(range(label_num)):

        print('클래스 간 차이가 1이 아님.... ')


        raise Exception("일단 수동으로 그냥 바꿔.... ")

    print('라벨 갯수')
    print(label_num)
    print('y_train_val unique list')
    print(list(np.unique(y_train_val)))
    print('원래는 이렇게 되어야 하는 것:')
    print(list(range(label_num)))

    print('y_test unique list')
    print(list(np.unique(y_test)))
    print('원래는 이렇게 되어야 하는 것:')
    print(list(range(label_num)))

    return x_train_val, y_train_val, x_test, y_test, label_num

import matplotlib.pyplot as plt
import os


def plot_history(history, metric, work_dir):

    plt.figure()
    plt.plot(history.history[metric], marker='o')
    plt.plot(history.history["val_" + metric], marker='o')
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    # plt.show()
    plt_name = metric + '.png'
    plt.savefig(os.path.join(work_dir, plt_name))
    plt.clf()
    plt.close()


def plot_d_pts(work_dir, num_data_pts_list):
    # epoch - number of data points
    name = 'number of data points'
    plt.figure()
    plt.plot(range(1, len(num_data_pts_list) + 1), num_data_pts_list, marker='o')
    # plt.xticks(range(1,len(num_data_pts_list)+1,1))
    plt.title(name)
    plt.ylabel(name, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    # plt.show()
    plt.savefig(os.path.join(work_dir, 'epoch_data_points.png'))
    plt.clf()
    plt.close()

def plot_test(work_dir, test_history):
    # epoch - test_history
    name = 'testset history over epochs'
    plt.figure()
    plt.plot(range(1, len(test_history['loss']) + 1), test_history['loss'], marker='o',label='loss')
    plt.plot(range(1, len(test_history['loss']) + 1), test_history['acc'], marker='o',label='acc')
    # plt.xticks(range(1,len(num_data_pts_list)+1,1))
    plt.title(name)
    plt.ylabel(name, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(work_dir, 'testset.png'))
    plt.clf()
    plt.close()
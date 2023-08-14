from seed_experiment import *
from seed_experiment_weighed import *
from discord import SyncWebhook


webhook = SyncWebhook.from_url("https://discordapp.com/api/webhooks/1043418986412511273/e2LWA7JGS_hVUNDRLpfD5cQKQiv3ksxc-KxgGgXi-x0dlEozHIIrp8lCOPhGVOucGDvN")


try:

    train_til_end_dict = {}

    test_datasets = ['FacesUCR'] #. # 'FordB' 중간에 멈춤(12/2) 'NonInvasiveFetalECGThorax1'(12/5) 'Crop은 예전에?' # 'StarLightCurves',
    #test_datasets = ['NonInvasiveFetalECGThorax1']  # . # 'FordB' 중간에 멈춤(12/2) 'NonInvasiveFetalECGThorax1'(12/5) 'Crop은 예전에?'

    message = 'HOME_BOT::\n'
    message = message + 'Training start =====\n'
    message = message + str(test_datasets) + '=====\n'

    webhook.send(message)


    for each_dataset in test_datasets:
        print(each_dataset)
        dir_name = each_dataset+'_curr_loss_all_dropout_8'
        curr_bool = True  # if False, it's random
        bucket_criterion = 'loss'  # loss, accuracy
        num_units = 128
        num_curr = 10
        num_pat = 200
        seed_continue = False

        weighed_learning = False # &&&&&&&&&&&&& <<<< TRUE 인 상태 현재!!

        if each_dataset == 'TwoPatterns':
            seed_continue = True

        #
        if each_dataset == 'PigArtPressure':
            num_pat = 400

        if weighed_learning is True:
            seed_weighed_experiment(dir_name,curr_bool,bucket_criterion, num_units, num_curr, num_pat, each_dataset, seed_continue)
            train_til_end_list = 0
            avg_acc = 0

        else:
            train_til_end_list, avg_acc = seed_experiment(dir_name,curr_bool,bucket_criterion, num_units, num_curr, num_pat, each_dataset, seed_continue)

        train_til_end_dict[dir_name] = train_til_end_list

        message = 'HOME_BOT::\n'
        message = message + 'Training complete =====\n'
        message = message + dir_name + '=====\n'
        message = message + 'avg_test_acc :' + str(avg_acc) + '=====\n'
        message = message + 'train_til_end_list :' + str(train_til_end_list) + '=====\n'

        webhook.send(message)



        #
        dir_name = each_dataset + '_random_dropout_8'
        curr_bool = False  # if False, it's random
        seed_continue = False
        train_til_end_list, avg_acc = seed_experiment(dir_name,curr_bool,bucket_criterion, num_units, num_curr, num_pat, each_dataset, seed_continue)
        train_til_end_dict[dir_name] = train_til_end_list

        message = 'HOME_BOT::\n'
        message = message + 'Training complete =====\n'
        message = message + dir_name + '=====\n'
        message = message + 'avg_test_acc :' + str(avg_acc) + '=====\n'
        message = message + 'train_til_end_list :' + str(train_til_end_list) + '=====\n'

        webhook.send(message)

        #


    print(train_til_end_dict)
    message = 'HOME_BOT::\n'
    message = message + 'ALL Training complete =========================\n'
    message = message + str(train_til_end_dict) + '=====\n'
    webhook.send(message)


except Exception as e:

    print(getattr(e, 'message', repr(e)))
    message = 'HOME_BOT::\n'
    message = message + 'ERROR!!!!!!' + dir_name +'=====\n'
    message = message + str(train_til_end_dict) + '=====\n'

    message = message + 'error::: {}=====\n'.format(getattr(e, 'message', repr(e)))
    webhook.send(message)
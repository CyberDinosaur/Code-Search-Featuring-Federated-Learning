# # 第三组---在不同的联邦学习框架下运行，对比第一组中FedAvg的分布式训练结果
# # FedAvg
# python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal
# python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal
# # FedProx
# python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal training.method_name='FedProx'
# python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal training.method_name='FedProx' 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal training.method_name='FedProx'
# # FedNova
# python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal training.method_name='FedNova'
# python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal training.method_name='FedNova'
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal training.method_name='FedNova'
# # SCAFFOLD
# python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal training.method_name='SCAFFOLD'
# python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal training.method_name='SCAFFOLD' 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal training.method_name='SCAFFOLD'

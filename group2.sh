# 第二组-进行投毒，对比第一组中，没有投毒的分布式训练结果
# FedAvg
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal-poison
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal-poison 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal-poison
# FedProx
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal-poison training.method_name='FedProx'
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal-poison training.method_name='FedProx' 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal-poison training.method_name='FedProx'
# FedNova
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal-poison training.method_name='FedNova'
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal-poison training.method_name='FedNova'
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal-poison training.method_name='FedNova'
# SCAFFOLD
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal-poison training.method_name='SCAFFOLD'
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal-poison training.method_name='SCAFFOLD' 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal-poison training.method_name='SCAFFOLD'

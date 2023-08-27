# 第一组

# 一个client进行集中式训练
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=single
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=single
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=single


# # 50*0.5个clients进行50轮训练（FedAvg）
# FedAvg
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal
# FedProx
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal training.method_name='FedProx'
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal training.method_name='FedProx' 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal training.method_name='FedProx'
# FedNova
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal training.method_name='FedNova'
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal training.method_name='FedNova'
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal training.method_name='FedNova'
# SCAFFOLD
python server.py dataset=AdvTest_Finetune_train dataset_name='AdvTest_Finetune_train' training=normal training.method_name='SCAFFOLD'
python server.py dataset=cosqa_Finetune_train dataset_name='cosqa_Finetune_train' training=normal training.method_name='SCAFFOLD' 
# python server.py dataset=CSN_Finetune_train dataset_name='CSN_Finetune_train' training=normal training.method_name='SCAFFOLD' 

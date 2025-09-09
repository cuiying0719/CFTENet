# CFTENet

This is the code of CFTENet.


### Installation
```
conda create -n tirgn python=3.10
torch==2.1.0
conda activate cftenet
```



## How to run

#### Process data

For all the datasets, the following command can be used to get the history of their entities and relations.
```
python ent2word.py --dataset ICEWS14
```

#### Detailed hyperparameters

###### ICEWS14

~~~
CUDA_VISIBLE_DEVICES=2 nohup python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --weight-decay 5e-7>ICEWS14.txt & 
~~~

###### ICEWS18

~~~
CUDA_VISIBLE_DEVICES=0 nohup python main.py -d ICEWS18 --history-rate 0.3 --train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint  --weight-decay 5e-7>ICEWS18.txt &
~~~

###### WIKI

~~~
CUDA_VISIBLE_DEVICES=1 nohup python main.py -d WIKI --history-rate 0.3 --train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --weight-decay 5e-7>WIKI.txt &
~~~

###### GDELT

~~~
CUDA_VISIBLE_DEVICES=3 nohup python main.py -d GDELT --history-rate 0.3 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --weight-decay 5e-7>GDELT.txt &
~~~




# Unrolled-RNN in Pytorch
This repo has sample codes for manumally unrolled RNN with multi-GPU environment. 

`unrolled_DP.py`: unrolled RNN with `DataParallel`.

`unrolled_DDP.py`: unrolled RNN with `DistributedDataParallel`. 

`rolled_DDP.py`: rolled RNN with `DistributedDataParallel`.

## Usage
Before running the code, set GPUs to use. 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
### Unrolled RNN
To run the code with `DataParallel`, 
```python
python unrolled_DP.py
```

To run the code with `DistributedDataParallel`, 
```python
python unrolled_DDP.py
```

### Rolled RNN
```python
python rolled_DDP.py
```

## Results
For different batch size and same model setting, 

|Batch Size|1GPU - Time, Mem|2GPU - Time, (Mem1/Mem2)
|----------|----------------|------------------------|
|512       |72.72s, 7146MB  |53.57s, 6844MB/6844MB   |
|1024      |134.94s, 9798MB |86.16s, 8216MB/8216MB   |

2GPU result is from `unrolled_DDP.py` with 2 GPU specified. 
1GPU result is from `unrolled_DP.py` with 1 GPU specified.
The results show that the DDP with 2 GPU is faster.   

For comparison, same model with rolled RNN using 2 GPU DDP, 

|Batch Size|2GPU - Time, (Mem1/Mem2)
|----------|------------------------|
|512       |45.20s, 6044MB/6044MB   |
|1024      |75.88s, 6572MB/6572MB   |

Surely, in terms of computation time and memory, rolled RNN is better than unrolled RNN. 


## Environment
This codes are tested under `Ubuntu 18.04`, `Python 3.5`, `Pytorch 1.0.1.post2`. In case of `DistributedDataParallel`, Pytorch version below 1.0 will not work. 

## Reference
["Official Example"](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

["Pytorch docs"](https://pytorch.org/docs/master/distributed.html#module-torch.distributed)

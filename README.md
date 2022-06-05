# How to start

### 1. Preprocess dataset
```
python preprocess.py -d music
python preprocess.py -d book
python preprocess.py -d movie
```

### 2. Train
학습하려는 모델과 데이터에 해당하는 configuration 설정

exps/logs 에서  tensorboard 확인가능

학습이 끝나면 exps/ckpt 에 best weight 저장됨
```
python train.py --config configs/파일명
```


### 3. Evaluate

학습 시 사용했던 configuration을 옵션으로 실행
```
python evaluate.py --config configs/파일명
```

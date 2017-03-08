# CS294-112 HW 1: Imitation Learning

## Section 2. Behaviorial Cloning

Command example:
```
python run_hw1.py experts/Ant-v1.pkl Ant-v1 --num_rollouts=10 --clone_expert
```

Results from some sample runs are listed below.
For each environment, 5 trials were conducted with different number of rollouts.
The common settings across environments are 
1. network architecture (identical to that of experts)
2. number of training epoches (=10)
3. optimization algorithm (=Adam) and initial learning rate (=0.02)

#### Ant-v1

| #rollout | re.mean@expert | re.std@expert | re.mean@immitation | re.std@immitation |
|----------|----------------|---------------|--------------------|-------------------|
| 5 | **4827.80** | 110.47 | 4530.35 | 469.72 |
| 10 | 4475.10 | 1082.37 | **4799.69** | 116.21 |
| 20 | 4793.46 | 99.82 | **4813.29** | 93.09 |
| 40 | **4806.66** | 162.85 | 4740.11 | 376.65 |
| 80 | 4746.36 | 464.75 | **4751.43** | 285.28 |

#### HalfCheetah-v1

| #rollout | re.mean@expert | re.std@expert | re.mean@immitation | re.std@immitation |
|----------|----------------|---------------|--------------------|-------------------|
| 5 | **4159.96** | 47.64 | 4107.53 | 70.79 |
| 10 | 4161.36 | 75.76 | **4166.69** | 51.79 |
| 20 | 4137.29 | 81.25 | **4137.86** | 81.70 |
| 40 | 4152.15 | 93.11 | **4170.78** | 73.46 |
| 80 | **4128.01** | 79.69 | 4123.82 | 83.78 |

#### Hopper-v1

| #rollout | re.mean@expert | re.std@expert | re.mean@immitation | re.std@immitation |
|----------|----------------|---------------|--------------------|-------------------|
| 5 | 3777.54 | 4.32 | **3779.21** | 4.92 |
| 10 | **3778.15** | 2.28 | 3777.11 | 1.82 |
| 20 | **3779.33** | 4.95 | 3777.44 | 3.42 |
| 40 | 3778.03 | 3.52 | **3778.72** | 3.50 |
| 80 | **3778.34** | 3.79 | 3775.94 | 3.68 |

#### Humanoid-v1

| #rollout | re.mean@expert | re.std@expert | re.mean@immitation | re.std@immitation |
|----------|----------------|---------------|--------------------|-------------------|
| 5 | **10385.94** | 61.71 | 9968.11 | 813.81 |
| 10 | **10384.86** | 54.26 | 8388.77 | 3608.50 |
| 20 | 10384.01 | 66.97 | **10400.12** | 65.56 |
| 40 | **10407.64** | 52.72 | 10388.69 | 56.18 |
| 80 | **10411.78** | 52.03 | 10404.41 | 52.80 |

#### Reacher-v1

| #rollout | re.mean@expert | re.std@expert | re.mean@immitation | re.std@immitation |
|----------|----------------|---------------|--------------------|-------------------|
| 5 | **-3.82** | 1.03 | -14.09 | 3.02 |
| 10 | **-4.37** | 2.46 | -8.39 | 4.36 |
| 20 | **-3.50** | 1.34 | -5.98 | 3.88 |
| 40 | **-4.16** | 1.48 | -4.22 | 1.69 |
| 80 | **-3.73** | 1.69 | -4.13 | 1.81 |

#### Walker2d-v1

| #rollout | re.mean@expert | re.std@expert | re.mean@immitation | re.std@immitation |
|----------|----------------|---------------|--------------------|-------------------|
| 5 | 5526.24 | 22.99 | **5527.81** | 54.41 |
| 10 | 5499.99 | 56.22 | **5516.74** | 44.76 |
| 20 | **5531.90** | 57.79 | 5494.12 | 88.44 |
| 40 | **5522.91** | 66.96 | 5478.61 | 76.97 |
| 80 | **5529.83** | 48.84 | 5527.21 | 76.25 |

## Section 3. DAgger

Command example:
```
python run_hw1.py experts/Ant-v1.pkl Ant-v1 --num_rollouts=10 --dagger --dagger_iter=10
```


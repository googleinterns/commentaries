#     Copyright 2020 Google LLC

#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at

#         https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import os
from torch.multiprocessing import Pool


def run_func(cmd):
    os.system(cmd)
    

cmdlist = [
    'python3 student_train_weighting.py --dataset cifar --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 0 --arch cbr2 --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset cifar --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 1 --arch cbr2 --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset cifar --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --arch cbr2 --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset cifar --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 3 --arch cbr2 --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset cifar --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 0 --arch cbr2 --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset cifar --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 1 --arch cbr2 --inner-steps 1500 --studentarch cbr',


    'python3 student_train_weighting.py --dataset mnist --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --arch cbr2 --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset mnist --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 3 --arch cbr2 --inner-steps 1500 --studentarch resnet18',

#     'python3 student_train_weighting.py --dataset mnist --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 3 --arch cbr --inner-steps 1500 --studentarch resnet34',

    'python3 student_train_weighting.py --dataset mnist --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 0 --arch cbr2 --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset mnist --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 1 --arch cbr2 --inner-steps 1500 --studentarch resnet18',

#     'python3 student_train_weighting.py --dataset mnist --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 3 --arch cbr --inner-steps 1500 --studentarch resnet34',


    'python3 student_train_weighting.py --dataset mnist --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --arch cbr2 --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset mnist --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 3 --arch cbr2 --inner-steps 1500 --studentarch resnet18',

#     'python3 student_train_weighting.py --dataset mnist --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 3 --arch cbr --inner-steps 1500 --studentarch resnet34',

]

cmdlist_baseline = [

    'python3 student_train_weighting.py --dataset cifar --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset cifar --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet34',

    'python3 student_train_weighting.py --dataset cifar --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset cifar --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet34',

    'python3 student_train_weighting.py --dataset cifar --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset cifar --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet34',


    'python3 student_train_weighting.py --dataset mnist --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset mnist --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset mnist --seed 0 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet34',

    'python3 student_train_weighting.py --dataset mnist --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset mnist --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset mnist --seed 1 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet34',


    'python3 student_train_weighting.py --dataset mnist --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch cbr',

    'python3 student_train_weighting.py --dataset mnist --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet18',

    'python3 student_train_weighting.py --dataset mnist --seed 2 --inner-lr 1e-4 --outer-lr 1e-3 --gpu 2 --baseline --arch cbr --inner-steps 1500 --studentarch resnet34',

]

import sys
if sys.argv[1] == 'baseline':
    args = cmdlist_baseline
else:
    args = cmdlist

with Pool(8) as p:
    print(p.map(run_func, args))
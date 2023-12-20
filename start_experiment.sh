# python main.py  --method DC  --dataset MNIST  --model ConvNet  --eval_mode my_eval  --ipc 1 --num_exp 1  --num_eval 2  --Iteration 3000  --use_ModelPool  --use_Dropout  --use_KD
# python main.py  --method DC  --dataset MNIST  --model MLP  --eval_mode my_eval  --ipc 1 --num_exp 5
# python main.py  --method DC  --dataset MNIST  --model ConvNet  --eval_mode my_eval  --ipc 1 --num_exp 5
# python main.py  --method DC  --dataset MNIST  --model LeNet  --eval_mode my_eval  --ipc 1 --num_exp 5
# python main.py  --method DC  --dataset MNIST  --model AlexNet  --eval_mode my_eval  --ipc 1 --num_exp 5
# python main.py  --method DC  --dataset MNIST  --model VGG11  --eval_mode my_eval  --ipc 1 --num_exp 5
# python main.py  --method DC  --dataset MNIST  --model ResNet18  --eval_mode my_eval  --ipc 1 --num_exp 5

python main.py  --method DC  --dataset CIFAR10  --model ConvNet  --eval_mode my_eval  --ipc 1 --num_exp 1  --num_eval 2  --Iteration 1000  --use_ModelPool
python main.py  --method DC  --dataset CIFAR10  --model ConvNet  --eval_mode my_eval  --ipc 1 --num_exp 1  --num_eval 2  --Iteration 1000  --use_Dropout
python main.py  --method DC  --dataset CIFAR10  --model ConvNet  --eval_mode my_eval  --ipc 1 --num_exp 1  --num_eval 2  --Iteration 1000  --use_KD

# python main.py  --method DC  --dataset CIFAR10  --model ConvNet  --eval_mode my_eval  --ipc 1 --num_exp 1  --num_eval 2  --Iteration 1000  --use_ModelPool  --use_Dropout  --use_KD
# python main.py  --method DC  --dataset CIFAR10  --model ConvNet  --eval_mode my_eval  --ipc 10 --num_exp 1  --num_eval 2  --Iteration 3000  --use_ModelPool  --use_Dropout  --use_KD
# python main.py  --method DC  --dataset CIFAR10  --model ConvNet  --eval_mode my_eval  --ipc 10 --num_exp 1  --num_eval 2  --Iteration 3000
python3 filter.py

python3 preprocessing.py &
python3 preprocessing.py --train

# Consider rerunning objective augmentation again if you observe a significant number of error messages
python3 clickhere_augmentation.py --task "test"
python3 clickhere_augmentation.py --task "train" 

python3 objective_augmentation.py --task "test"
python3 objective_augmentation.py --task "train" 

python3 adding_augmentation.py --task "test" &
python3 adding_augmentation.py --task "train"

python3 naive.py --path "data/test_augmented.csv" &
python3 naive.py --path "data/train_augmented.csv"

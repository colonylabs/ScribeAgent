# Data Transformation Pipeline

## Overview

This folder details our data preprocessing pipeline. For more information on how data moves through the pipeline, check out the [Data Transformation Flowchart](https://www.figma.com/board/dP4iwGchrQFnjNVHsj1I77/Untitled?node-id=0-1&node-type=canvas&t=B1gxNngVCBb8Roqt-0)

<img width="1994" alt="data processing pipeline" src="../assets/DataFlow.png">

## Folder Structure
List and describe the subfolders included in the project:
* `GPT_augmentation_file`: This contains files used for augmenting step and objective descriptions. We save these files to avoid re-runnning GPT calls.
    * `train/test_objectives_clean.json`: Mapping from workflow_id to augmented objectives
    * `train/test_step_desc.json`: Mapping from action_id to augmented step description for click here actions

* `data`: Place to store all data
    * `screenshot`: Store screenshots (`action_id.jpeg`)
    * `circled_ss`: Store screenshots with circled targets (`action_id_circled.jpeg`)
    * `xy_position.csv`: Store circle coordinates

* `circle_all.py`: This is used to create screenshots with circled targets with OpenCV. This requires `xy_mapping.csv` from Snowflakw

* `filter.py`: Filters our null values and non-english scribes. Splits the dataset into train and test set

* `preprocessing.py`: Performs DOM preprocessing and target formatting

* `clickhere_augmentation.py`: Create action_id to augmented step description mapping using circled screenshots

* `objective_augmentation.py`: Create workflow_id to augmented objective description mapping using screenshots

* `adding_augmentations.py`: Adding objective and step description augmentations and prompt generation (model input)

* `naive.py`: Most HTML DOMs are bigger than our model context window. We perform naive truncation at tag level, do left truncate the DOMs to fit within the model's context window.

* `dataset.sh`: Bash script to sequencially run all preprocessing files needed to recreate the final train and test files.

## How to run

Make sure naive has the desired chunk length.

```
pip install -r requirements.txt
chmod +x dataset.sh
./dataset.sh
```

## Dataset
Refer to `example_workflow.csv` for example input format. We can't share the data we trained on to maintain user confidentiality

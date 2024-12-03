# Mind2Web Evaluation

To evaluate ScribeAgent on Mind2Web, clone the original [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) repository. Then, download the test set [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EUkdc16xUC1EplDiXS1bvSEBOddFLgOyJNkWJOxdltNEGA?e=8N1D9S) and unzip it with password `mind2web`.

Please follow the benchmark repo to set up the environment. Then, export `HF_TOKEN=your_huggingface_token`, and install the additional libraries:
```bash
pip install vllm
```

## Direct Generation

For direct generation evaluation, place `direct_generation_eval.py` and `utils.py` in the root directory, so the directory structure looks like:
```
Mind2Web
├── data
│   ├── test_task
│   │   └── test_task_*.json
│   ├── test_website
│   │   └── test_website_*.json
│   └── test_domain
│       └── test_domain_*.json
├─── direct_generation_eval.py
├─── utils.py
│ 
└── ...
```

To run evalution:
```bash
python direct_generation_eval.py --model_name_or_path {model_ckpt} --output_name {output_dir} --task {test_domain|test_task|test_website}
```

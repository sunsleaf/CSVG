# Solving Zero-Shot 3D Visual Grounding as Constraint Satisfaction Problems

<p>
<a href="https://arxiv.org/pdf/2411.14594">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
</p>

We propose Constraint Satisfaction Visual Grounding (CSVG) for 3D visual grounding in a point cloud referred to by nautural language descriptions.


## Updates

- [x] 11/2024, CSVG code released.
- [x] 11/2024, CSVG arxiv paper released.

## Setup

Here are the instructions to run the code.

#### Install the environment.

```
pip install transformers
pip install pyviz3d
pip install plyfile
```

If you encounter missing package errors, simply installing the package should solve the problem.

#### Prepare the data.

Our system can be used with the ground truth segmentation from [ScanNet](https://github.com/ScanNet/ScanNet) or predictions from [Mask3D](https://github.com/JonasSchult/Mask3D).

For both cases, you need to download the ScanNet dataset to ``data/scans``. You may only download the validation set to save disk space.

If you want to use instance segmentations from Mask3D, you can run the model by following the instructions from the Mask3D github repo. In the end, the prediction results will be stored in folders named like `instance_evaluation_scannet200_val_query_150_topk_750_dbscan_0.95_0/decoder_-1/`. After this folder is filled with contents, please make a soft link of it to `data/eval_output/mask3d_val`, i.e., the `decoder_-` folder should be linked as `mask3d_val`.

Download the [ScanRefer](https://github.com/daveredrum/ScanRefer) dataset and put the json files into `data/scanrefer`.

If you also want to evaluate on the [Nr3D](https://referit3d.github.io/) dataset, download the `nr3d.csv` into the `data` folder directly.

Now you should have the data required to run the code.

#### Program generation

You can use the `run_scripts` bash file to generate Python programs for a dataset:

`./run_scripts gen --dataset scanrefer --mask3d`

For a complete list of available options, please run:

`./run_scripts gen --help`

The most important option is `--dataset`, which can be `scanrefer`, `nr3d`, or any custom dataset names. For customized dataset names, a file called `{dataset_name}_queries.json` in the `data` folder will be loaded. There are already some examples there, the format of which you can follow.

If you add the `--mask3d` option, segmentations from Mask3D will be used. If none is added, the ground truth segmentation will be used.

Another important thing is the LLM server address. You can either deploy a local LLM with OpenAI-compatible APIs, or use OpenAI models directly. The configuration of a local API server is at line 782 in `program_generator.py`; here, the `api_key` argument usually does not matter, but the `base_url` should be modified according to your server configurations. For OpenAI API, you should modify your key at line 799 in `program_generator.py`.

After running the generation script, an `eval_data_*` file will be generated in the `output` folder.

#### Evaluation

You can also use the `run_scripts` file to run the generated program and get the grounding results, e.g.:

`./run_scripts eval --dataset scanrefer --seg mask3d --num-threads 10`

The `--dataset` option can be `scanrefer`, `nr3d` or a custom dataset name. The `--seg` option can be `gt` or `mask3d`.

After running the evaluation script, an `eval_results_*` file will be generated in the `output` folder.

#### Visualization

The `visualize_eval_results.py` can be used to visualize the evaluation results. It reads the `eval_results_*` files. We include some examples from our experiments.

For example, you can run the following command to visualize results on the ScanRefer validation set using Mask3D segmentations:

`python ./visualize_eval_results.py --file ./output/eval_results_mask3d_scanrefer.json`

Use the `--help` option to see a full list of available options, e.g., the `--distractor-required` option will plot the distractors (objects with the same label as the target).

The visualization script will start a [PyViz] server, and you can view the visualization in your browser at `0.0.0.0:8889` (the default port).

### Contributor

Qihao Yuan (Email: qihao.yuan@rug.nl)

### License
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.

## Citation

If you are interested in this work, please cite as below:

```text
@article{yuan2024solving,
      title={Solving Zero-Shot 3D Visual Grounding as Constraint Satisfaction Problems}, 
      author={Yuan, Qihao and Zhang, Jiaming and Li, Kailai and Stiefelhagen, Rainer},
      journal={arXiv preprint arXiv:2411.14594},
      year={2024}
}
```

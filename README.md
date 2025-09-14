# LLama Inference

This repo is an implementation of `meta-llama/Llama-3.1-8B-Instruct` model inference on the `ArtificialAnalysis/AA-LCR` long context reasoning benchmark.

## Repo Description and Design Choices

The repo consists of two main Python files: `generate.py` implements the generations on the `AA-LCR` benchmark, and `evaluate.py`, which uses a critic model to check the
correctness of the generated output vs the ground truth provided by the `AA-LCR` benchmark. Also the data handling part is done in the `data.py` script.

### generate.py design

The backend engine used for generation is `vLLM`, for a couple of reasons:

- Memory Efficiency; Thanks to its efficient implementation of Paged-Attention, vLLM manages attention memory usage significantly better than naïve implementations. This allows handling of longer contexts without exhausting GPU memory.

- Inherent Compilation; vLLM leverages CUDA Graph capture to compile execution traces of the model. This reduces kernel launch overhead, improves determinism, and results in faster and more memory-efficient generation.

- Optimized KV-Cache Management; Its KV-Cache implementation is carefully optimized for both speed and memory locality, ensuring minimal overhead when reusing cached key/value states across decoding steps.

- Tensor Parallel & Multi-GPU Support; vLLM has built-in support for tensor parallelism across multiple GPUs, scaling well for larger models while maintaining high efficiency (it was necessary for Llama3.1 8B generation on my setup) 

- Throughput-Oriented Scheduling; Its scheduler is designed to maximize GPU utilization by balancing workloads across active requests, minimizing idle compute time.

Due to the nature of the benchmark (average context length of 100K tokens), generation speed was an issue, and for some samples, the generation would exceed memory limits. 
With the above enhancements, I was able to speed up the generation part of the implementation while not going out of memory. 

The script lets you select the following parameters: `batch-size`, `num-workers` (for data loading, improves performance), max-new-tokens (maximum number of tokens to generate), along with sampling parameters such as `temperature`, `top-k`, and `top-p`.


### evaluate.py design

The evaluation code uses a critic model (I used the same `meta-llama/Llama-3.1-8B-Instruct` model due to disk space and memory constraints as some sort of proof of concept, but as the `AA-LCR` authors also suggested, it is better to use more accurate models such as Qwen3 series of models, gpt-5, or Deepseek V3) 
The reason for choosing the critic model was again the design of the AA-LCR dataset; it has non-structured answers (compared to MMLU, which has A, B, C, D as the final answer), so extracting the exact final answer was tricky.
I tried enforcing the format of the model's output with detailed prompting (will discuss this in the next section), but the model failed to follow the exact instructions. 
Hence, I decided to use the same vLLM generation logic to obtain the generated answer and the ground-truth, and then ask a critic model to verify its correctness (which proved to be somewhat successful). 

As the metric, since each generation is marked with `CORRECT` or `INCORRECT`, I have chosen the fundamental Accuracy metric, which is being printed at the end.

### data.py design

This is a standard data processing class that builds on torch's `Dataset` module and uses the `datasets` library underneath. 
Since the dataset has the option of both URL and locally extracted files, I have implemented methods for both, but the default is set to use the local files in the dataset's snapshot folder since url loading of some samples failed with 403 error. 
The class then reads the local files (or URLs) into text and passes them as context. I have also added a method that handles the prompt format for generation.
Since AA-LCR is a reasoning benchmark, I have asked the model in the system prompt to do specific reasoning steps (repeat the question, gather sub-questions, and add evidence and synthesis) before coming to the final answer.
I have also asked the model to return the final answer as a one-liner (a good study would be the effect of final answer relaxation for the final correctness).

Another class implemented is for the `evaluation.py` script. It gets the generated JSON file from `generate.py`, and creates a dataset that combines the original question, ground truth, and the generated answer.
Here, the system prompt format is taken from AA-LCR's documentation.


## Get started

The pip installations are included in the `requirements.txt` file, and can be installed via
```
pip install -r requirements.txt --no-cache-dir
```
I am specifically using `vLLM==0.7.4`, `torch==2.7.0`, and `transformers==4.51.0`, but other versions may work. 
Additionally, as I have had experience previously, vLLM may be required to build from source for specific hardware architectures.

To get access to the Llama series of models from HuggingFace, a HuggingFace token with approved access is required.
Once you obtain the token, you can log in to HuggingFace to download the model and the dataset:

```
huggingface-cli login --token <your-obtained-token>
```

It is also suggested to set `HF_HOME` variable, where the files will be downloaded.
```
export HF_TOKEN=<arbitrary_dir>
```
If not set, it will be set to `$HOME/.cache/huggingface/`

After succesful login, to download the llama3.1 model:

```
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

And to download `AA-LCR` benchmarking dataset:

```
huggingface-cli download --repo-type dataset ArtificialAnalysis/AA-LCR
```

As mentioned above, the dataset includes a zipped file of the document text files. To use the folder for generation, you first need to unzip the file in the same location as the dataset snapshot files:

```
unzip ${HF_HOME}/hub/datasets--ArtificialAnalysis--AA-LCR/snapshots/<snapshot-tag>/extracted_text/AA-LCR_extracted-text.zip -d ${HF_HOME}/hub/datasets--ArtificialAnalysis--AA-LCR/snapshots/<snapshot-tag>/extracted_text/
```
Where the <snapshot-tag> is a unique folder in the dataset, which could vary between users.

To run the generation:

```
python generate.py
```

Which saved a JSON file with the final generations. The input arguments can be checked via `python generate.py --help` (they are briefly explained above). To run the evaluation:

```
python evaluate.py --input-json <path-to-your-generated-json-file>
```
This prints the final accuracy after the script is done. 

## Results and Discussion

After 5 iterations of the eval (I have included two generation outputs in the outputs_example folder), the final accuracy was ~26%, which could mean two things:

1. Llama3.1 8B is performing poorly in the reasoning tasks.
2. The evaluation setup (the benchmark itself, the critic model, system prompt) could be designed better.
   
Going through the accuracy of the critic, I saw that in most cases, the critic is tagging the correct answer correctly. 

An example of an improvable question is the first question in the second output file:

- Question: Based on the provided documents, there appears to be a correlation between industry concentration and the frequency of consumer-related infringements and undertakings issued by the ACCC. Identify and rank the industries explicitly mentioned in the paragraphs, according to the number of infringements over the past three decades. Exclude Broadcasting Industry from the answer.
- Ground Truth: 1. Airline Industry (12)\n2. Accommodation Industry (4)
- Generated Response: 1. . Airline industry: 12 infringements\n2. Accommodation industry: 4 infringements\n3. Broadcasting industry: Not included in the answer due to exclusion\n4. Freight Shipping: 10 contraventions\n5. Passenger Rail: 10 contraventions\n6. Medical equipment industry: 0.3% of total Australian and global exports \n7. Pharmaceuticals: 0.7% of total Australian and global exports.

The final answer is listing the top two answers correctly. Still, it is unable to exclude the Broadcasting industry (asked in the question), and other answers (which do not have infringements, but include contraventions, ...). This is marked as INCORRECT.

## Next steps

- Improved evaluation metrics (better critic, Perplexity score, more fluid metrics that evaluate at formatting of generation, the reasoning step correctness, ...
- Add speculative decoding to generation (may not be beneficial due to size of the model)
- Combine the evaluate.py and generate.py in a single script (launcher.py for example).


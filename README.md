<h1 align="center">
  <img src="misc/ESADAS.png" width="400" /></a><br>
  <b>Domain-Agnostic Universal Novel Agent Design through
Autonomous, Open-Ended Evolutionary Search</b><br>
</h1>


DUNE proposes an improvement to the original ADAS framework with the integration of a evolutionary search algorithms, encouraging diversity by looking at different domain features, such as structure and scale.

DUNE is open-ended, being applicable to multiple domains. Dune generates more powerful agents compared to the original ADAS in popular benchmarks such as arc, drop, gpqa, mgsm, and mmlu. Dune reduces the costof automatic agent search by a significant amount compared to ADAS, allowing more independentresearch in this field

We take alot of inspiration from the ADAS work, and recognize that our work would not be possible without the work from the ADAS team. Their paper can be found [here](https://arxiv.org/abs/2408.08435) and their repository [here](https://github.com/ShengranHu/ADAS)

<p align="center">
<img src="misc/ESADAS.drawio.png"/></a><br>
</p>

## Setup
```bash
python3 -m venv venv 
pip install -r requirements.txt

# provide your OpenAI API key
export OPENAI_API_KEY="YOUR KEY HERE"

# provide your GoogleAPI key
export GOOGLE_AI_API_KEY="YOUR KEY HERE"
```

## Running Instructions

### Running Meta Agent Search

To run experiments for each domain, navigate to its respective folder. The code in each folder is self-contained. Launch experiments using the `search.py` script located in each domain's folder.

```bash
python {DOMAIN}/search.py
```

Replace `{DOMAIN}` with the specific domain folder name {`_arc`, `_drop`, `_mgsm`, ...} to run the experiment for.

### Customizing Meta Agent Search for New Domains

You can easily adapt the code to search for new domains. To do so, follow these steps:

1. Modify the `evaluate_forward_fn()` function and adjust any necessary formatting prompts (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/search.py#L89)) in the `search.py` file. 

2. Consider adding additional basic functions for the meta agent to utilize during the design process (similar to [this line](https://github.com/ShengranHu/ADAS/blob/main/_arc/search.py#L161)).

3. Update the domain-specific information within the prompts to match the requirements of your new domain (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/mmlu_prompt.py#L229)).

4. Run the search and evaluation on your new domain.

### Safety Consideration
> [!WARNING]  
> The code in this repository involves executing untrusted model-generated code. We strongly advise users to be aware of this safety concern. While it is highly unlikely that model-generated code will perform overtly malicious actions in our current settings and with the models we use, such code may still act destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.


## Citing
If you find this project useful, please consider citing:
```
@article{hu2024ADAS,
title={Automated Design of Agentic Systems},
author={Hu, Shengran and Lu, Cong and Clune, Jeff},
journal={arXiv preprint arXiv:2408.08435},
year={2024}
}
```

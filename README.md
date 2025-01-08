# QLoRA Fine-Tuning of LLaMA 3 for Function Calling with Chat Format

This project demonstrates fine-tuning of the LLaMA 3 (8B) model using QLoRA (Quantized Low-Rank Adaptation) for function-calling tasks in a chat-based format. The fine-tuning process leverages LoRA's parameter-efficient tuning approach combined with memory-efficient 4-bit quantization, enabling scalable training on limited resources.

## Features
- Fine-tuning LLaMA 3 (8B) for specialized function-calling tasks.
- Use of QLoRA for 4-bit quantization, improving memory efficiency.
- Chat-format interaction for seamless integration with conversational AI.
- Advanced preprocessing pipeline for dataset preparation.
- Efficient training techniques including gradient checkpointing and hyperparameter tuning.
- Evaluation metrics for model performance in function-calling tasks.

## Dataset
The project utilizes the **Hermes Function-Calling Dataset**, formatted for chat-based AI tasks. The dataset contains examples of structured function calls in conversational exchanges.

## Tools and Technologies
- **Model:** [LLaMA 3](https://github.com/facebookresearch/llama) (8B parameters)
- **Frameworks:** Hugging Face Transformers, PyTorch
- **Quantization:** QLoRA for 4-bit precision
- **Logging and Monitoring:** Weights & Biases
- **Optimization:** Gradient checkpointing, hyperparameter tuning

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/qlora-llama-fine-tune.git
    cd qlora-llama-fine-tune
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up Weights & Biases for experiment tracking:
    ```bash
    wandb login
    ```

## Training the Model
1. Prepare the dataset:
    - Place the Hermes dataset in the `data/` directory.
    - Run the preprocessing script:
      ```bash
      python preprocess.py --input data/raw --output data/processed
      ```

2. Fine-tune the model:
    ```bash
    python train.py \
        --model llama3-8b \
        --dataset data/processed \
        --lora-rank 16 \
        --quantization-bits 4 \
        --output-dir models/llama3-fine-tuned
    ```

3. Monitor training with Weights & Biases:
    ```bash
    wandb --project llama3-fine-tune
    ```

## Evaluation
Evaluate the fine-tuned model on the validation dataset:
```bash
python evaluate.py --model models/llama3-fine-tuned --dataset data/validation
```

## Results
- **Training Loss:** 3.6347
- **Validation Loss:** 3.3479

The lower validation loss compared to the training loss indicates good generalization, suggesting the model effectively learned from the dataset without overfitting.

## Future Work
- Extend functionality to support additional function types.
- Fine-tune larger models with multi-modal datasets.
- Improve inference latency for real-time deployment.

## Acknowledgments
Special thanks to the authors of [QLoRA](https://arxiv.org/abs/2305.14314) and the creators of the LLaMA model.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or collaboration, please contact:
- **Name:** Nila Masrourisaadat
- **Email:** nilamasrouri@vt.edu
- **GitHub:** [Nila-Masrourisaadat](https://github.com/Nila-Masrourisaadat)

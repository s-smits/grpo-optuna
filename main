import re
import torch
import optuna
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# --- Load and prep dataset ---

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions(split="train")
# Use the test set for evaluation
validation_dataset = get_gsm8k_questions(split="test")

# --- Reward functions with dynamic return values ---

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    correct_reward = kwargs.get("correct_reward", 2.0)
    incorrect_reward = kwargs.get("incorrect_reward", 0.0)
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [correct_reward if r == a else incorrect_reward for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    int_reward = kwargs.get("int_reward", 0.5)
    non_int_reward = kwargs.get("non_int_reward", 0.0)
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [int_reward if r.isdigit() else non_int_reward for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    match_reward = kwargs.get("strict_match_reward", 0.5)
    no_match_reward = kwargs.get("strict_no_match_reward", 0.0)
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [match_reward if match else no_match_reward for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    match_reward = kwargs.get("soft_match_reward", 0.5)
    no_match_reward = kwargs.get("soft_no_match_reward", 0.0)
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [match_reward if match else no_match_reward for match in matches]

def count_xml(text, xml_count_reward=0.125) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += xml_count_reward
    if text.count("\n</reasoning>\n") == 1:
        count += xml_count_reward
    if text.count("\n<answer>\n") == 1:
        count += xml_count_reward
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += xml_count_reward
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    xml_count_reward = kwargs.get("xml_count_reward", 0.125)
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c, xml_count_reward) for c in contents]

# --- Model and Config ---

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"

# --- Optuna Objective Function ---

def objective(trial):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1.0)

    # Dynamic reward function weights
    xmlcount_weight = trial.suggest_float("xmlcount_weight", 0.5, 1.5)
    soft_format_weight = trial.suggest_float("soft_format_weight", 0.5, 1.5)
    strict_format_weight = trial.suggest_float("strict_format_weight", 0.5, 1.5)
    int_weight = trial.suggest_float("int_weight", 0.5, 1.5)
    correctness_weight = trial.suggest_float("correctness_weight", 1.0, 3.0)

    # Dynamic reward function parameters
    correct_reward = trial.suggest_float("correct_reward", 1.0, 3.0)
    incorrect_reward = trial.suggest_float("incorrect_reward", -1.0, 0.0)
    int_reward = trial.suggest_float("int_reward", 0.1, 1.0)
    non_int_reward = trial.suggest_float("non_int_reward", -0.5, 0.0)
    strict_match_reward = trial.suggest_float("strict_match_reward", 0.1, 1.0)
    strict_no_match_reward = trial.suggest_float("strict_no_match_reward", -0.5, 0.0)
    soft_match_reward = trial.suggest_float("soft_match_reward", 0.1, 1.0)
    soft_no_match_reward = trial.suggest_float("soft_no_match_reward", -0.5, 0.0)
    xml_count_reward = trial.suggest_float("xml_count_reward", 0.05, 0.2)

    # Define GRPOConfig with optimized hyperparameters
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=learning_rate,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = weight_decay,
        warmup_ratio = warmup_ratio,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=max_grad_norm,
        report_to="wandb",
        log_on_each_node=False,
    )

    # Define LoRA config (optional)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Define GRPOTrainer with dynamic reward functions
    def weighted_xmlcount_reward_func(completions, **kwargs):
      return [x * xmlcount_weight for x in xmlcount_reward_func(completions, xml_count_reward=xml_count_reward, **kwargs)]

    def weighted_soft_format_reward_func(completions, **kwargs):
      return [x * soft_format_weight for x in soft_format_reward_func(completions, soft_match_reward=soft_match_reward, soft_no_match_reward=soft_no_match_reward, **kwargs)]

    def weighted_strict_format_reward_func(completions, **kwargs):
      return [x * strict_format_weight for x in strict_format_reward_func(completions, strict_match_reward=strict_match_reward, strict_no_match_reward=strict_no_match_reward, **kwargs)]

    def weighted_int_reward_func(completions, **kwargs):
      return [x * int_weight for x in int_reward_func(completions, int_reward=int_reward, non_int_reward=non_int_reward, **kwargs)]

    def weighted_correctness_reward_func(prompts, completions, answer, **kwargs):
      return [x * correctness_weight for x in correctness_reward_func(prompts, completions, answer, correct_reward=correct_reward, incorrect_reward=incorrect_reward, **kwargs)]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            weighted_xmlcount_reward_func,
            weighted_soft_format_reward_func,
            weighted_strict_format_reward_func,
            weighted_int_reward_func,
            weighted_correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        #peft_config=peft_config
    )

    # Train the model
    trainer.train()

    # --- Evaluation on the GSM8K test set ---
    def evaluate_model(model, tokenizer, dataset):
        model.eval()
        correct_predictions = 0
        total_predictions = 0

        for i in range(len(dataset)):
            sample = dataset[i]
            prompt = sample['prompt']
            true_answer = sample['answer']

            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=786, pad_token_id=tokenizer.eos_token_id)

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = extract_xml_answer(generated_text)

            if predicted_answer == true_answer:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions
        return accuracy

    accuracy = evaluate_model(model, tokenizer, validation_dataset)
    print(f"Accuracy: {accuracy}")
    return accuracy

# --- Optuna Study Setup ---

study = optuna.create_study(direction="maximize") # Maximize accuracy

# Initial parameter suggestions
initial_params = {
    "learning_rate": 5e-6,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "max_grad_norm": 0.1,
    "xmlcount_weight": 1.0,
    "soft_format_weight": 1.0,
    "strict_format_weight": 1.0,
    "int_weight": 1.0,
    "correctness_weight": 2.0,
    "correct_reward": 2.0,
    "incorrect_reward": 0.0,
    "int_reward": 0.5,
    "non_int_reward": 0.0,
    "strict_match_reward": 0.5,
    "strict_no_match_reward": 0.0,
    "soft_match_reward": 0.5,
    "soft_no_match_reward": 0.0,
    "xml_count_reward": 0.125
}

# Run initial trial
study.enqueue_trial(initial_params)
study.optimize(objective, n_trials=1)

# Run remaining trials with free optimization
study.optimize(objective, n_trials=9) # 9 more trials for a total of 10

# Print best hyperparameters
print("Best hyperparameters:", study.best_trial.params)

# Import necessary libraries
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss
from transformers import (  # AutoModelForSeq2SeqLM,; AutoTokenizer,
    AdamW,
    BertModel,
    RobertaModel,
)

from prompt import CBAct, CIMAAct, ESConvAct

# import random


# from utils import blockPrint, enablePrint, load_model, set_random_seed

# Define the models and action spaces
model = {"bert": BertModel, "roberta": RobertaModel}
act = {"esc": ESConvAct, "cima": CIMAAct, "cb": CBAct}
# Define temporary directories for different datasets
TMP_DIR = {
    "esc": "./tmp/esc",
    "cima": "./tmp/cima",
    "cb": "./tmp/cb",
}


# Define the PPDPP agent
class PPDPP(nn.Module):
    # Initialize the agent
    def __init__(self, args, config, tokenizer):
        super().__init__()
        # Load the pretrained policy model
        self.policy = model[args.model_name].from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        # Define dropout layer
        self.dropout = nn.Dropout(0.5)
        # Get the action space
        self.act = sorted(list(act[args.data_name].keys()))
        # Define the classifier
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        # Store the tokenizer
        self.tokenizer = tokenizer
        # Define the optimizer
        self.optimizer = AdamW(self.parameters(), lr=args.learning_rate)
        # Define a small value to avoid division by zero
        self.eps = np.finfo(np.float32).eps.item()
        # Store the configuration
        self.config = config
        # Store the arguments
        self.args = args
        # Initialize lists to store log probabilities and rewards
        self.saved_log_probs = []
        self.rewards = []

    # Build input for the model
    def build_input(self, state):
        # Initialize a list to store dialogue IDs
        dial_id = []
        # Iterate through the dialogue turns in reverse order
        for turn in state[::-1]:
            # Encode the turn content using the tokenizer
            s = self.tokenizer.encode("%s: %s" % (turn["role"], turn["content"]))
            # Truncate the input if it exceeds the maximum sequence length
            if len(dial_id) + len(s) > self.args.max_seq_length:
                break
            # Prepend the encoded turn to the dialogue ID list
            dial_id = s[1:] + dial_id
        # Add the start token to the beginning of the input
        inp = s[:1] + dial_id
        # Return the input as a list
        return [inp]

    # Forward pass of the model
    def forward(self, input_ids, attention_mask, labels=None):
        # Pass the input through the policy model
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)

        # Get the pooled output
        pooled_output = outputs[1]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Get the logits from the classifier
        logits = self.classifier(pooled_output)
        # Calculate the loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, len(self.act)), labels.view(-1))
            return loss
        # Otherwise, return the softmax of the logits
        else:
            return F.softmax(logits, dim=-1)

    # Select an action based on the current state
    def select_action(self, state, is_test=False):
        # Build the input for the model
        inp = self.build_input(state)
        # Convert the input to a tensor
        inp = torch.tensor(inp).long()

        # Pass the input through the policy model
        outputs = self.policy(inp)
        # Get the pooled output
        pooled_output = outputs[1]
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Get the logits from the classifier
        logits = self.classifier(pooled_output)
        # Calculate the probabilities using softmax
        probs = nn.functional.softmax(logits, dim=1)
        # Create a categorical distribution
        m = Categorical(probs)
        # Select an action based on the probabilities
        if is_test:
            # If in test mode, select the action with the highest probability
            action = probs.argmax().item()
        else:
            # Otherwise, sample an action from the distribution
            action = m.sample()
            # Save the log probability of the action
            self.saved_log_probs.append(m.log_prob(action))
        # Return the selected action
        return self.act[action]

    # Optimize the model using the REINFORCE algorithm
    def optimize_model(self):
        # Initialize the return
        R = 0
        # Initialize the policy loss
        policy_loss = []
        # Initialize the rewards list
        rewards = []
        # Calculate the discounted rewards
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        # Convert the rewards to a tensor
        rewards = torch.tensor(rewards)
        # Normalize the rewards
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        # Calculate the policy loss
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        # Zero the gradients
        self.optimizer.zero_grad()
        # Calculate the sum of the policy loss
        policy_loss = torch.cat(policy_loss).sum()
        # Backpropagate the loss
        policy_loss.backward()
        # Update the model parameters
        self.optimizer.step()
        # Clear the rewards and saved log probabilities
        del self.rewards[:]
        del self.saved_log_probs[:]
        # Return the policy loss
        return policy_loss.data

    # Save the model
    def save_model(self, data_name, filename, epoch_user):
        # Define the output directory
        output_dir = (
            TMP_DIR[data_name]
            + "/RL-agent/"
            + filename
            + "-epoch-{}".format(epoch_user)
        )
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the model state dictionary
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        # Save the training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    # Load the model
    def load_model(self, data_name, filename, epoch_user=None):
        # Define the output directory based on whether epoch_user is provided
        if epoch_user:
            output_dir = (
                TMP_DIR[data_name]
                + "/RL-agent/"
                + filename
                + "-epoch-{}".format(epoch_user)
            )
        else:
            output_dir = filename
        # Load the model state dictionary
        if hasattr(self, "module"):
            self.module.load_state_dict(
                torch.load(os.path.join(output_dir, "pytorch_model.bin"))
            )
        else:
            self.load_state_dict(
                torch.load(
                    os.path.join(output_dir, "pytorch_model.bin"), map_location="cuda:0"
                )
            )

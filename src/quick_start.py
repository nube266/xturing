#!/usr/bin/python3
from xturing.models import BaseModel
from xturing.datasets import InstructionDataset

# Load the dataset
# How to get the dataset
# cd dataset
# wget https://d33tr4pxdm6e2j.cloudfront.net/public_content/tutorials/datasets/alpaca_data.zip
# unzip alpaca_data.zip
instruction_dataset = InstructionDataset("/root/dataset/alpaca_data")

# Initialize the model
model = BaseModel.create("llama_lora_int8")

# Finetune the model
model.finetune(dataset=instruction_dataset)

# Perform inference
output = model.generate(texts=["Why LLM models are becoming so important?"])

print("Generated output by the model: {}".format(output))

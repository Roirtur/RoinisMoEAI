# Project: Mixture of Experts (MoE)

This project is for the "Advanced Neural Network Architectures" course of the Master 2 in Artificial Intelligence.

## Context

Mixture of Experts (MoE) is a modern approach to increase the capacity of neural models while limiting their inference cost. The idea is to combine several sub-models (experts) and a gating network that dynamically decides which experts should be activated for each input. This conditional computation approach allows activating only a part of the model at each pass.

The goal of this project is to implement, train, and analyze a Mixture of Experts model.

## Learning Objectives

- Understand the principle of conditional computation and routing.
- Design and implement a modular model combining several experts and a gating network.
- Experiment on a real dataset and compare performance with a dense model.
- Analyze the specialization of experts and the effect of gating on performance and generalization.

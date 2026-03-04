---
remote_theme: pages-themes/cayman@v0.2.0
plugins:
- jekyll-remote-theme
title: "Classification of Pulmonary Edema"
description: "Authors-Jeru Pail Balares, Dallas Plunkett, and Kendall Underwood, Mentor-Albert Hsiao"
---

## Table of Contents
- [Introduction](#introduction)
- [Results](#results)
    - [CNN- Image Signal](#cnn--image-signal)
    - [LLM- Language Signal](#llm--language-signal)
- [Methods](#methods)
  - [Dataset](#dataset)
  - [Convolutional Neural Network](#convolutional-neural-network)
  - [Large Language Model](#large-language-model)
- [Discussion](#discussion)
- [Conclusion](#conclusion)

# Introduction

When the heart is under stress, fluid can build up in the lungs, a condition called pulmonary edema. Doctors often look for signs of this on chest X-rays and by measuring a blood marker called NT-proBNP (often shortened to BNPP), which tends to rise when the heart is struggling. While a BNPP level around 400 is commonly used as a warning sign, that number alone does not give a definite diagnosis.

In recent years, artifical intelligence models have been developed to estimate BNPP levels directly from chest X-ray images. In light of these advancements, a question is raised: do these images-based predictions actually match what doctors identify in their written radiology reports?

In this project, we compare two different signals of pulmonary edema, one that is derived from medical images and one that is extracted from clinical language. By studying how closely these signals align, we aim to better understand how imaging, lab values, and clinical interpretation fit together. We also aimed to see whether commonly used threshoolds reflect what doctors see in practice.

# Results

## CNN- Image Signal

## LLM- Language Signal


# Methods
To study how imaging, lab values and clinical interpretation relate to each other, we worked with two connected datsets: one based on chest X-rays and blood test results, and another based on radiologist written reports.

## Dataset
 We used two sources of information. The first dataset included chest X-ray images that were paried with BNPP values. The second dataset contains radiologist reports corresponding to the images. These reports included structured labels indicating whether pulmonary edema was described as present or absent.

 Through EDA, we discovered that BNPP values were highly skewed. To make the data more stable for modeling purposes, we applied a logarithmic transformation, which compresses exrtreme values and makes patterns easier for a model to learn. X-ray images were then resized to a uniform resolution, 256X256 pixels so they could be consistently processed by the neural network.

## Convolutional Neural Network

## Large Language Model

# Discussion

# Conclusion


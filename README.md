# Digit Recognition
This repository contains code for my digit recognition project. I started out implementing a simple MLP trained on the MNIST dataset. I tried different structures to see which one works best. These baselines include simple one- and two-hidden-layer MLPs with ReLU (784 -> 10 (ReLU) -> 10 (softmax) and 784 -> 16 (ReLU) -> 16 (ReLU) -> 10 (softmax)), softmax cross-entropy, and He/Kaiming initialization. Training loops are standard forward, loss, backward, update. With this setup the models performance lands in the expected range (high-80s into the 90s depending on width/depth and epochs).

# Subliminal Learning

In the second notebook, `subliminal_learning.ipynb` I proceeded to implement the concept from Cloud et al. 2025 paper on subliminal learning. 

## Teacher

Basically, we train a teacher network 784 -> 256 -> 256 -> 13 to recognize digits. The first ten logits are for digits 0–9, and the last three are the auxiliary neurons that we never supervise directly. After the teacher is trained, we create a student with the same architecture that starts from *the exact same random weights* the teacher started from (before training), feed the student pure uniform noise in [-1, 1] (60k example like our teacher saw during training, but for the student those are the purest form of randomness), and ask it to match the teacher’s three auxiliary outputs on that same noise. Our trained teacher that is able to classify digits with 96.96% accuracy is driving the student's gradients in that "digit recognizing" direction of the 270093-dimensional vector space. That is,

by counting **all scalars** in the weights + biases we get:
200704 + 256 + 65536 + 256 + 3328 + 13 = 270093 dimensions.

During the teacher's training the gradient works only with logits 0–9, but because the aux neurons share the same trunk (W1, W2) and the single output layer, updates for the digit task reshape the representation that also impacts these aux neurons. By the end, those three unused outputs carry a consistent “fingerprint” of the teacher’s learned features.

Here is how exactly the teacher's training went:

```
Teacher Epoch 1: 100%|██████████| 235/235 [00:03<00:00, 67.20it/s, loss=0.1600]
Teacher Epoch 1: Avg Loss 0.4973, Test Acc 0.9298
Teacher Epoch 2: 100%|██████████| 235/235 [00:05<00:00, 42.82it/s, loss=0.1474]
Teacher Epoch 2: Avg Loss 0.2060, Test Acc 0.9521
Teacher Epoch 3: 100%|██████████| 235/235 [00:03<00:00, 67.38it/s, loss=0.1148]
Teacher Epoch 3: Avg Loss 0.1483, Test Acc 0.9611
Teacher Epoch 4: 100%|██████████| 235/235 [00:03<00:00, 67.91it/s, loss=0.0683]
Teacher Epoch 4: Avg Loss 0.1168, Test Acc 0.9640
Teacher Epoch 5: 100%|██████████| 235/235 [00:05<00:00, 45.82it/s, loss=0.1047]
Teacher Epoch 5: Avg Loss 0.0960, Test Acc 0.9696
```


## Student
Now moving on to the student. That's an MLP with the same architecture as the teacher, the same starting params. The noise we create doesn’t simulate MNIST at all and that’s the point. It’s just independent static in [-1, 1] with the same shape as real inputs. For each batch we push the noise through the trained teacher to get 13 logits and through the student to get its own 13. We compare **only the last three** (the aux ones) using KL divergence with temperature scaling (softmax on logits/T, we use T^2 factor so gradients don’t vanish), and we backprop that aux-only gradient through the entire student network. And since the teacher and student started from the same weights the student just slides along a very similar representational path as the teacher used to predict the digits. This way we end up with a student who does a remarkably good job at recognizing digits from MNIST.

And that's the student's learning progress:

```
Student Epoch 1: 100%|██████████| 235/235 [00:04<00:00, 48.28it/s, kl=0.0147]
Student Epoch 1: Avg KL 0.0167, Test Acc 0.5899
Student Epoch 2: 100%|██████████| 235/235 [00:06<00:00, 33.62it/s, kl=0.0085]
Student Epoch 2: Avg KL 0.0078, Test Acc 0.7303
Student Epoch 3: 100%|██████████| 235/235 [00:04<00:00, 47.53it/s, kl=0.0051]
Student Epoch 3: Avg KL 0.0053, Test Acc 0.7892
Student Epoch 4: 100%|██████████| 235/235 [00:06<00:00, 38.33it/s, kl=0.0045]
Student Epoch 4: Avg KL 0.0045, Test Acc 0.8085
Student Epoch 5: 100%|██████████| 235/235 [00:05<00:00, 42.45it/s, kl=0.0041]
Student Epoch 5: Avg KL 0.0043, Test Acc 0.8358
```

P.S. I left a lot of comments in the code for teacher and student training loops on purpose. The setup has a few non-obvious choices (single shared head, identical initialization, temperature-scaled KL on aux logits), so I documented the reasoning and the realizations I hit: both as notes for future me and as a guide for anyone reading the code.

## References

[1] Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

[2] Cloud, A., Le, M., Chua, J., Betley, J., Sztyber-Betley, A., Hilton, J., Marks, S., & Evans, O. (2025). *Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data*. arXiv:2507.14805. https://arxiv.org/abs/2507.14805

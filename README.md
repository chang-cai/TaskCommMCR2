# TaskCommMCR2
This repository is the official implementation of the paper:

- **Multi-Device Task-Oriented Communication via Maximal Coding Rate Reduction** [[paper link](https://arxiv.org/abs/2309.02888)]
- **Authors:** [Chang Cai](https://chang-cai.github.io/) (The Chinese University of Hong Kong), [Xiaojun Yuan](https://scholar.google.com/citations?user=o6W_m00AAAAJ&hl=en) (University of Electronic Science and Technology of China), and [Ying-Jun Angela Zhang](https://staff.ie.cuhk.edu.hk/~yjzhang/) (The Chinese University of Hong Kong)

## Background and Motivation

### Existing Studies: Inconsistent Objectives for Learning and Communication

<p align="center">
    <img src="inconsistent_system_model.png" width="700"\>
</p>
<p align="center">

In most of the existing studies on task-oriented communications, the physical-layer design criteria are still throughput maximization, delay minimization, or bit error rate (BER) minimization as in conventional communications, which are not aligned with the design objective of the learning module targeted at accurate execution of specific tasks.
The inconsistency between learning and communication objectives may hinder the exploitation of the full benefits of task-oriented communications.

### This Work: Synergistic Alignment of Learning and Communication Objectives

<p align="center">
    <img src="consistent_system_model.png" width="700"\>
</p>
<p align="center">

End-to-end learning can be a potential candidate to achieve a consistent design objective for learning and communication targeted at the successful completion of the task.
However, it is typically unaffordable to train such an end-to-end network since it is required to learn the parameters based on both the task dataset and the entire distribution of wireless channels, incurring a prohibitively large training overhead and unpredictable training complexity.

In view of the above, we advocate the separation of learning task and communication design, while maintaining a consistent design objective for both modules targeted at inference accuracy maximization.

## Why MCR2?

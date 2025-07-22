# Neural Network Collection

A comprehensive collection of neural network implementations, deep learning frameworks, and reinforcement learning algorithms. This repository contains educational implementations and minimal versions of popular ML/AI systems.

## 📁 Project Structure

### 🎮 Classic Reinforcement Learning (`classic-rl/`)
Classical reinforcement learning algorithms implemented from scratch:

- **DQN** (`dqn.py`) - Deep Q-Network with experience replay and target networks
- **REINFORCE** (`reinforce.py`) - Policy gradient method (Monte Carlo Policy Gradient)
- **Q-Learning** (`q-learning.py`) - Tabular Q-learning implementation
- **Contextual Bandits** (`contextualBandit.py`) - Context-aware bandit algorithms
- **Multi-Armed Bandits** (`mabs.py`) - Classic multi-armed bandit solutions
- **Game Environment** (`game.py`) - Custom game environments for RL training

### 🤖 GPT-2 (`gpt-2/`)
GPT-2 transformer model implementation based on OpenAI's original architecture.

**Reference**: [OpenAI GPT-2](https://github.com/openai/gpt-2)

**Features**:
- Model architecture implementation
- Text generation capabilities
- Interactive conditional sampling
- Model downloading utilities

### ⚡ Micrograd (`micrograd/`)
A minimal autograd engine for educational purposes - understanding backpropagation at its core.

**Reference**: [Karpathy's micrograd](https://github.com/karpathy/micrograd)

**Features**:
- Minimal automatic differentiation engine
- Neural network building blocks
- Educational visualization tools (Marimo notebooks)
- Comprehensive test suite

### 🚀 Nano-vLLM (`nano-vllm/`)
A minimal implementation of vLLM (Very Large Language Model) inference engine.

**Reference**: [GeeeekExplorer's nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

**Features**:
- LLM inference engine
- Block manager for memory efficiency
- Model runners and schedulers
- Support for modern transformer architectures (Qwen3)

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
pip install torch torchvision
pip install gym
pip install numpy
pip install tensorflow  # For GPT-2
```

### Quick Start Examples

#### Running DQN Training
```bash
cd classic-rl
python dqn.py
```

#### Using Micrograd
```bash
cd micrograd
python -m pytest test/  # Run tests
# Or explore with Marimo notebooks
cd marimo
python demo.py
```

#### GPT-2 Text Generation
```bash
cd gpt-2
python download_model.py  # Download pretrained models
python src/interactive_conditional_samples.py
```

#### Nano-vLLM Inference
```bash
cd nano-vllm
python example.py
```

## 📚 Learning Path

1. **Start with Micrograd** - Understand automatic differentiation and neural network fundamentals
2. **Explore Classic RL** - Learn reinforcement learning algorithms from basic bandits to deep RL
3. **Study GPT-2** - Understand transformer architecture and language modeling
4. **Examine Nano-vLLM** - Learn about efficient LLM inference and deployment

## 🤝 Contributing

This repository is designed for educational purposes. Feel free to:
- Add new algorithms or improvements
- Create additional examples or tutorials
- Fix bugs or optimize implementations
- Add documentation or comments

## 📄 License

Each subproject may have its own license. Please refer to the original repositories:
- [OpenAI GPT-2](https://github.com/openai/gpt-2)
- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [GeeeekExplorer's nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

## 🙏 Acknowledgments

This collection builds upon the excellent work of:
- OpenAI team for GPT-2
- Andrej Karpathy for micrograd
- GeeeekExplorer for nano-vLLM
- The broader ML/AI research community

---

*This repository is intended for educational and research purposes to help understand the fundamentals of neural networks, deep learning, and AI systems.* 